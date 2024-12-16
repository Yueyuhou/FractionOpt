import os
import pathlib
from pathlib import Path
import datetime
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter
import math
import pandas as pd
import argparse

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net

from Net.ACnet import Actor, Critic

from Utils.readAndWrite.read_config_file import get_config_info
import torch
from torch import nn

import RT_env
import RT_env.envs.RT_env_V1

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

torch.cuda.set_device(0)


def sample_ppo_params(trial, config_info):
    """Sampler for PPO hyperparameters."""
    hyper_parameters = {'task': config_info['training_config']['task']}
    hidden_sizes = trial.suggest_categorical("hidden_sizes", ["small", "medium", "large"])
    hidden_sizes = {
        "small": [64, 64, 64, 64],
        "medium": [128, 128, 128, 128],
        "large": [256, 256, 256, 256],
    }[hidden_sizes]
    hyper_parameters['hidden_sizes'] = hidden_sizes

    activation_fn = trial.suggest_categorical(
        'activation_fn', ['relu', 'leaky_relu'])
    activation_fn = {"relu": nn.ReLU,
                     "leaky_relu": nn.LeakyReLU}[activation_fn]
    hyper_parameters['activation_fn'] = activation_fn

    hyper_parameters['lr'] = trial.suggest_loguniform("lr", 1e-5, 1e-4)
    hyper_parameters['eps_clip'] = trial.suggest_categorical("eps_clip", [0.2, 0.3, 0.4])
    hyper_parameters['vf_coef'] = trial.suggest_uniform("vf_coef", 0, 1)
    hyper_parameters['ent_coef'] = trial.suggest_loguniform("ent_coef", 0.00001, 0.1)
    hyper_parameters['max_grad_norm'] = trial.suggest_categorical(
        "max_grad_norm", [0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    hyper_parameters['gae_lambda'] = trial.suggest_categorical(
        "gae_lambda", [0.9, 0.93, 0.96, 0.99, 1.0])
    hyper_parameters['batch_size'] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512])
    hyper_parameters['gamma'] = trial.suggest_categorical(
        "gamma", [0.5, 0.7, 0.9, 0.95, 0.99, 0.999])
    hyper_parameters['reward_normalization'] = False
    hyper_parameters['repeat_per_collect'] = trial.suggest_categorical(
        "repeat_per_collect", [8, 16, 32, 64])

    return hyper_parameters


def test_ppo(config_info, hyper_parameters):
    print(config_info['training_config']['training_mode'])
    data_path = Path(config_info["data_config"]['data_path'])
    df = pd.read_excel(data_path)

    task = config_info['training_config']['task']
    env = gym.make(task, df_=df, config_info=config_info)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n

    hidden_sizes = config_info['training_config']['hidden_sizes']
    device = config_info['training_config']['device']

    np.random.seed(1)
    seed_list = np.random.choice(10000, 2000, replace=False)
    train_seed_list = seed_list[:1000].tolist()
    test_seed_list = seed_list[1000:].tolist()

    # SubprocVectorEnv  DummyVectorEnv
    train_envs = SubprocVectorEnv([lambda: gym.make(task, df_=df, config_info=config_info)
                                   for _ in range(config_info['training_config']['training_num'])])

    test_envs = SubprocVectorEnv([lambda: gym.make(task, df_=df, config_info=config_info)
                                  for _ in range(config_info['training_config']['test_num'])])

    train_envs.seed(train_seed_list[:config_info['training_config']['training_num']])
    test_envs.seed(test_seed_list[:config_info['training_config']['test_num']])

    train_envs = VectorEnvNormObs(train_envs, update_obs_rms=True)
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=True)

    # model

    # norm_layer = None
    # activation = nn.LeakyReLU nn.ReLU
    # act_args = 0.3
    net = Net(state_shape, hidden_sizes=hidden_sizes, activation=hyper_parameters['activation_fn'], device=device)
    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    lr = config_info['training_config']['lr']
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    LR_lambda = lambda epoch: 0.99 ** epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, LR_lambda)

    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        action_space=env.action_space,
        eps_clip=config_info['training_config']['eps_clip'],
        dual_clip=config_info['training_config']['dual_clip'],
        value_clip=config_info['training_config']['value_clip'],
        advantage_normalization=config_info['training_config']['norm_adv'],
        recompute_advantage=config_info['training_config']['recompute_adv'],
        vf_coef=config_info['training_config']['vf_coef'],
        ent_coef=config_info['training_config']['ent_coef'],
        max_grad_norm=config_info['training_config']['max_grad_norm'],
        gae_lambda=config_info['training_config']['gae_lambda'],
        max_batchsize=config_info['training_config']['max_batchsize'],
        discount_factor=config_info['training_config']['gamma'],
        reward_normalization=config_info['training_config']['rew_norm'],
        deterministic_eval=config_info['training_config']['deterministic_eval'],
        observation_space=env.observation_space,
        action_scaling=False,
        lr_scheduler=lr_scheduler)

    # collector
    buffer_size = config_info['training_config']['buffer_size']
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    l_name = config_info["training_config"]["log_name"]
    mode = config_info['training_config']['training_mode']
    log_name = os.path.join(task[-2:] + l_name + mode, now)
    log_dir = Path(config_info['training_config']['logdir'])
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    log_path = os.path.join(log_dir, log_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # trainer
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config_info['training_config']['epoch'],
        step_per_epoch=config_info['training_config']['step_per_epoch'],
        repeat_per_collect=hyper_parameters['repeat_per_collect'],
        episode_per_test=config_info['training_config']['test_num'],
        batch_size=hyper_parameters['batch_size'],
        step_per_collect=config_info['training_config']['step_per_collect'],
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_best_fn=None,
        logger=logger,
        save_checkpoint_fn=None,
        resume_from_log=config_info['training_config']['resume'],
        verbose=True,
        show_progress=True
    )

    for epoch, epoch_stat, info in result:
        yield epoch, epoch_stat, info


def objective(trial, config_info):
    hyper_parameters = sample_ppo_params(trial, config_info)
    i = 0
    loss = 0
    for epoch, epoch_stat, info in test_ppo(config_info, hyper_parameters):
        i += 1
        loss = epoch_stat['test_reward']
        trial.report(loss, i)
        if trial.should_prune():
            raise optuna.TrialPruned
        # print(epoch_stat['test_reward'], epoch_stat['test_reward_std'])
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='opt_tuner',
        description='opt_tuner',
    )
    parser.add_argument('--path', default=None)
    parser.add_argument('--train_mode', default='de-escalation', help='de-escalation or escalation')
    parser.add_argument('--vol_threshold', default=73.0, help='cutoff vol_threshold')
    parser.add_argument('--log_name')
    parser.add_argument('--ab_ratio', type=float, default=10.0, help='alpha_beta_ratio')
    args = parser.parse_args()

    if args.path is None:
        config_path = Path(r"../Utils/config_env_HN.yml")
    else:
        config_path = Path(args.path)

    print("config_path is :", config_path)

    config_info = get_config_info(config_path)
    config_info['training_config']['training_mode'] = args.train_mode
    config_info['tumor_rt_parameters']['cutoff_response'] = args.vol_threshold
    config_info['training_config']['log_name'] = args.log_name
    config_info['tumor_rt_parameters']['ab_tumor'] = args.ab_ratio

    print(config_info['tumor_rt_parameters']['cutoff_response'])
    print(config_info['training_config']['log_name'])
    print(config_info['tumor_rt_parameters']['ab_tumor'])

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=10))
    study.optimize(lambda trial: objective(trial, config_info), n_trials=35)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv("../Data/frac_log/ppo_env2/opt_history/" + config_info['training_config']['log_name']
                                    + args.train_mode + ".csv")


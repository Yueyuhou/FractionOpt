import os
import pprint
import pathlib
from pathlib import Path
import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter
import math
import argparse

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from Net.ACnet import Actor, Critic

from Utils.readAndWrite.read_config_file import get_config_info
import torch
from torch import nn

import RT_env.envs.RT_env_V1

torch.cuda.set_device(0)


def train(config_info):
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
    train_envs = DummyVectorEnv([lambda: gym.make(task, df_=df, config_info=config_info)
                                   for _ in range(config_info['training_config']['training_num'])])
    train_envs.seed(train_seed_list[:config_info['training_config']['training_num']])

    if not config_info['training_config']['resume']:
        test_envs = DummyVectorEnv([lambda: gym.make(task, df_=df, config_info=config_info)
                                      for _ in range(config_info['training_config']['test_num'])])
        test_envs.seed(test_seed_list[:config_info['training_config']['test_num']])
    else:
        test_envs = DummyVectorEnv([lambda: gym.make(task, df_=df, config_info=config_info,
                                                     record_flag=True)
                                    for _ in range(config_info['training_config']['test_num'])])
        test_envs.seed(list(range(config_info['training_config']['test_num'])))

    train_envs = VectorEnvNormObs(train_envs, update_obs_rms=True)
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=True)

    # model

    # norm_layer = None
    # activation = nn.LeakyReLU nn.ReLU
    # act_args = 0.3
    net = Net(state_shape, hidden_sizes=hidden_sizes, activation=nn.ReLU, device=device)
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
    if config_info['training_config']['resume']:
        # load from existing checkpoint
        log_path_ = r"../Data/frac_log/ppo_env2/v1_train_fine_tune_de-escalation"
        print(f"Loading agent under {log_path_}")
        ckpt_path = os.path.join(log_path_, "checkpoint_500.pth")  # "best_policy.pth" "checkpoint_100.pth"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            policy.load_state_dict(checkpoint["model"])
            train_envs.set_obs_rms(checkpoint["train_obs_rms"])
            test_envs.set_obs_rms(checkpoint["train_obs_rms"])
            print("Successfully restore policy and optim.")
        else:
            raise "Fail to restore policy."
    else:
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

    def save_best_fn(policy):
        ckpt_path = os.path.join(log_path, f"best_policy.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "train_obs_rms": train_envs.get_obs_rms(),
                "test_obs_rms": test_envs.get_obs_rms()
            }, ckpt_path  # "obs_rms": train_envs.get_obs_rms()
        )

    def stop_fn(mean_rewards):
        pass

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        if epoch % 10 == 0:
            torch.save(
                {
                    "model": policy.state_dict(),
                    "train_obs_rms": train_envs.get_obs_rms(),
                    "test_obs_rms": test_envs.get_obs_rms()
                }, ckpt_path  # "obs_rms": train_envs.get_obs_rms()
            )
        return ckpt_path

    # trainer
    if not config_info['training_config']['resume']:
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=config_info['training_config']['epoch'],
            step_per_epoch=config_info['training_config']['step_per_epoch'],
            repeat_per_collect=config_info['training_config']['repeat_per_collect'],
            episode_per_test=config_info['training_config']['test_num']//10,
            batch_size=config_info['training_config']['batch_size'],
            step_per_collect=config_info['training_config']['step_per_collect'],
            train_fn=None,
            test_fn=None,
            stop_fn=None,
            save_best_fn=save_best_fn,
            logger=logger,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=config_info['training_config']['resume'],
            verbose=True,
            show_progress=True
        ).run()
        pprint.pprint(result)
    else:
        print("Begin to evaluating policy: ")
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=5000)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ppo_trainer',
        description='ppo_trainer',
    )
    parser.add_argument('--path', default=None)
    parser.add_argument('--train_mode', default='de-escalation', help='de-escalation or escalation')
    parser.add_argument('--vol_threshold', default=73.0, help='cutoff vol_threshold')
    parser.add_argument('--log_name', default='log')
    parser.add_argument('--ab_ratio', type=float, default=10.0, help='alpha_beta_ratio')
    args = parser.parse_args()

    if args.path is None:
        config_path = Path(r"../Utils/config_env_HN.yml")
    else:
        config_path = Path(args.path)

    config_info = get_config_info(config_path)
    config_info['training_config']['training_mode'] = args.train_mode
    config_info['tumor_rt_parameters']['cutoff_response'] = args.vol_threshold
    config_info['training_config']['log_name'] = args.log_name
    config_info['tumor_rt_parameters']['ab_tumor'] = args.ab_ratio

    train(config_info)
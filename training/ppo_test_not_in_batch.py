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
from tianshou.utils import RunningMeanStd

from Net.ACnet import Actor, Critic

from Utils.readAndWrite.read_config_file import get_config_info
import torch
from torch import nn

import RT_env

torch.cuda.set_device(0)


def test_per_patient(config_info, test_data_excel=None, df_=None):
    if test_data_excel is None and df_ is None:
        raise "Please provide a valid data source."

    if df_ is None:
        df = pd.read_excel(test_data_excel)

    if test_data_excel is None:
        df = df_

    task = config_info['training_config']['task']
    myenv = gym.make(task, df_=df, config_info=config_info, record_flag=True)
    myenv.seed(1)
    state_shape = myenv.observation_space.shape
    action_shape = myenv.action_space.n

    hidden_sizes = config_info['training_config']['hidden_sizes']
    device = config_info['training_config']['device']
    np.random.seed(1)
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
        action_space=myenv.action_space,
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
        observation_space=myenv.observation_space,
        action_scaling=False,
        lr_scheduler=lr_scheduler)

    # read log file.
    log_path_ = r"../Data/frac_log/ppo_env2/v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation/240527-143210"
    print(f"Loading agent under {log_path_}")
    ckpt_path = os.path.join(log_path_, "checkpoint_100.pth")  # "best_policy.pth" "checkpoint_100.pth"

    obs_rms = RunningMeanStd()
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(checkpoint["model"])
        obs_rms = checkpoint["train_obs_rms"]
        print("Successfully restore policy and optim.")
    else:
        raise "Fail to restore policy."

    # begin to test
    for i in range(df.shape[0]):
        obs, info = myenv.reset(options={'patient_idx': i})
        while True:
            norm_obs = obs_rms.norm(obs)
            act = policy(Batch(obs=np.array([norm_obs]), info=info)).act[0]
            obs, rew, terminated_flag, _, info = myenv.step(act)
            if terminated_flag:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ppo_trainer',
        description='ppo_trainer',
    )
    parser.add_argument('--config_path', default=None)
    parser.add_argument('--train_mode', default='de-escalation', help='de-escalation or escalation')
    parser.add_argument('--data_path', default='test_data_excel')
    args = parser.parse_args()

    if args.config_path is None:
        config_path = Path(r"../Utils/config_env_HN.yml")
    else:
        config_path = Path(args.config_path)

    config_info = get_config_info(config_path)
    config_info['training_config']['training_mode'] = args.train_mode

    # test_data_directory = Path(r"../Data/add_noise_para")
    # test_data_excel = test_data_directory.joinpath('n_v2_train_ppo_18o_D6a_HN_LRC_60Gy_de-escalation_noise_0.05.xlsx')
    test_data_excel = args.data_path

    test_per_patient(config_info, test_data_excel)

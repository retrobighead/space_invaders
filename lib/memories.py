# TODO: import 見直し

import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
from gym.spaces.box import Box

class RolloutStorage(object):
    """
        Advantage 学習するためのクラス
        Advantage Learning: 
            Q(s_t, a_t) の学習のために, 1ステップ先の行動価値関数 Q(s_{t+1}, a) より先のステップを使用する
            2ステップの場合, Q(s_t, a_t) -> R(t+1) + gamma * R(t+2) + (gamma^2) * max_a [Q(s_{t+2}, a)]
    """
    def __init__(self, num_steps, num_processes, observation_shape, device):
        self.observations = torch.zeros(num_steps+1, num_processes, *observation_shape).to(device)
        self.masks = torch.ones(num_steps+1, num_processes, 1).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, 1).long().to(device)
        
        # discounted reward sum
        self.returns = torch.zeros(num_steps+1, num_processes, 1).to(device)
        self.index = 0
    
    def insert(self, current_observations, action, reward, mask):
        """
            次の index に transition を格納する
        """
        self.observations[self.index + 1].copy_(current_observations)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)
        
        self.index = (self.index + 1) % NUM_ADVANCED_STEP
        
    def after_update(self):
        """
            Advantage する step 数が完了したら, 最新のものを index0 に格納
        """
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
    
    def compute_returns(self, next_value):
        """
            Advantage するステップ中の各ステップの割引報酬和を計算する
        """
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = GAMMA * self.returns[ad_step+1] + GAMMA * self.masks[ad_step+1] + self.rewards[ad_step]
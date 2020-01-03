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

# TODO: ハイパーパラメータの配置場所

lr = 7e-4
eps = 1e-5
alpha = 0.99

value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic # network of Net class
        
        ## if loading saved join parameters
        # filename = 'weight.pth'
        # params = torch.load(filename, map_location='cpu')
        # self.actor_critic.load_state_dict(params)
        
        # defines optimizing algorithm of neural network
        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)
        
    def update(self, rollouts, num_advanced_step, num_processes):
        '''
            Advanced 
        '''
        observation_shape = rollouts.observations.size()[2:]
        num_steps = num_advanced_step
        
        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *observation_shape),
            rollouts.actions.view(-1, 1))
        
        # size
        # rollouts.observations[:-1].view(-1, *observation_shape) -- torch.Size([80, 4, 84, 84])
        # rollouts.actions.view(-1, 1)                            -- torch.Size([80, 1])
        # values                                                  -- torch.Size([80, 1])
        # action_log_probs                                        -- torch.Size([80, 1])
        # dist_entropy                                            -- torch.Size([])
        
        values = values.view(num_steps, num_processes, 1)                      # -- torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)  # -- torch.Size([5, 16, 1])
        advantages = rollouts.returns[:-1] - values                            # -- torch.Size([5, 16, 1])
        value_loss = advantages.pow(2).mean()
        
        action_gain = (advantageous.detach()*action_log_probs).mean()
        total_loss = (value_loss*value_loss_coef - action_gain - dist_entropy*entropy_coef)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_clitic.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        
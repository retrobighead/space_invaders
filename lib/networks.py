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

def init(module, gain):
    '''
        this function initialize the join parameters of the network
    '''
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0) # 重みを全て0で初期化
    
    return module

class Flatten(nn.Module):
    '''
        this class flattens the output image of convolutional Layer
    '''
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Net(nn.Module):
    def __init__(self, n_out, num_stack_frame):
        super(Net, self).__init__()
        
        # function initializing the join parameters
        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))
        self.conv = nn.Sequential(
            # size = (input_size - kernel_size + 2*padding_size)/stride_size + 1
            init_(nn.Conv2d(num_stack_frame, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(64*7*7, 512)),
            nn.ReLU()
        )
        
        # update function initializing the join parameters
        def init_(module): return init(module, gain=1.0)
        self.critic = init_(nn.Linear(512, 1))
        
        def init_(module): return init(module, gain=0.01)
        self.actor = init_(nn.Linear(512, n_out))
        
        self.train()
        
    def forward(self, x):
        '''
            forward propagation of neural network
        '''
        inp = x / 255. 
        conv_output = self.conv(inp)
        critic_output = self.critic(conv_output)
        actor_output = self.actor(critic_output)
        
        return critic_output, actor_output
    
    def act(self, x):
        '''
            decide stochastically action from observation x
        '''
        val, actor_output = self(x)
        probs = F.softmax(actor_output, dim=1)
        action = probs.multinomial(num_samples=1)
        
        return action
    
    def get_value(self, x):
        '''
            compute state value from observation x
        '''
        val, _ = self(x)
        
        return val
        
    def evaluate_actions(self, x, actions):
        '''
            Return 
            - state value
            - log probability and entropy of actions
            from observation x
        '''
        val, actor_output = self(x)
        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        
        probs = F.softmax(actor_output, dim=1)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        
        return value, action_log_probs, dist_entropy
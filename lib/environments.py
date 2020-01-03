# TODO: import 見直し

import numpy as np
from collections import deque
from tqdm import tqdm # progress bar 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
from gym.spaces.box import Box

import cv2
cv2.ocl.setUseOpenCL(False)

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

import networks
import agents
import memories


"""
    Main Environment
""" 
class Environment:
    def run(self, env_name, num_processes, num_stack_frame, num_advanced_step, num_updates):
        
        def make_env(env_id, seed, rank):
            def _thunk():
                """
                    _thunk がマルチプロセス環境の SubprocVecEnv を実行するのに必要らしい
                """

                env = gym.make(env_id)
                env = NoOpResetEnvironment(env, no_op_max=30)
                env = MaxAndSkipEnvironment(env, skip=4)
                env.seed(seed + rank)
                env = EpisodicLifeEnvironment(env)
                env = WarpFrame(env)
                env = WrapByTorch(env)

                return env

            return _thunk
        
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        print(device)
        
        seed_num = 1
        torch.manual_seed(seed_num)
        if use_cuda:
            torch.cuda.manual_seed(seed_num)
        
        # construct environments
        torch.set_num_threads(seed_num)
        envs = [ make_env(env_name, seed_num, i) for i in range(num_processes) ]
        envs = SubprocVecEnv(envs)  # マルチプロセス実行環境
        
        # create instance of Brain class shared by all agents
        n_out = envs.action_space.n
        actor_critic = networks.Net(n_out, num_stack_frame).to(device)
        global_brain = agents.Brain(actor_critic)
        
        # create variables 
        observation_shape = envs.observation_space.shape                                    # (1, 84, 84)
        observation_shape = (observation_shape[0]*num_stack_frame, *observation_shape[1:])  # (4, 84, 84)
        
        current_observation = torch.zeros(num_processes, *observation_shape).to(device)
        rollouts = memories.RolloutStorage(num_advanced_step, num_processes, observation_shape, device)
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])
        
        # initialize and start the environment 
        observation = envs.reset()
        observation = torch.from_numpy(observation).float()     # torch.Size([16, 1, 84, 84])
        current_observation[:, -1:] = observation
        
        # 
        rollouts.observations[0].copy_(current_observation)
        
        # run
        for j in tqdm(range(num_updates)):
            for step in range(num_advanced_step):
                # calculate action
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])
                    
                cpu_actions = action.squeeze(1).cpu().numpy() # torch.Tensor -> numpy.array
                
                observation, reward, done, info = envs.step(cpu_actions)
                
                # translate reward into torch.Tensor
                # change size (16,) -> (16, 1)
                reward = np.expand_dims(np.stack(reward), 1)
                reward = torch.from_numpy(reward).float()
                episode_rewards += reward
                
                # for parallel execution environments, mask is 0(done) or 1(not done)
                masks = torch.FloatTensor([[0.] if done_ else [1.] for done_ in done])
                
                # update rewards at the last trial
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
                
                masks = masks.to(device)
                
                # apply masks to current observation
                current_observation *= masks.unsqueeze(2).unsqueeze(2)
                
                # stacking the frames: torch.Size([16, 1, 84, 84])
                observation = torch.from_numpy(observation).float()
                current_observation[:, :-1] = current_observation[:, 1:]
                current_observation[:, -1:] = observation
                
                rollouts.insert(current_observation, action.data, reward, masks)
                
            # calculating the state value expected from advanced last step
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1]).detach()
            
            rollouts.compute_returns(next_value)
            
            # update network and rollouts
            global_brain.update(rollouts)
            rollouts.after_update()
            
            # logs
            if j%100 == 0:
                print("finished frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                      .format(j*num_processes*num_advanced_step, 
                              final_rewards.mean(), 
                              final_rewards.median(), 
                              final_rewards.min(),
                              final_rewards.max()))
            
            if j%12500 == 0:
                torch.save(global_brain.actor_critic.state_dict(), 'weight_' + str(j) + '.pth')
            
        torch.save(global_brain.actor_critic.state_dict(), 'weight_end.pth')

""" 
    Environments
"""
class NoOpResetEnvironment(gym.Wrapper):
    def __init__(self, env, no_op_max=30):
        """ 
            No-Operation 環境の実装
            環境リセット後に何もしない状態を [0, no_op_max] の間の数フレームだけ続けることで, 
            様々な状態からの学習を開始するための設定
            
            Parameters: 
                env (gym environment) -- OpenAI Gym の Environment 環境
                no_op_max (int)       -- [0, no_op_max] のランダムでフレーム数が設定される 
        """
        gym.Wrapper.__init__(self, env)
        
        self.no_op_max = no_op_max
        self.override_num_loops = None
        self.no_action = 0
        
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        """ 
            環境のリセット (env.reset()) を実行後, 
            何もしない状態を [0, no_op_max] ステップだけ実行する
        """
        self.env.reset(**kwargs)
        
        if self.override_num_loops is not None:
            noops = self.override_num_loops
        else:
            noops = np.random.randint(1, self.no_op_max + 1)
        assert noops > 0
        
        observation = None
        for _ in range(noops):
            observation, _, done, _ = self.env.step(self.no_action)
            if done:
                self.env.reset(**kwargs)
        
        return observation
    
    def step(self, action):
        return self.env.step(action)
    

class EpisodicLifeEnvironment(gym.Wrapper):
    def __init__(self, env):
        """
            Episodic Life 環境の実装
            Agentが複数機を持つ場合, 毎度リセットすると同じ状態からの学習に偏ってしまうため, 
            1機減っても環境をリセットせず, 継続して学習を行う
            
            Parameters: 
                env (gym environment) -- OpenAI Gym の Environment 環境
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        """
            done を更新して返り値を設定
        """
        observation, reward, done, info = self.env.step(action)
        self.was_real_done = done
        
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and 0 < lives:
            done = True
        
        self.lives = lives 
        return observation, reward, done, info
        
    def reset(self, **kwargs):
        """
            全ての機がなくなった場合, 完全にリセットする
        """
        if self.was_real_done:
            observation = self.env.reset(**kwargs)
        else:
            observation, _, _, _ = self.env.step(0)
            
        self.lives = self.env.unwrapped.ale.lives()
        return observation

    
class MaxAndSkipEnvironment(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
            高フレームレートで進行するゲームについて, 
            数フレーム単位で行動をまとめることで学習を簡単化する
        """
        gym.Wrapper.__init__(self, env)
        self._observation_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        
    def step(self, action):
        """
            action を self._skip 分繰り返す
            reward は合計して, 最後の2フレーム分の observation を最大化する
        """
        total_reward = 0.
        done = None
        
        for i in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._observation_buffer[0] = observation
            if i == self._skip - 1:
                self._observation_buffer[1] = observation
            total_reward += reward
            
            if done:
                break
        
        max_observation = self._observation_buffer.max(axis=0)
        
        return max_observation, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


"""
    Observations
""" 
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
            Observation が画像なので, 84x84 のグレイスケール画像に変換する
        """
        gym.ObservationWrapper.__init__(self, env)
        self.image_width = 84
        self.image_height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, 1), dtype=np.uint8)
        
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return frame[:, :, None]
    
class WrapByTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
            PyTorch のミニバッチのインデックス順に変更する
            [縦, 横, 色]
        """
        super(WrapByTorch, self).__init__(env)
        observation_shape = self.observation_space.shape
        self.observation_space = Box(self.observation_space.low[0, 0, 0],
                                     self.observation_space.high[0, 0, 0],
                                     [observation_shape[2], observation_shape[1], observation_shape[0]],
                                     dtype=self.observation_space.dtype)
    
    def observation(self, observation):
        return observation.transpose(2, 0, 1)
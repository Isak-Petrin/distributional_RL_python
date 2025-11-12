
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.envs.registration import register
import math

class RandomWalk(gym.Env):
    def __init__(
        self,     
        param: dict | None = None,
      ):
        self.l = 8
        self.start = self.l // 2
        self.observation_space = gym.spaces.Discrete(self.l)
        self.action_space = gym.spaces.Discrete(2)
    
    def _get_obs(self):
        return self.state
    def _get_info(self):
        return {'l': self.l}

        
    def reset(self,seed=None, options=None):
        self.state = self.start
        obs = self._get_obs()
        info = self._get_info()
        return obs,info
    
    def step(self, action: int):
        
        self.done = False
        self.reward = -1
        
        if action:
            self.state += 1
        else:
            self.state -= 1
        
        if self.state in (0, self.l-1):
            self.done = True
            self.reward = np.random.normal(loc = 5)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, self.reward, self.done, False, info
    
        
        
        
    
        
        
        
    
    
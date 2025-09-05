
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.envs.registration import register

class ToyEnv(gym.Env):
    def __init__(
        self,     
        param: dict | None = None,
      ):
        
        if param == None:
            self.gamma = 0.9
            
            self.ps = 0.1
            self.rs = 0.5
            
            self.pr = 0.05
            self.rr = 0.5
        #0 = not risky, 1 = risky
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
    
    def _get_obs(self):
        return {"agent": self.state}
    def _get_info(self):
        return {'gamma': self.gamma, 'ps': self.ps, 'rs': self.rs, 'pr': self.pr, 'rr': self.pr}

        
    def reset(self,seed=None, options=None):
        self.state = 0
        observations = self._get_obs()
        info = self._get_info()
        return observations,info
    
    def step(self, action: int):
        
        self.done = False
        
        if action:
            self.reward = self.rr
            if self.pr > np.random.random():
                self.done = True
                self.state = 1
        else:
            self.reward = self.rs
            if self.ps > np.random.random():
                self.done = True
                self.state = 1
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, self.reward, self.done, False, info 
        
        
            
        
        
        
        
        
    
        
        
        
    
    
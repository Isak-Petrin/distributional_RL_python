from collections import defaultdict
import gymnasium as gym
import numpy as np


class SimplePolicy:
    def __init__(
        self,
        env: gym.Env,
        policy: int,
    ):
        self.policy = policy
        self.env = env
        
    def get_action(self):
        if self.policy:
            return 1
        else:
            return 0
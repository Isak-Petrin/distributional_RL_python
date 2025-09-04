import gymnasium as gym
import sys
import gymnasium_env
import numpy as np
from SimplePolicyAgent import SimplePolicy
import matplotlib.pyplot as plt
from tqdm import tqdm

class O_CatTD:
    def __init__(
        self,
        env: gym.Env,
        alpha: int,
        theta_range: list[int,int],
        resolution: int,
        policy: int,
        gamma: float,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.min_theta = theta_range[0]
        self.max_theta = theta_range[1]
        self.resolution = resolution
        self.theta_lst = np.linspace(self.min_theta, self.max_theta, resolution)
        self.m = len(self.theta_lst)
        self.p = [np.ones(self.m) * (1/self.m) for _ in range(self.env.observation_space.n)]
        self.policy = policy
        
    def find_i_star(self,g):
        bool_lst = self.theta_lst <= g
        i_star = np.where(bool_lst)[0].max()
        return i_star
                
        
    def learn_distribution(self, n_episodes: int):
        for episode in tqdm(range(n_episodes)):
            
            obs, info = self.env.reset()
            s = obs['agent']
            if self.policy:
                a = 1
            else:
                a = 0
            done = False
            g = 0
            t = 0
            while not done:
                p_tilde = np.zeros(self.m)
                next_obs, reward, done, _ , info, = self.env.step(a)
                sn = next_obs['agent']
                for j in range(self.m):
                    if done:
                        g = reward
                    else:
                        g = reward + self.gamma * self.theta_lst[j]
                    if g <= self.min_theta:
                        p_tilde[0] += self.p[sn][j]
                    elif g >= self.max_theta:
                        p_tilde[self.m - 1] += self.p[sn][j]
                    else:
                        i_star = self.find_i_star(g)
                        zeta = (g - self.theta_lst[i_star]) / (self.theta_lst[i_star + 1] - self.theta_lst[i_star])
                        p_tilde[i_star] += (1-zeta) * self.p[sn][j]
                        p_tilde[i_star + 1] += zeta * self.p[sn][j]
                
                for i in range(self.m):
                    self.p[s][i] = (1 - self.alpha) * self.p[s][i] + self.alpha * p_tilde[i]
                t += 1
                    
                    
                        
                        
toy_env = gym.make("gymnasium_env/toy_example")
                        
catTD = O_CatTD(env = toy_env, alpha = 0.05,theta_range=[0,20],resolution=100, policy = 0, gamma=0.95)
catTD.learn_distribution(10000)

x = catTD.theta_lst
y = catTD.p[0]
w = np.diff(catTD.theta_lst)[0]
plt.bar(x,y,width=w)
plt.show()

                        
                        
                        
                    
                
                
                
            
        
        
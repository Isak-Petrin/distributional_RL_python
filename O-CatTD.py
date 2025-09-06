import gymnasium as gym
import sys
import gymnasium_env
import numpy as np
from SimplePolicyAgent import SimplePolicy
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class O_CatTD:
    def __init__(
        self,
        env: gym.Env,
        alpha: float,
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
        self.plot_lst = []
        
        
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

            t = 0
            while not done:
                p_tilde = np.zeros(self.m, dtype=np.float64)
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
                    
            self.plot_lst.append(np.array(copy.deepcopy(self.p)))

                
                    
                    
                        
                        
toy_env = gym.make("gymnasium_env/toy_example")
                        
catTD = O_CatTD(env = toy_env, alpha = 0.05,theta_range=[1,6],resolution = 6, policy = 1, gamma=0.95)
catTD.learn_distribution(51)

def geometric(k,p = 0.85):
    return 1*((1-p)**(k-1) * p)
    

x = catTD.theta_lst
y1 = catTD.plot_lst[0][0]
y3 = catTD.plot_lst[10][0]
y4 = catTD.plot_lst[50][0]

geo = geometric(x)  # reference geometric distribution

fig, axs = plt.subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column

axs[0].plot(x, y1, label="Episode 0")
axs[0].plot(x, geo, label="Geometric(x)", linestyle="--")
axs[0].set_title("Distribution at Episode 0")
axs[0].legend()

axs[1].plot(x, y3, label="Episode 25")
axs[1].plot(x, geo, label="Geometric(x)", linestyle="--")
axs[1].set_title("Distribution at Episode 25")
axs[1].legend()

axs[2].plot(x, y4, label="Episode 50")
axs[2].plot(x, geo, label="Geometric(x)", linestyle="--")
axs[2].set_title("Distribution at Episode 50")
axs[2].legend()

plt.tight_layout()
plt.show()
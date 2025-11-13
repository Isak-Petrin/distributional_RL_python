import numpy as np
from qrtd_lambda import QRTDlAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import sys
import gymnasium_env

episodes = 1000
gamma = 0.95
alpha = 0.2
length = 8
bootstrap = False
x0 = 2
m = 15
kappa = 2

agent1 = QRTDlAgent(m = m,obs_space=length,l = .1 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent2 = QRTDlAgent(m = m,obs_space=length,l = .3 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent3 = QRTDlAgent(m = m,obs_space=length,l = .4 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent4 = QRTDlAgent(m = m,obs_space=length,l = .6 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent5 = QRTDlAgent(m = m,obs_space=length,l = .8 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)

agents = [agent1,agent4]

env = gym.make("gymnasium_env/random_walk")
vs = {agent: [] for agent in agents}

for episode in tqdm(range(episodes)):
    obs, _ = env.reset()
    for agent in agents:
        vs[agent].append(agent.q[4].mean())
        agent.reset(x0 = obs)
        
    done = False
    
    while not done:
        action = np.random.randint(2)
        info = env.step(action)
        
        obs = info[0]
        reward = info[1] 
        done = info[2]
    
        for agent in agents:
            agent.update(x=obs, r = reward, done = done)
        
        
        
for agent in agents:            
    plt.plot(range(episodes), vs[agent], label=f'$\lambda$ = {agent.l}')

plt.legend()
plt.show()

import numpy as np
from qrtd_lambda import QRTDlAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import sys
import gymnasium_env

episodes = 5000
gamma = 0.95
alpha = 0.1
length = 8
bootstrap = False
x0 = 4
m = 10

agent1 = QRTDlAgent(m = m,obs_space=length,l = .2 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent2 = QRTDlAgent(m = m,obs_space=length,l = .4 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent3 = QRTDlAgent(m = m,obs_space=length,l = .5 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent4 = QRTDlAgent(m = m,obs_space=length,l = .6 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent5 = QRTDlAgent(m = m,obs_space=length,l = .8 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)

agents = [agent1,agent2,agent3,agent4,agent5]

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

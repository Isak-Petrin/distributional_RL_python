import numpy as np
from qrtd_lambda import QRTDlAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import sys
import gymnasium_env
from qrtd import env_q
from n_step_qrtd import env_q_nstep
from trash import env_q_lambda

episodes = 100000
gamma = 0.95
alpha = 0.05
length = 7
bootstrap = False
x0 = 3
m = 10
kappa = 2

agent1 = QRTDlAgent(m = m,obs_space=length,l = .1 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent2 = QRTDlAgent(m = m,obs_space=length,l = .3 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent3 = QRTDlAgent(m = m,obs_space=length,l = .4 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent4 = QRTDlAgent(m = m,obs_space=length,l = .6 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
agent5 = QRTDlAgent(m = m,obs_space=length,l = .8 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap, kappa = kappa)
n_step1 = env_q_nstep(m=m, obs_space=length, gamma=gamma, alpha=alpha, n=1)
n_step2 = env_q_nstep(m=m, obs_space=length, gamma=gamma, alpha=alpha, n=2)
n_step3 = env_q_nstep(m=m, obs_space=length, gamma=gamma, alpha=alpha, n=8)
agent6 = env_q(m = m, obs_space = length, gamma = gamma, alpha=alpha)
q1 = env_q_lambda(m=m, obs_space=length, gamma=gamma, alpha=alpha, lam=0.9)
q2 = env_q_lambda(m=m, obs_space=length, gamma=gamma, alpha=alpha, lam=0.6)
q3 = env_q_lambda(m=m, obs_space=length, gamma=gamma, alpha=alpha, lam=0.4)
q4 = env_q_lambda(m=m, obs_space=length, gamma=gamma, alpha=alpha, lam=0.2)

agents = [q1,q2,q3,q4]

env = gym.make("gymnasium_env/random_walk")
vs = {agent: [] for agent in agents}

for episode in tqdm(range(episodes)):
    obs, _ = env.reset()
    for agent in agents:
        vs[agent].append(agent.q[3].theta.mean())
        agent.reset(x0 = obs)
        
    done = False
    prev = length // 2
    while not done:
        action = np.random.randint(2)
        info = env.step(action)
        
        obs = info[0]
        reward = info[1] 
        done = info[2]
    
        for agent in agents:
                agent.step(x=prev, x_next=obs, done = done, r = reward)
        prev = obs
        
        
        
for agent in agents:            
    plt.plot(range(episodes), vs[agent], label=f'$\lambda$ = {agent.name}')
plt.legend()
plt.show()

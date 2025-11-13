import numpy as np
from qrtd_lambda import QRTDlAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

step = 10000
gamma = 0.95
alpha = 0.05
length = 5
bootstrap = False
s = np.random.normal(size=step) * gamma**(length-1)
m = 20
taus = [(2*i - 1) / (2*m) for i in range(1,m+1)]
emp = [np.quantile(s, tau) for tau in taus]


x0 = 0
agent1 = QRTDlAgent(m = m,obs_space=length,l = .2 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent2 = QRTDlAgent(m = m,obs_space=length,l = .4 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent3 = QRTDlAgent(m = m,obs_space=length,l = .5 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent4 = QRTDlAgent(m = m,obs_space=length,l = .6 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)
agent5 = QRTDlAgent(m = m,obs_space=length,l = .8 ,gamma = gamma, alpha = alpha, x0 = x0, bootstrap=bootstrap)


agent_lst = [agent1,agent5]

loss_lst = {agent: [] for agent in agent_lst}
mean_lst = {agent: [] for agent in agent_lst}

fig,axs = plt.subplots(1,2,figsize = (30,10))

for episode in tqdm(range(step)):

    r = np.random.normal()
    obs_lst = [(i+1,r if (i + 1) == (length - 1) else 0, (i+1) == (length - 1)) for i in range(length-1)]

    for obs in obs_lst:
        x,r,done = obs

        for agent in agent_lst:
          agent.update(x = x, r = r, done = done)
          if done == True:
            agent.reset(x0 = x0)
            agent.episode += 1

    
    for agent in agent_lst:
      est = agent.get_theta(x = 0)
      u = emp-est
      rho = u * (taus - (u < 0))
      total_loss = np.sum(rho)
      loss_lst[agent].append(total_loss)
      mean_lst[agent].append(agent.get_mean(x = 0))  

for agent in agent_lst:            
    axs[0].plot(range(step), loss_lst[agent], label=f'$\lambda$ = {agent.l}')
    axs[1].plot(range(step), mean_lst[agent], label=f'$\lambda$ = {agent.l}')

axs[0].set_xscale('log')
axs[1].set_xscale('log')

axs[0].set_title('Quantile Regression Loss')
axs[1].set_title('Estimated Mean of Quantiles')

axs[0].set_xlabel('Step')
axs[1].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[1].set_ylabel('Mean')

axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.show()

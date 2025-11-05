from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
class quantile:
    def __init__(self, m: int):
        self.m = m
        self.theta = np.linspace(-2,2,self.m)
    
    def get_theta(self):
        return self.theta
    def set_theta(self,theta):
        self.theta = theta
    
    def add_theta(self, diff):
        self.theta += diff

class container:
    def __init__(self, x0):
        self.hist = []
        self.x_hist = [x0]
        self.r_hist = []
    
    def add(self, obs: tuple):
        self.hist.append(obs)
        self.x_hist.append(obs[0])
        self.r_hist.append(obs[1])
        return 
    def get_xs(self):
        return self.x_hist
    def get_rs(self):
        return self.r_hist
    
    
class QRTDlAgent:
    def __init__(self, m: int, action_space: int, gamma: float, alpha: float, x0):
        self.m = m
        self.q = {x: quantile(m = self.m) for x in range(action_space)}
        self.tau = np.asarray([(2*i - 1) / (2*self.m) for i in range(1,self.m+1)])
        self.gamma = gamma
        self.alpha0 = alpha
        self.container = container()
    
    def get_q(self, x):
        return self.q[x]
    
    def current_alpha(self):
        return np.exp(-self.t * 1e-3)
    
    def get_theta(self, x):
        return self.q[x].get_theta()
    
    def add_diff(self, x, diff):
        self.q[x].add_theta(diff)
    
    def get_tau(self):
        return self.tau
    
    def project_monotone(self, x):
        theta = self.q[x].get_theta()
        theta = np.maximum.accumulate(theta)
        self.q[x].set_theta(theta)
    
    def store(self,x,r, done):
        self.container.add((x,r,done))
    
    def update(self,x,r,done):
        self.store(x,r,done)
        
    def get_target(self, t,h):
        xs = self.container.get_xs()
        rs = self.container.get_rs()
        x_b = xs[-1]
        
        target = 0
        for i in range():
            target[]
        
        
        
        
        
        


loss_history = []
step = 8000
r = np.random.normal(size = step)

q = env_q(m = 10, action_space=2, gamma=1, alpha=0.01)

# true target quantiles of N(0,1) at the tau levels
tau_vals = q.get_tau()
true_q = norm.ppf(tau_vals)  # shape (m,)

for i in tqdm(range(step)):
    # 1-step update on a terminal transition with reward r[i]
    q.update_q(x = 0, x_n = 1, r = r[i], done=True)
    
    # current estimate of quantiles
    est = q.get_theta(x = 0)  # shape (m,)
    
    # compute quantile (pinball) loss against the TRUE quantiles
    # u_i = q*_i - theta_i
    u = true_q - est  # shape (m,)
    # rho_tau(u) = u * ( tau - 1{u < 0} )
    rho = u * (tau_vals - (u < 0))
    
    total_loss = np.sum(rho)
    loss_history.append(total_loss)

# plot loss over steps
plt.plot(np.arange(1, step+1), loss_history)
plt.xlabel("Training step")
plt.ylabel("Quantile loss vs true Normal quantiles")
plt.title("Convergence of learned quantiles")
plt.show()

print(q.get_theta(x=0))
print(true_q)
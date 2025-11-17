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
    
class env_q:
    def __init__(self, m: int, obs_space: int, gamma: float, alpha: float):
        self.m = m
        self.q = {x: quantile(m = self.m) for x in range(obs_space)}
        self.tau = np.asarray([(2*i - 1) / (2*self.m) for i in range(1,self.m+1)])
        self.gamma = gamma
        self.alpha0 = alpha
        self.l = "qrtd"
        self.t = 1
    
    def reset(self, x0):
        return
    
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
    
    def update(self, x, x_n, r, done):
        theta_x = self.get_theta(x=x)
        theta_xn = self.get_theta(x = x_n)
        grad = np.zeros(self.m)
        alpha = self.current_alpha()
        for i in range(self.m):
            
            for j in range(self.m):
                if done:
                    g = r
                else:
                    g = r + self.gamma * theta_xn[j]
                grad[i] += (self.alpha0 / self.m) * (self.tau[i] - (g < theta_x[i]))
        
        self.add_diff(x = x, diff = grad)
        self.project_monotone(x)
        self.t += 1
    
"""
loss_history = []
step = 8000
r = np.random.normal(size = step)

q = env_q(m = 10, obs_space=2, gamma=1, alpha=0.01)

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
"""
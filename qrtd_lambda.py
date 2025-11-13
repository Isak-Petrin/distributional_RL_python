from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import savgol_filter
class quantile:
    def __init__(self, m: int):
        self.m = m
        self.theta = np.linspace(0.5,1.5,self.m)
    
    def get_theta(self):
        return self.theta
    def set_theta(self,theta):
        self.theta = theta
    def add_theta(self, diff):
        self.theta += diff
    def sample(self):
        return np.random.choice(self.theta)
    def mean(self):
        return np.mean(self.theta)

class container:
    def __init__(self, x0):
        self.hist = []
        self.xs = [x0]
        self.rs = []
        self.ds = []
    
    def add(self, obs: tuple):
        self.hist.append(obs)
        self.xs.append(obs[0])
        self.rs.append(obs[1])
        self.ds.append(obs[2])
        
    def get_xs(self):
        return self.xs
    def get_rs(self):
        return self.rs
    def get_x(self,t):
        return self.xs[t]
    def get_d(self,t):
        return self.ds[t]
    
    def reset(self, x0):
        self.hist = []          
        self.xs = [x0]      
        self.rs = []       
        self.ds = [] 
    
    
class QRTDlAgent:
    def __init__(self, m: int, l: float ,obs_space: int, gamma: float, alpha: float, x0, bootstrap: bool, kappa: bool):
        self.m = m
        self.l = l
        self.bootstrap = bootstrap
        self.q = {x: quantile(m = self.m) for x in range(obs_space)}
        self.tau = np.asarray([(2*i - 1) / (2*self.m) for i in range(1,self.m+1)])
        self.gamma = gamma
        self.N = {x: 0 for x in range(obs_space)}
        self.container = container(x0=x0)
        self.trace = lambda s: (1 - self.l) * self.l**(s - 1)
        self.t = 0
        self.alpha0 = alpha
        self.episode = 0
        self.kappa = kappa
        
    def reset(self,x0):
        self.t = 0
        self.episode += 1
        self.container.reset(x0)
    def get_mean(self,x):
        return self.q[x].mean()
    
    def get_ds(self,t):
        return self.container.get_d(t)
    
    def get_q(self, x):
        return self.q[x]
    def get_xt(self,t):
        x = self.container.get_x(t = t)
        return x,self.q[x].get_theta()
    
    def _huber_grad(self, u):
        k = float(self.kappa)  # ensure numeric
        return np.where(np.abs(u) <= k, u / k, np.sign(u))
    

    def get_alpha(self, x, trc):
        # either do not mutate N here...
        eff_visits = self.N[x] + trc
        alpha_x = self.alpha0 / ((1.0 + eff_visits) ** 0.8)
        return max(alpha_x, 1e-4)

    
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
    
    def update(self, x, r, done: bool):
        self.t += 1
        self.store(x, r, done)

        for i in range(self.t):
            h = self.t - i
            target = self.get_target(t=i, h=h)          # scalar if bootstrap else vector (len m)
            x_i, theta = self.get_xt(t=i)               # theta shape: (m,)
            trc   = self.trace(s=h)
            alpha = self.get_alpha(x=x_i, trc=trc)

            if self.bootstrap:
                # target is scalar -> deltas per quantile
                delta = target - theta                  # shape: (m,)
                weight = np.abs(self.tau - (delta < 0).astype(float))  # |tau - 1{delta<0}|
                g = weight * self._huber_grad(delta)    # elementwise
                grad = alpha * trc * g                  # optional: / self.m for stability
            else:
                delta = target[None, :] - theta[:, None]                       # (m, m)
                weight = np.abs(self.tau[:, None] - (delta < 0).astype(float)) # (m, m)
                g = weight * self._huber_grad(delta)                           # (m, m)
                grad = self.alpha0 * trc * g.mean(axis=1)                          # average over j -> (m,)

            self.add_diff(x=x_i, diff=grad)

        
    def get_target(self, t,h):
        rs = self.container.get_rs()
        xb = self.container.get_x(t = t+h)
        if self.bootstrap:
            target = 0
            for i,k in enumerate(range(t,t+h)):
                target += self.gamma**i * rs[k]
                if self.get_ds(t = k) == True:
                    return target
            return target + self.q[xb].sample() * self.gamma**h
        else:
            target = np.zeros(self.m)
            for i,k in enumerate(range(t,t+h)):
                target += self.gamma**i * np.ones(self.m) * rs[k]
                if self.get_ds(t = k) == True:
                    return target
            return target + self.get_theta(xb) * self.gamma**h

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class quantile:
    def __init__(self, m: int):
        self.m = m
        self.theta = np.linspace(-2, 2, self.m)
    
    def get_theta(self):
        return self.theta
    
    def set_theta(self, theta):
        self.theta = theta
    
    def add_theta(self, diff):
        self.theta += diff

class env_q_nstep:
    def __init__(self, m: int, obs_space: int, gamma: float, alpha: float, n: int):
        self.m = m
        self.q = {x: quantile(m=self.m) for x in range(obs_space)}
        self.tau = np.asarray([(2 * i - 1) / (2 * self.m) for i in range(1, self.m + 1)])
        self.gamma = gamma
        self.alpha0 = alpha
        self.l = "n_step"
        self.t = 1
        self.n = n

        self.buffer = []
        self.name = "n_step"

    def reset(self, x0):
        self.buffer = []
  
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

    def _update_from_buffer_front(self):
  

        k_max = min(self.n, len(self.buffer))

        # state at tid
        x_t, _, done_t, _ = self.buffer[0]


        G = 0.0
        discount = 1.0
        terminal_reached = False

        for k in range(k_max):
            x_k, r_k, done_k, _ = self.buffer[k]
            G += discount * r_k
            if done_k:
                terminal_reached = True

                break
            discount *= self.gamma  
        
        theta_x = self.get_theta(x_t)
        grad = np.zeros(self.m)

        if terminal_reached:

            for i in range(self.m):
                for j in range(self.m):

                    grad[i] += (self.alpha0 / self.m) * (self.tau[i] - (g < theta_x[i]))
        else:

            _, _, _, x_tn = self.buffer[k_max - 1]
            theta_xn = self.get_theta(x_tn)

            for i in range(self.m):
                for j in range(self.m):
                    g = G + discount * theta_xn[j]
                    grad[i] += (self.alpha0 / self.m) * (self.tau[i] - (g < theta_x[i]))

        self.add_diff(x_t, grad)
        self.project_monotone(x_t)
        self.t += 1


        self.buffer.pop(0)

    def step(self, x, r, done, x_next):

        self.buffer.append((x, r, done, x_next))


        if len(self.buffer) >= self.n:
            self._update_from_buffer_front()


        if done:
            while len(self.buffer) > 0:
                self._update_from_buffer_front()


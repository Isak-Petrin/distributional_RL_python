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


class env_q_lambda:
    """
    QR-TD(λ) implemented as in Section 2.5 of your paper (forward-view, G_{t:h}, e(h-t)).
    
    - Tabular quantiles per state.
    - Episodic: we collect a whole trajectory, then apply the QR-TD(λ) sweep at episode end.
    """
    def __init__(self, m: int, obs_space: int, gamma: float, alpha: float, lam: float):
        self.m = m
        self.q = {x: quantile(m=self.m) for x in range(obs_space)}
        self.tau = np.asarray([(2 * i - 1) / (2 * self.m) for i in range(1, self.m + 1)])
        self.gamma = gamma
        self.alpha0 = alpha
        self.lam = lam
        self.l = "qr_td_lambda"
        self.name = "qr_td_lambda"

        # episode buffers
        self.ep_states = []   # x_0, x_1, ..., x_T
        self.ep_rewards = []  # r_0, ..., r_{T-1} (reward for x_t -> x_{t+1})

    def reset(self, x0):
        """
        Call at the beginning of an episode with initial state x0.
        """
        self.ep_states = [x0]
        self.ep_rewards = []
        return

    def get_q(self, x):
        return self.q[x]
    
    def current_alpha(self):
        # constant α as in the paper; you can plug in a schedule if you want
        return self.alpha0
    
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

    # ---------- core TD(λ) logic ----------

    def _episode_update_lambda(self, terminal_bootstrap_zero: bool = True):
        """
        Apply QR-TD(λ) updates for the *completed* episode stored in ep_states / ep_rewards.
        Implements eqs. (2.23)–(2.27) from your text.

        terminal_bootstrap_zero=True:
            for the final state x_T, use θ_T^T = 0 (pure MC at the end).
            If False, we bootstrap from the current θ(x_T).
        """
        states = self.ep_states        # length T+1
        rewards = self.ep_rewards      # length T
        T = len(rewards)
        if T == 0:
            return  # nothing happened in this episode

        alpha = self.current_alpha()

        # base_theta[h] ≈ θ_h^h in the text: quantiles at time h used for bootstrapping
        base_theta = []
        for s in states:
            base_theta.append(self.get_theta(s).copy())
        
        # handle terminal bootstrap
        if terminal_bootstrap_zero:
            base_theta[-1] = np.zeros(self.m)

        # theta_temp[t] ≈ θ_t^0 initially; will be updated to θ_t^h across h
        theta_temp = [th.copy() for th in base_theta]

        # Forward-view λ-sweep:
        # h = 1,...,T ; t = 0,...,h-1
        for h in range(1, T + 1):
            # horizon index h corresponds to state index h and reward indices [0..h-1]
            theta_hh = base_theta[h]  # θ_h^h

            # loop over all starting times t < h
            for t in range(0, h):
                # compute discounted reward sum Σ_{k=1}^{h-t} γ^{k-1} r_{t+k-1}
                G_base = 0.0
                discount = 1.0
                for k in range(t, h):
                    G_base += discount * rewards[k]
                    discount *= self.gamma
                # now discount = γ^{h-t}

                theta_prev = theta_temp[t]  # θ_t^{h-1}, shape (m,)

                # smoothed quantile gradient wrt θ_t^{h-1}, eq. (2.24)
                grad = np.zeros(self.m)
                for i in range(self.m):
                    for j in range(self.m):
                        g = G_base + discount * theta_hh[j]  # G_{t:h} "sample" using θ_h^h_j
                        grad[i] += (1.0 / self.m) * (self.tau[i] - (g < theta_prev[i]))

                # eligibility weight e(h-t) = (1-λ) λ^{h-t-1}, eq. (2.25)
                lag = h - t
                e = (1.0 - self.lam) * (self.lam ** (lag - 1))

                # update θ_t^h, eq. (2.27)
                theta_temp[t] = theta_prev + alpha * e * grad

        # After all horizons, assign θ_t^T back to parameter vectors of visited states.
        # If a state is visited multiple times, the *later* time index overwrites the earlier,
        # which is one way to share parameters in the time-indexed derivation.
        for t, s in enumerate(states[:-1]):  # x_0,...,x_{T-1}; skip terminal x_T
            self.q[s].set_theta(theta_temp[t])
            self.project_monotone(s)

    def step(self, x, r, done, x_next):
        """
        Call this every environment transition.

        x      : current state x_t
        r      : reward r_t
        done   : whether this transition ends the episode
        x_next : next state x_{t+1}
        """
        # append transition to episode buffer
        # we assume `reset(x0)` was called before the first step so that ep_states[-1] == x
        if len(self.ep_states) == 0:
            self.ep_states.append(x)
        self.ep_rewards.append(r)
        self.ep_states.append(x_next)

        if done:
            # perform QR-TD(λ) sweep over the completed episode
            self._episode_update_lambda(terminal_bootstrap_zero=True)
            # clear buffers; you must call reset(x0) before the next episode
            self.ep_states = []
            self.ep_rewards = []

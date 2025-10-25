import math
import random
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class ToyEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, param: dict | None = None):
        super().__init__()
        if param is None:
            self.gamma = 0.9
            self.ps = 0.3
            self.rs = 1
            self.pr = 0.5
            self.rr = 1
        self.observation_space = spaces.Discrete(2)  # state in {0,1}
        self.action_space = spaces.Discrete(2)       # 0=safe, 1=risky

    def _get_obs(self):
        return {"agent": self.state}

    def _get_info(self):
        return {"gamma": self.gamma, "ps": self.ps, "rs": self.rs, "pr": self.pr, "rr": self.rr}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        terminated = False
        if action == 1:  # risky
            reward = self.rr
            if self.pr > np.random.random():
                terminated = True
                self.state = 1
        else:            # safe
            reward = self.rs
            if self.ps > np.random.random():
                terminated = True
                self.state = 1
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info


@dataclass
class Config:
    num_quantiles: int = 51
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    replay_size: int = 50_000
    min_replay: int = 1_000
    train_steps: int = 20_000
    target_update_interval: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 5_000  # linear decay
    huber_kappa: float = 1.0
    device: str = "cpu"
    seed: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.buffer: Deque = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.int64),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.int64),
            np.array(d, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


class QRDQN(nn.Module):
    """
    Very small network:
      - Embedding for the discrete state {0,1}
      - MLP to |A| * Nquantiles outputs
    """
    def __init__(self, n_states: int, n_actions: int, n_quantiles: int):
        super().__init__()
        embed_dim = 16
        hidden = 64
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        self.embed = nn.Embedding(n_states, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions * n_quantiles)

    def forward(self, state_idx: torch.Tensor) -> torch.Tensor:
        # state_idx: [B,] integers in {0,...,n_states-1}
        x = self.embed(state_idx)               # [B, E]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.head(x)                      # [B, A*N]
        out = out.view(-1, self.n_actions, self.n_quantiles)  # [B, A, N]
        return out

    def q_expected(self, state_idx: torch.Tensor) -> torch.Tensor:
        # Expected Q = mean over quantiles
        quantiles = self.forward(state_idx)     # [B, A, N]
        return quantiles.mean(dim=-1)           # [B, A]


def quantile_huber_loss(pred_quantiles: torch.Tensor,
                        target_quantiles: torch.Tensor,
                        taus: torch.Tensor,
                        kappa: float) -> torch.Tensor:
    """
    pred_quantiles:   [B, N] current network for (s,a)
    target_quantiles: [B, N] target samples r + gamma * next quantiles
    taus:             [N]    quantile fractions (e.g. (0.5+arange(N))/N)
    returns scalar loss
    """
    B, N = pred_quantiles.shape
    # Pairwise TD errors: [B, N_pred, N_tgt]
    td = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # [B, N, N]
    abs_td = torch.abs(td)

    if kappa > 0.0:
        huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    else:
        huber = abs_td  # L1

    # Indicator for td < 0
    tau = taus.view(1, N, 1)  # [1, N, 1]
    weight = torch.abs(tau - (td.detach() < 0).float())  # stop-grad through indicator

    loss = (weight * huber).mean(dim=2).sum(dim=1).mean() / N  # average over j, sum over i, average over batch
    return loss


class Agent:
    def __init__(self, env: gym.Env, cfg: Config):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.N = cfg.num_quantiles

        self.net = QRDQN(self.n_states, self.n_actions, self.N).to(self.device)
        self.tgt = QRDQN(self.n_states, self.n_actions, self.N).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        # Uniform quantile fractions (midpoints)
        self.taus = torch.tensor((np.arange(self.N) + 0.5) / self.N, dtype=torch.float32, device=self.device)

        self.replay = ReplayBuffer(cfg.replay_size, seed=cfg.seed)
        self.global_step = 0

    def select_action(self, state_idx: int, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor([state_idx], dtype=torch.long, device=self.device)
            q = self.net.q_expected(s)  # [1, A]
            return int(torch.argmax(q, dim=1).item())

    def train_step(self):
        if len(self.replay) < self.cfg.min_replay:
            return None

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)

        s  = torch.tensor(s, dtype=torch.long, device=self.device)
        a  = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        r  = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)            # [B,1]
        s2 = torch.tensor(s2, dtype=torch.long, device=self.device)
        done = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)          # [B,1]

        # Current quantiles for chosen actions: [B, N]
        all_q = self.net(s)                               # [B, A, N]
        pred = all_q.gather(1, a.expand(-1, 1, self.N)).squeeze(1)  # [B, N]

        with torch.no_grad():
            # Double DQN action selection via online net's expected values
            next_q_mean = self.net.q_expected(s2)         # [B, A]
            next_actions = torch.argmax(next_q_mean, dim=1, keepdim=True)  # [B,1]

            # Target quantiles from target net: [B, N]
            next_all = self.tgt(s2)                       # [B, A, N]
            next_q = next_all.gather(1, next_actions.unsqueeze(-1).expand(-1, 1, self.N)).squeeze(1)  # [B, N]

            gamma = self.cfg.gamma * (1.0 - done)         # [B,1]
            target = r + gamma * next_q                   # broadcast to [B, N]

        loss = quantile_huber_loss(pred, target, self.taus, self.cfg.huber_kappa)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()

        if self.global_step % self.cfg.target_update_interval == 0:
            self.tgt.load_state_dict(self.net.state_dict())

        return float(loss.item())

    def linear_epsilon(self, step: int) -> float:
        eps = self.cfg.eps_end + max(0.0, (self.cfg.eps_start - self.cfg.eps_end) *
                                     (1.0 - min(1.0, step / self.cfg.eps_decay_steps)))
        return eps

    def run(self):
        ep_returns = []
        obs, info = self.env.reset(seed=self.cfg.seed)
        gamma_from_env = info.get("gamma", None)
        if gamma_from_env is not None:
            self.cfg.gamma = float(gamma_from_env)  # sync with env's gamma

        state = obs["agent"]
        ep_ret = 0.0

        for step in range(1, self.cfg.train_steps + 1):
            self.global_step = step
            eps = self.linear_epsilon(step)
            action = self.select_action(state, eps)

            obs2, reward, terminated, truncated, _ = self.env.step(action)
            state2 = obs2["agent"]
            done = terminated or truncated

            self.replay.push(state, action, reward, state2, done)
            ep_ret += reward

            loss = self.train_step()

            state = state2
            if done:
                ep_returns.append(ep_ret)
                obs, _ = self.env.reset()
                state = obs["agent"]
                ep_ret = 0.0

            if step % 1000 == 0:
                last10 = np.mean(ep_returns[-10:]) if ep_returns else float("nan")
                print(f"step {step:6d} | eps {eps:.3f} | last10_return {last10:.3f} | replay {len(self.replay)} | loss {loss}")

        return ep_returns


if __name__ == "__main__":
    # Make env and train
    env = ToyEnv()
    cfg = Config(device="cpu", train_steps=10000, min_replay=200, batch_size=64, seed=42)
    agent = Agent(env, cfg)
    returns = agent.run()

    print("Training finished.")
    if len(returns) > 0:
        print("Average return (last 50 episodes):", np.mean(returns[-50:]))



    def get_action_quantiles(agent, state_idx=0):
        """
        Returns a dict: {action: np.array of shape (N,)} of learned atom locations
        for a given discrete state index.
        """
        device = next(agent.net.parameters()).device
        with torch.no_grad():
            s = torch.tensor([state_idx], dtype=torch.long, device=device)
            # quantiles: [1, A, N]
            quantiles = agent.net(s).squeeze(0).detach().cpu().numpy()  # [A, N]
        return {a: quantiles[a] for a in range(quantiles.shape[0])}

    def plot_atoms_stem(agent, state_idx=0, title="Quantile atoms (Dirac impulses)"):
        aq = get_action_quantiles(agent, state_idx)
        N = agent.N
        actions = sorted(aq.keys())
        plt.figure(figsize=(8, 4))
        for a in actions:
            xs = np.arange(N) + (a * 0.12)  # slight horizontal offset per action
            ys = np.sort(aq[a])             # sort only for nicer left-to-right view
            markerline, stemlines, baseline = plt.stem(xs, ys, label=f"action {a}")
            plt.setp(markerline, markersize=4)
            plt.setp(stemlines, linewidth=1)
            # Expected value line:
            plt.axhline(np.mean(aq[a]), linestyle="--", alpha=0.5)
        plt.xlabel("atom index (ordered)")
        plt.ylabel("return value")
        plt.title(title + f" — state {state_idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_cdf(agent, state_idx=0, title="Empirical CDF from quantile atoms"):
        aq = get_action_quantiles(agent, state_idx)
        actions = sorted(aq.keys())
        plt.figure(figsize=(7, 4))
        for a in actions:
            atoms = np.sort(aq[a])
            N = len(atoms)
            # ECDF: after each atom i, CDF jumps to i/N (post-step)
            y = (np.arange(1, N + 1)) / N
            plt.step(atoms, y, where="post", label=f"action {a}")
            plt.axvline(np.mean(atoms), linestyle="--", alpha=0.5)
        plt.xlabel("return")
        plt.ylabel("CDF")
        plt.ylim(0, 1.01)
        plt.title(title + f" — state {state_idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_hist(agent, state_idx=0, bins=15, title="Histogram of quantile atoms (equal weights)"):
        aq = get_action_quantiles(agent, state_idx)
        actions = sorted(aq.keys())
        plt.figure(figsize=(7, 4))
        for a in actions:
            atoms = aq[a]
            weights = np.ones_like(atoms) / len(atoms)
            plt.hist(atoms, bins=bins, weights=weights, alpha=0.5, label=f"action {a}")
            plt.axvline(np.mean(atoms), linestyle="--", alpha=0.7)
        plt.xlabel("return")
        plt.ylabel("probability")
        plt.title(title + f" — state {state_idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()

 
    plot_atoms_stem(agent, state_idx=0)
    plot_cdf(agent, state_idx=0)
    plot_hist(agent, state_idx=0, bins=10)

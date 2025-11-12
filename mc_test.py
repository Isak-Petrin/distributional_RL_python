import numpy as np
import gymnasium as gym
import gymnasium_env   # your env module

env = gym.make("gymnasium_env/random_walk")
n_episodes = 200000
returns = []

for _ in range(n_episodes):
    obs, _ = env.reset()
    G = 0
    done = False
    # if you want the value *starting from a specific state*, you may need to
    # reset the env into that state or only record episodes that start there
    while not done:
        action = np.random.randint(2)  # same random policy as training
        obs, r, done, truncated, _ = env.step(action)
        G += r
    returns.append(G)

print("Empirical mean return:", np.mean(returns))

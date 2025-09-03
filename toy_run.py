import gymnasium as gym
import sys
import gymnasium_env
import numpy as np
from SimplePolicyAgent import SimplePolicy
import matplotlib.pyplot as plt


#env = gym.make("gymnasium_env/GridWorld-v0",render_mode = "human")
#env = gym.make("gymnasium_env/GridWorld-v0")

toy_env = gym.make("gymnasium_env/toy_example")
# Training hyperparameters
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 10000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration

env = gym.wrappers.RecordEpisodeStatistics(env=toy_env, buffer_length=n_episodes)

agent = SimplePolicy(
    env=toy_env,
    policy=1,
)

from tqdm import tqdm  # Progress bar

for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action()

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Move to next state
        done = terminated or truncated
        obs = next_obs
    
def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over a 500-episode window
rolling_length = 500
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

plt.tight_layout()
plt.show()
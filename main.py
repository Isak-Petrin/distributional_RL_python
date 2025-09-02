import gymnasium as gym

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/GridWorld-v0")
from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/custom_grid_world",
    entry_point="gymnasium_env.envs:CustomGridEnv",
)

register(
    id="gymnasium_env/toy_example",
    entry_point="gymnasium_env.envs:ToyEnv",
)

register(
    id="gymnasium_env/random_walk",
    entry_point="gymnasium_env.envs:RandomWalk",
)
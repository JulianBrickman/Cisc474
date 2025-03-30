import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import coverage_gridworld.env  # registers all custom envs like maze, safe, etc.
import random
from coverage_gridworld.env import CoverageGridworld

COLOR_IDS = {
    0: (0, 0, 0),       # BLACK
    1: (255, 255, 255), # WHITE
    2: (101, 67, 33),   # BROWN (wall)
    3: (160, 161, 161), # GREY (agent)
    4: (31, 198, 0),    # GREEN (enemy)
}

def generate_random_map(map_type="maze", grid_size=10):
    print(map_type)
    while True:
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        grid[0][0] = 3  # agent starts at top-left
        occupied = {(0, 0)}

        # Default parameters based on map type
        if map_type == "just_go":
            num_walls = 0
            num_enemies = 0
        elif map_type == "safe":
            num_walls = 15
            num_enemies = 0
        elif map_type == "maze":
            num_walls = 20
            num_enemies = 3
        elif map_type == "chokepoint":
            num_walls = 30
            num_enemies = 2
        elif map_type == "sneaky_enemies":
            num_walls = 5
            num_enemies = 6
        else:
            raise ValueError(f"Unknown map type: {map_type}")

        # Place walls
        while len(occupied) < 1 + num_walls:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (x, y) not in occupied:
                grid[y][x] = 2
                occupied.add((x, y))

        # Place enemies
        while len(occupied) < 1 + num_walls + num_enemies:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            if (x, y) not in occupied:
                grid[y][x] = 4
                occupied.add((x, y))

        # Validate with CoverageGridworld logic
        np_grid = np.array(grid)
        try:
            CoverageGridworld(predefined_map=np_grid)
            return np_grid.tolist()
        except SystemExit:
            print("❌ Invalid map generated. Retrying...")
            continue


def generate_multiple_maps_by_type():
    return (
        [generate_random_map("just_go") for _ in range(3)] +
        [generate_random_map("safe") for _ in range(3)] +
        [generate_random_map("maze") for _ in range(5)] +
        [generate_random_map("chokepoint") for _ in range(5)] +
        [generate_random_map("sneaky_enemies") for _ in range(5)]
    )

def make_curriculum_env(stage_maps):
    def _make():
        return gym.make("maze", predefined_map_list=stage_maps)
    return _make

def main():
    total_timesteps = 300_000

    all_maps = generate_multiple_maps_by_type()

    # Stage 1: Easy maps (just_go, safe)
    easy_maps = all_maps[0:6]
    env_easy = make_vec_env(make_curriculum_env(easy_maps), n_envs=4)
    model = PPO(
        policy="MlpPolicy",
        env=env_easy,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    print("\n\n--- Training on EASY maps ---\n")
    model.learn(total_timesteps=75_000)

    # Stage 2: Intermediate maps (maze, chokepoint)
    intermediate_maps = all_maps[6:16]
    env_intermediate = make_vec_env(make_curriculum_env(intermediate_maps), n_envs=4)
    model.set_env(env_intermediate)
    print("\n\n--- Training on INTERMEDIATE maps ---\n")
    model.learn(total_timesteps=100_000)

    # Stage 3: Hard map (sneaky_enemies)
    hard_maps = all_maps[16:]
    env_hard = make_vec_env(make_curriculum_env(hard_maps), n_envs=4)
    model.set_env(env_hard)
    print("\n\n--- Training on HARD maps ---\n")
    model.learn(total_timesteps=125_000)

    # Save final model
    model_name = "ppo_curriculum_all_300000"
    model.save(model_name)
    print(f"\nModel saved as {model_name}.zip")

if __name__ == "__main__":
    main()
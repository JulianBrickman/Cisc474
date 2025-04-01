# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# import coverage_gridworld.env  # registers all custom envs like maze, safe, etc.
# import random
# from coverage_gridworld.env import CoverageGridworld
# from datetime import datetime

# def is_map_coverable(predefined_map):
#     try:
#         # Instantiate the environment just to trigger internal validation logic
#         env = CoverageGridworld(predefined_map=np.array(predefined_map))
#         return True
#     except SystemExit:
#         return False

# COLOR_IDS = {
#     0: (0, 0, 0),       # BLACK
#     1: (255, 255, 255), # WHITE
#     2: (101, 67, 33),   # BROWN (wall)
#     3: (160, 161, 161), # GREY (agent)
#     4: (31, 198, 0),    # GREEN (enemy)
# }

# def generate_random_map(map_type="maze", grid_size=10, max_attempts=500):
#     base_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
#     base_grid[0][0] = 3  # Agent starts at top-left
#     occupied = {(0, 0)}

#     # Map config
#     config = {
#         "just_go": (0, 0),
#         "safe": (15, 0),
#         "maze": (20, 3),
#         "chokepoint": (30, 2),
#         "sneaky_enemies": (5, 6)
#     }

#     if map_type not in config:
#         raise ValueError(f"Unknown map type: {map_type}")
    
#     num_walls, num_enemies = config[map_type]
#     total_to_place = [('wall', num_walls), ('enemy', num_enemies)]

#     grid = [row[:] for row in base_grid]  # copy base
#     placed = 0
#     attempts = 0

#     while placed < sum(q for _, q in total_to_place) and attempts < max_attempts:
#         kind = 'wall' if placed < num_walls else 'enemy'
#         val = 2 if kind == 'wall' else 4

#         y, x = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
#         if (y, x) in occupied:
#             attempts += 1
#             continue

#         grid[y][x] = val
#         if is_map_coverable(grid):
#             occupied.add((y, x))
#             placed += 1
#         else:
#             # Undo this placement
#             grid[y][x] = 0
#         attempts += 1

#     if placed < sum(q for _, q in total_to_place):
#         raise RuntimeError(f"❌ Failed to generate a valid '{map_type}' map after {attempts} attempts.")
    
#     return grid



# def generate_multiple_maps_by_type():
#     types = [("just_go", 3), ("safe", 3), ("maze", 5), ("chokepoint", 5), ("sneaky_enemies", 5)]
#     all_maps = []

#     for map_type, count in types:
#         print(f"Generating {count} '{map_type}' maps...")
#         valid_maps = []
#         while len(valid_maps) < count:
#             try:
#                 new_map = generate_random_map(map_type)
#                 valid_maps.append(new_map)
#                 print(f"✅ {map_type} map {len(valid_maps)} generated")
#             except RuntimeError as e:
#                 print(f"⚠️ {e}")
#         all_maps.extend(valid_maps)

#     return all_maps



# def make_curriculum_env(stage_maps):
#     def _make():
#         return gym.make("maze", predefined_map_list=stage_maps)
#     return _make

# def main():
#     n_envs = 4
#     total_timesteps_easy = 100_000
#     total_timesteps_intermediate = 200_000
#     total_timesteps_hard = 200_000
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     print("\n🧠 Generating curriculum maps...")
#     all_maps = generate_multiple_maps_by_type()
#     print(f"✅ Total maps generated: {len(all_maps)}")
#     model = PPO.load("ppo_sneaky_final_300k")
#     # Easy maps from predefined curriculum
#     easy_maps = all_maps[0:3] + all_maps[3:6]  # just_go + safe

#     # === EASY STAGE ===
#     print("\n\n--- 🟢 Training on EASY maps ---\n")
#     env_easy = make_vec_env(lambda: gym.make("just_go", predefined_map_list=None), n_envs=4)
#     # model = PPO(
#     #     policy="MlpPolicy",
#     #     env=env_easy,
#     #     verbose=1,
#     #     tensorboard_log="./tensorboard_logs/sneaky_comp_300k/"
#     # )
#     model.set_env(env_easy)
#     model.learn(total_timesteps=total_timesteps_easy)
#     model.save(f"ppo_sneaky_easy_{timestamp}")
#     print(f"✅ Model saved: 'ppo_sneaky_easy_{timestamp}_.zip'")

#     # === INTERMEDIATE STAGE ===
#     # \for i in range(3):
#     print(f"\n\n=== 🔁 INTERMEDIATE Iteration/5 (safe) ===\n")
#     env_intermediate = make_vec_env(lambda: gym.make("safe", predefined_map_list=None), n_envs=4)
#     model.set_env(env_intermediate)
#     model.learn(total_timesteps=total_timesteps_intermediate // 2)
#     model.save(f"ppo_sneaky_intermediate_safe__{timestamp}")
#     print(f"✅ Model saved: 'ppo_sneaky_intermediate_safe__{timestamp}_.zip'")

#     # for i in range(3, 5):
#     #     print(f"\n\n=== 🔁 INTERMEDIATE Iteration {i+1}/5 (maze) ===\n")
#     env_intermediate = make_vec_env(lambda: gym.make("maze", predefined_map_list=None), n_envs=4)
#     model.set_env(env_intermediate)
#     model.learn(total_timesteps=total_timesteps_intermediate // 2)
#     model.save(f"ppo_sneaky_intermediate_maze__{timestamp}")
#     print(f"✅ Model saved: 'ppo_sneaky_intermediate_maze__{timestamp}_.zip'")

#     # === HARD STAGE (SNEAKY ENEMIES) ===
#     env_hard = make_vec_env(make_curriculum_env(None), n_envs=n_envs)
#     model.set_env(env_hard)
#     model.learn(total_timesteps=total_timesteps_hard)
#     model.save(f"ppo_sneaky_hard_")
#     print(f"✅ Model saved: 'ppo_sneaky_{timestamp}_hard_.zip'")

#     # Final model save
#     model.save("ppo_sneaky_final_300k")
#     print(f"\n🎯 Final model saved as 'ppo_sneaky_final_300k.zip' — fully trained on 300k steps!")

# if __name__ == "__main__":
#     main()
import os
import sys
import gymnasium as gym
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import coverage_gridworld
from multiprocessing import freeze_support
from datetime import datetime

# ---------------- Global Configuration ---------------- #

reward_functions = ["rf3"]
coverage_gridworld.custom.OBSERVATION_MODE = "window"
training_phases = [
    "sneaky_enemies",
    "sneaky_enemies",
    "sneaky_enemies",
]

phase_learning_rates = {
    "sneaky_enemies_1": 5e-4,  # Normal LR to kickstart
    "sneaky_enemies_2": 3e-4,  # Slightly lower to refine
    "sneaky_enemies_3": 2e-4
}

LEARNING_RATE = 4e-4
ENTROPY_COEF = 0.01
render_list = []
n_envs = 64
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a single base directory for this training run
base_dir = f"training_run_{timestamp}"
log_dir = os.path.join(base_dir, "logs")
model_dir = os.path.join(base_dir, "models")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ---------------- Logging Setup ---------------- #

def setup_logger():
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(message)s")
    
    # File handler for detailed logs
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)
    
    # Console handler for cleaner output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Log initial setup
    logger.info(f"Training run started at {timestamp}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model directory: {model_dir}")

def get_logger():
    return logging.getLogger("train_logger")

# ---------------- Helper Functions ---------------- #

def make_env_func(env_tag, render_mode, monitor_log_path):
    env = gym.make(env_tag, render_mode=render_mode,
                   predefined_map_list=None, activate_game_status=False)
    env = Monitor(env, filename=monitor_log_path, allow_early_resets=True)
    return env

def compute_l2_norm(model):
    l2_norm = sum(param.data.norm(2).item() **
                  2 for param in model.policy.parameters()) ** 0.5
    get_logger().info(f"L2 norm of model parameters: {l2_norm:.4f}")
    return l2_norm

# ---------------- Plateau LR and Training Callback ---------------- #

class PlateauLRCallback(BaseCallback):
    def __init__(self, patience=8, lr_factor=0.7, max_decreases=4, improvement_threshold=1.0, reward_window=8, warmup_rollouts=10, alpha=0.4, verbose=1):
        super().__init__(verbose)
        self.patience = patience
        self.lr_factor = lr_factor
        self.max_decreases = max_decreases
        self.improvement_threshold = improvement_threshold
        self.reward_window = reward_window
        self.warmup_rollouts = warmup_rollouts
        self.alpha = alpha
        self.ewma_baseline = None
        self.wait = 0
        self.lr_decrease_count = 0
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        if len(self.model.ep_info_buffer) > 0:
            current_mean_reward = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer])

            if self.rollout_count <= self.warmup_rollouts:
                get_logger().info(
                    f"[PlateauLRCallback] Warm-up rollout {self.rollout_count}/{self.warmup_rollouts} — skipping plateau check.")
                return

            if self.ewma_baseline is None:
                self.ewma_baseline = current_mean_reward
            else:
                new_ewma = self.alpha * current_mean_reward + \
                    (1 - self.alpha) * self.ewma_baseline
                self.ewma_baseline = max(self.ewma_baseline, new_ewma)

            get_logger().info(
                f"[Rollout {self.rollout_count}] Mean reward: {current_mean_reward:.2f} | EWMA baseline: {self.ewma_baseline:.2f}")

            if current_mean_reward > self.ewma_baseline + self.improvement_threshold:
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    for param_group in self.model.policy.optimizer.param_groups:
                        old_lr = param_group['lr']
                        new_lr = old_lr * self.lr_factor
                        param_group['lr'] = new_lr
                    self.model.learning_rate = new_lr
                    self.model.lr_schedule = lambda progress: new_lr
                    get_logger().info(
                        f"[PlateauLRCallback] Plateau detected. Reducing LR to {new_lr:.1e}")
                    self.lr_decrease_count += 1
                    self.wait = 0

    def _on_step(self) -> bool:
        return True

    def reached_plateau(self) -> bool:
        return self.lr_decrease_count >= self.max_decreases

# ---------------- Training Functions ---------------- #

def train_phase(model, phase_env_fns, phase_name, initial_timesteps=10000000):
    plateau_callback = PlateauLRCallback()
    plateau_callback.ewma_baseline = None
    plateau_callback.wait = 0
    plateau_callback.lr_decrease_count = 0
    plateau_callback.rollout_count = 0
    vec_env = SubprocVecEnv(phase_env_fns)
    model.set_env(vec_env)
    get_logger().info(f"--- Starting phase: {phase_name} ---")
    while not plateau_callback.reached_plateau():
        model.learn(total_timesteps=initial_timesteps,
                    reset_num_timesteps=False, callback=plateau_callback)

        if plateau_callback.ewma_baseline:
            get_logger().info(
                f"[{phase_name}] EWMA baseline: {plateau_callback.ewma_baseline:.2f} | LR decreases: {plateau_callback.lr_decrease_count}")

        model.save(os.path.join(model_dir, f"ppo_{phase_name}_intermediate"))
    vec_env.close()
    get_logger().info(f"--- Phase {phase_name} completed due to plateau ---")
    return model

def rehearsal_phase(model, seen_envs, rehearsal_timesteps=100000, n_envs=64):
    get_logger().info("Starting multi-map rehearsal phase")

    def mixed_env():
        env_tag = np.random.choice(seen_envs)
        return make_env_func(env_tag, None, os.path.join(log_dir, f"monitor_rehearsal_{env_tag}.csv"))
    env_fns = [mixed_env for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    model.set_env(vec_env)
    plateau_callback = PlateauLRCallback()
    model.learn(total_timesteps=rehearsal_timesteps,
                reset_num_timesteps=False, callback=plateau_callback)
    vec_env.close()
    compute_l2_norm(model)
    return model

# ---------------- Main Training Loop ---------------- #

def main():
    setup_logger()
    seen_envs = []
    for rf in reward_functions:
        get_logger().info(f"\n===== Training with Reward Function: {rf} =====")
        
        coverage_gridworld.custom.REWARD_FUNCTION = rf
        model_file = "./training_run_20250331_163031/models/ppo_sneaky_enemies_phase3.zip"
        
        for phase_idx, env_tag in enumerate(training_phases):
            render = "human" if env_tag in render_list else None
            phase_name = f"{env_tag}_phase{phase_idx+1}"
            seen_envs.append(env_tag)

            # Simplified monitor log path
            monitor_log_path = os.path.join(log_dir, f"monitor_{rf}_{phase_name}.csv")

            # Create vectorized envs
            env_fns = [lambda: make_env_func(env_tag, render, monitor_log_path) for _ in range(n_envs)]
            vec_env = SubprocVecEnv(env_fns)

            # Initialize or load model
            if phase_idx == 0:
                if os.path.exists(model_file + ".zip"):
                    get_logger().info(f"Loading existing model from {model_file}")
                    model = PPO.load(model_file, env=vec_env)
                else:
                    get_logger().info("Creating new model")
                    model = PPO("MultiInputPolicy", vec_env, verbose=1, 
                              learning_rate=LEARNING_RATE, ent_coef=ENTROPY_COEF)
            else:
                model.set_env(vec_env)

            # Adjust learning rate dynamically for each phase
            phase_key = f"{env_tag}_{phase_idx+1}"
            phase_lr = phase_learning_rates.get(phase_key, LEARNING_RATE)
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = phase_lr
            model.learning_rate = phase_lr
            model.lr_schedule = lambda progress: phase_lr
            get_logger().info(f"Reset learning rate to {phase_lr:.1e} for phase {phase_name}")
            
            # Train the phase
            model = train_phase(model, env_fns, phase_name)
            
            # Save model after each phase
            model.save(os.path.join(model_dir, f"ppo_{phase_name}"))
            compute_l2_norm(model)
            
            if len(seen_envs) > 1:
                model = rehearsal_phase(
                    model, seen_envs[:-1], rehearsal_timesteps=100000, n_envs=n_envs)
                model.save(os.path.join(model_dir, f"ppo_{phase_name}_after_rehearsal"))
                get_logger().info(f"Model updated after rehearsal saved to {os.path.join(model_dir, f'ppo_{phase_name}_after_rehearsal')}")
        
        get_logger().info(f"Training completed for reward function: {rf}")
    
    get_logger().info("\nAll training completed.")

if __name__ == '__main__':
    freeze_support()
    main()
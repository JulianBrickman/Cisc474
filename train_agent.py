# Import all the necessary libraries
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

# List of reward functions to try (currently only using rf4)
reward_functions = ["rf4"]

# Set the observation mode to use dictionary observations
coverage_gridworld.custom.OBSERVATION_MODE = "dict"

# List of training phases - we're focusing on sneaky enemies
training_phases = [
    # "just_go",
    # "safe",
    # "maze",
    # "chokepoint"
     "sneaky_enemies",
     "sneaky_enemies",
     "sneaky_enemies",
     "sneaky_enemies",
     "sneaky_enemies",
     "sneaky_enemies",
    
]

# Different learning rates for each phase of training
phase_learning_rates = {
    "sneaky_enemies_1": 4e-4,  # Normal LR to kickstart
    "sneaky_enemies_2": 3e-4,  # Normal LR to kickstart
    "sneaky_enemies_3": 2e-4,  # Normal LR to kickstart
    "sneaky_enemies_4": 1e-4,  # Normal LR to kickstart
    "sneaky_enemies_5": 5e-5,  # Normal LR to kickstart
    "sneaky_enemies_6": 4e-5,  # Slightly lower to refine
}

# Default learning rate and entropy coefficient
LEARNING_RATE = 4e-4
ENTROPY_COEF = 0.01

# List of environments to render (empty means no rendering)
render_list = []

# Number of parallel environments to train on
n_envs = 8

# Create timestamp for saving logs and models
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./training_logs2_{timestamp}"
model_dir = f"./models-{timestamp}"

# Create directories for logs and models
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Set up logging to both file and console
def setup_logger():
    # Create a logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set up file handler for logging to file
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
    fh.setLevel(logging.INFO)
    
    # Set up console handler for logging to screen
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Format the log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

# Get the logger instance
def get_logger():
    return logging.getLogger("train_logger")

# Create an environment with the given parameters
def make_env_func(env_tag, render_mode, monitor_log_path):
    # Create the environment
    env = gym.make(env_tag, render_mode=render_mode,
                   predefined_map_list=None, activate_game_status=False)
    # Wrap it with a monitor to track performance
    env = Monitor(env, filename=monitor_log_path, allow_early_resets=True)
    return env

# Calculate the L2 norm of the model parameters
def compute_l2_norm(model):
    # Sum up the squared norms of all parameters and take square root
    l2_norm = sum(param.data.norm(2).item() **
                  2 for param in model.policy.parameters()) ** 0.5
    get_logger().info(f"L2 norm of model parameters: {l2_norm:.4f}")
    return l2_norm

# Callback to adjust learning rate when training plateaus
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
        # Called at the end of each rollout
        self.rollout_count += 1
        
        # Skip if no episodes completed yet
        if len(self.model.ep_info_buffer) > 0:
            # Calculate current mean reward
            current_mean_reward = np.mean(
                [ep["r"] for ep in self.model.ep_info_buffer])

            # Skip plateau check during warmup
            if self.rollout_count <= self.warmup_rollouts:
                get_logger().info(
                    f"[PlateauLRCallback] Warm-up rollout {self.rollout_count}/{self.warmup_rollouts} — skipping plateau check.")
                return

            # Initialize or update the baseline
            if self.ewma_baseline is None:
                self.ewma_baseline = current_mean_reward
            else:
                new_ewma = self.alpha * current_mean_reward + \
                    (1 - self.alpha) * self.ewma_baseline
                self.ewma_baseline = max(self.ewma_baseline, new_ewma)

            # Log current performance
            get_logger().info(
                f"[Rollout {self.rollout_count}] Mean reward: {current_mean_reward:.2f} | EWMA baseline: {self.ewma_baseline:.2f}")

            # Check if we're improving
            if current_mean_reward > self.ewma_baseline + self.improvement_threshold:
                self.wait = 0  # Reset wait counter if improving
            else:
                self.wait += 1
                # Reduce learning rate if we've waited long enough
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

# Train the model for a specific phase
def train_phase(model, phase_env_fns, phase_name, initial_timesteps=100000):
    # Set up the plateau callback
    plateau_callback = PlateauLRCallback()
    plateau_callback.ewma_baseline = None
    plateau_callback.wait = 0
    plateau_callback.lr_decrease_count = 0
    plateau_callback.rollout_count = 0
    
    # Create vectorized environment
    vec_env = SubprocVecEnv(phase_env_fns)
    model.set_env(vec_env)
    
    # Start training phase
    get_logger().info(f"--- Starting phase: {phase_name} ---")
    
    # Train until we hit a plateau
    while not plateau_callback.reached_plateau():
        model.learn(total_timesteps=initial_timesteps,
                    reset_num_timesteps=False, callback=plateau_callback)

        # Log progress
        if plateau_callback.ewma_baseline:
            get_logger().info(
                f"[{phase_name}] EWMA baseline: {plateau_callback.ewma_baseline:.2f} | LR decreases: {plateau_callback.lr_decrease_count}")

        # Save intermediate model
        model.save(os.path.join(model_dir, f"ppo_{phase_name}_intermediate"))
    
    # Clean up
    vec_env.close()
    get_logger().info(f"--- Phase {phase_name} completed due to plateau ---")
    return model

# Rehearsal phase to train on previously seen environments
def rehearsal_phase(model, seen_envs, rehearsal_timesteps=100000, n_envs=4):
    get_logger().info("Starting multi-map rehearsal phase")

    # Create a function that randomly selects from seen environments
    def mixed_env():
        env_tag = np.random.choice(seen_envs)
        return make_env_func(env_tag, None, os.path.join(log_dir, f"monitor_rehearsal_{env_tag}.csv"))
    
    # Create vectorized environment
    env_fns = [mixed_env for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    model.set_env(vec_env)
    
    # Train with plateau callback
    plateau_callback = PlateauLRCallback()
    model.learn(total_timesteps=rehearsal_timesteps,
                reset_num_timesteps=False, callback=plateau_callback)
    
    # Clean up and return model
    vec_env.close()
    compute_l2_norm(model)
    return model

# Main training function
def main():
    # Set up logging
    setup_logger()
    seen_envs = []
    
    # Train for each reward function
    for rf in reward_functions:
        get_logger().info(f"\n===== Training with Reward Function: {rf} =====")
        coverage_gridworld.custom.REWARD_FUNCTION = rf
        
        # Load existing model
        model_file = "model_20250401_165803.h5"
        model = PPO.load(model_file)
        
        # Train for each phase
        for phase_idx, env_tag in enumerate(training_phases):
            # Set up rendering
            render = "human" if env_tag in render_list else None
            phase_name = f"{env_tag}_phase{phase_idx+1}"
            seen_envs.append(env_tag)

            # Set up logging path
            monitor_log_path = os.path.join(
                log_dir, f"monitor_{rf}_{env_tag}_phase{phase_idx+1}+{timestamp}.csv")

            # Create vectorized environments
            env_fns = [lambda: make_env_func(env_tag, render, monitor_log_path) for _ in range(n_envs)]
            vec_env = SubprocVecEnv(env_fns)

            # Set environment for model
            model.set_env(vec_env)

            # Adjust learning rate for this phase
            phase_key = f"{env_tag}_{phase_idx+1}"
            phase_lr = phase_learning_rates.get(phase_key, LEARNING_RATE)
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = phase_lr
            model.learning_rate = phase_lr
            model.lr_schedule = lambda progress: phase_lr

            # Log learning rate change
            get_logger().info(
                f"Reset learning rate to {phase_lr:.1e} for phase {phase_name}.")
            
            # Train the phase
            model = train_phase(model, env_fns, phase_name)
            
            # Save model
            model.save(os.path.join(model_dir, f"ppo_{phase_name}+{timestamp}"))
            compute_l2_norm(model)
            model.save(f"model_{timestamp}.h5")
            
            # Do rehearsal if we've seen multiple environments
            if len(seen_envs) > 1:
                model = rehearsal_phase(
                    model, seen_envs[:-1], rehearsal_timesteps=100000, n_envs=n_envs)
                model.save(model_file)
                get_logger().info(
                    f"Model updated after rehearsal saved to model_{timestamp}.h5")
        
        get_logger().info(f"Training completed for reward function: {rf}")
    
    get_logger().info("\nAll training completed.")

# Run the main function if this script is executed directly
if __name__ == '__main__':
    freeze_support()
    main()
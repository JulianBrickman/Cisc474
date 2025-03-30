import gymnasium as gym
import coverage_gridworld  # This registers our environments
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np

def make_env(env_id):
    """Helper function to create environment"""
    def _init():
        env = gym.make(env_id, render_mode=None)  # No rendering during training
        return env
    return _init

# List of environments to train on
env_ids = ["just_go", "safe", "maze", "chokepoint", "sneaky_enemies"]

for env_id in env_ids:
    print(f"\nTraining on {env_id} environment...")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(env_id)])
    env = VecMonitor(env)  # Adds episode stats monitoring
    
    # Create PPO model with MultiInputPolicy for dictionary observation space
    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy to MultiInputPolicy
        env,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train the model
    total_timesteps = 100000
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # Save the trained model
    #model.save(f"coverage_gridworld_{env_id}")
    
    # Test the model
    print(f"\nTesting {env_id} model...")
    obs = env.reset()  # DummyVecEnv returns just the observation
    total_reward = 0
    done = False
    steps_taken = 0
    last_info = None
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # Vectorized env returns 4 values
        total_reward += reward[0]
        done = done[0]
        steps_taken += 1
        last_info = info[0]  # Store the last info dict
    
    print(f"Test episode results:")
    print(f"- Total reward: {total_reward}")
    print(f"- Steps taken: {steps_taken}")
    print(f"- Cells covered: {last_info['total_covered_cells']}/{last_info['coverable_cells']} ({last_info['total_covered_cells']/last_info['coverable_cells']*100:.1f}%)")
    print(f"- Cells remaining: {last_info['cells_remaining']}")

env.close() 
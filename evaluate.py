import gymnasium as gym
from stable_baselines3 import PPO
import coverage_gridworld.env  # Ensure environments like "maze" are registered

# Load your trained model (adjust the filename if needed)
model = PPO.load("ppo_maze_200000")  # or "ppo_safe_200000", etc.

# Create the environment with rendering enabled
env = gym.make("maze", render_mode="human")  # or "safe", "chokepoint", etc.

# Reset the environment to start a new episode
obs, _ = env.reset()
done = False

while not done:
    # Let the model predict the next action
    action, _states = model.predict(obs, deterministic=True)

    # Step the environment forward using the chosen action
    obs, reward, done, truncated, info = env.step(action)

# Close the environment window after the episode ends
env.close()

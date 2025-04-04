import gymnasium as gym
from stable_baselines3 import PPO
import coverage_gridworld.env  # Ensure custom envs are registered
import pandas as pd
import os
import time

def evaluate_model(model, env_name="sneaky_enemies", episodes=20, render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    results = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        was_spotted = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            if info.get("game_over", False):
                was_spotted = True

            if render:
                env.render()

        results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps,
            "spotted": was_spotted,
            "covered_cells": info.get("total_covered_cells"),
            "cells_remaining": info.get("cells_remaining"),
        })

    env.close()

    # Print summary
    print(f"\n📊 Evaluation Summary for {env_name}:")
    avg_reward = sum(r["reward"] for r in results) / episodes
    avg_steps = sum(r["steps"] for r in results) / episodes
    spotted_rate = sum(r["spotted"] for r in results) / episodes
    avg_covered = sum(r["covered_cells"] for r in results if r["covered_cells"] is not None) / episodes

    print(f"- Episodes: {episodes}")
    print(f"- Avg Reward: {avg_reward:.2f}")
    print(f"- Avg Steps: {avg_steps:.1f}")
    print(f"- Avg Covered Cells: {avg_covered:.1f}")
    print(f"- Spotted by Enemy: {spotted_rate*100:.1f}%")

    return results

def render_human_episode(model, env_name="sneaky_enemies", sleep_time=0.1):
    """Render a single episode like a human watching."""
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False

    print(f"\n🎮 Starting human-rendered episode in '{env_name}'...")
    time.sleep(1)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(sleep_time)  # Slow down for human viewing

    print("🏁 Episode finished.")
    env.close()

if __name__ == "__main__":
    # Load PPO model
    model_path = "./models-20250402_204729.zip"
    model = PPO.load(model_path)
    render_human_episode(model, env_name="sneaky_enemies", sleep_time=0.2)
   
    training_phases = [
        "sneaky_enemies"
    ]

    all_results = {}

    for env_name in training_phases:
        results = evaluate_model(model, env_name=env_name, episodes=20, render=False)
        df = pd.DataFrame(results)
        csv_filename = f"evaluation_results_{env_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📁 Results saved to '{csv_filename}'")
        all_results[env_name] = df

    print("\n✅ All evaluations complete.")

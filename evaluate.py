import gymnasium as gym
from stable_baselines3 import PPO
import coverage_gridworld.envTwo  # Ensure custom envs are registered
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
             "steps_remaining": info.get("steps_remaining"),
             'coverable_cells': info.get("coverable_cells")
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
    # List of model file paths
    model_paths = [
        'model_20250331_190457.h5',
        'model_20250331_204837.h5',
        'model_20250331_222729.h5',
        'model_20250401_011338.h5',
        'model_20250401_095150.h5',
        'model_20250401_114221.h5',
        'model_20250401_134848.h5',
        'model_20250401_165803.h5',
        'model_20250401_191253.h5',
        'model_20250401_234441.h5',
        'model_20250402_093929.h5',
        'model_20250402_124240.h5',
        'model_20250402_153117.h5',
        'model_20250402_193717.h5',
        'model_20250403_011102.h5',
        'model_20250403_110417.h5',
        'model_20250403_212015.h5'
    ]
    
    training_phases = [
        "sneaky_enemies"
    ]

    all_results = {}

    # Loop through model paths and evaluate each one
    for model_path in model_paths:
        print(f"\n🚀 Evaluating model: {model_path}")
        model = PPO.load(model_path)
        
        # Store results for each model in a separate dictionary for comparison
        model_name = os.path.basename(model_path)  # Get the model file name as the model name
        all_results[model_name] = {}

        for env_name in training_phases:
            results = evaluate_model(model, env_name=env_name, episodes=20, render=False)
            df = pd.DataFrame(results)
            csv_filename = f"evaluation_test_results_{env_name}_{model_name}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"📁 Results for model '{model_name}' saved to '{csv_filename}'")
            all_results[model_name][env_name] = df

    print("\n✅ All evaluations complete.")

    # Optionally, you can also combine results from different models into one large DataFrame for comparison:
    combined_results = []
    for model_name, model_results in all_results.items():
        for env_name, env_results in model_results.items():
            env_results['model'] = model_name  # Add model name as a column for comparison
            env_results['env'] = env_name  # Add environment name for comparison
            combined_results.append(env_results)

    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_csv_filename = "combined_evaluation_results.csv"
    combined_df.to_csv(combined_csv_filename, index=False)
    print(f"📁 Combined results saved to '{combined_csv_filename}'")

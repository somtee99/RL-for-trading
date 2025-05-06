import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from wrapper import get_env_wrapper

# ===== HYPERPARAMETER CONSTANTS =====
DEFAULT_SYMBOL = "AUDCAD"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_MODEL_PATH = f"../models/{DEFAULT_SYMBOL}/ppo_model_20250503_002220"
DEFAULT_EPISODES = 1
DEFAULT_RENDER = True
DEFAULT_MODEL_TYPE = "PPO"

# ===== MODEL TYPE MAPPING =====
MODEL_CLASSES = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
}

def evaluate_model(symbol, model_type, model_path, n_episodes, render):
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(MODEL_CLASSES.keys())}")

    # Load model dynamically
    model_class = MODEL_CLASSES[model_type]
    model = model_class.load(model_path)
    print(f'Loaded {model_type} at {model_path}')

    # Create environment
    env = get_env_wrapper(symbol)()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{symbol}/{model_type}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "evaluation_log.txt")
    log_file = open(log_file_path, "w")

    plt.figure(figsize=(10, 6))

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        balances = []
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if "balance" not in info:
                raise KeyError("Missing 'balance' in info dictionary. Please ensure your environment returns it.")

            balances.append(info["balance"])

            if render:
                env.render()

        print(f"Episode {ep+1} finished. Total reward: {total_reward:.2f}")
        log_file.write(f"Episode {ep+1} total reward: {total_reward:.2f}\n")

        # Plot for this episode
        plt.plot(balances, label=f"Episode {ep+1}")

    env.close()
    log_file.close()

    # Final plot
    plt.title(f"{model_type} Evaluation: Balance over Time ({n_episodes} episodes)")
    plt.xlabel("Timesteps")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(log_dir, "balance_plot.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"\nEvaluation logs saved to: {log_file_path}")
    print(f"Balance plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an RL model and plot balances.")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Symbol for the environment")
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, help="RL model type")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the trained model (no .zip)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", default=DEFAULT_RENDER, help="Render environment during evaluation")
    args = parser.parse_args()

    evaluate_model(args.symbol, args.model_type, args.model_path, args.episodes, args.render)

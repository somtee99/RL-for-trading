import os
import gym
from agents.dqn_agent import DQNAgent
from environments.scalping_env import ScalpingEnv

# Hyperparameters
TIMESTEPS = 1000000
LOG_DIR = "./logs/dqn"
SAVE_PATH = "./models/EURUSD/dqn_model"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize environment
symbol = "EURUSD"  # Example symbol, change as needed
env = ScalpingEnv(symbol=symbol)

agent = DQNAgent(env, log_dir=LOG_DIR)

# Train the agent
agent.train(TIMESTEPS)

# Save the model
agent.save(SAVE_PATH)

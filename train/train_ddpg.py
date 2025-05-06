import os
import gym
from agents.ddpg_agent import DDPGAgent
from environments.scalping_env import ScalpingEnv

# Hyperparameters
TIMESTEPS = 1000000
LOG_DIR = "./logs/ddpg"
SAVE_PATH = "./models/EURUSD/ddpg_model"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize environment
symbol = "EURUSD"  # Example symbol, change as needed
env = ScalpingEnv(symbol=symbol)

agent = DDPGAgent(env, log_dir=LOG_DIR)

# Train the agent
agent.train(TIMESTEPS)

# Save the model
agent.save(SAVE_PATH)

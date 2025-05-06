import os
import sys

# Ensure agents path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from td3_agent import TD3Agent

SYMBOL = "AUDCAD"
TIMESTEPS = 100000
LOG_DIR = "./logs/td3"
SAVE_PATH = f"../models/{SYMBOL}/td3_model"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Train and save model
agent = TD3Agent(symbol=SYMBOL, log_dir=LOG_DIR)
agent.train(TIMESTEPS)
agent.save(SAVE_PATH)
print(f"Model saved at {SAVE_PATH}")

import os
import sys
from datetime import datetime

# Ensure agents path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from ppo_agent import PPOAgent

SYMBOL = "AUDCAD"
TIMESTEPS = 5000000
LOG_DIR = "./logs/ppo"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"../models/{SYMBOL}/ppo_model_{timestamp}"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Train and save model
agent = PPOAgent(symbol=SYMBOL, log_dir=LOG_DIR)
agent.train(TIMESTEPS)
agent.save(SAVE_PATH)
print(f"Model saved at {SAVE_PATH}")


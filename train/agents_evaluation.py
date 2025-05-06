import os
import gym
import numpy as np
import pandas as pd
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environments.scalping_env import ScalpingEnv

# Hyperparameters
EVALUATE_TIMESTEPS = 50000
LOG_DIR = "./logs/agent_evaluation"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize environment
symbol = "EURUSD"  # Example symbol, change as needed
env = ScalpingEnv(symbol=symbol)

# Initialize agents
ddpg_agent = DDPGAgent(env, log_dir=LOG_DIR)
td3_agent = TD3Agent(env, log_dir=LOG_DIR)
dqn_agent = DQNAgent(env, log_dir=LOG_DIR)
ppo_agent = PPOAgent(env, log_dir=LOG_DIR)

# Function to evaluate agents
def evaluate_agent(agent, timesteps):
    rewards = []
    for _ in range(timesteps):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.model.predict(state)[0]  # For PPO, DQN etc.
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    return avg_reward

# Evaluate all agents
ddpg_avg_reward = evaluate_agent(ddpg_agent, EVALUATE_TIMESTEPS)
td3_avg_reward = evaluate_agent(td3_agent, EVALUATE_TIMESTEPS)
dqn_avg_reward = evaluate_agent(dqn_agent, EVALUATE_TIMESTEPS)
ppo_avg_reward = evaluate_agent(ppo_agent, EVALUATE_TIMESTEPS)

# Save evaluation results
evaluation_data = {
    "Agent": ["DDPG", "TD3", "DQN", "PPO"],
    "Average Reward": [ddpg_avg_reward, td3_avg_reward, dqn_avg_reward, ppo_avg_reward]
}
df = pd.DataFrame(evaluation_data)
df.to_csv(os.path.join(LOG_DIR, "agent_comparison.csv"), index=False)

import os
import gym
from agents.monte_carlo_agent import MonteCarloAgent
from environments.scalping_env import ScalpingEnv

# Hyperparameters
TIMESTEPS = 1000000
SAVE_PATH = "./models/EURUSD/monte_carlo_model"

# Ensure log directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize environment
symbol = "EURUSD"  # Example symbol, change as needed
env = ScalpingEnv(symbol=symbol)

agent = MonteCarloAgent(env.action_space)

# Train the agent
# Monte Carlo training typically doesn't use timesteps like other RL algorithms
# Instead, we train by episodic updates and sample from episodes
for episode in range(TIMESTEPS):
    episode_data = []
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
    agent.update(episode_data)

# Save the model
agent.save(SAVE_PATH)

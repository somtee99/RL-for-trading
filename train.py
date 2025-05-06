import argparse
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environments.scalping_env import ScalpingEnv

def train_agent(agent_type, symbol, timesteps):
    """
    Train the specified agent using the provided environment and timesteps.
    """
    env = ScalpingEnv(symbol=symbol)

    if agent_type == 'ddpg':
        agent = DDPGAgent(env, log_dir="./logs/ddpg")
    elif agent_type == 'td3':
        agent = TD3Agent(env, log_dir="./logs/td3")
    elif agent_type == 'monte_carlo':
        agent = MonteCarloAgent(env.action_space)
    elif agent_type == 'dqn':
        agent = DQNAgent(env, log_dir="./logs/dqn")
    elif agent_type == 'ppo':
        agent = PPOAgent(env, log_dir="./logs/ppo")
    else:
        raise ValueError("Invalid agent type.")

    # Train agent
    agent.train(timesteps)
    agent.save(f"models/{agent_type}_{symbol}_trained_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a trading agent.")
    parser.add_argument("agent_type", type=str, choices=['ddpg', 'td3', 'monte_carlo', 'dqn', 'ppo'], help="Type of agent to train.")
    parser.add_argument("symbol", type=str, help="Symbol to train the agent on.")
    parser.add_argument("timesteps", type=int, help="Number of timesteps for training.")

    args = parser.parse_args()

    train_agent(args.agent_type, args.symbol, args.timesteps)

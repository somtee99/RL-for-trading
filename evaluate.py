import argparse
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environments.scalping_env import ScalpingEnv

def evaluate_agent(agent_type, symbol, model_path):
    """
    Evaluate the performance of a pre-trained agent on the given environment.
    """
    env = ScalpingEnv(symbol=symbol)

    if agent_type == 'ddpg':
        agent = DDPGAgent(env, log_dir="./logs/ddpg")
        agent.load(model_path)
    elif agent_type == 'td3':
        agent = TD3Agent(env, log_dir="./logs/td3")
        agent.load(model_path)
    elif agent_type == 'monte_carlo':
        agent = MonteCarloAgent(env.action_space)
        agent.load(model_path)
    elif agent_type == 'dqn':
        agent = DQNAgent(env, log_dir="./logs/dqn")
        agent.load(model_path)
    elif agent_type == 'ppo':
        agent = PPOAgent(env, log_dir="./logs/ppo")
        agent.load(model_path)
    else:
        raise ValueError("Invalid agent type.")

    # Run evaluation
    total_reward = 0
    total_steps = 0
    while total_steps < 1000:  # Example of evaluation for 1000 steps
        action, _ = agent.model.predict(env.state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print(f"Total reward for {agent_type} agent: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trading agent.")
    parser.add_argument("agent_type", type=str, choices=['ddpg', 'td3', 'monte_carlo', 'dqn', 'ppo'], help="Type of agent to evaluate.")
    parser.add_argument("symbol", type=str, help="Symbol to evaluate the agent on.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model.")

    args = parser.parse_args()

    evaluate_agent(args.agent_type, args.symbol, args.model_path)

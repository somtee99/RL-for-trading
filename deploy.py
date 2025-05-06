import argparse
from deployment.live_runner import run_live_trading

def deploy_agent(agent_type, symbol, model_path):
    """
    Deploy a pre-trained agent to run live trading.
    """
    run_live_trading(agent_type, symbol, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a trading agent for live trading.")
    parser.add_argument("agent_type", type=str, choices=['ddpg', 'td3', 'monte_carlo', 'dqn', 'ppo'], help="Type of agent to deploy.")
    parser.add_argument("symbol", type=str, help="Symbol to deploy the agent on.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model.")

    args = parser.parse_args()

    deploy_agent(args.agent_type, args.symbol, args.model_path)

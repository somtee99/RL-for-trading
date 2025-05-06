import time
import logging
from deployment.mt_executor import execute_trade
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.monte_carlo_agent import MonteCarloAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environments.scalping_env import ScalpingEnv

# Setup logger
logger = logging.getLogger('live_runner')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('live_runner.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def run_live_trading(agent_type, symbol, model_path):
    """
    Run live trading for the specified agent type with a pre-trained model.
    """
    try:
        # Initialize environment
        env = ScalpingEnv(symbol=symbol)
        
        # Initialize agent
        if agent_type == 'ddpg':
            agent = DDPGAgent(env, log_dir=None)
            agent.load(model_path)
        elif agent_type == 'td3':
            agent = TD3Agent(env, log_dir=None)
            agent.load(model_path)
        elif agent_type == 'monte_carlo':
            agent = MonteCarloAgent(env.action_space)
            agent.load(model_path)  # Assuming Monte Carlo agent can load a model
        elif agent_type == 'dqn':
            agent = DQNAgent(env, log_dir=None)
            agent.load(model_path)
        elif agent_type == 'ppo':
            agent = PPOAgent(env, log_dir=None)
            agent.load(model_path)
        else:
            raise ValueError("Invalid agent type.")

        logger.info(f"Running live trading with {agent_type} agent using pre-trained model.")

        position = 0  # 0 = no position, 1 = long, -1 = short

        while True:
            # Predict action (Use the pre-trained model to make decisions)
            action, _ = agent.model.predict(env.state)  # Assuming predict method exists
            logger.info(f"Predicted Action: {action}")

            # Execute trade directly
            if action == 0:  # Hold
                if position != 0:
                    logger.info("Hold action executed. No position changes.")
                else:
                    logger.info("Hold action executed, but no position to hold.")

            elif action == 1:  # Buy (close short position)
                if position == -1:
                    # Close short position
                    position = 0  # No position after closing
                    execute_trade(symbol, "buy", 1)  # Close short position
                    logger.info("Close short position executed.")
                elif position == 0:
                    # Open long position
                    position = 1
                    execute_trade(symbol, "buy", 1)  # Open long position
                    logger.info("Open long position executed.")
                else:
                    logger.info("Buy action executed, but already in long position.")

            elif action == 2:  # Sell (close long position)
                if position == 1:
                    # Close long position
                    position = 0  # No position after closing
                    execute_trade(symbol, "sell", 1)  # Close long position
                    logger.info("Close long position executed.")
                elif position == 0:
                    # Open short position
                    position = -1
                    execute_trade(symbol, "sell", 1)  # Open short position
                    logger.info("Open short position executed.")
                else:
                    logger.info("Sell action executed, but already in short position.")

            time.sleep(60)  # Run every minute

    except Exception as e:
        logger.error(f"Error in live trading: {e}")

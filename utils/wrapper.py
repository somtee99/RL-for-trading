import os
import pandas as pd
import sys
import json

# Ensure utils path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../environments')))
from scalping_env import ScalpingEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from loader import load_data

TIMEFRAME_IN_MINUTES = 5
MARKET_CONFIG_PATH = '../environments/market_config.json'

def get_env_wrapper(symbol, market_config_path=MARKET_CONFIG_PATH):

    def wrapper():
        # Load market config from JSON
        with open(market_config_path, 'r') as f:
            market_config = json.load(f)

        # Load the market data
        data_file = f'../data/{symbol}{TIMEFRAME_IN_MINUTES}.csv' 
        data = load_data(data_file)

        # # Split the data into training (80%) and testing (20%)
        # train_size = int(0.8 * len(data))
        # train_data = data[:train_size]
        # test_data = data[train_size:]

        # Pass the data and config to the ScalpingEnv
        return ScalpingEnv(symbol=symbol, data=data, market_config=market_config)

    return wrapper

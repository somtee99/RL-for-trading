import alpaca_trade_api as tradeapi
import json
import logging
from deployment.deployment_config import load_deployment_config

# Hyperparameters
API_KEY = load_deployment_config()["alpaca_api_key"]
API_SECRET = load_deployment_config()["alpaca_api_secret"]
BASE_URL = load_deployment_config()["alpaca_base_url"]

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Setup logger
logger = logging.getLogger('alpaca_stream')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('alpaca_stream.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def stream_order(symbol, action, qty):
    """
    Sends order to Alpaca based on action (buy/sell).
    """
    try:
        if action == "buy":
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif action == "sell":
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        logger.info(f"Order placed: {action} {qty} {symbol}")
    except Exception as e:
        logger.error(f"Error placing order: {e}")


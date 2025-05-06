import MetaTrader5 as mt5
import logging
from deployment.deployment_config import load_deployment_config

# Hyperparameters
MT5_ACCOUNT = load_deployment_config()["mt5_account"]
MT5_PASSWORD = load_deployment_config()["mt5_password"]
MT5_SERVER = load_deployment_config()["mt5_server"]

# Initialize MT5
mt5.initialize()
mt5.login(MT5_ACCOUNT, MT5_PASSWORD, server=MT5_SERVER)

# Setup logger
logger = logging.getLogger('mt_executor')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('mt_executor.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

def execute_trade(symbol, action, volume):
    """
    Execute trade on MetaTrader 5 based on action (buy/sell).
    """
    try:
        if action == "buy":
            order = mt5.ORDER_TYPE_BUY
        elif action == "sell":
            order = mt5.ORDER_TYPE_SELL

        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": 0,
            "tp": 0,
            "deviation": 20,
            "magic": 234000,
            "comment": "Scalping trade",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC
        }
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to place order: {result.comment}")
        else:
            logger.info(f"Trade executed: {action} {volume} {symbol}")
    except Exception as e:
        logger.error(f"Error executing trade: {e}")

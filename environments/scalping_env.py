import numpy as np
import gym
from gym import spaces
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from scaler import scale_input_features
from loader import load_latest_lstm_model

# ----------------------------
# Hyperparameters
# ----------------------------
LOOKAHEAD_BARS = 3 # Number of future bars to look ahead for reward calculation
TRANSACTION_COST = 0.000
WINDOW_SIZE = 30  # should be same as LSTM seq length
ACCOUNT_BALANCE = 100
IMPOSSIBLE_ACTION_PENALTY = 0.1  # Penalty for trying to buy/sell when already in position
MAX_HOLD_TIME = 5 # Max time steps to hold a position
HOLDING_PENALTY = 0.01  # Penalty for holding too long
CLOSE_REWARD_COEF = 10  # Reward Coefficient for closing a position
BLOWN_ACCOUNT_PENALTY = 100  # Penalty for account blowout

class ScalpingEnv(gym.Env):
    def __init__(self, symbol, data, market_config, window_size=WINDOW_SIZE):
        super(ScalpingEnv, self).__init__()

        self.symbol = symbol
        self.data = data.reset_index(drop=True)
        self.market_config = market_config
        self.config = self.market_config[symbol]
        self.window_size = window_size
        self.current_step = self.window_size
        self.initial_balance = ACCOUNT_BALANCE
        self.balance = self.initial_balance
        
        # Initialize random seed (None by default)
        self.np_random = np.random.RandomState()

        # Load LSTM Model
        # self.lstm_model = load_latest_lstm_model(symbol)

        self.action_space = spaces.Discrete(4)  # 0 = hold, 1 = buy, 2 = sell, 3 = close
        self.position = 0
        self.entry_price = 0
        self.hold_time = 0

        # Initialize observation space shape after building one observation
        dummy_obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dummy_obs.shape[0],),
            dtype=np.float32
        )


    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.data.iloc[start:end]

        # Pad if window is too small
        if len(window) < self.window_size:
            pad_len = self.window_size - len(window)
            padding = pd.concat([self.data.iloc[[0]]] * pad_len)
            window = pd.concat([padding, window])

        # Scaled input features for observation
        scaled = scale_input_features(window)

        # Scale entry price using min/max from current window
        entry_price_scaled = 0.0
        if self.entry_price > 0:
            close_prices = window['close'].values
            min_close = np.min(close_prices)
            max_close = np.max(close_prices)

            # Avoid division by zero if all close prices are the same
            if max_close != min_close:
                entry_price_scaled = (self.entry_price - min_close) / (max_close - min_close)
            else:
                entry_price_scaled = 0.5  # neutral scaling when range is zero

        normalized_balance = self.balance / self.initial_balance

        return np.concatenate([
            scaled[-1],               # Last timestep's scaled features
            [self.position],          # Position
            [entry_price_scaled],     # Scaled entry price (local window)
            [normalized_balance]      # Normalized account balance
        ])


    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.hold_time = 0
        self.balance = self.initial_balance

        return self._get_observation()

    def step(self, action):
        reward = 0
        pnl = 0
        done = False
        current_price = self.data.iloc[self.current_step]['close']

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.hold_time = 0
            else:
                reward -= IMPOSSIBLE_ACTION_PENALTY

        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.hold_time = 0
            else:
                reward -= IMPOSSIBLE_ACTION_PENALTY

        elif action == 3:  # Close
            if self.position == 1:
                pnl = (current_price - self.entry_price) - TRANSACTION_COST
            elif self.position == -1:
                pnl = (self.entry_price - current_price) - TRANSACTION_COST
            else:
                reward -= IMPOSSIBLE_ACTION_PENALTY

            reward += pnl * CLOSE_REWARD_COEF
            self.balance += pnl
            self.position = 0
            self.entry_price = 0
            self.hold_time = 0

        # Holding penalty and reward calculation
        if self.position != 0:
            price_change = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            reward += price_change

            self.hold_time += 1
            if self.hold_time > MAX_HOLD_TIME:
                reward -= HOLDING_PENALTY * (self.hold_time - MAX_HOLD_TIME)

        # Lookahead bonus
        if self.current_step + LOOKAHEAD_BARS < len(self.data):
            future_price = self.data.iloc[self.current_step + LOOKAHEAD_BARS]['close']
            if self.position != 0:
                direction = self.position
                future_reward = direction * (future_price - current_price)
                reward += future_reward

        # Check for account blowout
        if self.balance <= 0:
            reward -= BLOWN_ACCOUNT_PENALTY
            done = True

        self.current_step += 1
        done = done or self.current_step >= len(self.data) - LOOKAHEAD_BARS

        return self._get_observation(), reward, done, {"balance": self.balance, "pnl": pnl}

    
    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.np_random.seed(seed)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Position: {'Long' if self.position == 1 else 'Short' if self.position == -1 else 'None'}")
        print(f"Current Price: {self.data.iloc[self.current_step]['close']:.4f}")
        print(f"Hold Time: {self.hold_time}")
        print(f"Entry Price: {self.entry_price:.4f}" if self.entry_price else "Entry Price: None")
        print(f"Balance: {self.balance:.2f}")
        print("-" * 40)
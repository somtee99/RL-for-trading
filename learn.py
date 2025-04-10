#!pip install tensorflow gym numpy pandas matplotlib seaborn stable-baselines3

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib
import gym
from gym import spaces

def load_and_resample_data(file_path):
    # Load the data from CSV
    data = pd.read_csv(file_path, sep='\t', header=None)

     # Rename columns to match the expected format
    data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    return pd.DataFrame(data)

# data_1d = load_and_resample_data('./EURCAD1440.csv')
# data_4h = load_and_resample_data('./EURCAD240.csv')
# data_60m = load_and_resample_data('./EURCAD60.csv')
data_30m = load_and_resample_data('C:/Users/okafo/Downloads/EURCAD30.csv')

def preprocess_data(data, window_size=60):
    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])
    
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data_scaled[i-window_size:i, 0])  # Last `window_size` closing prices
        y.append(data_scaled[i, 0])  # Next closing price

    X = np.array(X)
    y = np.array(y)

    # Reshape X for LSTM (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

X, y, scaler = preprocess_data(data_30m)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))  # Output layer

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

input_shape = (X.shape[1], 1)
lstm_model = build_lstm_model(input_shape)
lstm_model.summary()

class ForexTradingEnv(gym.Env):
    def __init__(self, data, model, window_size=60, initial_balance=50):
        super(ForexTradingEnv, self).__init__()

        self.data = data
        self.model = model
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size, 1), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0           # 0 = Flat, 1 = Long, -1 = Short
        self.entry_price = None
        self.done = False
        self.balance_history = [self.balance]
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step - self.window_size: self.current_step]

    def step(self, action):
        state = self._get_state()
        current_price = self.data[self.current_step][0]
        reward = 0
        account_blow_penalty = -1000

        # Long logic
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                # Close short
                reward = self.entry_price - current_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        # Short logic
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                # Close long
                reward = current_price - self.entry_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        # Append new balance
        self.balance_history.append(self.balance)

        # Account blow logic
        if self.balance < 0:
            reward += account_blow_penalty
            self.done = True
            print("❌ Account blown! Resetting environment.")
            return state, reward, self.done, {"reason": "account_blow"}

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return state, reward, self.done, {}

    def render(self):
        # Optionally plot or print current balance and position
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")

#!pip install 'shimmy>=2.0'

from stable_baselines3 import PPO

# Create the environment
env = ForexTradingEnv(data_30m[['close']].values, lstm_model)

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)


# After training, we can plot the balance history
plt.figure(figsize=(10, 6))
plt.plot(env.balance_history, label='Account Balance')
plt.title("Agent's Trading Performance Over Time")
plt.xlabel('Step')
plt.ylabel('Account Balance')
plt.legend()
plt.show()

# Test the agent after training
obs = env.reset()
for _ in range(100):  # Run the agent for 100 steps
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

# Plot the performance (balance history)
plt.figure(figsize=(10, 6))
plt.plot(env.balance_history, label='Account Balance')
plt.title("Test Agent's Trading Performance Over Time")
plt.xlabel('Step')
plt.ylabel('Account Balance')
plt.legend()
plt.show()

model.save("lstm_model.h5")  # Save the model for future use
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)  # Save the scaler for future use

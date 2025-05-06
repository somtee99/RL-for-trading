import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import datetime
import sys

# Ensure utils path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from data_loader import load_data

# Hyperparameters
SEQ_LENGTH = 30
LOOKAHEAD_TIMESTEPS = 5
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
SYMBOL = 'AUDCAD'
TIMEFRAME_IN_MINUTES = 5
LOG_DIR = './logs/lstm_training'

# LSTM model creation
def create_lstm_model(input_shape, lookahead_timesteps):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(lookahead_timesteps, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
    return model

# Extract close prices from CSV
def load_close_prices_from_data(csv_file_path):
    data = load_data(csv_file_path)
    return data['close'].values

# Prepare X, y sequences and split into train/test
def prepare_data(close_prices, seq_length, lookahead_timesteps):
    sequences = []
    targets = []

    for i in range(len(close_prices) - seq_length - lookahead_timesteps):
        x = close_prices[i:i + seq_length].reshape(-1, 1)
        y = close_prices[i + seq_length:i + seq_length + lookahead_timesteps]
        sequences.append(x)
        targets.append(y)

    X = np.array(sequences)
    y = np.array(targets)

    split_idx = int(0.8 * len(X))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

# Main training and evaluation routine
def train_lstm():
    file_path = f'../data/{SYMBOL}{TIMEFRAME_IN_MINUTES}.csv'
    close_prices = load_close_prices_from_data(file_path)

    X_train, y_train, X_test, y_test = prepare_data(close_prices, SEQ_LENGTH, LOOKAHEAD_TIMESTEPS)

    model = create_lstm_model((SEQ_LENGTH, 1), LOOKAHEAD_TIMESTEPS)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Evaluation on Test Set - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = f"../models/{SYMBOL}"
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.h5")
    model.save(model_save_path)

    print(f"Model saved at: {model_save_path}")
    return model

if __name__ == "__main__":
    train_lstm()

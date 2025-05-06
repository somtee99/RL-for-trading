import pandas as pd
import os
from tensorflow.keras.models import load_model

def load_latest_lstm_model(symbol):
    # Get all files in the folder for the given symbol
    model_dir = f"../models/{symbol}"
    
    # List all files in the directory
    all_files = os.listdir(model_dir)
    
    # Filter out files that match the pattern lstm_model_*.h5
    model_files = [f for f in all_files if f.startswith("lstm_model_") and f.endswith(".h5")]
    
    if not model_files:
        raise FileNotFoundError(f"No LSTM model found for symbol: {symbol}")
    
    # Sort the files based on timestamp, descending order to get the most recent model
    model_files.sort(reverse=True)  # Sort in descending order
    
    # Get the latest model file
    latest_model_file = model_files[0]
    
    # Load the model
    model_path = os.path.join(model_dir, latest_model_file)
    print(f"Loading the latest model: {latest_model_file}")
    
    lstm_model = load_model(model_path)
    
    return lstm_model

def load_data(file_path):
    """
    Load the data from CSV, clean it and prepare it for use in the environment.
    """
    data = pd.read_csv(file_path, sep='\t', header=None)

    # Rename columns to match the expected format
    data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    return data
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

SCALER_PATH = "scalers/minmax_scaler.pkl"

def scale_input_features(data, feature_range=(0, 1), fit_new=True):
    """
    Scales input features using MinMaxScaler.
    If fit_new is True, creates a new scaler and saves it.
    """
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    if fit_new:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(data)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(data)

    return scaled_data

def inverse_scale_features(scaled_data):
    """
    Inverses scaling using the saved scaler.
    """
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(scaled_data)

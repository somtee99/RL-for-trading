from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load model and scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")  # Save your scaler after training

app = FastAPI(title="Forex LSTM Trader")

# Define the structure of input data
class CandleData(BaseModel):
    close_prices: list  # e.g., last 60 close prices

def prepare_input(data):
    scaled = scaler.transform(np.array(data).reshape(-1, 1))
    return np.expand_dims(scaled, axis=0)  # shape (1, window_size, 1)

@app.post("/predict")
def predict(data: CandleData):
    close_prices = data.close_prices
    if len(close_prices) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 prices")

    try:
        input_seq = prepare_input(close_prices[-60:])
        prediction = model.predict(input_seq)[0][0]

        if prediction > close_prices[-1] * 1.001:
            action = "buy"
        elif prediction < close_prices[-1] * 0.999:
            action = "sell"
        else:
            action = "hold"

        return {
            "predicted_price": float(prediction),
            "last_price": close_prices[-1],
            "action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

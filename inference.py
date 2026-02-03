import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from fetch_data import fetch_klines
from preprocessing import wrangling
from train_models import SimpleRNN, LSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
FEATURE_COLS = ["open", "high", "low", "close", "volume"]


def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, model_class, input_size=5, hidden_size=64):
    # Load model
    model = model_class(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load scalers
    scaler_X = torch.load(scaler_X_path)
    scaler_y = torch.load(scaler_y_path)

    return model, scaler_X, scaler_y

def prepare_latest_window(df, scaler_X):
    X = scaler_X.transform(df[FEATURE_COLS].values)
    X_window = X[-WINDOW_SIZE:]  # last 30 rows
    X_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0)  # shape: (1, 30, 5)
    return X_tensor.to(DEVICE)



def predict_next_day(model, df, scaler_X, scaler_y):
    X_tensor = prepare_latest_window(df, scaler_X)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()
    pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
    return pred_price

def plot_prediction(df, predicted_price, symbol):
    recent_close = df["close"].values[-WINDOW_SIZE:]
    plt.figure(figsize=(10, 5))
    plt.plot(range(WINDOW_SIZE), recent_close, label="Last 30 Days Close")
    plt.axhline(predicted_price, linestyle="--", color="red", label="Predicted Next Close")
    plt.title(f"{symbol} Next-Day Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    os.makedirs("data/plot", exist_ok=True)
    plt.savefig(f"data/plot/{symbol.lower()}_future_prediction.png")
    plt.show()
    

def run_inference(symbol, model_path, scaler_X_path, scaler_y_path, model_class, start_date, interval):
    # 1. Fetch recent data (2026)
    df = fetch_klines(symbol=symbol, interval=interval,
                      start_date=start_date,
                      end_date=pd.Timestamp.today().strftime("%Y-%m-%d"))
    df = wrangling(df)

    # 2. Load model and scalers
    model, scaler_X, scaler_y = load_model_and_scalers(
        model_path, scaler_X_path, scaler_y_path, model_class
    )

    # 3. Predict next-day close
    next_day_price = predict_next_day(model, df, scaler_X, scaler_y)
    print(f"Predicted next-day close price for {symbol}: {next_day_price:.2f}")

    # 4. Plot prediction
    plot_prediction(df, next_day_price, symbol)

    return next_day_price
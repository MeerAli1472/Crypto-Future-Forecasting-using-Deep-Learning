# 🚀 Crypto Price Prediction Using RNN & LSTM

This project explores **deep learning–based time series forecasting** for cryptocurrency prices using **daily OHLCV data**.  
We compare **Simple RNN** and **LSTM** architectures on **BTC and ETH**, highlighting how model complexity interacts with market volatility.

---

## 📌 Key Highlights

- 📅 4 years of daily data (2021–2025)
- 🪟 Sliding window approach (30 days → predict next day)
- 🤖 Models: Simple RNN & LSTM (PyTorch)
- 📊 Coins: BTC (high volatility), ETH (lower volatility)
- 📈 Metric: R² Score
- 🔥 Insight: *Simpler models generalize better in highly volatile markets*

---

## 🧠 Motivation

Traditional ML models struggled to capture temporal dependencies with limited data.  
Deep learning models benefit from:
- longer histories
- nonlinear pattern learning
- sequential memory

However, **model capacity must match market behavior**, which this project demonstrates clearly.

---

## 🏗️ Project Architecture

**Just run Crypto forcasting using RNN & LSTM.ipynb by passing arguments.
```text
data_fetch.py      → Fetch OHLCV data from Binance API
preprocessing.py  → Cleaning, feature engineering, scaling, windowing
models.py         → RNN & LSTM architectures + training loops
inference.py      → Live prediction using latest market data

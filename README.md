# XAUUSD AI Resonance Trading Engine (Level 5)

An advanced, fully autonomous algorithmic trading system for MetaTrader 5 (MT5), specializing in XAUUSD (Gold). The system is powered by a **Level 5 Multi-Timeframe Resonance Engine** that combines extreme micro-momentum (M1) with macro-trend alignment (D1) using Ensemble Machine Learning (XGBoost + Random Forest).

## 🌟 Key Features

* **Full-Spectrum Timeframe Analysis (MTF):** Simultaneously processes data from M1, M5, M15, M30, H1, H4, and D1 to detect market resonance.
* **31-Dimension Feature Engineering:** Incorporates technical indicators (MACD, CCI, ADX, Bollinger Bands), candlestick morphology, and session timing (US/Asia markets).
* **Ensemble Machine Learning:** Combines the high precision of Random Forest with the aggressive trend-capturing capabilities of XGBoost.
* **Dynamic Sniper Mode:** Allows real-time adjustment of AI confidence thresholds and model weighting (e.g., 70% XGB / 30% RF).
* **Interactive UI:** A modern, dark-themed GUI control panel built with Tkinter for seamless monitoring and parameter adjustment.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch MetaTrader 5**: Ensure MT5 is running and logged into your account.
3. **Start the Engine**:
   ```bash
   python main.py
   ```
   *Connect to MT5 via the UI and adjust your lot size and AI weights via the intuitive slider.*

## 🧠 AI Training Pipeline

To retrain the AI models with the latest market data:

1. **Export History Data**: 
   Pulls historical ticks and bars from MT5.
   ```bash
   python tools/export_history.py
   ```
2. **Generate Dataset**: 
   Aligns all timeframes (M1 to D1) using `merge_asof` to prevent look-ahead bias and calculates the 31-dimension feature set.
   ```bash
   python ml_engine/create_dataset.py
   ```
3. **Train Models**: 
   Trains independent Buy/Sell models for both XGBoost and Random Forest.
   ```bash
   python ml_engine/train_model.py
   ```
4. **Evaluate Performance**: 
   Runs a high-precision, tick-level simulated backtest.
   ```bash
   python ml_engine/ml_backtest_m1.py
   ```

## ⚠️ Disclaimer
This software is for educational and research purposes only. Algorithmic trading carries significant risk. Past performance in backtests is not indicative of future results.

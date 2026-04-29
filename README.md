# XAUUSD AI Resonance Trading Engine (Level 5)

An advanced, fully autonomous algorithmic trading system for MetaTrader 5 (MT5), specializing in XAUUSD (Gold). The system runs two independent AI engines simultaneously — one optimized for maximum profit, one for maximum win rate — each with its own empirically-derived parameters.

## 🌟 Key Features

* **Dual-Model Independent Execution:** Two AI agents run in parallel, each placing trades based on their own specialized logic and tracking performance separately.
  * **Model A (Profit Engine — XGB 100%):** Threshold 0.60 · TP 10.0 · SL 8.0 — aggressive trend follower targeting large moves.
  * **Model B (Win-Rate Engine — RF 90% / XGB 10%):** Threshold 0.65 · TP 5.0 · SL 8.0 — ultra-precise sniper with ~99% historical win rate.
* **Full-Spectrum Timeframe Analysis (MTF):** Simultaneously processes M1, M5, M15, M30, H1, H4, and D1 to detect multi-timeframe resonance.
* **32-Dimension Feature Engineering:** Incorporates technical indicators (MACD, CCI, ADX, Bollinger Bands), candlestick morphology, session timing (US/Asia), and micro-momentum (M1 EMA slope).
* **Real-Time CSV Trade Tracking (`trade_records.csv`):** Every order is logged with model name, ticket, TP/SL, and outcome. The system auto-reconciles closed trades via MT5 history in the background.
* **Grand Simulation Engine (`hyper_optimizer.py`):** NumPy-vectorized grid search across 2,970+ parameter combinations (TP/SL/Threshold/Mix ratio) to empirically discover optimal settings.
* **Minimal Dark-Mode GUI:** Live per-model Buy/Sell confidence scores, today's P&L from CSV, and a real-time scanning animation — no clutter.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch MetaTrader 5**: Ensure MT5 is running and logged into your account.
3. **Configure Lot Sizes** in `main.py` (default both `0.1`):
   ```python
   ml_ai = MLResonanceStrategy(..., lot_size_A=0.1, lot_size_B=0.1, ...)
   ```
4. **Start the Engine**:
   ```bash
   python main.py
   ```
   Connect to MT5 via the GUI. Set **A手數** (Model A — Profit Engine) and **B手數** (Model B — Win-Rate Engine) independently in the control bar, then click **Start**.

## 🧠 AI Training Pipeline

To retrain the AI models with the latest market data:

1. **Export History Data**:
   ```bash
   python tools/export_history.py
   ```
2. **Generate Dataset** (32-dimension features with look-ahead-safe labeling):
   ```bash
   python ml_engine/create_dataset.py
   ```
3. **Train Models** (XGBoost + Random Forest, independent Buy/Sell models):
   ```bash
   python ml_engine/train_model.py
   ```
4. **Backtest & Validate** (tick-level M1 simulation):
   ```bash
   python ml_engine/ml_backtest_m1.py
   ```
5. **Run Grand Simulation** (grid search across all TP/SL/Threshold/Mix combinations):
   ```bash
   python ml_engine/hyper_optimizer.py
   ```

## 📂 Project Structure

```
gold/
├── core/
│   ├── engine.py          # Strategy engine & tick loop
│   ├── execution.py       # MT5 order management
│   ├── mt5_connector.py   # MT5 connection & data fetching
│   └── trade_tracker.py   # CSV-based per-model performance tracking
├── ml_engine/
│   ├── create_dataset.py  # Feature engineering & label generation
│   ├── train_model.py     # XGBoost + Random Forest training
│   ├── ml_backtest_m1.py  # M1-resolution backtest engine
│   └── hyper_optimizer.py # 2970-combination grand simulation
├── models/                # Trained model files (.pkl)
├── strategies/
│   └── ml_resonance_strategy.py  # Dual-model live execution logic
├── tools/
│   └── export_history.py  # MT5 history data exporter
├── ui/
│   └── app_gui.py         # Tkinter dark-mode control panel
└── main.py                # Entry point
```

## ⚠️ Disclaimer
This software is for educational and research purposes only. Algorithmic trading carries significant risk. Past backtest performance is not indicative of future results.

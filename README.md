# 黃金 AI 自動交易系統 (XAUUSD)

針對 MetaTrader 5 (MT5) 平台開發的全自動化 XAUUSD（黃金）量化交易系統。  
系統搭載**雙引擎 AI 架構**，以 XGBoost + Random Forest 集成學習驅動，融合 M1 至 D1 全時區共振訊號，同時運行兩套獨立策略：一套追求最大利潤、一套追求極高勝率。

---

## ✨ 核心功能

- **全時區布林同步 (MT5 Parity)**：手寫母體標準差公式 (ddof=0)，全線統一採用 BB(21, 2.1) 偏差，確保 AI 與 MT5 圖表 100% 數值對齊。
- **動態盤整評分**：採用 48 根 K 棒 (4小時) 動態窗口，靈活捕捉市場規律切換，防止歷史極端波動干擾。
- **解鎖動能策略**：優化 PPT Stage 4-1/4-2 邏輯，移除滯後的大時區趨勢限制，第一時間捕捉 M5 強力突破行情。
- **即時指標監控面板**：GUI 仪表板新增 Live Indicators 區塊，即時顯示 EMA12、BB(21, 2.1) 與盤整分數。
- **雙引擎獨立執行**：兩個 AI 代理各自獨立下單，互不干擾：
  - **模型 A（利潤引擎，XGB 100%）**：信心門檻 0.60、TP 10.0、SL 8.0，積極捕捉大波段。
  - **模型 B（勝率引擎，RF 90% / XGB 10%）**：信心門檻 0.65、TP 5.0、SL 8.0，歷史勝率接近 99%。
- **全時區多框架分析（MTF）**：同時讀取 M1、M5、M15、M30、H1、H4、D1，偵測多時區共振進場訊號。
- **32 維特徵工程**：涵蓋技術指標（MACD、CCI、ADX、布林通道）、K 棒型態、美亞時段標記、M1 微觀動能斜率。
- **實盤 CSV 績效追蹤（`trade_records.csv`）**：每筆訂單自動記錄模型名稱、Ticket、TP/SL 與損益，每 10 個 Tick 自動與 MT5 歷史對帳。
- **一鍵 GUI 重訓**：點擊「🔄 重訓模型」按鈕即可在背景執行資料匯出 → 特徵工程 → 模型訓練，完成後自動熱重載新模型，EA 無需重啟。
- **深度參數最佳化（`hyper_optimizer.py`）**：NumPy 向量化網格搜索，覆蓋 2,970+ 種 TP/SL/門檻/混合比例組合。
- **簡潔深色 GUI**：即時顯示每個模型的多空信心分數、今日損益（直接來自 CSV）與掃描動畫。

---

## 🚀 快速啟動

1. **安裝依賴套件**：
   ```bash
   pip install -r requirements.txt
   ```

2. **啟動 MetaTrader 5**：確認 MT5 已登入您的帳號。

3. **設定初始手數**（可在 GUI 啟動後隨時調整）：
   在 `main.py` 中設定兩個模型各自的手數：
   ```python
   ml_ai = MLResonanceStrategy(..., lot_size_A=0.1, lot_size_B=0.1, ...)
   ```

4. **啟動 EA**：
   ```bash
   python main.py
   ```
   連線 MT5 後，在控制列分別調整 **A手數**（利潤引擎）和 **B手數**（勝率引擎），按下「▶ 啟動 EA」即開始自動交易。

---

## 🧠 AI 訓練流程

**手動重訓**（或直接在 GUI 點「🔄 重訓模型」）：

1. **匯出歷史資料**（從 MT5 拉取最新 12 個月 K 棒）：
   ```bash
   python tools/export_history.py
   ```

2. **生成資料集**（32 維特徵 + 無未來資料污染標籤）：
   ```bash
   python ml_engine/create_dataset.py
   ```

3. **訓練模型**（XGBoost + Random Forest，多空各自訓練）：
   ```bash
   python ml_engine/train_model.py
   ```

4. **回測驗證**（M1 分鐘級精度模擬）：
   ```bash
   python ml_engine/ml_backtest_m1.py
   ```

5. **深度網格搜索**（最優 TP/SL/門檻組合）：
   ```bash
   python ml_engine/hyper_optimizer.py
   ```

> **一鍵全流程**（等同 GUI 按鈕）：
> ```bash
> python ml_engine/evolve_brain.py
> ```

---

## 📂 專案結構

```
gold/
├── core/
│   ├── engine.py            # 策略引擎與 Tick 驅動迴圈
│   ├── execution.py         # MT5 下單管理
│   ├── mt5_connector.py     # MT5 連線與資料取得
│   └── trade_tracker.py     # CSV 逐筆損益追蹤與自動對帳
├── ml_engine/
│   ├── create_dataset.py    # 特徵工程與標籤生成
│   ├── train_model.py       # XGBoost + Random Forest 訓練
│   ├── ml_backtest_m1.py    # M1 精度回測引擎
│   ├── evolve_brain.py      # 一鍵全流程重訓腳本
│   └── hyper_optimizer.py   # 2970+ 組合深度網格搜索
├── models/                  # 訓練好的模型檔案 (.pkl)
├── strategies/
│   └── ml_resonance_strategy.py  # 雙模型實盤執行邏輯
├── tools/
│   ├── export_history.py    # MT5 歷史資料匯出工具
│   └── backfill_csv.py      # 一次性補填歷史損益工具
├── ui/
│   └── app_gui.py           # Tkinter 深色控制面板
└── main.py                  # 程式進入點
```

---

## ⚠️ 免責聲明

本系統僅供研究與教育目的。量化交易具有顯著風險，歷史回測績效不代表未來實際交易結果。使用前請充分了解相關風險。

import os
import sys
import logging
import time

# 將專案根目錄加入路徑，方便匯入
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.export_history import export_data
from ml_engine.create_dataset import create_dataset
from ml_engine.train_model import train
import MetaTrader5 as mt5

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def evolve():
    logging.info("🚀 開始執行 AI 大腦進化程序...")
    start_time = time.time()

    # 1. 匯出數據
    logging.info("--- 階段 1/3: 匯出最新歷史數據 ---")
    # 這裡我們維持匯出過去 12 個月，進行一年期大回測
    success = export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, tf_name="M5", months_back=12)
    if not success:
        logging.error("數據匯出失敗，終止進化。")
        return
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, tf_name="M15", months_back=12)
    mt5.shutdown()

    # 2. 建立資料集
    logging.info("--- 階段 2/3: 特徵工程與標籤生成 ---")
    create_dataset()

    # 3. 訓練模型
    logging.info("--- 階段 3/3: 雙核心模型特訓 ---")
    train()

    total_time = time.time() - start_time
    logging.info(f"✅ AI 大腦進化完成！總耗時: {total_time:.1f} 秒")
    logging.info("現在重啟 EA 即可使用最新進化後的大腦。")

if __name__ == "__main__":
    evolve()

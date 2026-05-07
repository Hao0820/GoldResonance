import os
import sys
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.export_history import export_data
from ml_engine.create_dataset import create_dataset
from ml_engine.train_model import train
import MetaTrader5 as mt5

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def evolve():
    logging.info("=" * 50)
    logging.info("AI 大腦進化程序啟動")
    logging.info("注意: 執行前請先關閉主交易程式 (main.py)")
    logging.info("=" * 50)
    start_time = time.time()

    # 階段 1: 匯出多時區歷史數據
    logging.info("--- 階段 1/4: 匯出多時區歷史數據 ---")
    try:
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1,  tf_name="M1",  months_back=3)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5,  tf_name="M5",  months_back=6)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, tf_name="M15", months_back=12)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M30, tf_name="M30", months_back=12)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1,  tf_name="H1",  months_back=36)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H4,  tf_name="H4",  months_back=36)
        export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_D1,  tf_name="D1",  months_back=36)
        export_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5,  tf_name="M5",  months_back=6)
        export_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M30, tf_name="M30", months_back=12)
    finally:
        # 匯出完成後才斷開 MT5，後續步驟不需要 MT5 連線
        mt5.shutdown()
        logging.info("MT5 連線已關閉，後續階段使用本機 CSV 數據。")

    # 階段 2: 特徵工程與標籤生成 (不需要 MT5)
    logging.info("--- 階段 2/4: 特徵工程與標籤生成 ---")
    create_dataset()

    # 階段 3: 訓練模型 (不需要 MT5)
    logging.info("--- 階段 3/4: 雙核心模型特訓 ---")
    train()

    # 階段 4:
    
    total_time = time.time() - start_time
    logging.info("=" * 50)
    logging.info(f"AI 大腦進化完成！總耗時: {total_time:.1f} 秒")
    logging.info("=" * 50)

if __name__ == "__main__":
    evolve()

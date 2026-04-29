import MetaTrader5 as mt5
import pandas as pd
import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, tf_name="M5", months_back=6):
    """
    匯出指定商品與週期的歷史數據到 CSV
    """
    if not mt5.initialize():
        logging.error("MT5 初始化失敗")
        return False
        
    logging.info(f"成功連線 MT5，準備匯出 {symbol} {tf_name} 數據...")
    
    # 設定時間範圍 (現在往前推 N 個月)
    utc_to = datetime.datetime.now(datetime.timezone.utc)
    utc_from = utc_to - datetime.timedelta(days=30 * months_back)
    
    logging.info(f"時間範圍: {utc_from.strftime('%Y-%m-%d')} ~ {utc_to.strftime('%Y-%m-%d')}")
    
    # 強力同步下載模式 (解決資料未下載問題)
    logging.info(f"正在與經紀商伺服器同步 {symbol} {tf_name} 數據...")
    
    # 先隨便抓一筆，觸發終端機開始下載
    mt5.copy_rates_from(symbol, timeframe, datetime.datetime.now(), 1)
    
    # 嘗試抓取 (最多重試 5 次)
    rates = None
    for attempt in range(5):
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        if rates is not None and len(rates) > 0:
            break
        logging.info(f"等待同步中... (第 {attempt+1} 次嘗試)")
        time.sleep(2)
    
    if rates is None or len(rates) == 0:
        logging.error(f"❌ 嚴重錯誤：經紀商伺服器未提供 {symbol} {tf_name} 的歷史數據。請確認您的 MT5 已開啟該交易品種的圖表並切換到 M1 週期手動滾動一下。")
        mt5.shutdown()
        return False
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 確保目錄存在
    os.makedirs('history_data', exist_ok=True)
    
    # 儲存到 CSV
    filename = f"history_data/{symbol}_{tf_name}.csv"
    df.to_csv(filename, index=False)
    
    logging.info(f"✅ 成功匯出 {len(df)} 筆資料到 {filename}")
    return True

if __name__ == "__main__":
    import time
    # 匯出 M1 (用於精確回測)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1, tf_name="M1", months_back=3)
    # 匯出 M5, M15, M30, H1
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, tf_name="M5", months_back=3)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, tf_name="M15", months_back=3)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M30, tf_name="M30", months_back=3)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, tf_name="H1", months_back=6)
    # 匯出 H4 與 D1 (大趨勢背景)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H4, tf_name="H4", months_back=12)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_D1, tf_name="D1", months_back=12)
    
    mt5.shutdown()
    logging.info("匯出完成！")

import MetaTrader5 as mt5
import pandas as pd
import datetime
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, tf_name="M5", months_back=6):
    """
    匯出指定商品與週期的歷史數據到 CSV
    """
    if not mt5.initialize():
        logging.error("MT5 初始化失敗")
        return False
        
    # 確保商品在市場報價中 (沒在裡面會抓不到)
    if not mt5.symbol_select(symbol, True):
        logging.error(f"無法選取商品 {symbol}")
        return False
        
    logging.info(f"成功連線 MT5，準備匯出 {symbol} {tf_name} 數據...")
    
    # 設定時間範圍
    utc_to = datetime.datetime.now(datetime.timezone.utc)
    utc_from = utc_to - datetime.timedelta(days=int(30 * months_back))
    
    logging.info(f"嘗試抓取範圍: {utc_from.strftime('%Y-%m-%d')} ~ {utc_to.strftime('%Y-%m-%d')}")
    
    # 嘗試抓取 (最多重試 5 次)
    rates = None
    for attempt in range(5):
        # 方法 A: 依日期範圍抓取
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        
        # 方法 B (備案): 如果日期範圍抓不到，嘗試抓取最後 90,000 根 (這會強制伺服器同步)
        if rates is None or len(rates) == 0:
            logging.info(f"日期範圍抓取失敗，嘗試強制同步最後 90,000 根...")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 90000)
            
        if rates is not None and len(rates) > 0:
            break
        logging.info(f"等待同步中... (第 {attempt+1} 次嘗試)")
        time.sleep(2)
    
    if rates is None or len(rates) == 0:
        logging.error(f"❌ 錯誤：經紀商未提供 {symbol} {tf_name} 數據。")
        # 不要中斷，回傳 False 讓上層決定
        return False
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 自動偵測路徑：定位到 gold/ 下的 history_data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir = os.path.join(base_dir, 'history_data')
    os.makedirs(history_dir, exist_ok=True)
    
    # 儲存到 CSV
    filename = os.path.join(history_dir, f"{symbol}_{tf_name}.csv")
    df.to_csv(filename, index=False)
    
    logging.info(f"✅ 成功匯出 {len(df)} 筆資料到 {filename}")
    return True

if __name__ == "__main__":
    import time
    # XAUUSD 全時區 (根據經紀商實際限制調整長度)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1,  tf_name="M1",  months_back=3)   # M1 通常只有 3-6 個月
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5,  tf_name="M5",  months_back=6)  # M5 給到 1 年通常沒問題
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, tf_name="M15", months_back=12)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M30, tf_name="M30", months_back=12)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1,  tf_name="H1",  months_back=36)  # H1 以上通常有多年歷史
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H4,  tf_name="H4",  months_back=36)
    export_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_D1,  tf_name="D1",  months_back=36)
    
    # EURUSD 代理指標 (DXY 負相關)
    export_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5,  tf_name="M5",  months_back=12)
    export_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M30, tf_name="M30", months_back=12)
    
    mt5.shutdown()
    logging.info("匯出完成！")

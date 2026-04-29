import MetaTrader5 as mt5
import pandas as pd
import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def deep_sync_m1(symbol="XAUUSD", count=150000):
    if not mt5.initialize():
        logging.error("MT5 初始化失敗")
        return
    
    logging.info(f"🚀 開始深度挖掘 {symbol} M1 數據 (目標: {count} 根)...")
    
    # 使用 copy_rates_from_pos 往往比指定日期範圍更容易觸發下載
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    
    if rates is None or len(rates) == 0:
        # 如果失敗，嘗試檢查一下是不是商品名稱問題 (例如有些券商叫 GOLD)
        logging.error(f"❌ 無法透過數量抓取 M1 數據。錯誤代碼: {mt5.last_error()}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    first_date = df['time'].iloc[0]
    last_date = df['time'].iloc[-1]
    
    logging.info(f"✅ 成功挖掘到 {len(df)} 筆 M1 資料！")
    logging.info(f"📅 資料範圍: {first_date} ~ {last_date}")
    
    os.makedirs('history_data', exist_ok=True)
    filename = f"history_data/{symbol}_M1.csv"
    df.to_csv(filename, index=False)
    logging.info(f"💾 已儲存至 {filename}")
    
    mt5.shutdown()

if __name__ == "__main__":
    deep_sync_m1(count=80000)

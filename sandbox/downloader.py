import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging

logger = logging.getLogger(__name__)

def download_xauusd_m5():
    """下載過去兩年的 XAUUSD M5 數據並儲存為 parquet"""
    if not mt5.initialize():
        print("MT5 初始化失敗")
        return

    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5
    
    # 計算時間範圍 (2024-01-01 到現在)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    print(f"正在從 MT5 抓取 {symbol} M5 數據 (從 {start_date.date()} 到 {end_date.date()})...")
    
    # 分段下載以提高穩定性
    current_end = end_date
    all_rates = []
    
    while current_end > start_date:
        current_start = current_end - timedelta(days=60) # M5 點位較少，可以一次抓多一點
        if current_start < start_date:
            current_start = start_date
            
        print(f"  > 下載分段: {current_start.date()} ~ {current_end.date()}")
        rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
        
        if rates is not None and len(rates) > 0:
            df_segment = pd.DataFrame(rates)
            all_rates.append(df_segment)
        else:
            print(f"    警告: 分段 {current_start.date()} 獲取失敗或無資料")
            
        current_end = current_start
        time.sleep(0.1)
        
    if not all_rates:
        print("錯誤: 未抓取到任何資料")
        mt5.shutdown()
        return

    print("正在合併數據並去重...")
    df = pd.concat(all_rates).drop_duplicates(subset=['time']).sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 儲存
    os.makedirs('data', exist_ok=True)
    save_path = 'data/xauusd_m5_backtest.parquet'
    df.to_parquet(save_path)
    
    print(f"下載完成！共 {len(df)} 行數據。儲存於: {save_path}")
    mt5.shutdown()

if __name__ == "__main__":
    download_xauusd_m5()

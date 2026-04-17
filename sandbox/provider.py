import MetaTrader5 as mt5
from datetime import datetime
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

class SandboxConnector:
    def __init__(self, data_path, initial_time=None):
        import pandas as pd
        # 用戶提供 m1 資料，我們將其視為主資料源
        self.df_m1 = pd.read_parquet(data_path)
        self.df_m1['time'] = pd.to_datetime(self.df_m1['time'])
        
        # 效能極速優化：預先從 M1 重採樣 H1, M30, M5 數據
        print("⚡ 正在從 M1 數據重採樣多時區資料，請稍後...")
        temp_df = self.df_m1.set_index('time')
        
        self.df_h1 = temp_df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna().reset_index()
        
        self.df_m30 = temp_df.resample('30min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna().reset_index()
        
        self.df_m5 = temp_df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
        }).dropna().reset_index()
        
        # 效能核心優化：預先計算所有指標
        print("⚡ 正在計算全時區指標 (BB, EMA)，這將大幅提升回測速度...")
        for df in [self.df_m1, self.df_m5, self.df_m30, self.df_h1]:
            # 1. Bollinger Bands
            bb = BollingerBands(close=df['close'], window=21, window_dev=2.1)
            df['bb_h'] = bb.bollinger_hband()
            df['bb_m'] = bb.bollinger_mavg()
            df['bb_l'] = bb.bollinger_lband()
            
            # 2. 12 EMA
            ema = EMAIndicator(close=df['close'], window=12)
            df['ema_12'] = ema.ema_indicator()
        
        print(f"✅ 數據預處理完成。M1 筆數: {len(self.df_m1)}")

        self.connected = True
        # 初始化時間：從 M1 的第 3000 根開始 (確保 H1/M30/M5 指標有足夠資料預熱)
        self.current_time = initial_time if initial_time else self.df_m1['time'].iloc[3000]
        self.last_price = None # 當前模擬報價 (Tick)
        self.executor = None 
        
    def connect(self):
        """沙盒模式模擬連接成功"""
        return True
        
    def get_account_info(self):
        return {"balance": 10000.0, "equity": 10000.0}

    def get_symbol_info(self, symbol):
        # 使用最後一個 Tick 價格
        bid = self.last_price if self.last_price else 0.0
        return {"ask": bid + 3.0, "bid": bid}

    def get_rates(self, symbol, timeframe, count=100):
        if timeframe == mt5.TIMEFRAME_H1:
            df = self.df_h1
        elif timeframe == mt5.TIMEFRAME_M30:
            df = self.df_m30
        elif timeframe == mt5.TIMEFRAME_M5:
            df = self.df_m5
        else:
            df = self.df_m1
            
        # 核心優化：使用二分搜尋 (searchsorted) 尋找索引，複雜度從 O(N) 降為 O(log N)
        # 這確保了在 10 萬筆資料的迴圈中，每一拍都能瞬間定位
        idx = df['time'].searchsorted(self.current_time, side='right')
        result = df.iloc[max(0, idx - count):idx].copy()
        
        # 核心優化：將當前最後一個 Bar 的收盤價替換為「當前 Tick 價格」
        # 這確保了策略計算指標時使用的是模擬中的真實價格，而非歷史收盤價
        if not result.empty and self.last_price is not None:
            result.iloc[-1, result.columns.get_loc('close')] = self.last_price
            # 同時動態更新最高與最低 (避免 Tick 價格超出 OHLC 範圍導致指標異常)
            if self.last_price > result.iloc[-1]['high']: result.iloc[-1, result.columns.get_loc('high')] = self.last_price
            if self.last_price < result.iloc[-1]['low']: result.iloc[-1, result.columns.get_loc('low')] = self.last_price
            
        return result

    def get_positions_count_by_magic(self, magic):
        if self.executor:
            return self.executor.get_positions_count_by_magic(magic)
        return 0 

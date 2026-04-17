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

        # 🚀 效能終極優化：將所有數據轉換為 Numpy 陣列並建立索引映射表
        # 這能避免在 10 萬次循環中不斷調用 Pandas 搜尋
        self._prepare_numpy_arrays()

        self.connected = True
        self.current_time = initial_time if initial_time else self.df_m1['time'].iloc[3000]
        self.last_price = 0.0 # 當前模擬報價 (Tick)
        self.executor = None 
        
    def _prepare_numpy_arrays(self):
        """將 DataFrame 轉換為快速的 Numpy 結構並建立時區映射"""
        import numpy as np
        print("🚀 正在建立極速索引映射表 (Numpy Matrix)...")
        # 1. 基礎價格與時間 (M1)
        self.m1_times = self.df_m1['time'].values
        self.m1_closes = self.df_m1['close'].values
        
        # 2. 建立 M1 -> 其他時區的索引映射 (讓搜尋變成 O(1))
        # 例如：M1 的第 5000 根對應到 H1 的哪一根？
        self.m5_map = self.df_m5['time'].searchsorted(self.m1_times, side='right')
        self.m30_map = self.df_m30['time'].searchsorted(self.m1_times, side='right')
        self.h1_map = self.df_h1['time'].searchsorted(self.m1_times, side='right')
        
        # 3. 將所有指標轉換為 Dictionary 存儲 Numpy Array
        self.np_data = {}
        for tf, df in [("H1", self.df_h1), ("M30", self.df_m30), ("M5", self.df_m5), ("M1", self.df_m1)]:
            self.np_data[tf] = {
                'time': df['time'].values,
                'open': df['open'].values,
                'high': df['high'].values,
                'low': df['low'].values,
                'close': df['close'].values,
                'bb_h': df['bb_h'].values,
                'bb_m': df['bb_m'].values,
                'bb_l': df['bb_l'].values,
                'ema_12': df['ema_12'].values
            }
        print("✅ 極速矩陣建立完成。")        
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
        """
        獲取數據接口。
        如果是模擬循環中調用了核心指標，會使用預先計算好的 Numpy。
        """
        if not hasattr(self, '_current_m1_idx'):
            # 退回到慢速模式 (非模擬場景)
            return self._get_rates_pandas(timeframe, count)
        
        return self._get_rates_numpy(timeframe, count)

    def _get_rates_numpy(self, timeframe, count):
        import pandas as pd
        import numpy as np
        
        # 1. 根據 M1 索引定位目標時區的索引 (O(1) 查表)
        m1_idx = self._current_m1_idx
        if timeframe == mt5.TIMEFRAME_H1:
            target_idx = self.h1_map[m1_idx]
            data = self.np_data["H1"]
        elif timeframe == mt5.TIMEFRAME_M30:
            target_idx = self.m30_map[m1_idx]
            data = self.np_data["M30"]
        elif timeframe == mt5.TIMEFRAME_M5:
            target_idx = self.m5_map[m1_idx]
            data = self.np_data["M5"]
        else:
            target_idx = m1_idx + 1
            data = self.np_data["M1"]

        # 2. 切片取出數據
        start = max(0, target_idx - count)
        end = target_idx
        
        # 3. 組裝成 DataFrame (維持相容性)
        res_df = pd.DataFrame({
            'time': data['time'][start:end],
            'open': data['open'][start:end],
            'high': data['high'][start:end],
            'low': data['low'][start:end],
            'close': data['close'][start:end],
            'tick_volume': 0,
            'bb_h': data['bb_h'][start:end],
            'bb_m': data['bb_m'][start:end],
            'bb_l': data['bb_l'][start:end],
            'ema_12': data['ema_12'][start:end]
        })
        
        # 4. Tick 價格注入
        if not res_df.empty:
            res_df.iloc[-1, res_df.columns.get_loc('close')] = self.last_price
            
        return res_df

    def _get_rates_pandas(self, timeframe, count):
        if timeframe == mt5.TIMEFRAME_H1:
            df = self.df_h1
        elif timeframe == mt5.TIMEFRAME_M30:
            df = self.df_m30
        elif timeframe == mt5.TIMEFRAME_M5:
            df = self.df_m5
        else:
            df = self.df_m1
            
        idx = df['time'].searchsorted(self.current_time, side='right')
        result = df.iloc[max(0, idx - count):idx].copy()
        
        if not result.empty:
            result.iloc[-1, result.columns.get_loc('close')] = self.last_price
        return result

    def get_positions_count_by_magic(self, magic):
        if self.executor:
            return self.executor.get_positions_count_by_magic(magic)
        return 0 

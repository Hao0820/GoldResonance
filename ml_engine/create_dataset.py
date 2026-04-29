import pandas as pd
import numpy as np
import ta
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def create_dataset():
    logging.info("讀取歷史數據...")
    try:
        df_m5 = pd.read_csv("history_data/XAUUSD_M5.csv")
        df_m15 = pd.read_csv("history_data/XAUUSD_M15.csv")
    except FileNotFoundError:
        logging.error("找不到歷史數據，請確認 history_data 資料夾存在。")
        return

    df_m5['time'] = pd.to_datetime(df_m5['time'])
    df_m15['time'] = pd.to_datetime(df_m15['time'])

    logging.info("計算 M15 特徵...")
    bb_m15 = ta.volatility.BollingerBands(close=df_m15['close'], window=21, window_dev=2.1)
    df_m15['m15_bb_h'] = bb_m15.bollinger_hband()
    df_m15['m15_bb_l'] = bb_m15.bollinger_lband()
    df_m15['m15_bb_m'] = bb_m15.bollinger_mavg()
    df_m15['m15_ema_12'] = ta.trend.EMAIndicator(close=df_m15['close'], window=12).ema_indicator()
    df_m15['m15_ema_slope'] = df_m15['m15_ema_12'] - df_m15['m15_ema_12'].shift(5)
    
    # 萃取 M15 相對距離特徵
    df_m15['m15_dist_h'] = df_m15['high'] - df_m15['m15_bb_h']
    df_m15['m15_dist_l'] = df_m15['low'] - df_m15['m15_bb_l']
    df_m15['m15_dist_m'] = df_m15['close'] - df_m15['m15_bb_m']

    logging.info("計算 M5 特徵...")
    bb_m5 = ta.volatility.BollingerBands(close=df_m5['close'], window=21, window_dev=2.1)
    df_m5['m5_bb_h'] = bb_m5.bollinger_hband()
    df_m5['m5_bb_l'] = bb_m5.bollinger_lband()
    df_m5['m5_bb_m'] = bb_m5.bollinger_mavg()
    df_m5['m5_ema_12'] = ta.trend.EMAIndicator(close=df_m5['close'], window=12).ema_indicator()
    df_m5['m5_ema_slope'] = df_m5['m5_ema_12'] - df_m5['m5_ema_12'].shift(5)
    
    df_m5['m5_rsi_14'] = ta.momentum.RSIIndicator(close=df_m5['close'], window=14).rsi()
    df_m5['m5_atr_14'] = ta.volatility.AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14).average_true_range()
    
    # K 棒型態特徵
    df_m5['body_size'] = df_m5['close'] - df_m5['open']
    df_m5['upper_shadow'] = df_m5['high'] - np.maximum(df_m5['open'], df_m5['close'])
    df_m5['lower_shadow'] = np.minimum(df_m5['open'], df_m5['close']) - df_m5['low']
    df_m5['body_ratio'] = df_m5['body_size'] / (df_m5['high'] - df_m5['low'] + 0.0001)
    
    # 時間特徵
    df_m5['hour'] = df_m5['time'].dt.hour
    df_m5['day_of_week'] = df_m5['time'].dt.dayofweek

    # ===== 新增 6 個高品質特徵 =====
    
    # 1. 成交量比率 (>2 代表機構異常進場)
    vol_ma20 = df_m5['tick_volume'].rolling(20).mean()
    df_m5['volume_ratio'] = df_m5['tick_volume'] / (vol_ma20 + 1)
    
    # 2. 點差大小 (MT5 OHLCV 有內建 spread 欄位)
    if 'spread' in df_m5.columns:
        df_m5['spread'] = df_m5['spread'].astype(float)
    else:
        # Fallback: 用 ATR 較正規化的高低差作代理
        df_m5['spread'] = (df_m5['high'] - df_m5['low']) / (df_m5['m5_atr_14'] + 0.001)
    
    # 3. 歐美盤重疊時段 (13:00-17:00 UTC，流動性最強)
    df_m5['session_overlap'] = df_m5['hour'].apply(lambda x: 1 if 13 <= x <= 17 else 0)
    
    # 4. RSI 背離 (價格創新高但 RSI 沒跟上 = 看空)
    price_new_high = df_m5['close'] > df_m5['close'].shift(5)
    rsi_new_high   = df_m5['m5_rsi_14'] > df_m5['m5_rsi_14'].shift(5)
    price_new_low  = df_m5['close'] < df_m5['close'].shift(5)
    rsi_new_low    = df_m5['m5_rsi_14'] < df_m5['m5_rsi_14'].shift(5)
    df_m5['rsi_divergence'] = np.where(price_new_high & ~rsi_new_high, -1,
                              np.where(price_new_low  & ~rsi_new_low,   1, 0)).astype(float)
    
    # 5. 價格跟 VWAP 的距離 (量价平均價格代理)
    typical_price = (df_m5['high'] + df_m5['low'] + df_m5['close']) / 3
    vwap = (typical_price * df_m5['tick_volume']).rolling(20).sum() / df_m5['tick_volume'].rolling(20).sum()
    df_m5['price_vs_vwap'] = df_m5['close'] - vwap
    
    # 6. 吞噬型 K 棒型態 (看多吞噬=+1, 看空吞噬=-1)
    prev_body = df_m5['body_size'].shift(1)
    curr_body = df_m5['body_size']
    bullish_engulf = (curr_body > 0) & (prev_body < 0) & (df_m5['close'] > df_m5['open'].shift(1)) & (df_m5['open'] < df_m5['close'].shift(1))
    bearish_engulf = (curr_body < 0) & (prev_body > 0) & (df_m5['close'] < df_m5['open'].shift(1)) & (df_m5['open'] > df_m5['close'].shift(1))
    df_m5['pattern_engulf'] = np.where(bullish_engulf, 1, np.where(bearish_engulf, -1, 0)).astype(float)

    # 合併 M15 到 M5 (Forward Fill)
    m15_features = ['time', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope']
    df = pd.merge_asof(
        df_m5.sort_values('time'),
        df_m15[m15_features].sort_values('time'),
        on='time',
        direction='backward'
    )
    
    # M5 距離特徵 (要在 merge 之後算，避免名稱衝突)
    df['m5_dist_h'] = df['high'] - df['m5_bb_h']
    df['m5_dist_l'] = df['low'] - df['m5_bb_l']
    df['m5_dist_m'] = df['close'] - df['m5_bb_m']
    df['m5_dist_ema'] = df['close'] - df['m5_ema_12']
    
    # 生成標籤 Labels (往未來探索 60 根 K 棒 = 5 小時)
    logging.info("生成預測標籤 (Labels)...")
    tp_dist = 7.0
    sl_dist = 4.0
    look_forward = 60
    
    label_buy = np.zeros(len(df))
    label_sell = np.zeros(len(df))
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    for i in range(len(df) - look_forward):
        entry_price = closes[i]
        
        # 判斷多單結果
        buy_tp = entry_price + tp_dist
        buy_sl = entry_price - sl_dist
        for j in range(i + 1, i + look_forward):
            if lows[j] <= buy_sl:
                break # 失敗
            if highs[j] >= buy_tp:
                label_buy[i] = 1 # 成功
                break
                
        # 判斷空單結果
        sell_tp = entry_price - tp_dist
        sell_sl = entry_price + sl_dist
        for j in range(i + 1, i + look_forward):
            if highs[j] >= sell_sl:
                break # 失敗
            if lows[j] <= sell_tp:
                label_sell[i] = 1 # 成功
                break
                
    df['label_buy'] = label_buy
    df['label_sell'] = label_sell
    
    # 清除 NaN
    # --- 新增時區與多時區特徵 (Level 4) ---
    # 1. 技術指標 (Level 3)
    df['m5_adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['m5_macd_hist'] = ta.trend.MACD(df['close']).macd_diff()
    df['m5_cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    df['m5_bb_width'] = (df['m5_bb_h'] - df['m5_bb_l']) / df['m5_bb_m']
    df['prev_body_size'] = df['body_size'].shift(1)
    df['prev_close_change'] = df['close'].diff()
    df['m15_ema_long'] = ta.trend.EMAIndicator(df['close'], window=192).ema_indicator()

    # 2. 市場時段標記
    df['is_us_session'] = df['time'].dt.hour.apply(lambda x: 1 if 13 <= x <= 21 else 0)
    df['is_asia_session'] = df['time'].dt.hour.apply(lambda x: 1 if 0 <= x <= 8 else 0)
    
    # 3. 匯入 M30, H1, H4 與 D1 資料
    m30_df = pd.read_csv("history_data/XAUUSD_M30.csv")
    m30_df['time'] = pd.to_datetime(m30_df['time'])
    m30_df['m30_rsi'] = ta.momentum.RSIIndicator(m30_df['close'], window=14).rsi()
    m30_df['m30_trend'] = ta.trend.EMAIndicator(m30_df['close'], window=20).ema_indicator().diff()
    
    h1_df = pd.read_csv("history_data/XAUUSD_H1.csv")
    h1_df['time'] = pd.to_datetime(h1_df['time'])
    h1_df['h1_trend'] = ta.trend.EMAIndicator(h1_df['close'], window=12).ema_indicator().diff()
    
    h4_df = pd.read_csv("history_data/XAUUSD_H4.csv")
    h4_df['time'] = pd.to_datetime(h4_df['time'])
    h4_df['h4_ema_20'] = ta.trend.EMAIndicator(h4_df['close'], window=20).ema_indicator()
    h4_df['h4_trend'] = h4_df['h4_ema_20'].diff()
    
    d1_df = pd.read_csv("history_data/XAUUSD_D1.csv")
    d1_df['time'] = pd.to_datetime(d1_df['time'])
    d1_df['d1_rsi'] = ta.momentum.RSIIndicator(d1_df['close'], window=14).rsi()

    # 4. 匯入 M1 並計算微觀動能
    m1_df = pd.read_csv("history_data/XAUUSD_M1.csv")
    m1_df['time'] = pd.to_datetime(m1_df['time'])
    m1_df['m1_ema_5'] = ta.trend.EMAIndicator(m1_df['close'], window=5).ema_indicator()
    m1_df['m1_momentum'] = m1_df['m1_ema_5'].diff()
    
    # 5. 使用 merge_asof 對齊全時區
    df = df.sort_values('time')
    df = pd.merge_asof(df, m1_df[['time', 'm1_momentum']], on='time', direction='backward')
    df = pd.merge_asof(df, m30_df[['time', 'm30_rsi', 'm30_trend']], on='time', direction='backward')
    df = pd.merge_asof(df, h1_df[['time', 'h1_trend']], on='time', direction='backward')
    df = pd.merge_asof(df, h4_df[['time', 'h4_trend']], on='time', direction='backward')
    df = pd.merge_asof(df, d1_df[['time', 'd1_rsi']], on='time', direction='backward')

    # --- 清理並選擇欄位 ---
    df = df.dropna().reset_index(drop=True)
    
    # 終極全時區特徵清單 (38 維度)
    feature_cols = [
        'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 
        'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'hour', 'day_of_week',
        'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema',
        'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
        'prev_body_size', 'prev_close_change',
        'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
        'm30_rsi', 'm30_trend', 'm1_momentum',
        # 新增高品質特徵 (+6)
        'volume_ratio', 'spread', 'session_overlap',
        'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
    ]
    
    dataset = df[feature_cols + ['label_buy', 'label_sell', 'time', 'close']].copy()
    
    output_path = "ml_dataset.csv"
    dataset.to_csv(output_path, index=False)
    
    logging.info(f"✅ 資料集建立完成！共 {len(dataset)} 筆訓練資料，儲存至 {output_path}")
    logging.info(f"多單勝率樣本數: {dataset['label_buy'].sum()} / {len(dataset)} ({(dataset['label_buy'].sum()/len(dataset))*100:.1f}%)")
    logging.info(f"空單勝率樣本數: {dataset['label_sell'].sum()} / {len(dataset)} ({(dataset['label_sell'].sum()/len(dataset))*100:.1f}%)")

if __name__ == "__main__":
    create_dataset()

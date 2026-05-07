import pandas as pd
import numpy as np
import ta
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.stage_classifier import add_stage_labels

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def create_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir = os.path.join(base_dir, "history_data")
    
    logging.info("讀取歷史數據...")
    try:
        df_m5 = pd.read_csv(os.path.join(history_dir, "XAUUSD_M5.csv"))
        df_m15 = pd.read_csv(os.path.join(history_dir, "XAUUSD_M15.csv"))
    except FileNotFoundError:
        logging.error("找不到歷史數據，請確認 history_data 資料夾存在。")
        return

    df_m5['time'] = pd.to_datetime(df_m5['time'])
    df_m15['time'] = pd.to_datetime(df_m15['time'])

    logging.info("計算 M15 特徵...")
    bb_m15 = ta.volatility.BollingerBands(close=df_m15['close'], window=21, window_dev=2.1)
    df_m15['m15_bb_h']  = bb_m15.bollinger_hband()
    df_m15['m15_bb_l']  = bb_m15.bollinger_lband()
    df_m15['m15_bb_m']  = bb_m15.bollinger_mavg()
    # 第二層外帶（對應圖上白色虛線外帶）
    bb_m15_o = ta.volatility.BollingerBands(close=df_m15['close'], window=55, window_dev=2.0)
    df_m15['m15_bb_oh'] = bb_m15_o.bollinger_hband()
    df_m15['m15_bb_ol'] = bb_m15_o.bollinger_lband()
    df_m15['m15_ema_12']    = ta.trend.EMAIndicator(close=df_m15['close'], window=12).ema_indicator()
    df_m15['m15_ema_slope'] = df_m15['m15_ema_12'] - df_m15['m15_ema_12'].shift(5)
    df_m15['m15_dist_h']  = df_m15['high']  - df_m15['m15_bb_h']
    df_m15['m15_dist_l']  = df_m15['low']   - df_m15['m15_bb_l']
    df_m15['m15_dist_m']  = df_m15['close'] - df_m15['m15_bb_m']
    df_m15['m15_dist_oh'] = df_m15['high']  - df_m15['m15_bb_oh']  # 距外帶上軌
    df_m15['m15_dist_ol'] = df_m15['low']   - df_m15['m15_bb_ol']  # 距外帶下軌

    logging.info("計算 M5 基礎與 PPT 專家特徵...")
    bb_m5 = ta.volatility.BollingerBands(close=df_m5['close'], window=21, window_dev=2.1)
    df_m5['m5_bb_h'] = bb_m5.bollinger_hband()
    df_m5['m5_bb_l'] = bb_m5.bollinger_lband()
    df_m5['m5_bb_m'] = bb_m5.bollinger_mavg()
    # 第二層外帶（對應圖上白色虛線外帶）
    bb_m5_o = ta.volatility.BollingerBands(close=df_m5['close'], window=55, window_dev=2.0)
    df_m5['m5_bb_oh'] = bb_m5_o.bollinger_hband()
    df_m5['m5_bb_ol'] = bb_m5_o.bollinger_lband()
    df_m5['m5_ema_12']    = ta.trend.EMAIndicator(close=df_m5['close'], window=12).ema_indicator()
    df_m5['m5_ema_slope'] = df_m5['m5_ema_12'] - df_m5['m5_ema_12'].shift(5)
    df_m5['m5_rsi_14']    = ta.momentum.RSIIndicator(close=df_m5['close'], window=14).rsi()
    df_m5['m5_atr_14']    = ta.volatility.AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14).average_true_range()
    
    df_m5['body_size'] = df_m5['close'] - df_m5['open']
    df_m5['upper_shadow'] = df_m5['high'] - np.maximum(df_m5['open'], df_m5['close'])
    df_m5['lower_shadow'] = np.minimum(df_m5['open'], df_m5['close']) - df_m5['low']
    df_m5['body_ratio'] = df_m5['body_size'] / (df_m5['high'] - df_m5['low'] + 0.0001)
    df_m5['hour'] = df_m5['time'].dt.hour
    df_m5['day_of_week'] = df_m5['time'].dt.dayofweek

    # PPT 專家特徵
    df_m5['ema12_dist'] = df_m5['close'] - df_m5['m5_ema_12']
    df_m5['mid_bb_dist'] = df_m5['close'] - df_m5['m5_bb_m']
    significant_low_shadow = (df_m5['lower_shadow'] > (df_m5['body_size'].abs() * 0.5)).astype(int)
    df_m5['consecutive_low_shadows'] = significant_low_shadow.rolling(3).sum()
    significant_high_shadow = (df_m5['upper_shadow'] > (df_m5['body_size'].abs() * 0.5)).astype(int)
    df_m5['consecutive_high_shadows'] = significant_high_shadow.rolling(3).sum()
    df_m5['pullback_ratio'] = (df_m5['close'] - df_m5['open'].shift(1)) / (df_m5['close'].shift(1) - df_m5['open'].shift(1) + 0.0001)
    df_m5['is_breaking_upper'] = (df_m5['close'] > df_m5['m5_bb_h']).astype(int)
    df_m5['is_breaking_lower'] = (df_m5['close'] < df_m5['m5_bb_l']).astype(int)
    # 大紅K（上漲大K）/ 大綠K（下跌大K）
    # PPT 台灣慣例：紅K = 上漲（收盤 > 開盤），綠K = 下跌（收盤 < 開盤）
    avg_body = df_m5['body_size'].abs().rolling(20).mean()
    df_m5['is_big_red_k']   = ((df_m5['body_size'] > 0) & (df_m5['body_size'].abs() > avg_body * 1.5)).astype(int)
    df_m5['is_big_green_k'] = ((df_m5['body_size'] < 0) & (df_m5['body_size'].abs() > avg_body * 1.5)).astype(int)
    # 距外帶距離（第二層 BB）
    df_m5['m5_dist_oh'] = df_m5['high']  - df_m5['m5_bb_oh']  # 距外帶上軌（突破信號強度）
    df_m5['m5_dist_ol'] = df_m5['low']   - df_m5['m5_bb_ol']  # 距外帶下軌
    # 盤整評分：BB 寬度標準化（越窄 = 越盤整，2/3/4 階切換指標）
    bb_w_raw = df_m5['m5_bb_h'] - df_m5['m5_bb_l']
    df_m5['consolidation_score'] = 1.0 - (bb_w_raw / (bb_w_raw.rolling(100).max() + 0.0001))

    # 補回失蹤的高品質技術指標 (Fix KeyError)
    df_m5['m5_adx'] = ta.trend.ADXIndicator(df_m5['high'], df_m5['low'], df_m5['close']).adx()
    df_m5['m5_macd_hist'] = ta.trend.MACD(df_m5['close']).macd_diff()
    df_m5['m5_cci'] = ta.trend.CCIIndicator(df_m5['high'], df_m5['low'], df_m5['close']).cci()
    df_m5['m5_bb_width'] = (df_m5['m5_bb_h'] - df_m5['m5_bb_l']) / (df_m5['m5_bb_m'] + 0.0001)
    df_m5['prev_body_size'] = df_m5['body_size'].shift(1)
    df_m5['prev_close_change'] = df_m5['close'].diff()
    df_m5['is_us_session'] = df_m5['hour'].apply(lambda x: 1 if 13 <= x <= 21 else 0)
    df_m5['is_asia_session'] = df_m5['hour'].apply(lambda x: 1 if 0 <= x <= 8 else 0)

    vol_ma20 = df_m5['tick_volume'].rolling(20).mean()
    df_m5['volume_ratio'] = df_m5['tick_volume'] / (vol_ma20 + 1)
    df_m5['session_overlap'] = df_m5['hour'].apply(lambda x: 1 if 13 <= x <= 17 else 0)
    # spread: MT5 匯出的 CSV 通常含有 spread 欄位 (整數點差)；若無則以 0 補齊
    if 'spread' not in df_m5.columns:
        df_m5['spread'] = 0
    
    price_prev5 = df_m5['close'].shift(5)
    rsi_prev5   = df_m5['m5_rsi_14'].shift(5)
    df_m5['rsi_divergence'] = np.where((df_m5['close'] > price_prev5) & (df_m5['m5_rsi_14'] < rsi_prev5), -1,
                              np.where((df_m5['close'] < price_prev5) & (df_m5['m5_rsi_14'] > rsi_prev5), 1, 0)).astype(float)
    
    typical_price = (df_m5['high'] + df_m5['low'] + df_m5['close']) / 3
    vwap = (typical_price * df_m5['tick_volume']).rolling(20).sum() / (df_m5['tick_volume'].rolling(20).sum() + 1)
    df_m5['price_vs_vwap'] = df_m5['close'] - vwap
    
    bullish_engulf = (df_m5['body_size'] > 0) & (df_m5['body_size'].shift(1) < 0) & (df_m5['close'] > df_m5['open'].shift(1))
    bearish_engulf = (df_m5['body_size'] < 0) & (df_m5['body_size'].shift(1) > 0) & (df_m5['close'] < df_m5['open'].shift(1))
    df_m5['pattern_engulf'] = np.where(bullish_engulf, 1, np.where(bearish_engulf, -1, 0)).astype(float)

    # 合併多時區
    m15_features = ['time', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope', 'm15_dist_oh', 'm15_dist_ol']
    df = pd.merge_asof(df_m5.sort_values('time'), df_m15[m15_features].sort_values('time'), on='time', direction='backward')
    
    df['m5_dist_h'] = df['high'] - df['m5_bb_h']
    df['m5_dist_l'] = df['low'] - df['m5_bb_l']
    df['m5_dist_m'] = df['close'] - df['m5_bb_m']
    df['m5_dist_ema'] = df['close'] - df['m5_ema_12']
    
    m30_df = pd.read_csv(os.path.join(history_dir, "XAUUSD_M30.csv"))
    m30_df['time'] = pd.to_datetime(m30_df['time'])
    m30_df['m30_rsi'] = ta.momentum.RSIIndicator(m30_df['close'], window=14).rsi()
    m30_df['m30_trend'] = ta.trend.EMAIndicator(m30_df['close'], window=20).ema_indicator().diff()
    
    h1_df = pd.read_csv(os.path.join(history_dir, "XAUUSD_H1.csv"))
    h1_df['time'] = pd.to_datetime(h1_df['time'])
    h1_df['h1_trend'] = ta.trend.EMAIndicator(h1_df['close'], window=12).ema_indicator().diff()
    
    h4_df = pd.read_csv(os.path.join(history_dir, "XAUUSD_H4.csv"))
    h4_df['time'] = pd.to_datetime(h4_df['time'])
    h4_df['h4_trend'] = ta.trend.EMAIndicator(h4_df['close'], window=20).ema_indicator().diff()
    # H4 布林帶（第 4-1/4-2 階進場依據 H4 軌道位置）
    bb_h4 = ta.volatility.BollingerBands(close=h4_df['close'], window=21, window_dev=2.1)
    h4_df['h4_bb_dist_m'] = h4_df['close'] - bb_h4.bollinger_mavg()   # 距中軌
    h4_df['h4_bb_dist_h'] = h4_df['high']  - bb_h4.bollinger_hband()  # 距上軌
    h4_df['h4_bb_dist_l'] = h4_df['low']   - bb_h4.bollinger_lband()  # 距下軌
    
    d1_df = pd.read_csv(os.path.join(history_dir, "XAUUSD_D1.csv"))
    d1_df['time'] = pd.to_datetime(d1_df['time'])
    d1_df['d1_rsi'] = ta.momentum.RSIIndicator(d1_df['close'], window=14).rsi()

    m1_df = pd.read_csv(os.path.join(history_dir, "XAUUSD_M1.csv"))
    m1_df['time'] = pd.to_datetime(m1_df['time'])
    m1_df['m1_momentum'] = ta.trend.EMAIndicator(m1_df['close'], window=5).ema_indicator().diff()
    
    df = pd.merge_asof(df, m1_df[['time', 'm1_momentum']], on='time', direction='backward')
    df = pd.merge_asof(df, m30_df[['time', 'm30_rsi', 'm30_trend']], on='time', direction='backward')
    df = pd.merge_asof(df, h1_df[['time', 'h1_trend']], on='time', direction='backward')
    df = pd.merge_asof(df, h4_df[['time', 'h4_trend', 'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l']], on='time', direction='backward')
    df = pd.merge_asof(df, d1_df[['time', 'd1_rsi']], on='time', direction='backward')

    # ── 標籤生成 ─────────────────────────────────────────────────────────
    # 進場點改用「下一根 K 棒開盤價」
    #   PPT 交易者是「看到訊號 K 棒收盤後，下一根開盤才進場」
    FIXED_TP = 5.0      # 止盈：5 點
    FIXED_SL = 5.0      # 止損：5 點
    look_forward = 120  # 最多看 120 根 M5 = 10 小時

    h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    label_win_buy  = np.zeros(len(df), dtype=int)
    label_win_sell = np.zeros(len(df), dtype=int)

    # 最後 look_forward+1 根無法標記（需要未來資料）
    for i in range(len(df) - look_forward - 1):
        entry = o[i + 1]   # ← 下一根開盤進場（PPT 真實進場點）
        tp_b, sl_b = entry + FIXED_TP, entry - FIXED_SL
        tp_s, sl_s = entry - FIXED_TP, entry + FIXED_SL

        wb, ws = 0, 0
        # 從 i+1 起算（進場那根 K 棒本身也可能觸及 TP/SL）
        for j in range(i + 1, i + look_forward):
            if h[j] >= tp_b: wb = 1; break
            if l[j] <= sl_b: wb = 0; break

        for j in range(i + 1, i + look_forward):
            if l[j] <= tp_s: ws = 1; break
            if h[j] >= sl_s: ws = 0; break

        label_win_buy[i]  = wb
        label_win_sell[i] = ws

    df['label_win_buy']  = label_win_buy
    df['label_win_sell'] = label_win_sell

    win_rate_b = label_win_buy.mean() * 100
    win_rate_s = label_win_sell.mean() * 100
    logging.info(f"📊 標籤統計 → 多單自然勝率: {win_rate_b:.1f}%  空單自然勝率: {win_rate_s:.1f}%")

    # 清理與輸出
    df = df.dropna().reset_index(drop=True)
    feature_cols = [
        'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'hour', 'day_of_week', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema', 'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
        'prev_body_size', 'prev_close_change', 'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
        'm30_rsi', 'm30_trend', 'm1_momentum', 'volume_ratio', 'spread', 'session_overlap', 'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
        'ema12_dist', 'mid_bb_dist', 'consecutive_low_shadows', 'consecutive_high_shadows', 'pullback_ratio', 'is_breaking_upper', 'is_breaking_lower',
        # ── 新增：對應專家圖示的特徵 ──
        'is_big_red_k', 'is_big_green_k',          # 大紅K / 大綠K (4-2階)
        'm5_dist_oh', 'm5_dist_ol',                # M5 距外帶距離 (3階上下軌)
        'm15_dist_oh', 'm15_dist_ol',              # M15 距外帶距離
        'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l',  # H4 布林帶位置 (4-1/4-2階)
        'consolidation_score',                     # 盤整程度 (2階/3階切換)
    ]

    # ── 階段分類標籤（PPT 規則 → AI 訓練權重）──────────────────────────────
    logging.info("--- 附加階段分類標籤 (2/3/4-1/4-2 階) ---")
    df = add_stage_labels(df)
    # pattern_stage, signal_direction, sample_weight 已加入 df

    export_cols = feature_cols + [
        'label_win_buy', 'label_win_sell',
        'pattern_stage', 'signal_direction', 'sample_weight',
        'time', 'close'
    ]
    df[export_cols].to_csv(
        os.path.join(base_dir, "ml_dataset.csv"), index=False)
    stage_counts = df['pattern_stage'].value_counts().to_dict()
    ppt_n = sum(v for k, v in stage_counts.items() if k != 0)
    logging.info(
        f"✅ 資料集建立完成！{len(feature_cols)} 維特徵，TP={FIXED_TP}/SL={FIXED_SL}。\n"
        f"   全量樣本: {len(df):,}  PPT訊號樣本: {ppt_n:,} ({ppt_n/len(df)*100:.1f}%)\n"
        f"   階段分佈: {stage_counts}"
    )


if __name__ == "__main__":
    create_dataset()

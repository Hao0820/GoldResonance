"""
Iron Wall AI 回測引擎
────────────────────────────────────────────────
讀取 XAUUSD_M1.csv → 重採樣為所有時間框架 →
逐根 M5 K 棒計算 44 維特徵 → AI 預測 TP/SL/信心率 →
使用 M1 逐根模擬進出場結果 → 輸出完整報表
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import ta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# ── 路徑 ──
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_DIR = os.path.join(BASE_DIR, "history_data")
MODELS_DIR  = os.path.join(BASE_DIR, "ml_engine", "models")

# ── 回測參數 ──
CONFIDENCE_THRESHOLD = 0.65   # 信心率門檻
FIXED_TP             = 5.0    # 固定止盈點數
FIXED_SL             = 5.0    # 固定止損點數
WARM_UP_BARS         = 120    # 指標暖機所需 M5 K 棒數


FEATURE_COLS = [
    'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
    'hour', 'day_of_week', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
    'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema', 'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
    'prev_body_size', 'prev_close_change', 'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
    'm30_rsi', 'm30_trend', 'm1_momentum', 'volume_ratio', 'spread', 'session_overlap',
    'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
    'ema12_dist', 'mid_bb_dist', 'consecutive_low_shadows', 'consecutive_high_shadows',
    'pullback_ratio', 'is_breaking_upper', 'is_breaking_lower',
    'is_big_red_k', 'is_big_green_k',
    'm5_dist_oh', 'm5_dist_ol',
    'm15_dist_oh', 'm15_dist_ol',
    'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l',
    'consolidation_score',
]


def resample_ohlcv(m1_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """從 M1 重採樣為指定時間框架"""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'}
    if 'spread' in m1_df.columns:
        agg['spread'] = 'mean'
    df = m1_df.set_index('time').resample(rule).agg(agg).dropna(subset=['open'])
    return df.reset_index()


def classify_ppt_stage(feat: dict, prev_feat: dict | None) -> tuple:
    """
    即時 PPT 階段分類（與 stage_classifier.py 規則完全同步）
    需要「前一根」的 is_big_red_k, is_big_green_k, is_breaking_upper/lower。
    
    回傳: (stage, signal_dir)
      stage      : 0=無訊號, 2=2階, 3=3階, 41=4-1階, 42=4-2階
      signal_dir : 1=多, -1=空, 0=無
    """
    if prev_feat is None:
        return 0, 0

    # 當前特徵
    cs     = feat['consolidation_score']
    mbd    = feat['mid_bb_dist']
    e12d   = feat['ema12_dist']
    pbr    = feat['pullback_ratio']
    cls_sh = feat['consecutive_low_shadows']
    chs_sh = feat['consecutive_high_shadows']
    brk_u  = feat['is_breaking_upper']
    brk_d  = feat['is_breaking_lower']
    rsi5   = feat['m5_rsi_14']
    rsi30  = feat['m30_rsi']
    h4t    = feat['h4_trend']
    h4bdm  = feat['h4_bb_dist_m']
    slope  = feat['m5_ema_slope']
    d5h    = feat['m5_dist_h']
    d5l    = feat['m5_dist_l']
    d5oh   = feat['m5_dist_oh']
    d5ol   = feat['m5_dist_ol']
    body   = feat['body_size']
    lshdw  = feat['lower_shadow']
    ushdw  = feat['upper_shadow']

    # 前一根特徵（4-1/4-2 階需要）
    prev_big_r = prev_feat.get('is_big_red_k', 0)
    prev_big_g = prev_feat.get('is_big_green_k', 0)
    prev_brk_u = prev_feat.get('is_breaking_upper', 0)
    prev_brk_d = prev_feat.get('is_breaking_lower', 0)

    # 4-2 階
    if prev_big_r == 1 and abs(e12d) < 3.5 and lshdw > abs(body) * 0.25 and d5l > -6.0:
        return 42, 1
    if prev_big_g == 1 and abs(e12d) < 3.5 and ushdw > abs(body) * 0.25 and d5h < 6.0:
        return 42, -1

    # 4-1 階
    if cs < 0.35 and prev_brk_u == 1 and 0.20 < pbr < 0.45 and slope > 0:
        return 41, 1
    if cs < 0.35 and prev_brk_d == 1 and -0.45 < pbr < -0.20 and slope < 0:
        return 41, -1

    # 3 階（逆勢外帶）
    if cs > 0.40 and d5ol <= 1.5 and rsi5 < 38 and rsi30 < 42 and cls_sh >= 1:
        return 3, 1
    if cs > 0.40 and d5oh >= -1.5 and rsi5 > 62 and rsi30 > 58 and chs_sh >= 1:
        return 3, -1

    # 2 階（盤整突破順勢）
    if cs > 0.55 and mbd > 0 and 0.28 < pbr < 0.65 and cls_sh >= 1 and brk_u == 0 and 38 < rsi5 < 68:
        return 2, 1
    if cs > 0.55 and mbd < 0 and -0.65 < pbr < -0.28 and chs_sh >= 1 and brk_d == 0 and 32 < rsi5 < 62:
        return 2, -1

    return 0, 0


def compute_features(i: int, m5: pd.DataFrame, m15: pd.DataFrame, m30: pd.DataFrame,
                     h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame,
                     m1: pd.DataFrame) -> dict | None:
    """計算第 i 根 M5 K 棒的 44 維特徵（與實盤策略完全一致）"""
    # 取前 500 根以確保 EMA 與 BB55 (外帶) 充分暖機收斂，與實盤策略一致
    win = m5.iloc[max(0, i - 499): i + 1]
    if len(win) < 55:
        return None

    live      = win.iloc[-1]
    m5_close  = win['close']
    bar_time  = pd.to_datetime(live['time'])

    ema12       = ta.trend.EMAIndicator(m5_close, window=12).ema_indicator()
    ema_slope   = ema12.iloc[-1] - ema12.iloc[-5]
    rsi         = ta.momentum.RSIIndicator(m5_close, window=14).rsi().iloc[-1]
    atr         = ta.volatility.AverageTrueRange(win['high'], win['low'], m5_close, window=14).average_true_range().iloc[-1]
    body        = live['close'] - live['open']
    u_shadow    = live['high'] - max(live['open'], live['close'])
    l_shadow    = min(live['open'], live['close']) - live['low']
    body_ratio  = body / (live['high'] - live['low'] + 1e-6)
    hour        = bar_time.hour
    dow         = bar_time.dayofweek

    # M15
    m15_win = m15[m15['time'] <= bar_time].tail(500)
    if len(m15_win) < 56:
        return None
    m15_bb_m = m15_win['close'].rolling(21).mean().iloc[-1]
    m15_bb_std = m15_win['close'].rolling(21).std(ddof=0).iloc[-1]
    m15_bb_h = m15_bb_m + 2.1 * m15_bb_std
    m15_bb_l = m15_bb_m - 2.1 * m15_bb_std
    
    m15_bb_o_m = m15_win['close'].rolling(55).mean().iloc[-1]
    m15_bb_o_std = m15_win['close'].rolling(55).std(ddof=0).iloc[-1]
    m15_bb_o_h = m15_bb_o_m + 2.0 * m15_bb_o_std
    m15_bb_o_l = m15_bb_o_m - 2.0 * m15_bb_o_std

    m15_ema  = ta.trend.EMAIndicator(m15_win['close'], window=12).ema_indicator()
    m15_dh   = live['close'] - m15_bb_h
    m15_dl   = live['close'] - m15_bb_l
    m15_dm   = live['close'] - m15_bb_m
    m15_es   = m15_ema.iloc[-1] - m15_ema.iloc[-5]
    m15_doh  = live['high'] - m15_bb_o_h  # 距M15外帶上軌
    m15_dol  = live['low']  - m15_bb_o_l  # 距M15外帶下軌

    # M5 BB（中帶 + 外帶）
    if len(win) < 56:
        return None
    bb_m = m5_close.rolling(21).mean().iloc[-1]
    bb_std = m5_close.rolling(21).std(ddof=0).iloc[-1]
    bb_h = bb_m + 2.1 * bb_std
    bb_l = bb_m - 2.1 * bb_std
    
    bb_o_m = m5_close.rolling(55).mean().iloc[-1]
    bb_o_std = m5_close.rolling(55).std(ddof=0).iloc[-1]
    bb_o_h = bb_o_m + 2.0 * bb_o_std
    bb_o_l = bb_o_m - 2.0 * bb_o_std

    m5_dh   = live['high'] - bb_h
    m5_dl   = live['low']  - bb_l
    m5_dm   = live['close'] - bb_m
    m5_de   = live['close'] - ema12.iloc[-1]
    bb_w    = (bb_h - bb_l) / (bb_m + 1e-6)
    m5_doh  = live['high'] - bb_o_h  # 距外帶上軌
    m5_dol  = live['low']  - bb_o_l  # 距外帶下軌
    # 盤整評分
    bb_w_all = 4.2 * m5_close.rolling(21).std(ddof=0)
    con_score = float(1.0 - (bb_w_all.iloc[-1] / (bb_w_all.iloc[-48:].max() + 1e-6)))
    adx     = ta.trend.ADXIndicator(win['high'], win['low'], m5_close).adx().iloc[-1]
    macd_h  = ta.trend.MACD(m5_close).macd_diff().iloc[-1]
    cci     = ta.trend.CCIIndicator(win['high'], win['low'], m5_close).cci().iloc[-1]

    prev_body   = win['close'].iloc[-2] - win['open'].iloc[-2]
    prev_chg    = win['close'].iloc[-1] - win['close'].iloc[-2]

    # 多時框指標
    def safe_diff(df_tf, window, ema_window=12):
        w = df_tf[df_tf['time'] <= bar_time].tail(500)
        return ta.trend.EMAIndicator(w['close'], window=ema_window).ema_indicator().diff().iloc[-1] if len(w) >= ema_window + 1 else 0.0
    def safe_rsi(df_tf, window=14):
        w = df_tf[df_tf['time'] <= bar_time].tail(500)
        return ta.momentum.RSIIndicator(w['close'], window=window).rsi().iloc[-1] if len(w) >= window + 1 else 50.0

    h1_trend  = safe_diff(h1, 12, 12)
    h4_win    = h4[h4['time'] <= bar_time].tail(500)
    h4_trend  = ta.trend.EMAIndicator(h4_win['close'], window=20).ema_indicator().diff().iloc[-1] if len(h4_win) >= 21 else 0.0
    # H4 布林帶距離
    if len(h4_win) >= 22:
        h4_bb_m = h4_win['close'].rolling(21).mean().iloc[-1]
        h4_bb_std = h4_win['close'].rolling(21).std(ddof=0).iloc[-1]
        h4_bb_h = h4_bb_m + 2.1 * h4_bb_std
        h4_bb_l = h4_bb_m - 2.1 * h4_bb_std
        h4_bdm     = float(h4_win['close'].iloc[-1] - h4_bb_m)
        h4_bdh     = float(h4_win['high'].iloc[-1]  - h4_bb_h)
        h4_bdl     = float(h4_win['low'].iloc[-1]   - h4_bb_l)
    else:
        h4_bdm, h4_bdh, h4_bdl = 0.0, 0.0, 0.0
    d1_rsi    = safe_rsi(d1)
    m30_rsi   = safe_rsi(m30)
    m30_win   = m30[m30['time'] <= bar_time].tail(500)
    m30_trend = ta.trend.EMAIndicator(m30_win['close'], window=20).ema_indicator().diff().iloc[-1] if len(m30_win) >= 21 else 0.0
    m1_win    = m1[m1['time'] <= bar_time].tail(20)
    m1_mom    = ta.trend.EMAIndicator(m1_win['close'], window=5).ema_indicator().diff().iloc[-1] if len(m1_win) >= 6 else 0.0

    vol_ratio   = live['tick_volume'] / (win['tick_volume'].rolling(20).mean().iloc[-1] + 1)
    spread_val  = float(live.get('spread', 0)) if 'spread' in live.index else 0.0
    is_us       = 1 if 13 <= hour <= 21 else 0
    is_asia     = 1 if 0  <= hour <= 8  else 0
    overlap     = 1 if 13 <= hour <= 17 else 0

    p5       = m5_close.iloc[-6] if len(win) >= 6 else live['close']
    rsi5     = ta.momentum.RSIIndicator(m5_close, window=14).rsi().iloc[-6] if len(win) >= 20 else rsi
    rsi_div  = -1.0 if (live['close'] > p5 and rsi < rsi5) else (1.0 if (live['close'] < p5 and rsi > rsi5) else 0.0)

    tp_price = (win['high'] + win['low'] + m5_close) / 3
    vwap     = (tp_price * win['tick_volume']).rolling(20).sum().iloc[-1] / (win['tick_volume'].rolling(20).sum().iloc[-1] + 1)
    p_vs_v   = live['close'] - vwap

    b_prev   = win['close'].iloc[-2] - win['open'].iloc[-2]
    o_prev   = win['open'].iloc[-2]
    engulf   = 1.0 if (body > 0 and b_prev < 0 and live['close'] > o_prev) else (-1.0 if (body < 0 and b_prev > 0 and live['close'] < o_prev) else 0.0)

    abs_body   = (win['close'] - win['open']).abs()
    l_sh       = np.minimum(win['open'], win['close']) - win['low']
    u_sh       = win['high'] - np.maximum(win['open'], win['close'])
    con_low    = float((l_sh.iloc[-3:] > abs_body.iloc[-3:] * 0.5).sum())
    con_high   = float((u_sh.iloc[-3:] > abs_body.iloc[-3:] * 0.5).sum())
    # 大紅K（上漲）/ 大綠K（下跌） — PPT 台灣慣例
    avg_body_20 = abs_body.rolling(20).mean().iloc[-1]
    is_big_red   = 1.0 if (body > 0 and abs(body) > avg_body_20 * 1.5) else 0.0
    is_big_green = 1.0 if (body < 0 and abs(body) > avg_body_20 * 1.5) else 0.0

    pullback   = (live['close'] - win['open'].iloc[-2]) / (win['close'].iloc[-2] - win['open'].iloc[-2] + 1e-6)
    brk_up     = 1.0 if live['close'] > bb_h else 0.0
    brk_dn     = 1.0 if live['close'] < bb_l else 0.0

    return {
        'm5_ema_slope': ema_slope, 'm5_rsi_14': rsi, 'm5_atr_14': atr,
        'body_size': body, 'upper_shadow': u_shadow, 'lower_shadow': l_shadow, 'body_ratio': body_ratio,
        'hour': hour, 'day_of_week': dow,
        'm15_dist_h': m15_dh, 'm15_dist_l': m15_dl, 'm15_dist_m': m15_dm, 'm15_ema_slope': m15_es,
        'm5_dist_h': m5_dh, 'm5_dist_l': m5_dl, 'm5_dist_m': m5_dm, 'm5_dist_ema': m5_de,
        'm5_adx': adx, 'm5_macd_hist': macd_h, 'm5_cci': cci, 'm5_bb_width': bb_w,
        'prev_body_size': prev_body, 'prev_close_change': prev_chg,
        'is_us_session': is_us, 'is_asia_session': is_asia,
        'h1_trend': h1_trend, 'h4_trend': h4_trend, 'd1_rsi': d1_rsi,
        'm30_rsi': m30_rsi, 'm30_trend': m30_trend, 'm1_momentum': m1_mom,
        'volume_ratio': vol_ratio, 'spread': spread_val, 'session_overlap': overlap,
        'rsi_divergence': rsi_div, 'price_vs_vwap': p_vs_v, 'pattern_engulf': engulf,
        'ema12_dist': m5_de, 'mid_bb_dist': m5_dm,
        'consecutive_low_shadows': con_low, 'consecutive_high_shadows': con_high,
        'pullback_ratio': pullback, 'is_breaking_upper': brk_up, 'is_breaking_lower': brk_dn,
        # 新增專家特徵
        'is_big_red_k': is_big_red, 'is_big_green_k': is_big_green,
        'm5_dist_oh': m5_doh, 'm5_dist_ol': m5_dol,
        'm15_dist_oh': m15_doh, 'm15_dist_ol': m15_dol,
        'h4_bb_dist_m': h4_bdm, 'h4_bb_dist_h': h4_bdh, 'h4_bb_dist_l': h4_bdl,
        'consolidation_score': con_score,
    }



def simulate_trade(entry: float, direction: str, tp_pts: float, sl_pts: float,
                   entry_idx: int, m1_arr: np.ndarray, m1_times: np.ndarray) -> tuple[str, float, pd.Timestamp]:
    """用 M1 逐根模擬交易，持倉直到 TP 或 SL 觸發為止（無強制超時）"""
    tp_p = entry + tp_pts if direction == 'buy' else entry - tp_pts
    sl_p = entry - sl_pts if direction == 'buy' else entry + sl_pts

    for j in range(entry_idx + 1, len(m1_arr)):
        h, l    = m1_arr[j, 0], m1_arr[j, 1]
        exit_time = pd.Timestamp(m1_times[j])
        if direction == 'buy':
            if h >= tp_p: return 'win',    tp_pts,  exit_time
            if l <= sl_p: return 'loss',  -sl_pts,  exit_time
        else:
            if l <= tp_p: return 'win',    tp_pts,  exit_time
            if h >= sl_p: return 'loss',  -sl_pts,  exit_time

    # 資料尾端仍未觸發：以最後收盤價結算
    last_close = m1_arr[-1, 2]
    exit_time  = pd.Timestamp(m1_times[-1])
    pnl = (last_close - entry) if direction == 'buy' else (entry - last_close)
    return 'timeout', round(pnl, 2), exit_time


def run_backtest():
    logger.info("=" * 60)
    logger.info("Iron Wall AI 回測引擎啟動")
    logger.info("=" * 60)

    # ── 載入模型 ──
    try:
        models = {
            'win_b': joblib.load(os.path.join(MODELS_DIR, 'win_buy.pkl')),
            'win_s': joblib.load(os.path.join(MODELS_DIR, 'win_sell.pkl')),
        }
        logger.info("✅ 成功載入 AI 信心率大腦 (win_buy / win_sell)")
    except FileNotFoundError as e:
        logger.error(f"❌ 模型不存在，請先訓練：{e}")
        return

    # ── 載入 M1 ──
    m1_path = os.path.join(HISTORY_DIR, "XAUUSD_M1.csv")
    if not os.path.exists(m1_path):
        logger.error(f"找不到 {m1_path}，請先匯出歷史數據。")
        return

    logger.info("📂 載入 M1 歷史數據...")
    m1_df = pd.read_csv(m1_path, parse_dates=['time']).sort_values('time').reset_index(drop=True)
    if 'spread' not in m1_df.columns:
        m1_df['spread'] = 0
    logger.info(f"M1 資料：{len(m1_df):,} 根  ({m1_df['time'].min()} ~ {m1_df['time'].max()})")

    # ── 重採樣 ──
    logger.info("⚙️  重採樣各時間框架...")
    m5_df  = resample_ohlcv(m1_df, '5min')
    m15_df = resample_ohlcv(m1_df, '15min')
    m30_df = resample_ohlcv(m1_df, '30min')
    h1_df  = resample_ohlcv(m1_df, '60min')
    h4_df  = resample_ohlcv(m1_df, '4h')
    d1_df  = resample_ohlcv(m1_df, '1D')
    logger.info(f"M5={len(m5_df):,}  M15={len(m15_df):,}  M30={len(m30_df):,}  H1={len(h1_df):,}")

    # M1 numpy array for fast simulate (columns: high, low, close)
    m1_arr  = m1_df[['high', 'low', 'close']].values
    m1_times = m1_df['time'].values  # numpy datetime64

    # ── 逐根 M5 回測 ──
    trades            = []
    in_position_until = pd.Timestamp.min
    total_bars        = len(m5_df) - WARM_UP_BARS
    prev_feat         = None   # 保留上一根特徵（PPT 4-1/4-2 階判斷需要）
    ppt_skip_count    = 0      # 跳過的非 PPT K 棒統計

    logger.info(f"🔄 開始逐根 M5 回測（共 {total_bars:,} 根）「PPT 嚴格模式 + AI 信心率 雙重確認」")


    for i in range(WARM_UP_BARS, len(m5_df)):
        bar_time = pd.Timestamp(m5_df.iloc[i]['time'])

        if bar_time <= in_position_until:
            continue

        feat = compute_features(i, m5_df, m15_df, m30_df, h1_df, h4_df, d1_df, m1_df)
        if feat is None:
            prev_feat = None
            continue

        # ── PPT 階段過濾（與實盤策略完全對齊）──────────────────────
        stage, signal_dir = classify_ppt_stage(feat, prev_feat)
        prev_feat = feat    # 更新前一根累積
        if stage == 0:
            ppt_skip_count += 1
            continue

        X = pd.DataFrame([feat])[FEATURE_COLS]

        # 信心率預測（固定 TP=8 / SL=5 條件下的獲勝機率）
        c_b = models['win_b'].predict_proba(X)[0, 1]
        c_s = models['win_s'].predict_proba(X)[0, 1]

        # ── PPT 方向鎖定：只保留訊號方向的信心率 ────────────────
        if signal_dir == 1:
            c_s = 0.0   # PPT 訊號是多，空單信心率強制歸零
        elif signal_dir == -1:
            c_b = 0.0   # PPT 訊號是空，多單信心率強制歸零

        can_buy  = c_b >= CONFIDENCE_THRESHOLD
        can_sell = c_s >= CONFIDENCE_THRESHOLD

        if not can_buy and not can_sell:
            continue

        if can_buy and can_sell:
            direction = 'buy' if c_b >= c_s else 'sell'
        elif can_buy:
            direction = 'buy'
        else:
            direction = 'sell'

        conf        = c_b if direction == 'buy' else c_s
        entry_price = float(m5_df.iloc[i]['close'])
        entry_idx   = np.searchsorted(m1_times, bar_time.to_datetime64())

        outcome, profit, exit_time = simulate_trade(
            entry_price, direction, FIXED_TP, FIXED_SL, entry_idx, m1_arr, m1_times)

        trades.append({
            'time': bar_time, 'direction': direction,
            'entry': entry_price, 'tp_pts': FIXED_TP, 'sl_pts': FIXED_SL,
            'confidence': round(conf * 100, 1), 'outcome': outcome, 'profit': profit,
            'exit_time': exit_time,
        })

        # 透過實際離場時間註册，離場後立刻可開下一單
        in_position_until = exit_time

        if len(trades) % 50 == 0:
            logger.info(f"  進度：{i - WARM_UP_BARS:,}/{total_bars:,}  已成交 {len(trades)} 筆...")

    # ── 報表 ──
    if not trades:
        logger.warning("❌ 回測完成，無任何交易。PPT 訊號太少或信心率門檻過高。")
        logger.info(f"   PPT 跳過 K 棒: {ppt_skip_count:,} 根")
        return

    df = pd.DataFrame(trades)
    wins     = df[df['outcome'] == 'win']
    losses   = df[df['outcome'] == 'loss']
    timeouts = df[df['outcome'] == 'timeout']
    total    = len(df)
    wr       = len(wins) / total * 100

    report = f"""
{'═'*60}
🏆  Iron Wall AI 回測報告
{'═'*60}
  回測區間 : {df['time'].min().date()} ~ {df['time'].max().date()}
  總交易數 : {total:,} 筆   多單: {len(df[df['direction']=='buy']):,}  空單: {len(df[df['direction']=='sell']):,}
  PPT略過 K棒: {ppt_skip_count:,} 根（非 PPT 訊號）
{'─'*60}
  勝  率   : {wr:6.1f}%
  獲利單   : {len(wins):,} 筆   平均獲利: +{wins['profit'].mean():.2f} 點
  虧損單   : {len(losses):,} 筆   平均虧損:  {losses['profit'].mean():.2f} 點
  超時單   : {len(timeouts):,} 筆
{'─'*60}
  總盈虧   : {df['profit'].sum():+.1f} 點
  平均 TP  : {df['tp_pts'].mean():.2f} 點
  平均 SL  : {df['sl_pts'].mean():.2f} 點
  平均信心 : {df['confidence'].mean():.1f}%
{'═'*60}"""
    logger.info(report)

    out = os.path.join(BASE_DIR, "backtest_result.csv")
    df.to_csv(out, index=False, encoding='utf-8-sig')
    logger.info(f"📄 詳細紀錄已存至: {out}")


if __name__ == "__main__":
    run_backtest()

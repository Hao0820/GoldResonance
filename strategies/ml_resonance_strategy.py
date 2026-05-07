import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import os
import time
import logging

import ta
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MLResonanceStrategy(BaseStrategy):
    def __init__(self, name="AI_Gold", symbol="XAUUSD", lot_size=0.1, trade_tracker=None):
        super().__init__(name)
        self.symbol = symbol
        self.lot_size = lot_size
        self.trade_tracker = trade_tracker

        # --- 策略參數 ---
        self.magic_number = 88001
        self.status = "AI 信心率大腦載入中..."
        self.FIXED_TP  = 5.0   # 止盈：5 點（與訓練標籤一致）
        self.FIXED_SL  = 5.0   # 止損：5 點
        self.THRESHOLD = 0.65  # 信心率門檻
        self.conf_buy  = 0.0   # 多單信心率 (0.0 ~ 1.0)
        self.conf_sell = 0.0   # 空單信心率 (0.0 ~ 1.0)

        # --- 波動過濾器 ---
        # M5 布林中軌(21均) 5根斜率絕對值超過此門檻 → 視為暴力行情，跳過模型
        # XAUUSD M5：正常回踩 < 1.5，趨勢加速 1.5~3.0，爆發行情 > 3.0
        self.BB_SLOPE_LIMIT = 3.0    # 單位：點/根（可在 GUI 調整）
        self.bb_slope       = 0.0    # 最新 BB 中軌斜率（供 GUI 顯示）
        self.filter_blocked = False  # True = 本次被波動過濾擋下

        # --- 嚴格模式 (PPT 規則) ---
        self.PPT_STRICT_MODE  = True  # 必須符合 2/3/4階規則才進場
        self.current_stage    = 0     # 供 GUI 顯示目前的階段 (0=無)
        self.signal_dir       = 0     # 供 GUI 顯示訊號方向 (1=多, -1=空, 0=無)

        # --- GUI 統計 ---
        self.today_wins       = 0     # 今日獲利筆數
        self.today_losses     = 0     # 今日虧損筆數
        self.last_signal_time = None  # 上次觸發訊號時間（字串）

        self.models_loaded = False
        self._load_models()

    def _load_models(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "ml_engine", "models")
        try:
            self.models = {
                'win_b': joblib.load(os.path.join(models_dir, 'win_buy.pkl')),
                'win_s': joblib.load(os.path.join(models_dir, 'win_sell.pkl')),
            }
            self.status = "AI 信心率大腦就緒 (TP=5 / SL=5)"
            self.models_loaded = True
            logger.info("✅ 成功載入信心率大腦 win_buy.pkl / win_sell.pkl")
        except Exception as e:
            self.status = f"模型載入失敗: {e}"
            logger.error(f"❌ 模型載入失敗: {e}")
            self.models_loaded = False


    def on_tick(self):
        """每秒被引擎驅動一次：計算 44 維特徵 → AI 預測 TP/SL → 決策下單"""
        if not self.models_loaded:
            return

        # 完整拓取所有時區數據 (統一抓取 500 根，保證所有指標與 MT5 顯示同步收斂)
        rates_m1  = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1,  0, 500)
        rates_m5  = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5,  0, 500)
        rates_m15 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, 500)
        rates_m30 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M30, 0, 500)
        rates_h1  = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1,  0, 500)
        rates_h4  = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H4,  0, 500)
        rates_d1  = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_D1,  0, 500)

        if rates_m5 is None or rates_m1 is None: return

        m5_df = pd.DataFrame(rates_m5)
        m1_df = pd.DataFrame(rates_m1)
        m15_df = pd.DataFrame(rates_m15)
        m5_live = m5_df.iloc[-1]
        m5_close = m5_df['close']

        # --- 特徵工程 ---
        m5_ema_12 = ta.trend.EMAIndicator(m5_close, window=12).ema_indicator()
        m5_ema_slope = m5_ema_12.iloc[-1] - m5_ema_12.iloc[-5]
        m5_rsi = ta.momentum.RSIIndicator(m5_close, window=14).rsi().iloc[-1]
        m5_atr = ta.volatility.AverageTrueRange(m5_df['high'], m5_df['low'], m5_close, window=14).average_true_range().iloc[-1]
        
        body_size = m5_live['close'] - m5_live['open']
        upper_shadow = m5_live['high'] - max(m5_live['open'], m5_live['close'])
        lower_shadow = min(m5_live['open'], m5_live['close']) - m5_live['low']
        body_ratio = body_size / (m5_live['high'] - m5_live['low'] + 0.0001)

        current_time = pd.to_datetime(m5_live['time'], unit='s')
        hour, day_of_week = current_time.hour, current_time.dayofweek
        is_us_session = 1 if 13 <= hour <= 21 else 0
        is_asia_session = 1 if 0 <= hour <= 8 else 0
        session_overlap = 1 if 13 <= hour <= 17 else 0

        m15_bb = ta.volatility.BollingerBands(m15_df['close'], window=21)
        m15_bb_o = ta.volatility.BollingerBands(m15_df['close'], window=55)  # 外帶
        m15_dist_h  = m5_live['close'] - m15_bb.bollinger_hband().iloc[-1]
        m15_dist_l  = m5_live['close'] - m15_bb.bollinger_lband().iloc[-1]
        m15_dist_m  = m5_live['close'] - m15_bb.bollinger_mavg().iloc[-1]
        m15_dist_oh = m5_live['high']  - m15_bb_o.bollinger_hband().iloc[-1]
        m15_dist_ol = m5_live['low']   - m15_bb_o.bollinger_lband().iloc[-1]
        m15_ema_12 = ta.trend.EMAIndicator(m15_df['close'], window=12).ema_indicator()
        m15_ema_slope = m15_ema_12.iloc[-1] - m15_ema_12.iloc[-5]

        # 顯式計算 M5 Bollinger Bands (對齊 MT5: SMA, ddof=0, 偏差 2.1)
        m5_bb_m = m5_close.rolling(21).mean().iloc[-1]
        m5_bb_std = m5_close.rolling(21).std(ddof=0).iloc[-1]
        m5_bb_h = m5_bb_m + 2.1 * m5_bb_std
        m5_bb_l = m5_bb_m - 2.1 * m5_bb_std
        
        # 顯式計算外帶 (window=55, dev=2.0)
        m5_bb_o_m = m5_close.rolling(55).mean().iloc[-1]
        m5_bb_o_std = m5_close.rolling(55).std(ddof=0).iloc[-1]
        m5_bb_o_h = m5_bb_o_m + 2.0 * m5_bb_o_std
        m5_bb_o_l = m5_bb_o_m - 2.0 * m5_bb_o_std
        m5_dist_h   = m5_live['high']  - m5_bb_h
        m5_dist_l   = m5_live['low']   - m5_bb_l
        m5_dist_m   = m5_live['close'] - m5_bb_m
        m5_dist_ema = m5_live['close'] - m5_ema_12.iloc[-1]
        m5_dist_oh  = m5_live['high']  - m5_bb_o_h
        m5_dist_ol  = m5_live['low']   - m5_bb_o_l
        # 盤整評分 (用歷史 48 根(4小時)寬度的最大值，避免太久遠的極端波幅干擾)
        # 上軌 - 下軌 = 4.2 * 標準差
        bb_w_series = 4.2 * m5_close.rolling(21).std(ddof=0)
        consolidation_score = float(1.0 - (bb_w_series.iloc[-1] / (bb_w_series.iloc[-48:].max() + 1e-6)))

        m5_adx = ta.trend.ADXIndicator(m5_df['high'], m5_df['low'], m5_close).adx().iloc[-1]
        m5_macd_hist = ta.trend.MACD(m5_close).macd_diff().iloc[-1]
        m5_cci = ta.trend.CCIIndicator(m5_df['high'], m5_df['low'], m5_close).cci().iloc[-1]
        m5_bb_width = (m5_bb_h - m5_bb_l) / (m5_bb_m + 0.0001)

        prev_body_size = m5_df['close'].iloc[-2] - m5_df['open'].iloc[-2]
        prev_close_change = m5_df['close'].iloc[-1] - m5_df['close'].iloc[-2]

        h1_trend = ta.trend.EMAIndicator(pd.Series([r['close'] for r in rates_h1]), window=12).ema_indicator().diff().iloc[-1]
        # H4 趨勢 + 布林帶
        h4_close_s = pd.Series([r['close'] for r in rates_h4])
        h4_high_s  = pd.Series([r['high']  for r in rates_h4])
        h4_low_s   = pd.Series([r['low']   for r in rates_h4])
        h4_trend   = ta.trend.EMAIndicator(h4_close_s, window=20).ema_indicator().diff().iloc[-1]
        h4_bb_m = h4_close_s.rolling(21).mean().iloc[-1]
        h4_bb_std = h4_close_s.rolling(21).std(ddof=0).iloc[-1]
        h4_bb_h = h4_bb_m + 2.1 * h4_bb_std
        h4_bb_l = h4_bb_m - 2.1 * h4_bb_std
        h4_bb_dist_m = float(h4_close_s.iloc[-1] - h4_bb_m)
        h4_bb_dist_h = float(h4_high_s.iloc[-1]  - h4_bb_h)
        h4_bb_dist_l = float(h4_low_s.iloc[-1]   - h4_bb_l)

        d1_rsi = ta.momentum.RSIIndicator(pd.Series([r['close'] for r in rates_d1]), window=14).rsi().iloc[-1]
        m30_df = pd.DataFrame(rates_m30)
        m30_rsi, m30_trend = ta.momentum.RSIIndicator(m30_df['close'], window=14).rsi().iloc[-1], ta.trend.EMAIndicator(m30_df['close'], window=20).ema_indicator().diff().iloc[-1]
        m1_momentum = ta.trend.EMAIndicator(m1_df['close'], window=5).ema_indicator().diff().iloc[-1]

        volume_ratio, spread_val = float(m5_live['tick_volume']) / (m5_df['tick_volume'].rolling(20).mean().iloc[-1] + 1), float(m5_live['spread'])

        price_prev5 = m5_df['close'].iloc[-6]
        rsi_prev5 = ta.momentum.RSIIndicator(m5_close, window=14).rsi().iloc[-6]
        rsi_divergence = -1.0 if (m5_live['close'] > price_prev5 and m5_rsi < rsi_prev5) else (1.0 if (m5_live['close'] < price_prev5 and m5_rsi > rsi_prev5) else 0.0)

        vwap = ((m5_df['high'] + m5_df['low'] + m5_close) / 3 * m5_df['tick_volume']).rolling(20).sum().iloc[-1] / (m5_df['tick_volume'].rolling(20).sum().iloc[-1] + 1)
        price_vs_vwap = m5_live['close'] - vwap
        pattern_engulf = 1.0 if (body_size > 0 and prev_body_size < 0 and m5_live['close'] > m5_df['open'].iloc[-2]) else (-1.0 if (body_size < 0 and prev_body_size > 0 and m5_live['close'] < m5_df['open'].iloc[-2]) else 0.0)

        # PPT + 大K棒 + 外帶特徵
        ema12_dist, mid_bb_dist = m5_live['close'] - m5_ema_12.iloc[-1], m5_live['close'] - m5_bb_m
        m5_bodies_abs = (m5_df['close'] - m5_df['open']).abs()
        m5_l_shadows = np.minimum(m5_df['open'], m5_df['close']) - m5_df['low']
        m5_u_shadows = m5_df['high'] - np.maximum(m5_df['open'], m5_df['close'])
        consecutive_low_shadows  = float((m5_l_shadows.iloc[-3:] > m5_bodies_abs.iloc[-3:] * 0.5).sum())
        consecutive_high_shadows = float((m5_u_shadows.iloc[-3:] > m5_bodies_abs.iloc[-3:] * 0.5).sum())
        avg_body_20  = m5_bodies_abs.rolling(20).mean().iloc[-1]
        # PPT 台灣慣例：紅K = 上漲（body > 0），綠K = 下跌（body < 0）
        is_big_red_k   = 1.0 if (body_size > 0 and abs(body_size) > avg_body_20 * 1.5) else 0.0
        is_big_green_k = 1.0 if (body_size < 0 and abs(body_size) > avg_body_20 * 1.5) else 0.0
        prev_o, prev_c = m5_df['open'].iloc[-2], m5_df['close'].iloc[-2]
        pullback_ratio = (m5_live['close'] - prev_o) / (prev_c - prev_o + 0.0001)
        is_breaking_upper = (1.0 if m5_live['close'] > m5_bb_h else 0.0)
        is_breaking_lower = (1.0 if m5_live['close'] < m5_bb_l else 0.0)

        # PPT 嚴格模式所需的前一根特徵
        prev_body = prev_c - prev_o
        prev_avg_body_20 = m5_bodies_abs.rolling(20).mean().iloc[-2] if len(m5_bodies_abs) >= 20 else avg_body_20
        prev_big_red_k   = 1.0 if (prev_body > 0 and abs(prev_body) > prev_avg_body_20 * 1.5) else 0.0
        prev_big_green_k = 1.0 if (prev_body < 0 and abs(prev_body) > prev_avg_body_20 * 1.5) else 0.0
        prev_bb_m = m5_close.rolling(21).mean().iloc[-2]
        prev_bb_std = m5_close.rolling(21).std(ddof=0).iloc[-2]
        prev_bb_h = prev_bb_m + 2.1 * prev_bb_std
        prev_bb_l = prev_bb_m - 2.1 * prev_bb_std
        prev_breaking_upper = 1.0 if prev_c > prev_bb_h else 0.0
        prev_breaking_lower = 1.0 if prev_c < prev_bb_l else 0.0

        # 計算當前所屬 PPT 階段
        stage = 0
        signal_dir = 0
        
        # 4-2 階
        if prev_big_red_k == 1 and abs(ema12_dist) < 3.5 and lower_shadow > abs(body_size) * 0.25 and m5_dist_l > -6.0:
            stage = 42; signal_dir = 1
        elif prev_big_green_k == 1 and abs(ema12_dist) < 3.5 and upper_shadow > abs(body_size) * 0.25 and m5_dist_h < 6.0:
            stage = 42; signal_dir = -1
        # 4-1 階
        elif consolidation_score < 0.35 and prev_breaking_upper == 1 and 0.20 < pullback_ratio < 0.45 and m5_ema_slope > 0:
            stage = 41; signal_dir = 1
        elif consolidation_score < 0.35 and prev_breaking_lower == 1 and -0.45 < pullback_ratio < -0.20 and m5_ema_slope < 0:
            stage = 41; signal_dir = -1
        # 3 階
        elif consolidation_score > 0.40 and m5_dist_ol <= 1.5 and m5_rsi < 38 and m30_rsi < 42 and consecutive_low_shadows >= 1:
            stage = 3; signal_dir = 1
        elif consolidation_score > 0.40 and m5_dist_oh >= -1.5 and m5_rsi > 62 and m30_rsi > 58 and consecutive_high_shadows >= 1:
            stage = 3; signal_dir = -1
        # 2 階
        elif consolidation_score > 0.55 and mid_bb_dist > 0 and 0.28 < pullback_ratio < 0.65 and consecutive_low_shadows >= 1 and is_breaking_upper == 0 and 38 < m5_rsi < 68:
            stage = 2; signal_dir = 1
        elif consolidation_score > 0.55 and mid_bb_dist < 0 and -0.65 < pullback_ratio < -0.28 and consecutive_high_shadows >= 1 and is_breaking_lower == 0 and 32 < m5_rsi < 62:
            stage = 2; signal_dir = -1

        self.current_stage = stage
        self.signal_dir    = signal_dir   # 同步給 GUI 顯示方向

        # 53 維特徵對齊（與 create_dataset / train_model / backtest 完全一致）
        feature_cols = [
            'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
            'hour', 'day_of_week', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
            'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema', 'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
            'prev_body_size', 'prev_close_change', 'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
            'm30_rsi', 'm30_trend', 'm1_momentum', 'volume_ratio', 'spread', 'session_overlap', 'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
            'ema12_dist', 'mid_bb_dist', 'consecutive_low_shadows', 'consecutive_high_shadows', 'pullback_ratio', 'is_breaking_upper', 'is_breaking_lower',
            'is_big_red_k', 'is_big_green_k', 'm5_dist_oh', 'm5_dist_ol', 'm15_dist_oh', 'm15_dist_ol',
            'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l', 'consolidation_score',
        ]
        feature_data = {
            'm5_ema_slope': m5_ema_slope, 'm5_rsi_14': m5_rsi, 'm5_atr_14': m5_atr, 'body_size': body_size,
            'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow, 'body_ratio': body_ratio,
            'hour': hour, 'day_of_week': day_of_week, 'm15_dist_h': m15_dist_h, 'm15_dist_l': m15_dist_l,
            'm15_dist_m': m15_dist_m, 'm15_ema_slope': m15_ema_slope, 'm5_dist_h': m5_dist_h, 'm5_dist_l': m5_dist_l,
            'm5_dist_m': m5_dist_m, 'm5_dist_ema': m5_dist_ema, 'm5_adx': m5_adx, 'm5_macd_hist': m5_macd_hist,
            'm5_cci': m5_cci, 'm5_bb_width': m5_bb_width, 'prev_body_size': prev_body_size, 'prev_close_change': prev_close_change,
            'is_us_session': is_us_session, 'is_asia_session': is_asia_session, 'h1_trend': h1_trend, 'h4_trend': h4_trend,
            'd1_rsi': d1_rsi, 'm30_rsi': m30_rsi, 'm30_trend': m30_trend, 'm1_momentum': m1_momentum,
            'volume_ratio': volume_ratio, 'spread': spread_val, 'session_overlap': session_overlap,
            'rsi_divergence': rsi_divergence, 'price_vs_vwap': price_vs_vwap, 'pattern_engulf': pattern_engulf,
            'ema12_dist': ema12_dist, 'mid_bb_dist': mid_bb_dist, 'consecutive_low_shadows': consecutive_low_shadows,
            'consecutive_high_shadows': consecutive_high_shadows, 'pullback_ratio': pullback_ratio,
            'is_breaking_upper': is_breaking_upper, 'is_breaking_lower': is_breaking_lower,
            'is_big_red_k': is_big_red_k, 'is_big_green_k': is_big_green_k,
            'm5_dist_oh': m5_dist_oh, 'm5_dist_ol': m5_dist_ol,
            'm15_dist_oh': m15_dist_oh, 'm15_dist_ol': m15_dist_ol,
            'h4_bb_dist_m': h4_bb_dist_m, 'h4_bb_dist_h': h4_bb_dist_h, 'h4_bb_dist_l': h4_bb_dist_l,
            'consolidation_score': consolidation_score,
        }
        
        # 儲存當前指標數值供 GUI 顯示
        self.live_indicators = {
            'close': m5_live['close'],
            'ema12': m5_ema_12.iloc[-1],
            'ema12_dist': ema12_dist,
            'bb_h': m5_bb_h,
            'bb_l': m5_bb_l,
            'bb_m': m5_bb_m,
            'bb_o_h': m5_bb_o_h,
            'bb_o_l': m5_bb_o_l,
            'consolidation_score': consolidation_score,
            'h4_trend': h4_trend,
            'm5_rsi': m5_rsi,
        }

        X = pd.DataFrame([feature_data])[feature_cols]

        # ── 波動過濾器：依 PPT 階段分開判斷 ────────────────────────────────
        # 4-2 / 4-1 階 → 順勢交易，斜率陡反而是進場動力，不受波動過濾限制
        # 2 / 3 階    → 盤整/逆勢，斜率太陡代表行情不對，擋下
        # stage=0     → 無訊號，直接擋
        # ─────────────────────────────────────────────────────────────────
        trend_stages = {42, 41}   # 順勢階段：不受 BB 斜率限制
        calm_stages  = {2, 3}     # 盤整/逆勢階段：斜率過大時擋下

        # BB 中軌斜率計算（每根平均位移量）
        bb_mavg_series = m5_close.rolling(21).mean()
        if len(bb_mavg_series) >= 6:
            self.bb_slope = float(bb_mavg_series.iloc[-1] - bb_mavg_series.iloc[-6]) / 5.0
        else:
            self.bb_slope = 0.0

        slope_blocked = abs(self.bb_slope) > self.BB_SLOPE_LIMIT

        if stage in trend_stages:
            # 4-2 / 4-1：順勢，完全放行，不管斜率
            self.filter_blocked = False
        elif stage in calm_stages and slope_blocked:
            # 2 / 3 階遇到爆發行情：封鎖
            self.filter_blocked = True
            self.conf_buy  = 0.0
            self.conf_sell = 0.0
            logger.debug(
                f"⚡ 波動過濾（{stage}階）| BB斜率={self.bb_slope:+.3f} > ±{self.BB_SLOPE_LIMIT} | 略過"
            )
            return
        else:
            self.filter_blocked = False


        # ── AI 信心率預測：多/空單在固定 TP=5/SL=5 條件下的獲勝機率 ────────
        self.conf_buy  = self.models['win_b'].predict_proba(X)[0, 1]
        self.conf_sell = self.models['win_s'].predict_proba(X)[0, 1]

        # ── 嚴格 PPT 過濾 ──────────────────────────────────────────────────
        if getattr(self, 'PPT_STRICT_MODE', True):
            if stage == 0:
                self.conf_buy = 0.0
                self.conf_sell = 0.0
                self.filter_blocked = True
            else:
                # 只有符合 PPT 方向才保留信心率
                if signal_dir == 1:
                    self.conf_sell = 0.0
                elif signal_dir == -1:
                    self.conf_buy = 0.0

        # UI 展示
        self.model_buy_prob  = self.conf_buy
        self.model_sell_prob = self.conf_sell

        # ── 下單決策：信心率 >= 65% 才進場 ──────────────────────────────────
        can_buy  = self.conf_buy  >= self.THRESHOLD
        can_sell = self.conf_sell >= self.THRESHOLD

        if can_buy and can_sell:
            if self.conf_buy >= self.conf_sell:
                self.open_position(mt5.ORDER_TYPE_BUY,  "AI_Buy")
            else:
                self.open_position(mt5.ORDER_TYPE_SELL, "AI_Sell")
        elif can_buy:
            self.open_position(mt5.ORDER_TYPE_BUY,  "AI_Buy")
        elif can_sell:
            self.open_position(mt5.ORDER_TYPE_SELL, "AI_Sell")


    def open_position(self, order_type, comment):
        """以固定 TP=5 / SL=5 掛單到 MT5"""
        if self.has_open_position(self.magic_number): return
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None: return
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        tp = price + self.FIXED_TP if order_type == mt5.ORDER_TYPE_BUY else price - self.FIXED_TP
        sl = price - self.FIXED_SL if order_type == mt5.ORDER_TYPE_BUY else price + self.FIXED_SL
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol,
            "volume": self.lot_size, "type": order_type,
            "price": price, "sl": sl, "tp": tp,
            "magic": self.magic_number, "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            direction = '多' if order_type == mt5.ORDER_TYPE_BUY else '空'
            conf      = self.conf_buy if order_type == mt5.ORDER_TYPE_BUY else self.conf_sell
            logger.info(f"🚀 [{comment}] 進場成功 | {direction}單 | 信心率: {conf*100:.1f}% | 價: {price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
            if self.trade_tracker:
                try:
                    self.trade_tracker.log_open_trade(
                        ticket=res.order, model_name=comment,
                        order_type="BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL",
                        lots=self.lot_size, open_price=price, sl=sl, tp=tp
                    )
                except Exception as e:
                    logger.error(f"⚠️ 交易記錄寫入失敗: {e}")
        else:
            logger.error(f"❌ [{comment}] 進場失敗 | MT5 錯誤碼: {res.retcode} | {res.comment}")


    def has_open_position(self, magic):
        pos = mt5.positions_get(symbol=self.symbol)
        if pos:
            for p in pos:
                if p.magic == magic: return True
        return False

import MetaTrader5 as mt5
import logging
import pandas as pd
import numpy as np
import ta
import joblib
import os
import warnings

# 徹底攔截 AI 模型載入時的並行運算警告
warnings.filterwarnings("ignore", category=UserWarning)

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MLResonanceStrategy(BaseStrategy):
    def __init__(self, name="ML_Resonance_AI", symbol="XAUUSD", lot_size=0.1):
        super().__init__(name)
        self.symbol = symbol
        self.lot_size = lot_size
        self.max_positions = 1
        self.magic_number = 88005 # AI 專屬 Magic Number
        self.status = "🧠 AI 模型載入中..."
        self.last_signal_time = None
        self.sl_dist = 4.0
        self.tp_dist = 7.0
        self.xgb_weight = 0.7
        self.rf_weight = 0.3
        
        # 載入模型
        self._load_models()
        
    def _load_models(self):
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        try:
            self.xgb_buy = joblib.load(os.path.join(base_path, 'xgb_buy.pkl'))
            self.rf_buy = joblib.load(os.path.join(base_path, 'rf_buy.pkl'))
            self.xgb_sell = joblib.load(os.path.join(base_path, 'xgb_sell.pkl'))
            self.rf_sell = joblib.load(os.path.join(base_path, 'rf_sell.pkl'))
            
            # 強制將預測執行緒設為 1，解決 Parallels 虛擬機中的 joblib 平行運算警告
            self.xgb_buy.n_jobs = 1
            self.rf_buy.n_jobs = 1
            self.xgb_sell.n_jobs = 1
            self.rf_sell.n_jobs = 1
            
            self.status = "🧠 AI 大腦運轉中 (XGB+RF)"
            self.models_loaded = True
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            self.status = "❌ AI 模型缺失"
            self.models_loaded = False

    def get_data(self):
        # 取得全時區數據 (M1, M5, M15, M30, H1, H4, D1)
        m1_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M1, 60)
        m5_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M5, 300)
        m15_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M15, 100)
        m30_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M30, 60)
        h1_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_H1, 60)
        h4_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_H4, 50)
        d1_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_D1, 30)
        
        if any(df is None or df.empty for df in [m1_df, m5_df, m15_df, m30_df, h1_df, h4_df, d1_df]):
            return None, None, None, None, None, None, None
            
        return m1_df, m5_df, m15_df, m30_df, h1_df, h4_df, d1_df

    def get_current_price(self):
        info = self.connector.get_symbol_info(self.symbol)
        if info:
            return info.get('ask', 0.0), info.get('bid', 0.0)
        return 0.0, 0.0

    def on_tick(self, can_execute_new_trades: bool = True):
        if not self.models_loaded:
            return
            
        ask, bid = self.get_current_price()
        if ask == 0 or bid == 0:
            return

        m1_df, m5_df, m15_df, m30_df, h1_df, h4_df, d1_df = self.get_data()
        if m1_df is None or m5_df is None or len(m5_df) < 20: 
            return
            
        # 計算特徵 (與 create_dataset 邏輯保持 100% 一致)
        # M15 特徵
        m15_bb = ta.volatility.BollingerBands(close=m15_df['close'], window=21, window_dev=2.1)
        m15_bb_h = m15_bb.bollinger_hband().iloc[-1]
        m15_bb_l = m15_bb.bollinger_lband().iloc[-1]
        m15_bb_m = m15_bb.bollinger_mavg().iloc[-1]
        m15_ema = ta.trend.EMAIndicator(close=m15_df['close'], window=12).ema_indicator()
        m15_ema_curr = m15_ema.iloc[-1]
        m15_ema_prev5 = m15_ema.iloc[-6] if len(m15_ema) >= 6 else m15_ema_curr
        m15_ema_slope = m15_ema_curr - m15_ema_prev5
        
        m15_curr = m15_df.iloc[-1]
        m15_dist_h = m15_curr['high'] - m15_bb_h
        m15_dist_l = m15_curr['low'] - m15_bb_l
        m15_dist_m = m15_curr['close'] - m15_bb_m
        
        # M5 特徵
        m5_bb = ta.volatility.BollingerBands(close=m5_df['close'], window=21, window_dev=2.1)
        m5_bb_h = m5_bb.bollinger_hband().iloc[-1]
        m5_bb_l = m5_bb.bollinger_lband().iloc[-1]
        m5_bb_m = m5_bb.bollinger_mavg().iloc[-1]
        m5_ema = ta.trend.EMAIndicator(close=m5_df['close'], window=12).ema_indicator()
        m5_ema_curr = m5_ema.iloc[-1]
        m5_ema_prev5 = m5_ema.iloc[-6] if len(m5_ema) >= 6 else m5_ema_curr
        m5_ema_slope = m5_ema_curr - m5_ema_prev5
        
        m5_rsi = ta.momentum.RSIIndicator(close=m5_df['close'], window=14).rsi().iloc[-1]
        m5_atr = ta.volatility.AverageTrueRange(high=m5_df['high'], low=m5_df['low'], close=m5_df['close'], window=14).average_true_range().iloc[-1]
        
        m5_live = m5_df.iloc[-1]
        body_size = m5_live['close'] - m5_live['open']
        upper_shadow = m5_live['high'] - max(m5_live['open'], m5_live['close'])
        lower_shadow = min(m5_live['open'], m5_live['close']) - m5_live['low']
        body_ratio = body_size / (m5_live['high'] - m5_live['low'] + 0.0001)
        
        hour = m5_live['time'].hour
        day_of_week = m5_live['time'].dayofweek
        
        m5_dist_h = m5_live['high'] - m5_bb_h
        m5_dist_l = m5_live['low'] - m5_bb_l
        m5_dist_m = m5_live['close'] - m5_bb_m
        m5_dist_ema = m5_live['close'] - m5_ema_curr
        
        # 4. 新增高階特徵 (與訓練集完全一致)
        # ADX
        adx_m5 = ta.trend.ADXIndicator(m5_df['high'], m5_df['low'], m5_df['close'], window=14)
        m5_adx = adx_m5.adx().iloc[-1]
        
        # MACD
        macd_m5 = ta.trend.MACD(m5_df['close'])
        m5_macd_hist = macd_m5.macd_diff().iloc[-1]
        
        # CCI
        m5_cci = ta.trend.CCIIndicator(m5_df['high'], m5_df['low'], m5_df['close'], window=20).cci().iloc[-1]
        
        # BB Width
        m5_bb_width = (m5_bb_h - m5_bb_l) / (m5_bb_m + 0.0001)
        
        # 型態基礎
        prev_m5 = m5_df.iloc[-2]
        prev_body_size = prev_m5['close'] - prev_m5['open']
        prev_close_change = m5_live['close'] - prev_m5['close']
        
        # H1 大趨勢背景
        h1_ema_12 = ta.trend.EMAIndicator(h1_df['close'], window=12).ema_indicator()
        h1_trend = h1_ema_12.iloc[-1] - h1_ema_12.iloc[-2]

        # 5. 市場時段與多時區 (Level 4)
        is_us_session = 1 if 13 <= hour <= 21 else 0
        is_asia_session = 1 if 0 <= hour <= 8 else 0
        
        # H4 趨勢
        h4_ema_20 = ta.trend.EMAIndicator(h4_df['close'], window=20).ema_indicator()
        h4_trend = h4_ema_20.iloc[-1] - h4_ema_20.iloc[-2]
        
        # D1 強弱
        d1_rsi = ta.momentum.RSIIndicator(d1_df['close'], window=14).rsi().iloc[-1]

        # 6. 新增 M1 與 M30 特徵 (Level 5)
        # M30 特徵
        m30_rsi = ta.momentum.RSIIndicator(m30_df['close'], window=14).rsi().iloc[-1]
        m30_ema_20 = ta.trend.EMAIndicator(m30_df['close'], window=20).ema_indicator()
        m30_trend = m30_ema_20.iloc[-1] - m30_ema_20.iloc[-2]
        
        # M1 微觀動能
        m1_ema_5 = ta.trend.EMAIndicator(m1_df['close'], window=5).ema_indicator()
        m1_momentum = m1_ema_5.iloc[-1] - m1_ema_5.iloc[-2]

        # 打包成特徵陣列 (必須與訓練時的順序一模一樣，共 32 維度)
        features = pd.DataFrame([{
            'm5_ema_slope': m5_ema_slope, 'm5_rsi_14': m5_rsi, 'm5_atr_14': m5_atr,
            'body_size': body_size, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow, 'body_ratio': body_ratio,
            'hour': hour, 'day_of_week': day_of_week,
            'm15_dist_h': m15_dist_h, 'm15_dist_l': m15_dist_l, 'm15_dist_m': m15_dist_m, 'm15_ema_slope': m15_ema_slope,
            'm5_dist_h': m5_dist_h, 'm5_dist_l': m5_dist_l, 'm5_dist_m': m5_dist_m, 'm5_dist_ema': m5_dist_ema,
            'm5_adx': m5_adx, 'm5_macd_hist': m5_macd_hist, 'm5_cci': m5_cci, 'm5_bb_width': m5_bb_width,
            'prev_body_size': prev_body_size, 'prev_close_change': prev_close_change,
            'is_us_session': is_us_session, 'is_asia_session': is_asia_session, 'h1_trend': h1_trend, 'h4_trend': h4_trend, 'd1_rsi': d1_rsi,
            'm30_rsi': m30_rsi, 'm30_trend': m30_trend, 'm1_momentum': m1_momentum
        }])
        
        # AI 預測 (取得勝率 Probability)
        # XGBoost 和 Random Forest 的 predict_proba 返回 [敗率, 勝率]
        xgb_buy_prob = self.xgb_buy.predict_proba(features)[0][1]
        rf_buy_prob = self.rf_buy.predict_proba(features)[0][1]
        
        xgb_sell_prob = self.xgb_sell.predict_proba(features)[0][1]
        rf_sell_prob = self.rf_sell.predict_proba(features)[0][1]
        
        # 綜合預測勝率 (Ensemble) - 使用動態權重
        buy_score = (xgb_buy_prob * self.xgb_weight) + (rf_buy_prob * self.rf_weight)
        sell_score = (xgb_sell_prob * self.xgb_weight) + (rf_sell_prob * self.rf_weight)
        
        # 更新 UI 狀態
        self.status = f"🧠 AI 預測勝率 | 多單: {buy_score*100:.1f}% | 空單: {sell_score*100:.1f}% (XGB:{self.xgb_weight*100:.0f}%)"
        
        # 下單邏輯
        if not can_execute_new_trades: return
        if self.last_signal_time == m5_live['time']: return
        
        positions = mt5.positions_get(symbol=self.symbol)
        pos_count = len([p for p in positions if str(p.magic).startswith(str(self.magic_number))]) if positions else 0
        if pos_count >= self.max_positions: return
        
        # 設定極高的 AI 勝率門檻 (例如 65% 以上才下單，結合 XGBoost 與 Random Forest 的判斷)
        THRESHOLD = 0.65
        
        if buy_score > THRESHOLD and buy_score > sell_score:
            sl, tp = ask - self.sl_dist, ask + self.tp_dist
            logger.info(f"🚀 AI 發起多單進場! (預測勝率: {buy_score*100:.1f}%)")
            self.executor.send_order(self.symbol, mt5.ORDER_TYPE_BUY, self.lot_size, ask, sl, tp, comment=f"AI-BUY-{buy_score:.2f}", magic=self.magic_number)
            self.last_signal_time = m5_live['time']
            return
            
        if sell_score > THRESHOLD and sell_score > buy_score:
            sl, tp = bid + self.sl_dist, bid - self.tp_dist
            logger.info(f"🚀 AI 發起空單進場! (預測勝率: {sell_score*100:.1f}%)")
            self.executor.send_order(self.symbol, mt5.ORDER_TYPE_SELL, self.lot_size, bid, sl, tp, comment=f"AI-SELL-{sell_score:.2f}", magic=self.magic_number)
            self.last_signal_time = m5_live['time']
            return

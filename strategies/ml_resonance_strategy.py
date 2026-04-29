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
    def __init__(self, name="ML_Resonance_AI", symbol="XAUUSD", lot_size_A=0.1, lot_size_B=0.1, trade_tracker=None):
        super().__init__(name)
        self.symbol = symbol
        self.lot_size_A = lot_size_A
        self.lot_size_B = lot_size_B
        self.max_positions = 2  # 允許兩個模型各自持倉
        self.magic_number_A = 88001 # 模型A (利潤極大化)
        self.magic_number_B = 88002 # 模型B (勝率極大化)
        self.status = "雙引擎 AI 載入中..."
        self.model_a_buy = 0.0
        self.model_a_sell = 0.0
        self.model_b_buy = 0.0
        self.model_b_sell = 0.0
        self.last_signal_time_A = None
        self.last_signal_time_B = None
        self.trade_tracker = trade_tracker
        self.tick_counter = 0
        
        # 載入模型
        self._load_models()
        
    def _load_models(self):
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        try:
            # 載入所有集成專家
            self.xgb_buy = joblib.load(os.path.join(base_path, 'xgb_buy.pkl'))
            self.lgb_buy = joblib.load(os.path.join(base_path, 'lgb_buy.pkl'))
            self.rf_buy  = joblib.load(os.path.join(base_path, 'rf_buy.pkl'))
            self.stack_buy = joblib.load(os.path.join(base_path, 'stack_buy.pkl'))
            
            self.xgb_sell = joblib.load(os.path.join(base_path, 'xgb_sell.pkl'))
            self.lgb_sell = joblib.load(os.path.join(base_path, 'lgb_sell.pkl'))
            self.rf_sell  = joblib.load(os.path.join(base_path, 'rf_sell.pkl'))
            self.stack_sell = joblib.load(os.path.join(base_path, 'stack_sell.pkl'))
            
            # 效能優化 (多執行緒設為 1)
            for m in [self.xgb_buy, self.lgb_buy, self.rf_buy, self.xgb_sell, self.lgb_sell, self.rf_sell]:
                if hasattr(m, 'n_jobs'): m.n_jobs = 1
            
            self.status = "🧠 集成大腦運轉中 (XGB+LGB+RF+Stacking)"
            self.models_loaded = True
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            self.status = "❌ AI 模型缺失"
            self.models_loaded = False

    def reload_models(self):
        logger.info("🔄 開始重新載入 AI 模型...")
        self._load_models()
        if self.models_loaded:
            logger.info("✅ AI 模型重載完成！新大腦已上線。")

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

        # ===== 新增 6 個高品質即時特徵 =====
        # 1. 成交量比率
        vol_ma20 = m5_df['tick_volume'].rolling(20).mean().iloc[-1]
        volume_ratio = float(m5_live['tick_volume']) / (vol_ma20 + 1)
        
        # 2. 點差
        if 'spread' in m5_df.columns:
            spread_val = float(m5_live['spread'])
        else:
            spread_val = (m5_live['high'] - m5_live['low']) / (m5_atr + 0.001)
        
        # 3. 歐美盤重疊時段
        session_overlap = 1 if 13 <= hour <= 17 else 0
        
        # 4. RSI 背離
        rsi_series = ta.momentum.RSIIndicator(close=m5_df['close'], window=14).rsi()
        rsi_prev5   = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi_series.iloc[-1]
        price_prev5 = m5_df['close'].iloc[-6] if len(m5_df) >= 6 else m5_df['close'].iloc[-1]
        if m5_live['close'] > price_prev5 and m5_rsi <= rsi_prev5:
            rsi_divergence = -1.0
        elif m5_live['close'] < price_prev5 and m5_rsi >= rsi_prev5:
            rsi_divergence = 1.0
        else:
            rsi_divergence = 0.0
        
        # 5. 價格與 VWAP 的距離
        typical_price = (m5_df['high'] + m5_df['low'] + m5_df['close']) / 3
        vwap = (typical_price * m5_df['tick_volume']).rolling(20).sum().iloc[-1] / \
               (m5_df['tick_volume'].rolling(20).sum().iloc[-1] + 1)
        price_vs_vwap = m5_live['close'] - vwap
        
        # 6. 吞噬型 K 棒
        prev2_open  = m5_df['open'].iloc[-2]
        prev2_close = m5_df['close'].iloc[-2]
        prev2_body  = prev2_close - prev2_open
        if body_size > 0 and prev2_body < 0 and m5_live['close'] > prev2_open and m5_live['open'] < prev2_close:
            pattern_engulf = 1.0
        elif body_size < 0 and prev2_body > 0 and m5_live['close'] < prev2_open and m5_live['open'] > prev2_close:
            pattern_engulf = -1.0
        else:
            pattern_engulf = 0.0

        # 打包成特徵陣列 (共 37 維度)
        features = pd.DataFrame([{

            'm5_ema_slope': m5_ema_slope, 'm5_rsi_14': m5_rsi, 'm5_atr_14': m5_atr,
            'body_size': body_size, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow, 'body_ratio': body_ratio,
            'hour': hour, 'day_of_week': day_of_week,
            'm15_dist_h': m15_dist_h, 'm15_dist_l': m15_dist_l, 'm15_dist_m': m15_dist_m, 'm15_ema_slope': m15_ema_slope,
            'm5_dist_h': m5_dist_h, 'm5_dist_l': m5_dist_l, 'm5_dist_m': m5_dist_m, 'm5_dist_ema': m5_dist_ema,
            'm5_adx': m5_adx, 'm5_macd_hist': m5_macd_hist, 'm5_cci': m5_cci, 'm5_bb_width': m5_bb_width,
            'prev_body_size': prev_body_size, 'prev_close_change': prev_close_change,
            'is_us_session': is_us_session, 'is_asia_session': is_asia_session, 'h1_trend': h1_trend, 'h4_trend': h4_trend, 'd1_rsi': d1_rsi,
            'm30_rsi': m30_rsi, 'm30_trend': m30_trend, 'm1_momentum': m1_momentum,
            # 新增高品質特徵 (+6)
            'volume_ratio': volume_ratio, 'spread': spread_val, 'session_overlap': session_overlap,
            'rsi_divergence': rsi_divergence, 'price_vs_vwap': price_vs_vwap, 'pattern_engulf': pattern_engulf,
        }])
        
        # --- 多專家集成預測 (XGB + LGB + RF + Stacking) ---
        # 1. 取得各專家的初步勝率
        p_xgb_b = self.xgb_buy.predict_proba(features)[0][1]
        p_lgb_b = self.lgb_buy.predict_proba(features)[0][1]
        p_rf_b  = self.rf_buy.predict_proba(features)[0][1]
        
        p_xgb_s = self.xgb_sell.predict_proba(features)[0][1]
        p_lgb_s = self.lgb_sell.predict_proba(features)[0][1]
        p_rf_s  = self.rf_sell.predict_proba(features)[0][1]
        
        # 2. 由 Stacker (主審) 進行最終裁決
        meta_b = np.column_stack([[p_xgb_b, p_lgb_b, p_rf_b]])
        meta_s = np.column_stack([[p_xgb_s, p_lgb_s, p_rf_s]])
        
        final_buy_prob = self.stack_buy.predict_proba(meta_b)[0][1]
        final_sell_prob = self.stack_sell.predict_proba(meta_s)[0][1]
        
        # 3. 分配給 Model A 與 Model B
        # 現在兩者都使用最強的集成勝率
        model_A_buy_score = final_buy_prob
        model_A_sell_score = final_sell_prob
        
        model_B_buy_score = final_buy_prob
        model_B_sell_score = final_sell_prob
        
        # 儲存供 UI 讀取
        self.model_a_buy = model_A_buy_score
        self.model_a_sell = model_A_sell_score
        self.model_b_buy = model_B_buy_score
        self.model_b_sell = model_B_sell_score
        
        # 更新 UI 狀態
        self.status = "市場掃描中"
        
        if not can_execute_new_trades: return
        
        positions = mt5.positions_get(symbol=self.symbol)
        pos_A_count = len([p for p in positions if p.magic == self.magic_number_A]) if positions else 0
        pos_B_count = len([p for p in positions if p.magic == self.magic_number_B]) if positions else 0

        # === 執行模型 A (利潤極大化) ===
        # 配置: Threshold=0.60, TP=10.0, SL=8.0
        if pos_A_count == 0 and self.last_signal_time_A != m5_live['time']:
            if model_A_buy_score >= 0.60 and model_A_buy_score > model_A_sell_score:
                sl, tp = ask - 8.0, ask + 10.0
                res = self.executor.send_order(self.symbol, mt5.ORDER_TYPE_BUY, self.lot_size_A, ask, sl, tp, comment="Model_A_XGB", magic=self.magic_number_A)
                if res is not None:
                    logger.info(f"🚀 [模型 A - 利潤引擎] 多單進場 @ {ask:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | 信心: {model_A_buy_score*100:.1f}% | Ticket: {res.order}")
                    if self.trade_tracker:
                        self.trade_tracker.log_open_trade(res.order, "Model_A_XGB", "BUY", self.lot_size_A, ask, sl, tp)
                self.last_signal_time_A = m5_live['time']
                
            elif model_A_sell_score >= 0.60 and model_A_sell_score > model_A_buy_score:
                sl, tp = bid + 8.0, bid - 10.0
                res = self.executor.send_order(self.symbol, mt5.ORDER_TYPE_SELL, self.lot_size_A, bid, sl, tp, comment="Model_A_XGB", magic=self.magic_number_A)
                if res is not None:
                    logger.info(f"🚀 [模型 A - 利潤引擎] 空單進場 @ {bid:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | 信心: {model_A_sell_score*100:.1f}% | Ticket: {res.order}")
                    if self.trade_tracker:
                        self.trade_tracker.log_open_trade(res.order, "Model_A_XGB", "SELL", self.lot_size_A, bid, sl, tp)
                self.last_signal_time_A = m5_live['time']

        # === 執行模型 B (勝率極大化) ===
        # 配置: Threshold=0.65, TP=5.0, SL=8.0
        if pos_B_count == 0 and self.last_signal_time_B != m5_live['time']:
            if model_B_buy_score >= 0.65 and model_B_buy_score > model_B_sell_score:
                sl, tp = ask - 8.0, ask + 5.0
                res = self.executor.send_order(self.symbol, mt5.ORDER_TYPE_BUY, self.lot_size_B, ask, sl, tp, comment="Model_B_Mix", magic=self.magic_number_B)
                if res is not None:
                    logger.info(f"🎯 [模型 B - 勝率引擎] 多單進場 @ {ask:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | 信心: {model_B_buy_score*100:.1f}% | Ticket: {res.order}")
                    if self.trade_tracker:
                        self.trade_tracker.log_open_trade(res.order, "Model_B_Mix", "BUY", self.lot_size_B, ask, sl, tp)
                self.last_signal_time_B = m5_live['time']
                
            elif model_B_sell_score >= 0.65 and model_B_sell_score > model_B_buy_score:
                sl, tp = bid + 8.0, bid - 5.0
                res = self.executor.send_order(self.symbol, mt5.ORDER_TYPE_SELL, self.lot_size_B, bid, sl, tp, comment="Model_B_Mix", magic=self.magic_number_B)
                if res is not None:
                    logger.info(f"🎯 [模型 B - 勝率引擎] 空單進場 @ {bid:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | 信心: {model_B_sell_score*100:.1f}% | Ticket: {res.order}")
                    if self.trade_tracker:
                        self.trade_tracker.log_open_trade(res.order, "Model_B_Mix", "SELL", self.lot_size_B, bid, sl, tp)
                self.last_signal_time_B = m5_live['time']
        
        # 每 10 個 tick 更新一次 CSV 紀錄
        self.tick_counter += 1
        if self.tick_counter % 10 == 0 and self.trade_tracker:
            self.trade_tracker.update_closed_trades(self.connector)

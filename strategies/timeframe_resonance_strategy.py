import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TimeframeResonanceStrategy(BaseStrategy):
    def __init__(self, name="ResonanceStrategy", symbol="XAUUSD", lot_size=0.1):
        super().__init__(name)
        self.symbol = symbol
        self.lot_size = 1.0     # Fixed 1 lot
        self.max_positions = 1  # Fixed 1 position limit
        self.indicators = {}    # 用於向 GUI 顯示的指標數據快取
        self.status = "等待數據..."
        self._cached_dfs = (None, None, None)


    def calculate_indicators(self, df):
        # 在沙盒模擬中，我們需要即時更新軌道以產生「跳動感」
        # 即使預算列已存在，重新計算當前切片（如 100 筆）的效能影響極低
        pass

            
        # 1. Bollinger Bands (校準為 21, 2.1)
        bb = BollingerBands(close=df['close'], window=21, window_dev=2.1)
        df['bb_h'] = bb.bollinger_hband()
        df['bb_m'] = bb.bollinger_mavg()
        df['bb_l'] = bb.bollinger_lband()
        
        # 2. 12 EMA
        ema = EMAIndicator(close=df['close'], window=12)
        df['ema_12'] = ema.ema_indicator()
        return df

    def get_data(self):
        # 效能優化：如果模擬時間與價格都沒變，才回傳快取
        current_sim_time = self.connector.current_time
        current_price = self.connector.last_price
        
        cache_key = (current_sim_time, current_price)
        if hasattr(self, '_last_cache_key') and self._last_cache_key == cache_key and self._cached_dfs[0] is not None:
            return self._cached_dfs

        h1_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_H1, 60)
        m30_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M30, 60)
        m5_df = self.connector.get_rates(self.symbol, mt5.TIMEFRAME_M5, 150)
        
        if h1_df is None or m30_df is None or m5_df is None or h1_df.empty or m30_df.empty or m5_df.empty:
            return None, None, None
            
        h1_df = self.calculate_indicators(h1_df)
        m30_df = self.calculate_indicators(m30_df)
        m5_df = self.calculate_indicators(m5_df)
        
        # 更新快取
        self._last_cache_key = cache_key
        self._cached_dfs = (h1_df, m30_df, m5_df)
        return self._cached_dfs

    def get_current_price(self):
        info = self.connector.get_symbol_info(self.symbol)
        if info:
            return info.get('ask', 0.0), info.get('bid', 0.0)
        return 0.0, 0.0

    def on_tick(self, can_execute_new_trades: bool):
        if not can_execute_new_trades:
            self.status = "全域風控停機中"
            return
            
        ask, bid = self.get_current_price()
        if ask == 0 or bid == 0:
            self.status = "取得即時行情報價失敗"
            return

        h1_df, m30_df, m5_df = self.get_data()
        if h1_df is None or m30_df is None or m5_df is None:
            return

        # --- 數據監控更新 (不論是否進場，都要讓 GUI 看到最新數據) ---
        m30_curr = m30_df.iloc[-1]
        h1_curr = h1_df.iloc[-1]
        m5_live = m5_df.iloc[-1]
        
        self.indicators = {
            "H1": {"H": h1_curr['bb_h'], "M": h1_curr['bb_m'], "L": h1_curr['bb_l'], "EMA": h1_curr['ema_12']},
            "M30": {"H": m30_curr['bb_h'], "M": m30_curr['bb_m'], "L": m30_curr['bb_l'], "EMA": m30_curr['ema_12']},
            "M5": {"H": m5_live['bb_h'], "M": m5_live['bb_m'], "L": m5_live['bb_l'], "EMA": m5_live['ema_12']}
        }

        m5_signal_time = m5_df.iloc[-2]['time']
        if self.last_signal_time == m5_signal_time:
            self.status = "✅ 該 K 線已完成任務，冷卻中..."
            return

        # Check position limit by magic number
        pos_count = self.connector.get_positions_count_by_magic(self.magic_number)
        if pos_count > 0:
            self.status = f"已有持倉，等待出場 (持倉數: {pos_count})"
            return
            
        # 嚴格 SOP: 收整點K才進場。判定最後一根已完整收盤的 K 線 (iloc[-2])
        m5_closed = m5_df.iloc[-2]
        m5_prev = m5_df.iloc[-3]
        
        price = m5_closed['close']
        
        # M5 狙擊扳機：必須極度貼近軌道 (容赦 1.5 USD)
        threshold = 1.5 
        
        # === 尋找多單共振 (Long Resonance) ===
        # H1：當前 H1 必須是跌向支撐的紅K (開盤>當前價) 
        h1_is_red = h1_curr['open'] > price
        h1_bull_mid = max(h1_curr['ema_12'], h1_curr['bb_m'])
        h1_bull_low = h1_curr['bb_l']
        
        # H1 層級較大，給予 3 塊錢的緩衝區 (300點)
        h1_threshold = 3.0
        h1_touch_mid = abs(price - h1_bull_mid) < h1_threshold
        h1_touch_low = (price < h1_bull_low + h1_threshold) 
        
        h1_bull_ok = h1_is_red and (h1_touch_mid or h1_touch_low)
        
        # M30 中軌或下軌支撐 (解除紅綠K限制，僅看位置)
        m30_bull_mid = m30_curr['bb_m']
        m30_bull_low = m30_curr['bb_l']
        
        # M30 給予 2 塊錢的緩衝區 (200點)
        m30_threshold = 2.0
        m30_touch_mid = abs(price - m30_bull_mid) < m30_threshold
        m30_touch_low = (price < m30_bull_low + m30_threshold)
        
        m30_bull_ok = (m30_touch_mid or m30_touch_low)
        
        # M5 下軌測試 (檢查上一根收盤K線的最低點是否碰觸下軌)
        m5_bull_target = m5_closed['bb_l']
        m5_bull_ok = (m5_closed['low'] < m5_bull_target + threshold)
        
        # 型態 1：多頭吞沒 (前紅後綠，實體包覆)
        is_prev_bear = m5_prev['close'] < m5_prev['open']
        is_curr_bull = m5_closed['close'] > m5_closed['open']
        is_bull_engulfing = is_curr_bull and is_prev_bear and (m5_closed['open'] <= m5_prev['close']) and (m5_closed['close'] >= m5_prev['open'])
        
        # 型態 2：長下影線 (下影線 > 實體兩倍)
        body = abs(m5_closed['close'] - m5_closed['open'])
        lower_wick = min(m5_closed['close'], m5_closed['open']) - m5_closed['low']
        has_lower_wick = is_curr_bull and (lower_wick > body * 2) and body > 0
        
        if h1_bull_ok and m30_bull_ok and m5_bull_ok and (is_bull_engulfing or has_lower_wick):
            self.status = "多單共振達成！準備進場"
            
            # 使用最新的 ask 價格進場作多
            sl = ask - 5.0 # Fixed 5 USD SL
            tp = ask + 5.0 # Fixed 5 USD TP
            
            logger.info(f"[{self.name}] 完美多頭共振觸發！進場價(Ask): {ask}, 停損: {sl}, 停利: {tp}")
            self.executor.send_order(self.symbol, mt5.ORDER_TYPE_BUY, self.lot_size, ask, sl, tp, comment=self.name, magic=self.magic_number)
            self.last_signal_time = m5_signal_time # 鎖定這個時間戳
            return

        # === 尋找空單共振 (Short Resonance) ===
        # H1：當前 H1 必須是漲向壓力的綠K (開盤<當前價)
        h1_is_green = h1_curr['open'] < price
        h1_bear_mid = min(h1_curr['ema_12'], h1_curr['bb_m'])
        h1_bear_high = h1_curr['bb_h']
        
        h1_touch_mid_bear = abs(price - h1_bear_mid) < h1_threshold
        h1_touch_high = (price > h1_bear_high - h1_threshold) 
        
        h1_bear_ok = h1_is_green and (h1_touch_mid_bear or h1_touch_high)
        
        # M30 中軌或上軌壓力 (解除紅綠K限制，僅看位置)
        m30_bear_mid = m30_curr['bb_m']
        m30_bear_high = m30_curr['bb_h']
        
        m30_touch_mid_bear = abs(price - m30_bear_mid) < m30_threshold
        m30_touch_high = (price > m30_bear_high - m30_threshold)
        
        m30_bear_ok = (m30_touch_mid_bear or m30_touch_high)
        
        # M5 上軌測試 (檢查上一根收盤K線的最高點是否碰觸上軌)
        m5_bear_target = m5_closed['bb_h']
        m5_bear_ok = (m5_closed['high'] > m5_bear_target - threshold)
        
        # 型態 1：空頭吞沒 (前綠後紅，實體包覆)
        is_prev_bull = m5_prev['close'] > m5_prev['open']
        is_curr_bear = m5_closed['close'] < m5_closed['open']
        is_bear_engulfing = is_curr_bear and is_prev_bull and (m5_closed['open'] >= m5_prev['close']) and (m5_closed['close'] <= m5_prev['open'])
        
        # 型態 2：長上影線 (上影線 > 實體兩倍)
        upper_wick = m5_closed['high'] - max(m5_closed['close'], m5_closed['open'])
        body_bear = abs(m5_closed['close'] - m5_closed['open'])
        has_upper_wick = is_curr_bear and (upper_wick > body_bear * 2) and body_bear > 0
        
        if h1_bear_ok and m30_bear_ok and m5_bear_ok and (is_bear_engulfing or has_upper_wick):
            self.status = "空單共振達成！準備進場"
            
            # 使用最新的 bid 價格進場作空
            sl = bid + 5.0 # Fixed 5 USD SL
            tp = bid - 5.0 # Fixed 5 USD TP 
            
            logger.info(f"[{self.name}] 完美空頭共振觸發！進場價(Bid): {bid}, 停損: {sl}, 停利: {tp}")
            self.executor.send_order(self.symbol, mt5.ORDER_TYPE_SELL, self.lot_size, bid, sl, tp, comment=self.name, magic=self.magic_number)
            self.last_signal_time = m5_signal_time # 鎖定這個時間戳
            return

        # --- 產生詳細狀態文字 (階層式顯示) ---
        # 多頭狀態解析
        h1_dist_mid = abs(price - h1_bull_mid)
        h1_dist_low = abs(price - h1_bull_low)
        h1_msg = f"中:{h1_dist_mid:.1f} 下:{h1_dist_low:.1f}"
        
        if h1_bull_ok:
            long_status = "多頭[✅H1]"
            m30_dist_mid = abs(price - m30_bull_mid)
            m30_dist_low = abs(price - m30_bull_low)
            m30_msg = f"中:{m30_dist_mid:.1f} 下:{m30_dist_low:.1f}"
            
            if m30_bull_ok:
                long_status += " [✅M30]"
                if not m5_bull_ok:
                    d_m5 = abs(m5_closed['low'] - m5_bull_target)
                    long_status += f" [❌M5 下軌差 {d_m5:.1f}]"
                else:
                    if is_bull_engulfing: long_status += " [✅M5吞沒]"
                    elif has_lower_wick: long_status += " [✅M5長下影]"
                    else: long_status += " [❌M5欠缺型態]"
            else:
                long_status += f" [❌M30 {m30_msg}]"
        else:
            reason = "需為紅K" if not h1_is_red else h1_msg
            long_status = f"多頭[❌H1 {reason}]"

        # 空頭狀態解析
        h1_dist_b_mid = abs(price - h1_bear_mid)
        h1_dist_b_high = abs(price - h1_bear_high)
        h1_bear_msg = f"中:{h1_dist_b_mid:.1f} 上:{h1_dist_b_high:.1f}"
        
        if h1_bear_ok:
            short_status = "空頭[✅H1]"
            m30_dist_b_mid = abs(price - m30_bear_mid)
            m30_dist_b_high = abs(price - m30_bear_high)
            m30_bear_msg = f"中:{m30_dist_b_mid:.1f} 上:{m30_dist_b_high:.1f}"
            
            if m30_bear_ok:
                short_status += " [✅M30]"
                if not m5_bear_ok:
                    d_m5_b = abs(m5_bear_target - m5_closed['high'])
                    short_status += f" [❌M5 上軌差 {d_m5_b:.1f}]"
                else:
                    if is_bear_engulfing: short_status += " [✅M5吞沒]"
                    elif has_upper_wick: short_status += " [✅M5長上影]"
                    else: short_status += " [❌M5欠缺型態]"
            else:
                short_status += f" [❌M30 {m30_bear_msg}]"
        else:
            reason = "需為綠K" if not h1_is_green else h1_bear_msg
            short_status = f"空頭[❌H1 {reason}]"

        # 決定哪個方向「眼下最可能發生」
        if h1_is_red:
            self.status = f"{long_status}"
        else:
            self.status = f"{short_status}"

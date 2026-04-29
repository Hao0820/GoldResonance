import pandas as pd
import numpy as np
import joblib
import warnings
import os
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def run_m1_backtest():
    logging.info("--- 啟動集成大腦 M1 級別高精度回測 ---")
    
    # 載入資料與模型
    try:
        m1_df = pd.read_csv("history_data/XAUUSD_M1.csv")
        # 直接使用剛才產生的資料集
        m5_df = pd.read_csv("ml_dataset.csv") 
        
        # 載入所有集成組件
        models = {
            'xgb_b': joblib.load('models/xgb_buy.pkl'),
            'lgb_b': joblib.load('models/lgb_buy.pkl'),
            'rf_b':  joblib.load('models/rf_buy.pkl'),
            'stack_b': joblib.load('models/stack_buy.pkl'),
            'xgb_s': joblib.load('models/xgb_sell.pkl'),
            'lgb_s': joblib.load('models/lgb_sell.pkl'),
            'rf_s':  joblib.load('models/rf_sell.pkl'),
            'stack_s': joblib.load('models/stack_sell.pkl'),
        }
    except Exception as e:
        logging.error(f"資料或模型載入失敗: {e}")
        return

    m1_df['time'] = pd.to_datetime(m1_df['time'])
    m5_df['time'] = pd.to_datetime(m5_df['time'])
    m1_df = m1_df.sort_values('time')
    
    # 與訓練一致的 37 個特徵
    features = [
        'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 
        'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'hour', 'day_of_week',
        'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema',
        'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
        'prev_body_size', 'prev_close_change',
        'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
        'm30_rsi', 'm30_trend', 'm1_momentum',
        'volume_ratio', 'spread', 'session_overlap',
        'rsi_divergence', 'price_vs_vwap', 'pattern_engulf'
    ]
    
    X = m5_df[features]
    
    # 取得集成預測機率
    logging.info("計算集成專家預測值...")
    p_xgb_b = models['xgb_b'].predict_proba(X)[:, 1]
    p_lgb_b = models['lgb_b'].predict_proba(X)[:, 1]
    p_rf_b  = models['rf_b'].predict_proba(X)[:, 1]
    meta_b  = np.column_stack([p_xgb_b, p_lgb_b, p_rf_b])
    final_buy_probs = models['stack_b'].predict_proba(meta_b)[:, 1]
    
    p_xgb_s = models['xgb_s'].predict_proba(X)[:, 1]
    p_lgb_s = models['lgb_s'].predict_proba(X)[:, 1]
    p_rf_s  = models['rf_s'].predict_proba(X)[:, 1]
    meta_s  = np.column_stack([p_xgb_s, p_lgb_s, p_rf_s])
    final_sell_probs = models['stack_s'].predict_proba(meta_s)[:, 1]

    def simulate(tp, sl, threshold=0.75):
        trades, wins, losses = 0, 0, 0
        balance = 1000.0
        in_trade_until = pd.Timestamp.min
        
        for i in range(len(m5_df)):
            current_time = m5_df['time'].iloc[i]
            if current_time < in_trade_until: continue
            
            buy_sig = final_buy_probs[i] > threshold
            sell_sig = final_sell_probs[i] > threshold
            
            if buy_sig or sell_sig:
                entry_price = m5_df['close'].iloc[i]
                target_tp = entry_price + tp if buy_sig else entry_price - tp
                target_sl = entry_price - sl if buy_sig else entry_price + sl
                
                # 在 M1 資料找結果 (最多找 480 根 = 8 小時)
                future_m1 = m1_df[m1_df['time'] > current_time].head(480)
                
                outcome = None
                for _, m1 in future_m1.iterrows():
                    if buy_sig:
                        if m1['low'] <= target_sl: 
                            outcome = 'loss'; break
                        if m1['high'] >= target_tp: 
                            outcome = 'win'; break
                    else:
                        if m1['high'] >= target_sl: 
                            outcome = 'loss'; break
                        if m1['low'] <= target_tp: 
                            outcome = 'win'; break
                
                if outcome:
                    trades += 1
                    if outcome == 'win':
                        wins += 1; balance += (tp * 10) # 簡化計算
                    else:
                        losses += 1; balance -= (sl * 10)
                    in_trade_until = m1['time'] # 到這單結束前不進場
        
        wr = (wins/trades*100) if trades>0 else 0
        return trades, wins, losses, wr, balance

    # 執行三種方案
    configs = [
        {"name": "Sniper (1:2 RR)", "tp": 10.0, "sl": 5.0,  "threshold": 0.80},
        {"name": "Balanced (1:1.4 RR)", "tp": 7.0,  "sl": 5.0,  "threshold": 0.75},
        {"name": "HighWinRate (1:0.6 RR)", "tp": 5.0,  "sl": 8.0,  "threshold": 0.70}
    ]
    
    print("\n" + "="*70)
    print(f"{'Strategy Name':<25} | {'Count':<5} | {'W/L':<10} | {'WinRate':<8} | {'Profit':<10}")
    print("-"*70)
    
    for cfg in configs:
        t, w, l, wr, bal = simulate(cfg['tp'], cfg['sl'], cfg['threshold'])
        print(f"{cfg['name']:<23} | {t:<5} | {w}/{l:<7} | {wr:>5.1f}% | ${bal-1000:>8.2f}")
    
    print("="*70)
    print("註：預估獲利僅供參考，未包含點差與滑價。回測時間範圍取決於 M1 CSV 長度。")

if __name__ == "__main__":
    run_m1_backtest()

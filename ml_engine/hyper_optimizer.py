import pandas as pd
import numpy as np
import joblib
import itertools
import time

def run_grid_search():
    print("Start Level 5 AI Grand Simulation (Grid Search)...")
    
    # 1. 載入資料
    try:
        m5_df = pd.read_csv("ml_dataset.csv")
        m5_df['time'] = pd.to_datetime(m5_df['time'])
        m1_df = pd.read_csv("history_data/XAUUSD_M1.csv")
        m1_df['time'] = pd.to_datetime(m1_df['time'])
    except Exception as e:
        print(f"資料載入失敗: {e}")
        return

    features = [
        'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 
        'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'hour', 'day_of_week',
        'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema',
        'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
        'prev_body_size', 'prev_close_change',
        'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
        'm30_rsi', 'm30_trend', 'm1_momentum'
    ]
    X = m5_df[features]
    
    # 2. 載入模型並預先計算所有機率
    print("Loading models and pre-calculating probabilities...")
    xgb_buy = joblib.load('models/xgb_buy.pkl')
    xgb_sell = joblib.load('models/xgb_sell.pkl')
    rf_buy = joblib.load('models/rf_buy.pkl')
    rf_sell = joblib.load('models/rf_sell.pkl')
    
    xgb_buy_probs = xgb_buy.predict_proba(X)[:, 1]
    rf_buy_probs = rf_buy.predict_proba(X)[:, 1]
    xgb_sell_probs = xgb_sell.predict_proba(X)[:, 1]
    rf_sell_probs = rf_sell.predict_proba(X)[:, 1]

    # 3. 準備 M1 Numpy 向量化陣列 (加速 100 倍)
    print("Converting to Numpy arrays for acceleration...")
    m1_times = m1_df['time'].values
    m1_highs = m1_df['high'].values
    m1_lows = m1_df['low'].values
    m5_times = m5_df['time'].values
    m5_closes = m5_df['close'].values

    # 預先計算每個 M5 K棒在 M1 陣列中的起始點
    start_indices = np.searchsorted(m1_times, m5_times, side='right')
    
    # 4. 定義參數網格 (Deep Simulation)
    tp_grid = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    sl_grid = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    threshold_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
    
    # 權重組合 (100/0 到 0/100)
    mix_grid = [
        (1.0, 0.0, "XGB 100%"), (0.9, 0.1, "XGB 90%"), (0.8, 0.2, "XGB 80%"), 
        (0.7, 0.3, "XGB 70%"), (0.6, 0.4, "XGB 60%"), (0.5, 0.5, "Mix 50%"), 
        (0.4, 0.6, "RF 60%"), (0.3, 0.7, "RF 70%"), (0.2, 0.8, "RF 80%"), 
        (0.1, 0.9, "RF 90%"), (0.0, 1.0, "RF 100%")
    ]
    
    combinations = list(itertools.product(tp_grid, sl_grid, threshold_grid, mix_grid))
    print(f"Total {len(combinations)} combinations, starting hyper optimization...\n")
    
    results = []
    
    start_sim_time = time.time()
    
    for tp, sl, threshold, (w_xgb, w_rf, mix_name) in combinations:
        buy_probs = (xgb_buy_probs * w_xgb) + (rf_buy_probs * w_rf)
        sell_probs = (xgb_sell_probs * w_xgb) + (rf_sell_probs * w_rf)
        
        balance = 1000.0
        trades, wins, losses = 0, 0, 0
        in_trade_until = np.datetime64('1970-01-01')
        
        for i in range(len(m5_times)):
            current_time = m5_times[i]
            if current_time < in_trade_until:
                continue
                
            bp, sp = buy_probs[i], sell_probs[i]
            is_buy = bp >= threshold and bp > sp
            is_sell = sp >= threshold and sp > bp
            
            if not is_buy and not is_sell:
                continue
                
            entry_price = m5_closes[i]
            start_idx = start_indices[i]
            end_idx = min(start_idx + 300, len(m1_times)) # 找未來5小時
            
            high_slice = m1_highs[start_idx:end_idx]
            low_slice = m1_lows[start_idx:end_idx]
            
            if is_buy:
                target_tp = entry_price + tp
                target_sl = entry_price - sl
                tp_hits = np.where(high_slice >= target_tp)[0]
                sl_hits = np.where(low_slice <= target_sl)[0]
            else:
                target_tp = entry_price - tp
                target_sl = entry_price + sl
                tp_hits = np.where(low_slice <= target_tp)[0]
                sl_hits = np.where(high_slice >= target_sl)[0]
                
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 9999
            first_sl = sl_hits[0] if len(sl_hits) > 0 else 9999
            
            if first_tp == 9999 and first_sl == 9999:
                continue
                
            trades += 1
            # 獲利金額為 點數*10 (假設 0.1手)
            win_amount = tp * 10
            loss_amount = sl * 10
            
            if first_tp < first_sl:
                wins += 1
                balance += win_amount
                in_trade_until = m1_times[start_idx + first_tp]
            elif first_sl < first_tp:
                losses += 1
                balance -= loss_amount
                in_trade_until = m1_times[start_idx + first_sl]
            else:
                # 同一分鐘內碰到，算最壞情況 (停損)
                losses += 1
                balance -= loss_amount
                in_trade_until = m1_times[start_idx + first_sl]
                
        win_rate = (wins / trades * 100) if trades > 0 else 0
        profit = balance - 1000.0
        
        results.append({
            'Mix': mix_name,
            'Threshold': threshold,
            'TP': tp,
            'SL': sl,
            'Trades': trades,
            'WinRate': round(win_rate, 1),
            'Profit': round(profit, 1)
        })

    # 排序並輸出
    res_df = pd.DataFrame(results)
    
    # 過濾交易次數大於 50 次的才有意義
    res_df = res_df[res_df['Trades'] >= 50]
    
    # 策略A：利潤最大化
    top_profit = res_df.sort_values(by='Profit', ascending=False).head(5)
    # 策略B：勝率最高 (且利潤 > 0)
    top_wr = res_df[res_df['Profit'] > 0].sort_values(by='WinRate', ascending=False).head(5)
    
    print("="*60)
    print("[Strategy A] Maximize Profit (Top 5)")
    print(top_profit.to_string(index=False))
    
    print("\n" + "="*60)
    print("[Strategy B] Maximize Win Rate (Top 5)")
    print(top_wr.to_string(index=False))
    print("="*60)
    
    elapsed = time.time() - start_sim_time
    print(f"\nSimulation Complete! Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_grid_search()

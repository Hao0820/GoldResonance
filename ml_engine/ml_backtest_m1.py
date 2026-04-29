import pandas as pd
import joblib
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def run_m1_backtest():
    print("--- 啟動 M1 級別高精度回測 (3個月) ---")
    
    # 載入資料
    try:
        m1_df = pd.read_csv("history_data/XAUUSD_M1.csv")
        m5_df = pd.read_csv("ml_dataset.csv") # 這是我們剛才產生的 3個月特徵資料集
        
        xgb_buy = joblib.load('models/xgb_buy.pkl')
        xgb_sell = joblib.load('models/xgb_sell.pkl')
        rf_buy = joblib.load('models/rf_buy.pkl')
        rf_sell = joblib.load('models/rf_sell.pkl')
    except Exception as e:
        print(f"資料載入失敗: {e}")
        return

    m1_df['time'] = pd.to_datetime(m1_df['time'])
    m5_df['time'] = pd.to_datetime(m5_df['time'])
    
    # 確保 M1 數據按時間排序
    m1_df = m1_df.sort_values('time')
    
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
    
    # 預測機率
    xgb_buy_probs = xgb_buy.predict_proba(X)[:, 1]
    rf_buy_probs = rf_buy.predict_proba(X)[:, 1]
    xgb_sell_probs = xgb_sell.predict_proba(X)[:, 1]
    rf_sell_probs = rf_sell.predict_proba(X)[:, 1]
    
    # 7:3 綜合比例
    ensemble_buy_probs = (xgb_buy_probs * 0.7) + (rf_buy_probs * 0.3)
    ensemble_sell_probs = (xgb_sell_probs * 0.7) + (rf_sell_probs * 0.3)
    
    threshold = 0.65  # 回到平衡模式 (兼顧勝率與次數)
    tp_dist = 7.0
    sl_dist = 4.0
    
    def simulate_precision(buy_probs, sell_probs, name):
        balance = 1000.0
        trades = 0
        wins = 0
        losses = 0
        
        in_trade_until = pd.Timestamp.min
        
        # 為了加速運算，將 M1 轉為字典索引
        m1_data = m1_df.set_index('time')
        
        for i in range(len(m5_df)):
            current_time = m5_df['time'].iloc[i]
            
            if current_time < in_trade_until:
                continue
            
            is_buy = buy_probs[i] > threshold and buy_probs[i] > sell_probs[i]
            is_sell = sell_probs[i] > threshold and sell_probs[i] > buy_probs[i]
            
            if is_buy or is_sell:
                entry_price = m5_df['close'].iloc[i]
                target_tp = entry_price + tp_dist if is_buy else entry_price - tp_dist
                target_sl = entry_price - sl_dist if is_buy else entry_price + sl_dist
                
                # 從當前時間點開始，在 M1 數據中找結果
                # 我們往後找最多 300 根 M1 K棒 (5小時)
                future_m1 = m1_df[m1_df['time'] > current_time].head(300)
                
                outcome = None
                exit_time = current_time
                
                for _, m1_row in future_m1.iterrows():
                    if is_buy:
                        if m1_row['low'] <= target_sl:
                            outcome = 'loss'
                            exit_time = m1_row['time']
                            break
                        if m1_row['high'] >= target_tp:
                            outcome = 'win'
                            exit_time = m1_row['time']
                            break
                    else: # Sell
                        if m1_row['high'] >= target_sl:
                            outcome = 'loss'
                            exit_time = m1_row['time']
                            break
                        if m1_row['low'] <= target_tp:
                            outcome = 'win'
                            exit_time = m1_row['time']
                            break
                
                if outcome:
                    trades += 1
                    if outcome == 'win':
                        wins += 1
                        balance += 70.0
                    else:
                        losses += 1
                        balance -= 40.0
                    in_trade_until = exit_time # 冷卻直到這單結束
                
        win_rate = (wins/trades*100) if trades>0 else 0
        print(f"\n[{name}] - M1 級別高精度回測")
        print(f"初始資金: $1000.00")
        print(f"總交易次數: {trades} 次 (勝: {wins}, 敗: {losses})")
        print(f"精確勝率: {win_rate:.1f}%")
        print(f"淨利潤(三個月): ${balance - 1000:.2f}")

    print("="*55)
    simulate_precision(xgb_buy_probs, xgb_sell_probs, "方案 A: 100% XGBoost")
    simulate_precision(ensemble_buy_probs, ensemble_sell_probs, "方案 B: 7:3 綜合比例")
    print("="*55)

if __name__ == "__main__":
    run_m1_backtest()

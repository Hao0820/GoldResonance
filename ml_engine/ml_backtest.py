import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_backtest():
    print("載入歷史數據與模型...")
    try:
        df = pd.read_csv("ml_dataset.csv")
    except FileNotFoundError:
        print("找不到 ml_dataset.csv，請確認資料存在。")
        return
        
    xgb_buy = joblib.load('models/xgb_buy.pkl')
    xgb_sell = joblib.load('models/xgb_sell.pkl')
    rf_buy = joblib.load('models/rf_buy.pkl')
    rf_sell = joblib.load('models/rf_sell.pkl')
    
    features = [
        'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 
        'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
        'hour', 'day_of_week',
        'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema'
    ]
    
    X = df[features]
    y_buy = df['label_buy']
    y_sell = df['label_sell']
    
    print("計算預測機率...")
    # XGBoost 和 RF 都將預測機率回傳為陣列
    xgb_buy_probs = xgb_buy.predict_proba(X)[:, 1]
    rf_buy_probs = rf_buy.predict_proba(X)[:, 1]
    xgb_sell_probs = xgb_sell.predict_proba(X)[:, 1]
    rf_sell_probs = rf_sell.predict_proba(X)[:, 1]
    
    ensemble_buy_probs = (xgb_buy_probs + rf_buy_probs) / 2
    ensemble_sell_probs = (xgb_sell_probs + rf_sell_probs) / 2
    
    threshold = 0.65
    tp_profit = 70.0 # $70 for 0.1 lot (7 points)
    sl_loss = -40.0 # -$40 for 0.1 lot (4 points)
    
    def simulate(buy_probs, sell_probs, name):
        balance = 1000.0
        trades = 0
        wins = 0
        losses = 0
        
        # 簡單的進場冷卻機制 (避免在同一波段連續重複進場)
        # 假設持倉時間約為 12 根 K 棒 (1小時)
        in_trade_until = 0
        
        for i in range(len(df)):
            if i < in_trade_until:
                continue
                
            is_buy = buy_probs[i] > threshold and buy_probs[i] > sell_probs[i]
            is_sell = sell_probs[i] > threshold and sell_probs[i] > buy_probs[i]
            
            if is_buy:
                trades += 1
                if y_buy.iloc[i] == 1:
                    balance += tp_profit
                    wins += 1
                else:
                    balance += sl_loss
                    losses += 1
                in_trade_until = i + 12
                
            elif is_sell:
                trades += 1
                if y_sell.iloc[i] == 1:
                    balance += tp_profit
                    wins += 1
                else:
                    balance += sl_loss
                    losses += 1
                in_trade_until = i + 12
                
        win_rate = (wins/trades*100) if trades>0 else 0
        print(f"\n[{name}] - 過去 6 個月回測 (含訓練集)")
        print(f"初始資金: $1000.00")
        print(f"總交易次數: {trades} 次 (勝: {wins}, 敗: {losses})")
        print(f"實盤勝率: {win_rate:.1f}%")
        print(f"淨利潤(半年): ${balance - 1000:.2f}")
        print(f"預估年化淨利: ${(balance - 1000) * 2:.2f}")

    print("\n" + "="*55)
    print("--- AI Strategy Comparison Test ($1000, 0.1 lot) ---")
    print("="*55)
    
    # 測試 1: 100% XGBoost
    simulate(xgb_buy_probs, xgb_sell_probs, "方案 A: 100% XGBoost (激進全開)")
    
    # 測試 2: 70% XGB / 30% RF (目前的設定)
    simulate(ensemble_buy_probs, ensemble_sell_probs, "方案 B: 7:3 綜合大腦 (穩健平衡)")
    
    print("="*55 + "\n")

if __name__ == "__main__":
    run_backtest()

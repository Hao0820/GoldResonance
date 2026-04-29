import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import logging
import os
import warnings

# 隱藏訓練過程中的並行運算警告
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def train_and_evaluate(X_train, X_test, y_train, y_test, side="Buy"):
    logging.info(f"開始訓練 {side} 模型...")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    
    logging.info(f"--- {side} 模型評估結果 ---")
    logging.info(f"[XGBoost] 準確率(Accuracy): {xgb_acc*100:.1f}%, 精確率(Precision): {xgb_prec*100:.1f}%")
    logging.info(f"[Random Forest] 準確率(Accuracy): {rf_acc*100:.1f}%, 精確率(Precision): {rf_prec*100:.1f}%")
    
    return xgb_model, rf_model

def train():
    try:
        df = pd.read_csv("ml_dataset.csv")
    except FileNotFoundError:
        logging.error("找不到 ml_dataset.csv，請先執行 create_dataset.py")
        return
        
    # 定義特徵欄位 (必須與 create_dataset.py 一致，共 37 維度)
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
        # 新增高品質特徵 (+6)
        'volume_ratio', 'spread', 'session_overlap',
        'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
    ]
    
    X = df[features]
    y_buy = df['label_buy']
    y_sell = df['label_sell']
    
    # 切分訓練集與測試集 (80% 訓練, 20% 測試)
    # 因為是時間序列資料，我們不打亂順序 (shuffle=False)，避免未來數據污染過去
    X_train, X_test, y_buy_train, y_buy_test = train_test_split(X, y_buy, test_size=0.2, shuffle=False)
    _, _, y_sell_train, y_sell_test = train_test_split(X, y_sell, test_size=0.2, shuffle=False)
    
    # 訓練多單預測模型
    xgb_buy, rf_buy = train_and_evaluate(X_train, X_test, y_buy_train, y_buy_test, "多單 (Buy)")
    
    # 訓練空單預測模型
    xgb_sell, rf_sell = train_and_evaluate(X_train, X_test, y_sell_train, y_sell_test, "空單 (Sell)")
    
    # 儲存模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_buy, 'models/xgb_buy.pkl')
    joblib.dump(rf_buy, 'models/rf_buy.pkl')
    joblib.dump(xgb_sell, 'models/xgb_sell.pkl')
    joblib.dump(rf_sell, 'models/rf_sell.pkl')
    
    logging.info("✅ 4 個模型 (XGBoost & Random Forest 各多空) 已儲存至 models/ 資料夾！")

if __name__ == "__main__":
    train()

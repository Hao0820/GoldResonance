import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import joblib
import logging
import os
import warnings

# 隱藏警告
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def optimize_xgb(X, y):
    logging.info("🧬 正在使用 Optuna 優化 XGBoost 參數...")
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return precision_score(y_val, preds, zero_division=0)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # 為了速度設為 20 次，可增加
    return study.best_params

def optimize_lgb(X, y):
    logging.info("🧬 正在使用 Optuna 優化 LightGBM 參數...")
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return precision_score(y_val, preds, zero_division=0)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

def train_ensemble(X, y, side="Buy"):
    logging.info(f"--- 開始訓練 {side} 集成專家系統 ---")
    
    # 1. 拆分訓練集 (用於基礎模型) 與 驗證集 (用於訓練 Stacker)
    X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # 2. 優化並訓練基礎模型
    xgb_params = optimize_xgb(X_base, y_base)
    lgb_params = optimize_lgb(X_base, y_base)
    
    m_xgb = xgb.XGBClassifier(**xgb_params).fit(X_base, y_base)
    m_lgb = lgb.LGBMClassifier(**lgb_params).fit(X_base, y_base)
    m_rf  = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42).fit(X_base, y_base)
    
    # 3. 建立 Stacker (Logistic Regression)
    # 讓 Stacker 學習如何組合這三個模型的機率輸出
    p_xgb = m_xgb.predict_proba(X_meta)[:, 1]
    p_lgb = m_lgb.predict_proba(X_meta)[:, 1]
    p_rf  = m_rf.predict_proba(X_meta)[:, 1]
    
    meta_features = np.column_stack([p_xgb, p_lgb, p_rf])
    stacker = LogisticRegression().fit(meta_features, y_meta)
    
    # 評估最終效能
    final_probs = stacker.predict_proba(meta_features)[:, 1]
    final_preds = (final_probs > 0.5).astype(int)
    acc = accuracy_score(y_meta, final_preds)
    prec = precision_score(y_meta, final_preds, zero_division=0)
    
    logging.info(f"✅ {side} 集成完成! 準確率: {acc*100:.1f}%, 精確率: {prec*100:.1f}%")
    return m_xgb, m_lgb, m_rf, stacker

def train():
    try:
        df = pd.read_csv("ml_dataset.csv")
    except FileNotFoundError:
        logging.error("找不到資料集，請先執行 create_dataset.py")
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
        'm30_rsi', 'm30_trend', 'm1_momentum',
        'volume_ratio', 'spread', 'session_overlap',
        'rsi_divergence', 'price_vs_vwap', 'pattern_engulf'
    ]
    
    X = df[features]
    
    # 訓練多單
    xgb_b, lgb_b, rf_b, stack_b = train_ensemble(X, df['label_buy'], "多單")
    # 訓練空單
    xgb_s, lgb_s, rf_s, stack_s = train_ensemble(X, df['label_sell'], "空單")
    
    # 儲存所有大腦組件
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_b, 'models/xgb_buy.pkl')
    joblib.dump(lgb_b, 'models/lgb_buy.pkl')
    joblib.dump(rf_b,  'models/rf_buy.pkl')
    joblib.dump(stack_b,'models/stack_buy.pkl')
    
    joblib.dump(xgb_s, 'models/xgb_sell.pkl')
    joblib.dump(lgb_s, 'models/lgb_sell.pkl')
    joblib.dump(rf_s,  'models/rf_sell.pkl')
    joblib.dump(stack_s,'models/stack_sell.pkl')
    
    logging.info("🚀 多專家集成大腦訓練完成！所有專家與主審已入駐 models/ 資料夾。")

if __name__ == "__main__":
    train()

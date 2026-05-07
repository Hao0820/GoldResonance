"""
Iron Wall AI 訓練模組
────────────────────────────────────────────────
簡潔架構：固定 TP=5 / SL=5
  → 訓練兩個 XGBClassifier 分別預測多單/空單的勝率（信心率 %）
  → 信心率越高代表 AI 越確定這筆能賺
  → 輸出：win_buy.pkl、win_sell.pkl
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

FEATURE_COLS = [
    'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
    'hour', 'day_of_week', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
    'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema', 'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
    'prev_body_size', 'prev_close_change', 'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
    'm30_rsi', 'm30_trend', 'm1_momentum', 'volume_ratio', 'spread', 'session_overlap',
    'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
    'ema12_dist', 'mid_bb_dist', 'consecutive_low_shadows', 'consecutive_high_shadows',
    'pullback_ratio', 'is_breaking_upper', 'is_breaking_lower',
    # 專家圖示對應特徵
    'is_big_red_k', 'is_big_green_k',
    'm5_dist_oh', 'm5_dist_ol',
    'm15_dist_oh', 'm15_dist_ol',
    'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l',
    'consolidation_score',
]

def train_win_classifier(X, y, name: str, sample_weight=None):
    """訓練單一方向的勝率分類器，回傳訓練好的模型"""
    logging.info(f"--- 開始訓練 [{name}] 信心率大腦 ---")
    pos_rate = y.mean() * 100
    logging.info(f"    標籤勝率: {pos_rate:.1f}%  樣本數: {len(y):,}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 切割 sample_weight（與 X_tr 對齊）
    sw_tr = None
    if sample_weight is not None:
        sw_all = sample_weight.values if hasattr(sample_weight, 'values') else sample_weight
        sw_tr  = sw_all[:len(X_tr)]
        avg_w  = sw_tr.mean()
        logging.info(f"    使用 sample_weight（平均權重: {avg_w:.2f}，覆蓋 {(sw_tr > 1).sum():,} 筆高品質樣本）")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(y_val, preds) * 100
    prec  = precision_score(y_val, preds, zero_division=0) * 100
    auc   = roc_auc_score(y_val, probs) * 100

    logging.info(f"✅ [{name}] 完成！準確率: {acc:.1f}%  精確率: {prec:.1f}%  AUC: {auc:.1f}%")
    return model


def train():
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset    = os.path.join(base_dir, "ml_dataset.csv")
    models_dir = os.path.join(base_dir, "ml_engine", "models")

    try:
        df = pd.read_csv(dataset)
    except FileNotFoundError:
        logging.error("找不到 ml_dataset.csv，請先執行 create_dataset.py")
        return

    # 欄位檢查
    for col in ['label_win_buy', 'label_win_sell']:
        if col not in df.columns:
            logging.error(f"資料集缺少欄位 {col}，請重新執行 create_dataset.py")
            return

    X = df[FEATURE_COLS]

    # ───────────────────────────────────────────────────────────────
    # 修正核心：只用 PPT 訊號樣本訓練（pattern_stage != 0）
    # 原因：若用全量資料，AI 學到的是「全市場平均」而不是 PPT 語境
    #       PPT 訊號可能只占全部 K 棒的5~10%，開着訓練等於讓 AI 吃雜訊
    # ───────────────────────────────────────────────────────────────
    if 'pattern_stage' in df.columns:
        total_n = len(df)
        df = df[df['pattern_stage'] != 0].copy().reset_index(drop=True)
        ppt_n  = len(df)
        stage_dist = df['pattern_stage'].value_counts().to_dict()
        logging.info(
            f"🎯 PPT訊號過濾: {total_n:,} → {ppt_n:,} 樣本 "
            f"({ppt_n / total_n * 100:.1f}% 為 PPT K 棒)\n"
            f"   階段分佈: {stage_dist}"
        )
        if ppt_n < 200:
            logging.error("❌ PPT 樣本數量不足 200 筆，請檢查 stage_classifier 參數或历史資料量。")
            return
    else:
        logging.warning("⚠️  未找到 pattern_stage 欄位，將使用全量資料訓練（不建議）")

    X  = df[FEATURE_COLS]

    # 若資料集包含 stage sample_weight 則使用（由 stage_classifier 產生）
    sw = df['sample_weight'] if 'sample_weight' in df.columns else None
    if sw is not None:
        stage_dist = df['pattern_stage'].value_counts().to_dict() if 'pattern_stage' in df.columns else {}
        logging.info(f"📊 載入階段標籤 → 分佈: {stage_dist}")
    else:
        logging.info("⚠️  未找到 sample_weight 欄位，使用均等權重訓練")

    logging.info("=== Iron Wall AI 信心率大腦特訓開始 ===")
    logging.info(f"訓練樣本數: {len(X):,}  特徵維度: {len(FEATURE_COLS)}")
    logging.info("固定策略：TP = 5 點  SL = 5 點  下單門檻：信心率 ≥ 65%")

    model_buy  = train_win_classifier(X, df['label_win_buy'],  "多單勝率", sample_weight=sw)
    model_sell = train_win_classifier(X, df['label_win_sell'], "空單勝率", sample_weight=sw)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model_buy,  os.path.join(models_dir, 'win_buy.pkl'))
    joblib.dump(model_sell, os.path.join(models_dir, 'win_sell.pkl'))

    logging.info("=" * 50)
    logging.info(f"🚀 AI 信心率大腦特訓完成！模型已存至 {models_dir}")
    logging.info("模型清單：win_buy.pkl  win_sell.pkl")
    logging.info("重新啟動 main.py 即可使用最新大腦！")
    logging.info("=" * 50)


if __name__ == "__main__":
    train()

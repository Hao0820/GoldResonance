"""
Iron Wall AI — 持續演化訓練引擎
════════════════════════════════════════════════════════
設計理念：
  每一輪（Generation）= 訓練 → 回測 → 反饋 → 調整樣本權重 → 下一輪

  三種強化機制：
  ① 勝者加權（Winner Boost）
     回測中實際獲利的交易，找回對應的訓練樣本，提高其 sample_weight
  ② 敗者懲罰（Loser Decay）
     回測中虧損的交易，降低其訓練權重（讓模型不再模仿這類行為）
  ③ 信心率校準（Confidence Calibration）
     若某一 pattern_stage 的勝率 > 70%，提升整個 stage 的基礎權重
     若某一 pattern_stage 的勝率 < 45%，壓低整個 stage 的基礎權重

使用方式：
  python continuous_evolve.py               # 預設跑 10 代
  python continuous_evolve.py --gens 20     # 跑 20 代
  python continuous_evolve.py --gens 5 --skip-data  # 不重新拉資料，純訓練
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
import argparse
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "ml_engine", "models")
DATASET_CSV = os.path.join(BASE_DIR, "ml_dataset.csv")
BACKTEST_CSV = os.path.join(BASE_DIR, "backtest_result.csv")
HISTORY_LOG = os.path.join(BASE_DIR, "ml_engine", "evolution_history.json")

# ── 日誌 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "ml_engine", "evolve.log"),
                            encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'm5_ema_slope', 'm5_rsi_14', 'm5_atr_14', 'body_size', 'upper_shadow', 'lower_shadow', 'body_ratio',
    'hour', 'day_of_week', 'm15_dist_h', 'm15_dist_l', 'm15_dist_m', 'm15_ema_slope',
    'm5_dist_h', 'm5_dist_l', 'm5_dist_m', 'm5_dist_ema', 'm5_adx', 'm5_macd_hist', 'm5_cci', 'm5_bb_width',
    'prev_body_size', 'prev_close_change', 'is_us_session', 'is_asia_session', 'h1_trend', 'h4_trend', 'd1_rsi',
    'm30_rsi', 'm30_trend', 'm1_momentum', 'volume_ratio', 'spread', 'session_overlap',
    'rsi_divergence', 'price_vs_vwap', 'pattern_engulf',
    'ema12_dist', 'mid_bb_dist', 'consecutive_low_shadows', 'consecutive_high_shadows',
    'pullback_ratio', 'is_breaking_upper', 'is_breaking_lower',
    'is_big_red_k', 'is_big_green_k',
    'm5_dist_oh', 'm5_dist_ol',
    'm15_dist_oh', 'm15_dist_ol',
    'h4_bb_dist_m', 'h4_bb_dist_h', 'h4_bb_dist_l',
    'consolidation_score',
]

# ── 權重上下限 ───────────────────────────────────────────────────────────────
W_MIN   = 0.5    # 最低懲罰權重
W_MAX   = 5.0    # 最高獎勵權重
W_BOOST = 1.5    # 勝者每代加成倍率
W_DECAY = 0.7    # 敗者每代衰減倍率


# ════════════════════════════════════════════════════════════════════════════
# 1. 訓練
# ════════════════════════════════════════════════════════════════════════════
def train_one_model(X_tr, y_tr, X_val, y_val, sw_tr=None, name="model") -> tuple:
    """回傳 (model, acc, prec, auc)"""
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc  = accuracy_score(y_val, preds) * 100
    prec = precision_score(y_val, preds, zero_division=0) * 100
    auc  = roc_auc_score(y_val, probs) * 100
    logger.info(f"  [{name}] 準確率:{acc:.1f}%  精確率:{prec:.1f}%  AUC:{auc:.1f}%")
    return model, acc, prec, auc


def run_training(df: pd.DataFrame, gen: int) -> dict:
    """執行一輪訓練，回傳本輪指標 dict"""
    X  = df[FEATURE_COLS]
    sw = df['evolve_weight'] if 'evolve_weight' in df.columns else df.get('sample_weight')

    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_b_tr, y_b_val = df['label_win_buy'].iloc[:split],  df['label_win_buy'].iloc[split:]
    y_s_tr, y_s_val = df['label_win_sell'].iloc[:split], df['label_win_sell'].iloc[split:]
    sw_tr = sw.iloc[:split].values if sw is not None else None

    logger.info(f"  訓練樣本: {len(X_tr):,}  驗證: {len(X_val):,}  "
                f"平均權重: {sw_tr.mean():.3f}" if sw_tr is not None else "  均等權重")

    model_buy,  acc_b, prec_b, auc_b = train_one_model(X_tr, y_b_tr, X_val, y_b_val, sw_tr, "多單")
    model_sell, acc_s, prec_s, auc_s = train_one_model(X_tr, y_s_tr, X_val, y_s_val, sw_tr, "空單")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_buy,  os.path.join(MODELS_DIR, "win_buy.pkl"))
    joblib.dump(model_sell, os.path.join(MODELS_DIR, "win_sell.pkl"))

    # 備份此代模型
    gen_dir = os.path.join(MODELS_DIR, f"gen_{gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    joblib.dump(model_buy,  os.path.join(gen_dir, "win_buy.pkl"))
    joblib.dump(model_sell, os.path.join(gen_dir, "win_sell.pkl"))

    return {
        "gen": gen,
        "auc_buy": round(auc_b, 2), "auc_sell": round(auc_s, 2),
        "acc_buy": round(acc_b, 2), "acc_sell": round(acc_s, 2),
        "prec_buy": round(prec_b, 2), "prec_sell": round(prec_s, 2),
        "avg_weight": round(float(sw_tr.mean()), 4) if sw_tr is not None else 1.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 快速回測（輕量版，不用 M1 逐根，只用模型預測+歷史標籤）
# ════════════════════════════════════════════════════════════════════════════
def run_fast_backtest(df: pd.DataFrame) -> dict:
    """
    用已訓練的模型對驗證集預測，比對 label_win_buy/sell 作為快速回測。
    回傳 {win_rate_buy, win_rate_sell, stage_wr: {stage: wr}}
    """
    try:
        model_buy  = joblib.load(os.path.join(MODELS_DIR, "win_buy.pkl"))
        model_sell = joblib.load(os.path.join(MODELS_DIR, "win_sell.pkl"))
    except FileNotFoundError:
        logger.warning("  找不到模型，跳過回測")
        return {}

    split = int(len(df) * 0.8)
    val   = df.iloc[split:].copy()
    X_val = val[FEATURE_COLS]

    prob_b = model_buy.predict_proba(X_val)[:, 1]
    prob_s = model_sell.predict_proba(X_val)[:, 1]

    # 信心率 >= 65% 才計入
    THRESH = 0.65
    mask_b = prob_b >= THRESH
    mask_s = prob_s >= THRESH

    wr_b = val['label_win_buy'][mask_b].mean()  if mask_b.sum() > 0 else 0.0
    wr_s = val['label_win_sell'][mask_s].mean() if mask_s.sum() > 0 else 0.0
    trades_b = int(mask_b.sum())
    trades_s = int(mask_s.sum())

    logger.info(f"  快速回測 → 多單: {wr_b*100:.1f}% ({trades_b}筆)  "
                f"空單: {wr_s*100:.1f}% ({trades_s}筆)")

    # 分 stage 統計
    stage_wr = {}
    if 'pattern_stage' in val.columns:
        for stage in [0, 2, 3, 41, 42]:
            m = (val['pattern_stage'] == stage)
            mb = m & mask_b
            ms = m & mask_s
            wr = 0.0
            n  = int(mb.sum() + ms.sum())
            if n > 0:
                hits = int(val['label_win_buy'][mb].sum() + val['label_win_sell'][ms].sum())
                wr = hits / n
            stage_wr[stage] = {"wr": round(wr, 4), "n": n}
        stage_summary = {s: f"{v['wr']*100:.1f}%({v['n']})" for s, v in stage_wr.items()}
        logger.info(f"  各階段勝率: {stage_summary}")

    return {
        "win_rate_buy": round(float(wr_b), 4),
        "win_rate_sell": round(float(wr_s), 4),
        "trades_buy": trades_b,
        "trades_sell": trades_s,
        "stage_wr": stage_wr,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 反饋：根據回測結果調整 evolve_weight
# ════════════════════════════════════════════════════════════════════════════
def update_weights(df: pd.DataFrame, bt: dict) -> pd.DataFrame:
    """
    三層調整機制：
    A. 全局：多/空單整體勝率決定基礎乘數
    B. 階段：各 pattern_stage 的勝率決定 stage 乘數
    C. 初始化：若無 evolve_weight 則用 sample_weight 初始化
    """
    if 'evolve_weight' not in df.columns:
        base = df['sample_weight'] if 'sample_weight' in df.columns else pd.Series(1.0, index=df.index)
        df['evolve_weight'] = base.values.copy()

    w = df['evolve_weight'].values.copy()

    # ── A. 全局勝率調整 ──────────────────────────────────────────────────
    wr_b = bt.get("win_rate_buy",  0.5)
    wr_s = bt.get("win_rate_sell", 0.5)
    global_factor = 1.0
    if (wr_b + wr_s) / 2 > 0.65:
        global_factor = 1.02   # 整體好 → 輕微加強
        logger.info("  全局勝率優秀，整體權重小幅提升 +2%")
    elif (wr_b + wr_s) / 2 < 0.50:
        global_factor = 0.98   # 整體差 → 輕微下壓
        logger.info("  全局勝率偏低，整體權重小幅下調 -2%")
    w *= global_factor

    # ── B. 各 stage 獨立調整 ─────────────────────────────────────────────
    stage_wr = bt.get("stage_wr", {})
    if 'pattern_stage' in df.columns and stage_wr:
        for stage, info in stage_wr.items():
            wr = info.get("wr", 0.5)
            n  = info.get("n", 0)
            if n < 10:      # 樣本太少，不調
                continue
            mask = (df['pattern_stage'].values == stage)
            if wr > 0.70:
                # 這個 stage 很強 → 權重 ×W_BOOST
                w[mask] = np.clip(w[mask] * W_BOOST, W_MIN, W_MAX)
                logger.info(f"  Stage {stage} 勝率={wr*100:.1f}% 優秀 → 樣本加權 ×{W_BOOST}")
            elif wr < 0.45:
                # 這個 stage 很弱 → 權重 ×W_DECAY
                w[mask] = np.clip(w[mask] * W_DECAY, W_MIN, W_MAX)
                logger.info(f"  Stage {stage} 勝率={wr*100:.1f}% 偏低 → 樣本降權 ×{W_DECAY}")
            # 45%~70% 不動

    df['evolve_weight'] = w
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. 歷史記錄
# ════════════════════════════════════════════════════════════════════════════
def load_history() -> list:
    if os.path.exists(HISTORY_LOG):
        with open(HISTORY_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: list):
    os.makedirs(os.path.dirname(HISTORY_LOG), exist_ok=True)
    with open(HISTORY_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def print_progress(history: list):
    if len(history) < 2:
        return
    logger.info("\n" + "═" * 60)
    logger.info(f"  {'代':>4}  {'多AUC':>7}  {'空AUC':>7}  "
                f"{'多勝率':>7}  {'空勝率':>7}  {'平均權重':>8}")
    logger.info("─" * 60)
    for h in history:
        wr_b = h.get("win_rate_buy",  0) * 100
        wr_s = h.get("win_rate_sell", 0) * 100
        logger.info(
            f"  Gen{h['gen']:>3}  {h['auc_buy']:>6.1f}%  {h['auc_sell']:>6.1f}%  "
            f"  {wr_b:>5.1f}%    {wr_s:>5.1f}%  {h.get('avg_weight',1):.3f}"
        )
    # 最優指標
    best_gen = max(history, key=lambda x: (x.get("win_rate_buy", 0) + x.get("win_rate_sell", 0)))
    logger.info("─" * 60)
    logger.info(f"  🏆 最佳代數: Gen{best_gen['gen']}  "
                f"多勝率={best_gen.get('win_rate_buy',0)*100:.1f}%  "
                f"空勝率={best_gen.get('win_rate_sell',0)*100:.1f}%")
    logger.info("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 5. 主循環
# ════════════════════════════════════════════════════════════════════════════
def continuous_evolve(total_gens: int = 10, skip_data: bool = False):
    """
    主演化循環
    total_gens : 總訓練代數
    skip_data  : True = 跳過資料匯出/重建，直接用現有 ml_dataset.csv
    """
    logger.info("=" * 60)
    logger.info(f"🧬  Iron Wall 持續演化訓練引擎啟動")
    logger.info(f"    目標代數: {total_gens}  跳過資料重建: {skip_data}")
    logger.info("=" * 60)

    history = load_history()
    start_gen = len(history) + 1   # 從上次斷點繼續

    # ── 一次性準備資料 ───────────────────────────────────────────────────
    if not skip_data:
        logger.info("── 階段 0：資料準備 ──────────────────────────────")
        try:
            from ml_engine.create_dataset import create_dataset
            create_dataset()
        except Exception as e:
            logger.error(f"資料建立失敗: {e}，改用現有 ml_dataset.csv")

    if not os.path.exists(DATASET_CSV):
        logger.error(f"找不到 {DATASET_CSV}，請先執行 create_dataset.py")
        return

    logger.info("📂 載入資料集...")
    df = pd.read_csv(DATASET_CSV, parse_dates=['time'] if 'time' in pd.read_csv(DATASET_CSV, nrows=0).columns else [])
    logger.info(f"   資料筆數: {len(df):,}  特徵維度: {len(FEATURE_COLS)}")

    # ── 迭代演化 ─────────────────────────────────────────────────────────
    for gen in range(start_gen, start_gen + total_gens):
        t0 = time.time()
        logger.info(f"\n{'━'*60}")
        logger.info(f"🔁  Generation {gen} / {start_gen + total_gens - 1}  "
                    f"[{datetime.now().strftime('%H:%M:%S')}]")
        logger.info(f"{'━'*60}")

        # 訓練
        logger.info("── 訓練 ──")
        metrics = run_training(df, gen)

        # 快速回測
        logger.info("── 快速回測 ──")
        bt = run_fast_backtest(df)
        metrics.update(bt)

        # 調整權重
        logger.info("── 反饋調整 ──")
        df = update_weights(df, bt)

        # 記錄歷史
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["elapsed"]   = round(time.time() - t0, 1)
        history.append(metrics)
        save_history(history)

        # 打印進度表
        print_progress(history)
        logger.info(f"  本代耗時: {metrics['elapsed']}s")

    # ── 結束 ─────────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info(f"🎉  演化訓練完成！共執行 {total_gens} 代")
    logger.info(f"    最終模型已存至: {MODELS_DIR}")
    logger.info(f"    歷史記錄: {HISTORY_LOG}")
    logger.info("    重新啟動 main.py 即可使用最新大腦！")
    logger.info("═" * 60)


# ════════════════════════════════════════════════════════════════════════════
# 6. 單獨查看訓練歷史
# ════════════════════════════════════════════════════════════════════════════
def show_history():
    history = load_history()
    if not history:
        print("尚無訓練歷史，請先執行演化訓練。")
        return
    print_progress(history)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iron Wall 持續演化訓練")
    parser.add_argument("--gens",       type=int,  default=10,    help="訓練代數（預設10）")
    parser.add_argument("--skip-data",  action="store_true",       help="跳過資料重建，使用現有 ml_dataset.csv")
    parser.add_argument("--history",    action="store_true",       help="只顯示歷史記錄不訓練")
    args = parser.parse_args()

    if args.history:
        show_history()
    else:
        continuous_evolve(total_gens=args.gens, skip_data=args.skip_data)

"""
Iron Wall — PPT 交易階段分類器
────────────────────────────────────────────────
根據 2/3/4-1/4-2 四個操作階段的規則，對每根 M5 K棒進行標記：
  - pattern_stage:    0=無訊號, 2=2階, 3=3階, 41=4-1階, 42=4-2階
  - signal_direction: 1=多, -1=空, 0=無
  - sample_weight:    依階段給予訓練權重（42/41 > 3/2 > 0）

使用方式：
    from ml_engine.stage_classifier import add_stage_labels
    df = add_stage_labels(df)
    # 之後在 model.fit() 中: sample_weight=df['sample_weight'].values
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── 各階段 sample_weight ──────────────────────────────────────────────────────
STAGE_WEIGHT_MAP = {0: 1.0, 2: 1.5, 3: 1.5, 41: 2.0, 42: 2.0}


def classify_stage(df: pd.DataFrame) -> np.ndarray:
    """
    規則引擎：對 df 每根 M5 K棒標記操作階段。

    必要欄位（來自 create_dataset.py 輸出）：
        consolidation_score, mid_bb_dist, ema12_dist, pullback_ratio,
        consecutive_low_shadows, consecutive_high_shadows,
        is_breaking_upper, is_breaking_lower,
        is_big_red_k, is_big_green_k,
        m5_rsi_14, m30_rsi,
        h4_trend, h4_bb_dist_m,
        h1_trend, m5_ema_slope,
        m5_dist_h, m5_dist_l, m5_dist_oh, m5_dist_ol,
        body_size, lower_shadow, upper_shadow
    """
    n = len(df)
    stage = np.zeros(n, dtype=int)

    # 轉 numpy 加速
    cs      = df['consolidation_score'].values
    mbd     = df['mid_bb_dist'].values
    e12d    = df['ema12_dist'].values
    pbr     = df['pullback_ratio'].values
    cls_sh  = df['consecutive_low_shadows'].values
    chs_sh  = df['consecutive_high_shadows'].values
    brk_u   = df['is_breaking_upper'].values
    brk_d   = df['is_breaking_lower'].values
    big_r   = df['is_big_red_k'].values
    big_g   = df['is_big_green_k'].values
    rsi5    = df['m5_rsi_14'].values
    rsi30   = df['m30_rsi'].values
    h4t     = df['h4_trend'].values
    h4bdm   = df['h4_bb_dist_m'].values
    h1t     = df['h1_trend'].values
    slope   = df['m5_ema_slope'].values
    d5h     = df['m5_dist_h'].values
    d5l     = df['m5_dist_l'].values
    d5oh    = df['m5_dist_oh'].values
    d5ol    = df['m5_dist_ol'].values
    body    = df['body_size'].values
    lshdw   = df['lower_shadow'].values
    ushdw   = df['upper_shadow'].values

    for i in range(5, n):
        # ────────────────────────────────────────────────────────────────────
        # 4-2 階（優先級最高）— PPT 台灣慣例：紅K=上漲(close>open)，綠K=下跌(close<open)
        # 多單：前根大紅K（上漲大K），當前回踩 12均/中軌，收下影線
        is_42_long = (
            big_r[i - 1] == 1
            and abs(e12d[i]) < 3.5
            and lshdw[i] > abs(body[i]) * 0.25
            and d5l[i] > -6.0           # 未跌穿下軌
        )
        # 空單：前根大綠K（下跌大K），當前反彈至 12均/中軌，收上影線
        is_42_short = (
            big_g[i - 1] == 1
            and abs(e12d[i]) < 3.5
            and ushdw[i] > abs(body[i]) * 0.25
            and d5h[i] < 6.0            # 未突穿上軌
        )

        # ────────────────────────────────────────────────────────────────────
        # 4-1 階
        # 多單：前根剛破上軌，當前回踩 1/4~1/3
        is_41_long = (
            cs[i] < 0.35            # 非盤整，趨勢中
            and brk_u[i - 1] == 1       # 前根剛突破上軌
            and 0.20 < pbr[i] < 0.45   # 回踩 1/4 ~ 1/3
            and slope[i] > 0            # EMA 斜率向上
        )
        # 空單：前根剛破下軌，當前反彈 1/4~1/3
        is_41_short = (
            cs[i] < 0.35
            and brk_d[i - 1] == 1
            and -0.45 < pbr[i] < -0.20
            and slope[i] < 0
        )

        # ────────────────────────────────────────────────────────────────────
        # 3 階（逆勢，觸及外帶）
        # 多單：跌至外帶下軌，RSI 超賣，出現下影線
        is_3_long = (
            cs[i] > 0.40
            and d5ol[i] <= 1.5          # 觸碰或接近外帶下軌
            and rsi5[i] < 38
            and rsi30[i] < 42
            and cls_sh[i] >= 1
        )
        # 空單：漲至外帶上軌，RSI 超買，出現上影線
        is_3_short = (
            cs[i] > 0.40
            and d5oh[i] >= -1.5         # 觸碰或接近外帶上軌
            and rsi5[i] > 62
            and rsi30[i] > 58
            and chs_sh[i] >= 1
        )

        # ────────────────────────────────────────────────────────────────────
        # 2 階（盤整突破後順勢）
        # 多單：盤整後突破中軌向上，回踩 1/3~1/2，有下影線
        is_2_long = (
            cs[i] > 0.55
            and mbd[i] > 0
            and 0.28 < pbr[i] < 0.65
            and cls_sh[i] >= 1
            and brk_u[i] == 0           # 回踩（不是當前還在突破）
            and 38 < rsi5[i] < 68
        )
        # 空單：盤整後跌破中軌向下，反彈 1/3~1/2，有上影線
        is_2_short = (
            cs[i] > 0.55
            and mbd[i] < 0
            and -0.65 < pbr[i] < -0.28
            and chs_sh[i] >= 1
            and brk_d[i] == 0
            and 32 < rsi5[i] < 62
        )

        # ── 優先級賦值：42 > 41 > 3 > 2 ──
        if is_42_long or is_42_short:
            stage[i] = 42
        elif is_41_long or is_41_short:
            stage[i] = 41
        elif is_3_long or is_3_short:
            stage[i] = 3
        elif is_2_long or is_2_short:
            stage[i] = 2

    return stage


def get_signal_direction(df: pd.DataFrame, stage: np.ndarray) -> np.ndarray:
    """
    根據特徵方向判斷多（1）/ 空（-1）/ 無（0）。
    """
    n = len(df)
    direction = np.zeros(n, dtype=int)

    h4t   = df['h4_trend'].values
    h1t   = df['h1_trend'].values
    mbd   = df['mid_bb_dist'].values
    slope = df['m5_ema_slope'].values
    big_r = df['is_big_red_k'].values
    big_g = df['is_big_green_k'].values
    e12d  = df['ema12_dist'].values

    for i in range(5, n):
        if stage[i] == 0:
            continue

        long_score  = 0
        short_score = 0

        if h4t[i] > 0:    long_score  += 2
        if h4t[i] < 0:    short_score += 2
        if h1t[i] > 0:    long_score  += 1
        if h1t[i] < 0:    short_score += 1
        if mbd[i] > 0:    long_score  += 1
        if mbd[i] < 0:    short_score += 1
        if slope[i] > 0:  long_score  += 1
        if slope[i] < 0:  short_score += 1

        if i >= 1:
            if big_r[i - 1] == 1: long_score  += 2
            if big_g[i - 1] == 1: short_score += 2

        # 3 階逆勢：方向與趨勢相反
        if stage[i] == 3:
            long_score, short_score = short_score, long_score

        if long_score > short_score:
            direction[i] = 1
        elif short_score > long_score:
            direction[i] = -1

    return direction


def add_stage_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    對 ml_dataset 新增三個輔助欄位並回傳：
      - pattern_stage:   操作階段 (0/2/3/41/42)
      - signal_direction: 方向 (1/-1/0)
      - sample_weight:   訓練權重 (1.0 ~ 2.0)
    """
    required_cols = [
        'consolidation_score', 'mid_bb_dist', 'ema12_dist', 'pullback_ratio',
        'consecutive_low_shadows', 'consecutive_high_shadows',
        'is_breaking_upper', 'is_breaking_lower',
        'is_big_red_k', 'is_big_green_k',
        'm5_rsi_14', 'm30_rsi',
        'h4_trend', 'h4_bb_dist_m', 'h1_trend', 'm5_ema_slope',
        'm5_dist_h', 'm5_dist_l', 'm5_dist_oh', 'm5_dist_ol',
        'body_size', 'lower_shadow', 'upper_shadow',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"⚠️  stage_classifier: 缺少欄位 {missing}，將以 0 填補")
        for c in missing:
            df[c] = 0.0

    stage     = classify_stage(df)
    direction = get_signal_direction(df, stage)
    weight    = np.vectorize(STAGE_WEIGHT_MAP.get)(stage, 1.0)

    df = df.copy()
    df['pattern_stage']    = stage
    df['signal_direction'] = direction
    df['sample_weight']    = weight

    counts = {s: int((stage == s).sum()) for s in [0, 2, 3, 41, 42]}
    logger.info(
        f"📊 階段分佈 → 無訊號: {counts[0]:,}  "
        f"2階: {counts[2]:,}  3階: {counts[3]:,}  "
        f"4-1階: {counts[41]:,}  4-2階: {counts[42]:,}"
    )
    return df


if __name__ == "__main__":
    # 快速測試
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "ml_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['time'])
        df = add_stage_labels(df)
        print(df[['time', 'pattern_stage', 'signal_direction', 'sample_weight']].tail(20).to_string())
    else:
        print("請先執行 create_dataset.py 產生 ml_dataset.csv")

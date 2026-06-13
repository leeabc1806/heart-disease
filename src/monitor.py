"""
04 — 모니터링 & 데이터 드리프트 탐지

1. 추론 로깅 계측: 타임스탬프 / 모델 버전 / 입력 shape / 예측값 / 실제 정답 → 파일
2. 테스트셋 연속형 특성 분포를 인위적으로 이동 (chol 평균 +30, 분산 증가)
3. ks_2samp: 학습 분포 vs 드리프트 분포, p-value 보고, p < 0.05 플래그
4. 원본 vs 드리프트 balanced accuracy 비교 → 입력 드리프트 ↔ 성능 저하 연관성
5. 합성 타임스탬프 기반 시계열 지표 그래프
"""
import logging
import os
import pickle
import json
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # 비대화형 백엔드 (스크립트 실행용)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_data
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# ── 설정 ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
ALPHA = 0.05
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "monitor.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── 아티팩트 로드 ───────────────────────────────────────────────────────────────

def load_artifacts():
    """모델 + 학습/테스트 분할 재현 (train.py와 동일한 시드)."""
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    df = load_data()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = df[feature_cols], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return model_data, X_train, X_test, y_train, y_test


# ── 1. 추론 로깅 계측 ───────────────────────────────────────────────────────────

def log_inference(pipeline, X, y_true, model_name, version, tag=""):
    """추론 1배치를 로그로 기록: 타임스탬프 / 모델 버전 / 입력 shape / 예측 / 정답."""
    preds = pipeline.predict(X)
    acc = balanced_accuracy_score(y_true, preds) if y_true is not None else None

    logger.info(
        "INFERENCE%s | ts=%s | model=%s v%s | input_shape=%s | "
        "pred_pos=%d pred_neg=%d | balanced_acc=%s",
        f"[{tag}]" if tag else "",
        datetime.now().isoformat(timespec="seconds"),
        model_name, version, X.shape,
        int((preds == 1).sum()), int((preds == 0).sum()),
        f"{acc:.4f}" if acc is not None else "N/A",
    )
    # 행 단위 예측값 (정답 포함) 상세 기록
    if y_true is not None:
        logger.info("  predictions=%s", preds.tolist())
        logger.info("  y_true     =%s", list(y_true))
    return preds


# ── 2. 드리프트 주입 ────────────────────────────────────────────────────────────

def inject_drift(df, shifts):
    """
    연속형 특성에 분포 이동 주입.
    shifts: {feature: (mean_shift, std_scale)}
        - mean_shift: 평균에 더할 값
        - std_scale:  중심 기준 분산 확대 배수
    """
    drifted = df.copy()
    for feat, (mean_shift, std_scale) in shifts.items():
        col = drifted[feat].astype(float)
        mean = col.mean()
        drifted[feat] = mean + (col - mean) * std_scale + mean_shift
    return drifted


# ── 3. KS 검정 ─────────────────────────────────────────────────────────────────

def run_ks_tests(train_df, drifted_df, features, alpha=ALPHA):
    """각 연속형 특성에 ks_2samp: 학습 분포 vs 드리프트 분포."""
    rows = []
    for feat in features:
        a = train_df[feat].dropna()
        b = drifted_df[feat].dropna()
        stat, p = ks_2samp(a, b)
        rows.append({
            "feature": feat,
            "ks_stat": round(stat, 4),
            "p_value": p,
            "drift_flagged": bool(p < alpha),
        })
    return pd.DataFrame(rows)


# ── 5. 시계열 시각화 ────────────────────────────────────────────────────────────

def plot_drift_timeseries(pipeline, X_test, y_test, train_feat, feature="chol",
                          n_days=12, step=8):
    """합성 타임스탬프(일 단위)로 드리프트 점증 → balanced accuracy & KS p-value 추이."""
    base_ts = datetime.now()
    days, timestamps, ba_scores, ks_pvals, feat_means = [], [], [], [], []

    for d in range(n_days):
        shift = d * step
        X_d = X_test.copy()
        X_d[feature] = X_d[feature].astype(float) + shift
        ba = balanced_accuracy_score(y_test, pipeline.predict(X_d))
        _, p = ks_2samp(train_feat.dropna(), X_d[feature].dropna())

        days.append(d)
        timestamps.append(base_ts + timedelta(days=d))
        ba_scores.append(ba)
        ks_pvals.append(p)
        feat_means.append(X_d[feature].mean())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(timestamps, ba_scores, marker="o", color="crimson", linewidth=2)
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title(f"Performance degradation as '{feature}' drifts over time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timestamps, ks_pvals, marker="s", color="steelblue", linewidth=2,
                 label="KS p-value")
    axes[1].axhline(ALPHA, color="red", linestyle="--", label=f"alpha = {ALPHA}")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("KS p-value (log scale)")
    axes[1].set_xlabel("Synthetic timestamp (days)")
    axes[1].set_title(f"Drift detection: KS p-value for '{feature}'")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "fig_drift_monitoring.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("시계열 그래프 저장: %s", out_path)

    return pd.DataFrame({
        "day": days,
        "feature_mean": np.round(feat_means, 2),
        "ks_p_value": ks_pvals,
        "balanced_accuracy": np.round(ba_scores, 4),
    })


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    model_data, X_train, X_test, y_train, y_test = load_artifacts()
    pipeline = model_data["pipeline"]
    model_name = model_data.get("model_name", "unknown")
    version = model_data.get("version", "1.0")

    print("=" * 70)
    print("1. 추론 로깅 (원본 테스트셋)")
    print("=" * 70)
    log_inference(pipeline, X_test, y_test, model_name, version, tag="original")

    # ── 드리프트 주입: chol 평균 +30, 분산 1.5배 ──
    print("\n" + "=" * 70)
    print("2. 드리프트 주입 (chol 평균 +30, 분산 ×1.5)")
    print("=" * 70)
    shifts = {"chol": (30.0, 1.5)}
    X_drift = inject_drift(X_test, shifts)
    logger.info(
        "chol 평균: 원본=%.1f → 드리프트=%.1f | chol 표준편차: 원본=%.1f → 드리프트=%.1f",
        X_test["chol"].mean(), X_drift["chol"].mean(),
        X_test["chol"].std(), X_drift["chol"].std(),
    )
    log_inference(pipeline, X_drift, y_test, model_name, version, tag="drifted")

    # ── KS 검정: 학습 분포 vs 드리프트 분포 ──
    print("\n" + "=" * 70)
    print("3. KS 검정 (학습 분포 vs 드리프트 분포)")
    print("=" * 70)
    ks_df = run_ks_tests(X_train, X_drift, NUMERIC_FEATURES)
    print(ks_df.to_string(index=False))
    flagged = ks_df[ks_df["drift_flagged"]]["feature"].tolist()
    print(f"\n드리프트 플래그된 특성 (p < {ALPHA}): {flagged}")

    # ── 성능 비교 ──
    print("\n" + "=" * 70)
    print("4. 성능 비교 (원본 vs 드리프트)")
    print("=" * 70)
    ba_orig = balanced_accuracy_score(y_test, pipeline.predict(X_test))
    ba_drift = balanced_accuracy_score(y_test, pipeline.predict(X_drift))
    print(f"  원본    balanced accuracy: {ba_orig:.4f}")
    print(f"  드리프트 balanced accuracy: {ba_drift:.4f}")
    print(f"  성능 저하: {(ba_orig - ba_drift):.4f} ({(ba_orig - ba_drift) / ba_orig * 100:.1f}%)")

    # ── 시계열 ──
    print("\n" + "=" * 70)
    print("5. 시계열 드리프트 모니터링 (합성 타임스탬프)")
    print("=" * 70)
    ts_df = plot_drift_timeseries(pipeline, X_test, y_test, X_train["chol"], feature="chol")
    print(ts_df.to_string(index=False))

    summary_path = os.path.join(DATA_DIR, "monitoring_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "model_version": version,
            "drift": shifts,
            "ks_results": ks_df.to_dict(orient="records"),
            "original_balanced_accuracy": ba_orig,
            "drifted_balanced_accuracy": ba_drift,
            "timeseries": ts_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)
    logger.info("모니터링 요약 저장: %s", summary_path)

    print("\n로그 파일: logs/monitor.log")


if __name__ == "__main__":
    main()

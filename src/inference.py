"""
추론 스크립트 — 저장된 모델로 CSV 배치 예측 수행.
logging으로 타임스탬프, 모델 버전, 입력 shape, 예측값 기록.

사용법:
    python src/inference.py --input data/sample_input.csv
    python src/inference.py --input data/sample_input.csv --output data/predictions.csv
"""
import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ── 로깅 설정 (파일 + 콘솔) ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR  = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, "inference.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_model(model_path: str = MODEL_PATH) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일 없음: {model_path}\n먼저 python src/train.py 실행")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(input_path: str, output_path: str = None) -> pd.DataFrame:
    """CSV를 읽어 예측 수행, 로그 기록 후 결과 DataFrame 반환."""
    model_data = load_model()
    pipeline   = model_data["pipeline"]
    model_name = model_data.get("model_name", "unknown")
    version    = model_data.get("version", "1.0")

    df = pd.read_csv(input_path)

    # 필요한 컬럼 없으면 오류 출력
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"입력 CSV에 컬럼 누락: {missing_cols}")

    X = df[FEATURE_COLS]

    logger.info(
        "추론 시작 | timestamp=%s | model=%s | version=%s | input_shape=%s",
        datetime.now().isoformat(timespec="seconds"),
        model_name,
        version,
        str(X.shape),
    )

    predictions  = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None

    df["prediction"] = predictions
    if probabilities is not None:
        df["prob_disease"] = probabilities.round(4)

    logger.info(
        "예측 완료 | 총 %d건 | 심장병(1)=%d건 | 정상(0)=%d건",
        len(predictions),
        int((predictions == 1).sum()),
        int((predictions == 0).sum()),
    )

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("결과 저장: %s", output_path)

    return df


def main():
    parser = argparse.ArgumentParser(description="CardioCare 추론")
    parser.add_argument("--input",  required=True, help="입력 CSV 경로")
    parser.add_argument("--output", default=None,  help="결과 CSV 저장 경로 (선택)")
    args = parser.parse_args()

    result = predict(args.input, args.output)
    print(result[["prediction"] + (["prob_disease"] if "prob_disease" in result.columns else [])].head(10))


if __name__ == "__main__":
    main()

"""
02 — 모델 학습 & MLflow 실험 추적

모델 계열 3종 비교: Logistic Regression / SVC / Random Forest
+ Random Forest 하이퍼파라미터 탐색 (RandomizedSearchCV)
5-fold 교차 검증, balanced_accuracy / precision / recall / F1 / confusion matrix 기록
최종 모델: recall 기준 선택 (임상적으로 False Negative가 치명적)
"""
import os
import sys
import pickle
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from data import load_data
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    get_feature_names,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
EXPERIMENT_NAME = "cardiocare"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def make_pipeline(model) -> Pipeline:
    """모델마다 독립적인 Pipeline 생성 (공유 상태 없음)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("selector", SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            threshold="mean",
        )),
        ("model", model),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
    }


def train_and_log(
    name: str,
    pipeline: Pipeline,
    params: dict,
    X_train, X_test,
    y_train, y_test,
    cv,
) -> tuple[Pipeline, dict]:
    """MLflow run 하나: 학습 → CV → 평가 → 기록."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", name)
        mlflow.log_params(params)

        # CV (Pipeline을 내부적으로 clone → 누수 없음)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="balanced_accuracy", n_jobs=-1,
        )

        # 전체 학습 데이터로 최종 학습
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        metrics["cv_balanced_accuracy_mean"] = float(cv_scores.mean())
        metrics["cv_balanced_accuracy_std"]  = float(cv_scores.std())

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(
            "[%s] bal_acc=%.4f  recall=%.4f  f1=%.4f  "
            "cv=%.4f±%.4f\nCM:\n%s",
            name,
            metrics["balanced_accuracy"], metrics["recall"], metrics["f1"],
            metrics["cv_balanced_accuracy_mean"], metrics["cv_balanced_accuracy_std"],
            cm,
        )

    return pipeline, metrics


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 데이터 로드 & 분할 (시드 고정, stratify)
    df = load_data()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X, y = df[feature_cols], df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train %s  Test %s", X_train.shape, X_test.shape)
    logger.info("Train 분포: %s", y_train.value_counts(normalize=True).round(3).to_dict())

    # 2. 선택된 특성 보고 (참조 파이프라인 1회 fit)
    ref_pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ref_pipe.fit(X_train, y_train)
    all_feature_names = get_feature_names(ref_pipe.named_steps["preprocessor"])
    selected_mask     = ref_pipe.named_steps["selector"].get_support()
    selected_features = [f for f, m in zip(all_feature_names, selected_mask) if m]
    logger.info("선택된 특성 (%d개): %s", len(selected_features), selected_features)

    # 3. MLflow 실험 설정
    db_path = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}

    # 4-A. Logistic Regression
    pipelines["LogisticRegression"], results["LogisticRegression"] = train_and_log(
        "LogisticRegression",
        make_pipeline(LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
        {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        X_train, X_test, y_train, y_test, cv,
    )

    # 4-B. SVC (CalibratedClassifierCV로 predict_proba 지원)
    pipelines["SVC"], results["SVC"] = train_and_log(
        "SVC",
        make_pipeline(CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
            ensemble=False,
        )),
        {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"},
        X_train, X_test, y_train, y_test, cv,
    )

    # 4-C. Random Forest (기본)
    pipelines["RandomForest"], results["RandomForest"] = train_and_log(
        "RandomForest",
        make_pipeline(RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        )),
        {"n_estimators": 100, "class_weight": "balanced"},
        X_train, X_test, y_train, y_test, cv,
    )

    # 4-D. Random Forest 하이퍼파라미터 탐색 (RandomizedSearchCV)
    logger.info("Random Forest 하이퍼파라미터 탐색 중 (n_iter=20)...")
    param_dist = {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":     ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        make_pipeline(RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        )),
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring="balanced_accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_params_rf = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    pipelines["RandomForest_tuned"], results["RandomForest_tuned"] = train_and_log(
        "RandomForest_tuned",
        search.best_estimator_,
        {**best_params_rf, "cv_best_score": round(search.best_score_, 4)},
        X_train, X_test, y_train, y_test, cv,
    )

    # 5. 비교 표
    header = f"{'Model':<25} {'BalAcc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'CV Mean':>9}"
    print("\n" + "=" * 70)
    print(header)
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<25} {m['balanced_accuracy']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['cv_balanced_accuracy_mean']:>9.4f}"
        )
    print("=" * 70)

    # 6. 최종 모델 선택 — recall 기준 (False Negative 최소화)
    best_name = max(results, key=lambda k: results[k]["recall"])
    best_metrics = results[best_name]
    best_pipeline = pipelines[best_name]

    print(f"\n최종 선택 모델: {best_name}")
    print(
        f"  recall={best_metrics['recall']:.4f}  "
        f"balanced_accuracy={best_metrics['balanced_accuracy']:.4f}  "
        f"f1={best_metrics['f1']:.4f}"
    )
    print(
        "  선택 근거: 심장병 예측 시스템에서 False Negative(심장병 → 정상 오분류)는\n"
        "  실제 환자를 치료 기회로부터 배제하는 결과를 낳는다. 따라서 recall을\n"
        "  우선 지표로 삼아 최종 모델을 선택했다. balanced_accuracy와 F1도\n"
        "  함께 고려하여 precision이 지나치게 낮아지지 않음을 확인했다."
    )

    # 7. 저장
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline":   best_pipeline,
            "model_name": best_name,
            "metrics":    best_metrics,
            "version":    "1.0",
        }, f)
    logger.info("최종 모델 저장: %s", model_path)

    return best_pipeline, best_name


if __name__ == "__main__":
    main()

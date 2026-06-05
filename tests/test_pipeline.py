"""
CardioCare 파이프라인 단위 테스트 (unittest)

테스트 1 — 예측 결과 shape이 입력 shape와 일치하는지
테스트 2 — 예측 확률이 [0, 1] 범위이고 행별 합이 약 1인지
테스트 3 — 임상 범위 초과 입력값 검증 (chol, trestbps 등)
테스트 4 — 고정 시드에서 동일 입력 → 동일 출력 (결정론)
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import load_data
from preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    validate_input,
)

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")


def _build_test_pipeline() -> Pipeline:
    """빠른 단위 테스트용 소형 파이프라인 (n_estimators=10)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("selector", SelectFromModel(
            RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE),
            threshold="mean",
        )),
        ("model", RandomForestClassifier(
            n_estimators=10, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """파이프라인 1회 학습 후 클래스 전체에서 재사용."""
        df = load_data(data_dir=DATA_DIR)
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[feature_cols]
        y = df["target"]

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        cls.pipeline = _build_test_pipeline()
        cls.pipeline.fit(X_train, y_train)
        cls.X_sample = X_test.iloc[:10].reset_index(drop=True)

    # ── 테스트 1 ──────────────────────────────────────────────────────────────
    def test_prediction_shape_matches_input(self):
        """predict() 출력 shape[0]이 입력 행 수와 같아야 한다."""
        preds = self.pipeline.predict(self.X_sample)
        self.assertEqual(
            preds.shape[0],
            self.X_sample.shape[0],
            msg=f"입력 {self.X_sample.shape[0]}행인데 예측 결과 {preds.shape[0]}행",
        )

    # ── 테스트 2 ──────────────────────────────────────────────────────────────
    def test_probability_in_unit_interval_and_row_sum_one(self):
        """predict_proba() 값이 [0,1] 범위이고, 행합이 1에 수렴해야 한다."""
        proba = self.pipeline.predict_proba(self.X_sample)

        self.assertTrue(
            np.all(proba >= 0.0) and np.all(proba <= 1.0),
            msg=f"확률 범위 벗어남: min={proba.min():.4f}, max={proba.max():.4f}",
        )
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(row_sums)),
            atol=1e-6,
            err_msg="행별 확률 합이 1이 아님",
        )

    # ── 테스트 3 ──────────────────────────────────────────────────────────────
    def test_input_range_validation(self):
        """임상 범위(chol [0,600] 등) 초과 시 오류 감지, 정상 입력은 오류 없음."""
        normal = pd.DataFrame([{
            "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
            "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
        }])
        self.assertEqual(
            validate_input(normal), [],
            msg="정상 입력에서 오류가 발생해선 안 됨",
        )

        # chol=999 → [0, 600] 범위 초과
        bad = normal.copy()
        bad["chol"] = 999
        errors = validate_input(bad)
        self.assertGreater(
            len(errors), 0,
            msg="chol=999은 범위 초과로 오류를 반환해야 함",
        )
        self.assertTrue(
            any("chol" in e for e in errors),
            msg=f"오류 메시지에 'chol'이 없음: {errors}",
        )

    # ── 테스트 4 ──────────────────────────────────────────────────────────────
    def test_pipeline_is_deterministic(self):
        """동일 입력을 두 번 통과시켜도 완전히 동일한 출력이 나와야 한다."""
        pred1  = self.pipeline.predict(self.X_sample)
        pred2  = self.pipeline.predict(self.X_sample)
        proba1 = self.pipeline.predict_proba(self.X_sample)
        proba2 = self.pipeline.predict_proba(self.X_sample)

        np.testing.assert_array_equal(pred1, pred2, err_msg="predict() 결과가 비결정론적")
        np.testing.assert_array_equal(proba1, proba2, err_msg="predict_proba() 결과가 비결정론적")


if __name__ == "__main__":
    unittest.main(verbosity=2)

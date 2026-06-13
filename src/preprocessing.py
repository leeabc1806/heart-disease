"""
재사용 가능한 전처리 파이프라인.
반드시 학습 데이터에만 fit하고, 테스트/추론 데이터엔 transform만 적용한다.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 14개 특성 중 연속형 vs 범주형 구분
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor() -> ColumnTransformer:
    """
    연속형: 중앙값 대치 → StandardScaler
    범주형: 최빈값 대치 → OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


# 임상적으로 유효한 특성 범위 (단위: 원본 데이터 기준)
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":      (1,   120),
    "trestbps": (60,  300),
    "chol":     (1,   600),
    "thalach":  (50,  250),
    "oldpeak":  (-5,  10),
}

CATEGORICAL_VALUES: dict[str, set[float]] = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def validate_input(df: pd.DataFrame) -> list[str]:
    """필수 컬럼, 숫자 형식, 임상 범위와 범주값을 검증한다."""
    errors = []
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"필수 컬럼 누락: {missing}")

    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        invalid_type = df[col].notna() & values.isna()
        if invalid_type.any():
            errors.append(f"{col}: 숫자가 아닌 값 {int(invalid_type.sum())}개")
        mask = values.notna() & ((values < lo) | (values > hi))
        n = int(mask.sum())
        if n > 0:
            errors.append(f"{col}: {n}행이 [{lo}, {hi}] 범위 초과")

    for col, allowed in CATEGORICAL_VALUES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        invalid_type = df[col].notna() & values.isna()
        if invalid_type.any():
            errors.append(f"{col}: 숫자가 아닌 값 {int(invalid_type.sum())}개")
        invalid_value = values.notna() & ~values.isin(allowed)
        if invalid_value.any():
            bad = sorted(values[invalid_value].unique().tolist())
            errors.append(f"{col}: 허용되지 않은 값 {bad}")
    return errors


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """fit된 preprocessor에서 출력 특성명 목록 반환."""
    num_names = NUMERIC_FEATURES.copy()
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names

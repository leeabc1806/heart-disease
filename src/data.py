"""
Heart Disease 데이터 로더
4개 기관(Cleveland, Hungarian, Switzerland, VA) processed 파일을 합쳐
이진 분류용 DataFrame을 반환한다.
결측치는 '?'로 표기돼 있으며 NaN으로 변환한다.
타깃: num=0 → 0(정상), num>=1 → 1(심장병)
"""
import os
import numpy as np
import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

SOURCES = {
    "cleveland":  "processed.cleveland.data",
    "hungarian":  "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va":         "processed.va.data",
}

# UCI 원본에서 0은 실제 측정값이 아니라 미측정을 뜻하는 경우가 있다.
# 이 세 항목은 생리적으로 0이 될 수 없으므로 결측치로 정규화한다.
ZERO_AS_MISSING = ["trestbps", "chol", "thalach"]


def load_raw(data_dir: str = None) -> pd.DataFrame:
    """4개 데이터셋을 합쳐 원본 DataFrame 반환 (결측=NaN, 타깃 미이진화)."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..")

    frames = []
    for source, filename in SOURCES.items():
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
        df[COLUMNS] = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
        df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
        df["source"] = source
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_data(data_dir: str = None) -> pd.DataFrame:
    """타깃 이진화까지 완료한 DataFrame 반환 (num: 0→0, 1-4→1)."""
    df = load_raw(data_dir)
    df = df.dropna(subset=["num"]).copy()
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(df["target"].value_counts(normalize=True).round(3))
    print(df.head())

# CardioCare — Heart Disease Prediction

임상 데이터로부터 심장병 발병 가능성을 예측하는 종단간 ML 시스템.

> **윤리적 관점**: 이 시스템은 심장 전문의의 의사결정을 **보조**하는 도구입니다.
> 예측 결과는 참고용이며, 진단 및 치료 결정은 반드시 의료 전문가가 내려야 합니다.

---

## 데이터셋

UCI Heart Disease Dataset — Cleveland, Hungarian, Switzerland, VA 4개 기관 통합 (920행)  
출처: https://archive.ics.uci.edu/dataset/45/heart+disease

원본 파일(`processed.*.data`)이 저장소 루트에 포함되어 있습니다.

---

## 전체 재현 절차

```bash
# 1. 저장소 클론
git clone <repo-url>
cd heart+disease   # 또는 프로젝트 루트

# 2. 의존성 설치
pip install -r requirements.txt

# 3. EDA 노트북 실행 (선택)
cd notebooks
jupyter notebook 01_eda_preprocessing.ipynb
cd ..

# 4. 모델 학습 (MLflow 기록 포함)
python src/train.py

# 5. 단위 테스트 실행
python -m unittest discover -s tests -v

# 6. Docker 이미지 빌드 및 추론 실행
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0

# 사용자 정의 입력 파일로 추론
docker run --rm -v $(pwd)/data:/app/data cardiocare:1.0 \
    --input data/sample_input.csv

# 7. MLflow UI 확인
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

---

## 저장소 구조

```
├── data/
│   └── sample_input.csv          # 추론용 샘플 입력
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── data.py                   # 데이터 로더
│   ├── preprocessing.py          # sklearn Pipeline
│   ├── train.py                  # 학습 & MLflow 추적
│   ├── inference.py              # 배치 추론
│   └── monitor.py                # 드리프트 탐지
├── tests/
│   └── test_pipeline.py          # unittest 4개
├── models/
│   └── best_model.pkl            # 최종 모델 (train.py 실행 후 생성)
├── mlflow.db                     # MLflow SQLite 백엔드
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

## 모델 성능 요약

| 모델 | Balanced Accuracy | Recall | F1 |
|------|:-----------------:|:------:|:--:|
| Logistic Regression | 0.820 | 0.824 | 0.836 |
| **SVC (최종)** | **0.825** | **0.833** | **0.842** |
| Random Forest | 0.802 | 0.824 | 0.824 |
| Random Forest (tuned) | 0.808 | 0.824 | 0.828 |

**최종 모델: SVC** — recall 최고(0.833), False Negative 최소화

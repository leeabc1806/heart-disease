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
cd heart-disease

# 2. 의존성 설치
pip install -r requirements.txt

# 3. EDA 노트북 실행 (선택)
cd notebooks
jupyter notebook 01_eda_preprocessing.ipynb
cd ..

# 4. 모델 학습 (60/20/20 train/validation/test, MLflow 기록 포함)
python src/train.py

# 5. 단위 테스트 실행
python -m unittest discover -s tests -v

# 6. 드리프트 시뮬레이션 및 보고서 재생성
python src/monitor.py
python src/build_report.py

# 7. Docker 이미지 빌드 및 추론 실행
# ※ models/best_model.pkl이 저장소에 포함돼 있어 train.py 없이도 바로 빌드 가능
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0

# 사용자 정의 입력 파일로 추론
docker run --rm -v $(pwd)/data:/app/data cardiocare:1.0 \
    --input data/sample_input.csv

# 8. MLflow UI 확인 (학습 후 mlflow.db와 mlruns/ 생성)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

---

## 저장소 구조

```
├── data/
│   ├── sample_input.csv           # 추론용 샘플 입력
│   ├── training_summary.json      # 재현된 validation/test 결과
│   └── monitoring_summary.json    # 드리프트 시뮬레이션 결과
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── data.py                   # 데이터 로더
│   ├── preprocessing.py          # sklearn Pipeline
│   ├── train.py                  # 학습 & MLflow 추적
│   ├── inference.py              # 배치 추론
│   └── monitor.py                # 드리프트 탐지
├── tests/
│   └── test_pipeline.py          # unittest 7개
├── models/
│   └── best_model.pkl            # 학습된 최종 모델 (저장소에 포함 — 즉시 사용 가능)
├── docs/
│   ├── feature_store_and_registry.md
│   └── serving_and_retraining_strategy.md
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-runtime.txt      # 추론 컨테이너 최소 의존성
├── report.pdf                    # 최종 보고서
├── .github/workflows/ci.yml
└── README.md
```

---

## 모델 선택 및 최종 성능

후보 모델은 validation set으로만 비교하며 test set은 최종 모델 결정 후 한 번만 사용합니다.

| 모델 | Validation Balanced Accuracy | Validation Recall | Validation F1 | CV Mean |
|------|:----------------------------:|:-----------------:|:-------------:|:-------:|
| **Logistic Regression (선택)** | **0.801** | **0.833** | **0.825** | 0.771 |
| SVC | 0.773 | 0.814 | 0.802 | 0.754 |
| Random Forest | 0.778 | 0.824 | 0.808 | 0.775 |
| Random Forest (tuned) | 0.778 | 0.824 | 0.808 | **0.790** |

**독립 test 성능**: balanced accuracy 0.780, precision 0.804, recall 0.804, F1 0.804
혼동행렬: TN=62, FP=20, FN=20, TP=82

**Leave-one-site-out 검증**: balanced accuracy 0.728 ± 0.054. 기관별 분포 차이가 커서
무작위 분할 성능보다 낮으며, 새 병원 적용 전 외부 검증이 필요합니다.

---

## 특성 선택 결과

`SelectFromModel(RandomForestClassifier, threshold='mean')` — OHE 후 **9개 선택**

| 선택 컬럼 | 원본 특성 / 의미 | RF 중요도 |
|-----------|----------------|:---------:|
| age | 나이 (세) | 0.136 |
| thalach | 최대 심박수 (bpm) | 0.117 |
| chol | 혈중 콜레스테롤 (mg/dl) | 0.096 |
| oldpeak | 운동 ST 하강 (mm) | 0.080 |
| cp_4.0 | 흉통 유형 — 무증상 | 0.079 |
| trestbps | 안정 시 혈압 (mmHg) | 0.074 |
| exang_0.0 | 운동유발 협심증 없음 | 0.059 |
| exang_1.0 | 운동유발 협심증 있음 | 0.057 |
| cp_2.0 | 흉통 유형 — 비전형 협심증 | 0.040 |

5개 연속형(chol·age·thalach·oldpeak·trestbps) 전부와 흉통 유형·운동유발 협심증이 선택됨.
EDA에서 타깃 분리가 컸던 변수들과 일치하며, 임상 의미가 있는 특성만 남음.

원본에서 생리적으로 불가능한 `trestbps=0`, `chol=0`, `thalach=0`은 결측 sentinel로 간주해
`NaN`으로 정규화한 뒤, 학습 fold의 중앙값으로만 대치합니다.

---

## 서빙 및 재학습 전략 요약

자세한 내용: [`docs/serving_and_retraining_strategy.md`](docs/serving_and_retraining_strategy.md)

- **서빙 목표**: Model-as-a-Service (MaaS). 현재 구현은 검증이 포함된 CSV 배치 추론 CLI이며,
  중앙 추론 API는 운영 확장안입니다.
- **재학습 트리거**: KS p < 0.05 지속 또는 balanced accuracy 임계치 이하 → 재학습 트리거
- **정기 재학습**: 트리거 외 분기별 최신 데이터로 점진적 드리프트 대응
- **Human-in-the-loop**: 경계 확률(0.4–0.6) 구간 의사 검토 라우팅, 재학습 모델은 recall 게이팅 후 담당자 승인 필요

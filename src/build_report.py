"""
report.pdf 생성 스크립트 (reportlab).
EDA/학습/드리프트 결과와 그림을 결합해 8개 섹션 + AI 사용 부록으로 구성.

실행: python src/build_report.py  →  report.pdf
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_PATH = os.path.join(PROJECT_ROOT, "report.pdf")

# ── 한글 폰트 등록 ──────────────────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBold", "C:/Windows/Fonts/malgunbd.ttf"))

# ── 스타일 ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
TITLE = ParagraphStyle("KTitle", parent=styles["Title"], fontName="MalgunBold",
                       fontSize=20, leading=26, spaceAfter=4)
SUB = ParagraphStyle("KSub", parent=styles["Normal"], fontName="Malgun",
                     fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.grey,
                     spaceAfter=18)
H1 = ParagraphStyle("KH1", parent=styles["Heading1"], fontName="MalgunBold",
                    fontSize=14, leading=18, spaceBefore=14, spaceAfter=6,
                    textColor=colors.HexColor("#1a3e6e"))
H2 = ParagraphStyle("KH2", parent=styles["Heading2"], fontName="MalgunBold",
                    fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("KBody", parent=styles["Normal"], fontName="Malgun",
                      fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("KBullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CAPTION = ParagraphStyle("KCap", parent=styles["Normal"], fontName="Malgun",
                         fontSize=8.5, leading=11, alignment=TA_CENTER,
                         textColor=colors.grey, spaceBefore=2, spaceAfter=10)
CELL = ParagraphStyle("KCell", parent=BODY, fontSize=8.5, leading=11, alignment=TA_LEFT, spaceAfter=0)


def P(text, style=BODY):
    return Paragraph(text, style)


def fig(path, width_cm, caption):
    full = os.path.join(DATA_DIR, path)
    img = Image(full)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    img.hAlign = "CENTER"
    return [img, P(caption, CAPTION)]


def make_table(data, col_widths, header_bg="#1a3e6e", body_font="Malgun"):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "MalgunBold"),
        ("FONTNAME", (0, 1), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="CardioCare Final Report",
    )
    s = []  # story

    # ── 표지 ──
    s.append(Spacer(1, 4 * mm))
    s.append(P("CardioCare — 심장병 예측을 위한 종단간 ML 시스템", TITLE))
    s.append(P("End-to-End Machine Learning System for Heart Disease Prediction", SUB))
    s.append(P("기계학습 기말 프로젝트 · 컴퓨터공학과", SUB))
    s.append(Spacer(1, 4 * mm))

    # ── 1. 문제 정의 ──
    s.append(P("1. 문제 정의와 사용 목적", H1))
    s.append(P(
        "CardioCare는 UCI Heart Disease 데이터셋의 임상 변수(나이, 혈압, 콜레스테롤, "
        "최대 심박수, ST 하강 등 13개)로부터 심장병 발병 가능성을 예측하여 "
        "<b>심장 전문의의 의사결정을 보조</b>하는 시스템이다. 출력은 0/1 분류와 "
        "위험 확률값이며, 이는 추가 검사가 필요한 환자를 우선 선별하는 <b>참고 신호</b>로 사용된다.", BODY))
    s.append(P(
        "<b>알리되, 결정하지 않는다 (inform, not decide).</b> 이 시스템은 진단을 내리거나 "
        "치료를 결정하지 않는다. 최종 판단은 반드시 의료 전문가가 환자의 전체 맥락을 "
        "고려해 내린다. 모델은 그 판단을 위한 한 가지 입력일 뿐이며, 단독 의사결정 도구로 "
        "쓰여서는 안 된다(§8 윤리 참조).", BODY))
    s.append(P(
        "임상적으로 <b>False Negative(심장병 환자를 정상으로 오분류)</b>는 환자를 치료 기회로부터 "
        "배제하는 치명적 결과를 낳는다. 반면 False Positive는 추가 검사라는 비교적 가벼운 비용을 "
        "수반한다. 따라서 본 시스템은 recall(민감도)을 우선 지표로 설계했다.", BODY))

    # ── 2. EDA 핵심 결과 ──
    s.append(P("2. EDA 핵심 결과", H1))
    s.append(P(
        "Cleveland·Hungarian·Switzerland·VA 4개 기관 데이터를 통합해 <b>920행 × 13특성</b>을 "
        "구성하고, 다중 클래스 타깃(num 0~4)을 이진화(0=정상, ≥1=심장병)했다. "
        "타깃 분포는 심장병 55.3% vs 정상 44.7%로 <b>완만한 불균형(약 1.24:1)</b>이다. "
        "극단적이지 않으나, 임상적 비용 비대칭 때문에 단순 accuracy 대신 "
        "<b>balanced accuracy와 recall</b>을 핵심 지표로 채택하는 근거가 된다.", BODY))
    s += fig("fig_target_dist.png", 13.5, "그림 1. 타깃 클래스 분포 — 심장병(1) 55.3%, 정상(0) 44.7%")
    s.append(P(
        "결측값은 기관별로 패턴이 뚜렷하다. <b>slope·ca·thal</b>은 Switzerland/VA에서 결측 비율이 "
        "50~90%에 달하며, Switzerland의 <b>chol</b>은 다수가 0(측정 누락)으로 기록돼 있다. "
        "연속형 특성 boxplot에서는 chol과 trestbps에 이상치가 관찰된다.", BODY))
    s += fig("fig_boxplots.png", 14.5, "그림 2. 연속형 특성 Boxplot — chol·trestbps의 이상치 및 chol=0 기록 누락 확인")

    # ── 3. 전처리 결정 ──
    s.append(P("3. EDA 결과에 근거한 전처리 결정", H1))
    s.append(P(
        "EDA에서 드러난 결측·이상치 패턴을 근거로, 정보 손실을 최소화하면서 "
        "<b>데이터 누수 없이</b> 새 데이터에 재적용 가능한 <font name='MalgunBold'>sklearn.Pipeline</font>을 "
        "구성했다. 모든 변환기는 학습 fold에만 fit한다.", BODY))
    pre_tbl = [
        ["항목", "결정", "근거"],
        ["연속형 결측", "중앙값 대치", "이상치·왜도에 강건"],
        ["범주형 결측", "최빈값 대치", "가장 흔한 임상 상태로 보수적 대치"],
        ["스케일링", "StandardScaler", "LR·SVC에 필요, 학습 fold에만 fit (누수 방지)"],
        ["범주형 인코딩", "OneHotEncoder", "순서 없는 명목형 변수"],
        ["이상치", "삭제 안 함", "920행 한정 데이터, 대치로 영향만 완화"],
    ]
    s.append(make_table(pre_tbl, [3.2 * cm, 3.5 * cm, 8.3 * cm]))
    s.append(Spacer(1, 4 * mm))
    s.append(P(
        "<b>핵심: 데이터 누수 방지.</b> 스케일러·임퓨터·특성 선택기는 train/test 분할 "
        "이후 학습 데이터에만 fit하고, 테스트·추론 데이터에는 transform만 적용한다. "
        "교차 검증 시에도 Pipeline 전체가 각 fold마다 새로 fit되어 검증 fold의 정보가 "
        "전처리에 새지 않는다.", BODY))

    s.append(PageBreak())

    # ── 4. 모델 비교 ──
    s.append(P("4. 모델링 · MLflow 실험 및 최종 선택", H1))
    s.append(P(
        "데이터를 80/20으로 분할(시드 42, stratify)한 뒤, RandomForest 기반 "
        "<font name='MalgunBold'>SelectFromModel</font>(threshold='mean')로 특성을 선택했다. "
        "OHE 변환 후 28차원 중 <b>중요도 평균(0.036) 이상인 9개</b>가 선택됐다(표 1). "
        "5개 연속형(chol·age·thalach·oldpeak·trestbps) 전부와 흉통 유형(cp_4·cp_2)·"
        "운동유발 협심증(exang)이 포함됐으며, 이는 EDA에서 타깃과의 차이가 컸던 변수들과 일치한다.", BODY))
    feat_tbl = [
        ["선택 컬럼", "원본 특성 / 의미", "RF 중요도"],
        ["chol", "혈중 콜레스테롤 (mg/dl)", "0.132"],
        ["age", "나이 (세)", "0.113"],
        ["thalach", "최대 심박수 (bpm)", "0.106"],
        ["oldpeak", "운동 ST 하강 (mm)", "0.086"],
        ["cp_4.0", "흉통 유형 — 무증상", "0.082"],
        ["trestbps", "안정 시 혈압 (mmHg)", "0.073"],
        ["exang_0.0", "운동유발 협심증 없음", "0.062"],
        ["exang_1.0", "운동유발 협심증 있음", "0.054"],
        ["cp_2.0", "흉통 유형 — 비전형 협심증", "0.041"],
        ["(나머지 19개)", "sex·fbs·restecg·ca·thal·slope 등", "< 0.036 → 탈락"],
    ]
    s.append(make_table(feat_tbl, [3.5 * cm, 8.0 * cm, 3.5 * cm]))
    s.append(P(
        "표 1. RandomForest SelectFromModel 특성 선택 결과 — "
        "중요도 평균(threshold='mean', 0.036) 이상인 9개 선택, 나머지 19개 탈락.", CAPTION))
    s.append(P(
        "Logistic Regression·SVC·Random Forest 3개 계열을 학습하고, 모든 실행을 "
        "<b>MLflow</b>(SQLite 백엔드)에 파라미터·지표·모델 아티팩트·계열 태그와 함께 기록했다. "
        "전 모델에 5-fold 교차 검증을, 가장 유력한 RandomForest에는 RandomizedSearchCV"
        "(n_iter=20) 하이퍼파라미터 탐색을 적용했다.", BODY))
    model_tbl = [
        ["모델", "Bal. Acc", "Precision", "Recall", "F1", "CV Mean"],
        ["Logistic Regression", "0.820", "0.849", "0.824", "0.836", "0.779"],
        ["SVC  (최종 선택)", "0.825", "0.850", "0.833", "0.842", "0.797"],
        ["Random Forest", "0.802", "0.824", "0.824", "0.824", "0.777"],
        ["Random Forest (tuned)", "0.808", "0.832", "0.824", "0.828", "0.789"],
    ]
    t = make_table(model_tbl, [4.3 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 1.8 * cm, 1.9 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff3cd"))]))
    s.append(t)
    s.append(P("표 2. 테스트셋 모델 성능 비교 (MLflow 기록). 강조 행이 최종 선택 모델.", CAPTION))
    s.append(P(
        "최종 선택의 핵심 근거는 <b>혼동행렬의 False Negative</b>다. 아래 표는 4개 모델의 "
        "테스트셋(184건) 혼동행렬을 분해한 것이다.", BODY))
    cm_tbl = [
        ["모델", "TN", "FP", "FN ↓", "TP"],
        ["Logistic Regression", "67", "15", "18", "84"],
        ["SVC  (최종 선택)", "67", "15", "17", "85"],
        ["Random Forest", "64", "18", "18", "84"],
        ["Random Forest (tuned)", "65", "17", "18", "84"],
    ]
    t = make_table(cm_tbl, [4.3 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff3cd")),
        ("BACKGROUND", (3, 0), (3, -1), colors.HexColor("#f8d7da")),
        ("TEXTCOLOR", (3, 0), (3, 0), colors.white),
    ]))
    s.append(t)
    s.append(P("표 3. 모델별 혼동행렬 분해 — FN(심장병 미탐지) 열 강조. SVC가 FN=17로 최소.", CAPTION))
    s.append(P(
        "<b>최종 모델: SVC.</b> 임상적으로 가장 중요한 recall(0.833)이 최고이며, balanced "
        "accuracy(0.825)·F1(0.842)·CV 평균(0.797)도 가장 우수하다. 무엇보다 <b>False "
        "Negative가 17건으로 4개 모델 중 가장 적다</b> — 심장병 환자를 놓치는 빈도가 가장 낮다는 뜻이다. "
        "심장병 미탐지가 치명적인 본 문제에서 FN 최소화는 결정적 선택 기준이며, precision(0.850)도 "
        "충분히 높아 과도한 오경보 없이 균형을 유지한다.", BODY))

    # ── 5. 테스트 · 패키징 ──
    s.append(P("5. 테스트와 패키징 — 무엇을, 왜", H1))
    s.append(P("<b>단위 테스트 (unittest 4개)</b> — 사소하지 않은 실제 버그를 잡도록 설계:", BODY))
    test_tbl = [
        ["테스트", "검증 내용 / 잡아내는 버그"],
        ["1. 예측 shape", "predict() 행 수 = 입력 행 수. 배치 처리 시 행 누락/중복 탐지"],
        ["2. 확률 범위·합", "predict_proba ∈ [0,1], 행합 ≈ 1. 보정·인코딩 깨짐 탐지"],
        ["3. 입력 범위 검증", "chol∈[0,600] 등 임상 범위 초과 입력 차단 (잘못된 단위/오타)"],
        ["4. 결정론", "고정 시드에서 동일 입력 → 동일 출력. 재현성·시드 누락 탐지"],
    ]
    s.append(make_table(test_tbl, [3.4 * cm, 11.6 * cm]))
    s.append(Spacer(1, 3 * mm))
    s.append(P(
        "<b>Docker.</b> python:3.13-slim 기반으로 requirements 설치 → 코드·모델 복사 → "
        "추론 엔트리포인트를 실행한다. <font name='Malgun'>docker build -t cardiocare:1.0 .</font> 빌드 후 "
        "샘플 배치 입력으로 정상 추론된다. 비밀값은 포함하지 않는다.", BODY))
    s.append(P(
        "<b>CI (GitHub Actions).</b> 모든 push에서 의존성 설치 후 unittest를 실행한다. "
        "main 브랜치의 green 유지를 목표로 한다.", BODY))
    s.append(P(
        "<b>피처 스토어 / 모델 레지스트리.</b> 피처 스토어에는 <b>oldpeak</b>(ST 하강)를 등록한다 — "
        "타깃 상관이 높고 특성 선택에서 항상 채택되는 안정적 피처로, ECG 장비에서 자동 갱신되므로 "
        "최신성 관리가 필요하다. 모델 레지스트리에는 <b>recall 값</b>을 메타데이터로 기록해, "
        "'기존 배포 모델보다 recall이 낮은 버전은 Production 승격 불가'라는 안전 게이팅을 자동화한다.", BODY))

    s.append(PageBreak())

    # ── 6. 드리프트 · 재학습 ──
    s.append(P("6. 드리프트 결과 및 재학습 / 피드백 루프", H1))
    s.append(P(
        "테스트셋의 <b>chol 평균을 +30, 분산을 1.5배</b>로 인위 이동시킨 뒤, 각 연속형 특성에 대해 "
        "학습 분포와 <font name='MalgunBold'>scipy.stats.ks_2samp</font>를 수행했다.", BODY))
    ks_tbl = [
        ["특성", "KS 통계량", "p-value", "드리프트 플래그"],
        ["age", "0.041", "0.96", "—"],
        ["trestbps", "0.037", "0.99", "—"],
        ["chol", "0.286", "6.6e-11", "플래그 (p<0.05)"],
        ["thalach", "0.058", "0.71", "—"],
        ["oldpeak", "0.041", "0.96", "—"],
    ]
    t = make_table(ks_tbl, [3.2 * cm, 3.0 * cm, 3.5 * cm, 5.3 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#f8d7da"))]))
    s.append(t)
    s.append(P("표 4. KS 검정 결과 — 드리프트를 주입한 chol만 정확히 플래그됨.", CAPTION))
    s.append(P(
        "검정은 드리프트를 주입한 chol만 정확히 탐지했고(p=6.6e-11), 손대지 않은 특성은 모두 "
        "p≥0.05로 플래그되지 않아 <b>오탐 없이 작동</b>함을 확인했다. 입력 드리프트는 성능 저하로 "
        "이어져, balanced accuracy가 <b>0.825 → 0.807</b>로 떨어졌다(단일 이동 기준 −2.2%).", BODY))
    s += fig("fig_drift_monitoring.png", 13.5,
             "그림 3. 합성 타임스탬프 기반 드리프트 모니터링 — chol 평균이 점증할수록 "
             "balanced accuracy 하락(상단), KS p-value 급락(하단, 로그 스케일).")
    s.append(P("<b>재학습 / 피드백 전략</b>", H2))
    s.append(P("• <b>트리거 기반 재학습</b>: 핵심 특성의 KS p&lt;0.05가 일정 기간 지속되거나, "
               "라벨 확보 후 balanced accuracy가 임계치 이하로 떨어지면 재학습을 트리거한다.", BULLET))
    s.append(P("• <b>정기 재학습</b>: 트리거가 없어도 분기별로 최신 임상 데이터로 재학습해 "
               "점진적 드리프트에 대비한다.", BULLET))
    s.append(P("• <b>폭주 피드백 루프 위험</b>: 모델 예측을 그대로 라벨로 재사용하면, 모델이 "
               "놓친 양성(FN)이 학습 데이터에서도 음성으로 굳어져 <b>recall이 자기강화적으로 악화</b>된다. "
               "이를 막으려면 재학습 라벨은 모델 출력이 아니라 <b>의사의 확진/추적 결과 같은 실제 정답</b>만 "
               "사용해야 한다.", BULLET))
    s.append(P("• <b>Human-in-the-loop 지점</b>: (1) 경계 확률(예: 0.4~0.6) 구간은 의사 검토로 라우팅, "
               "(2) 재학습된 모델은 recall 게이팅 통과 후 사람이 승인해야 배포, "
               "(3) 드리프트 알람은 자동 재배포가 아니라 담당자에게 통지된다.", BULLET))

    s.append(Spacer(1, 4 * mm))

    # ── 7. 서빙 ──
    s.append(P("7. 서빙 선택: Model-as-a-Service", H1))
    s.append(P(
        "본 시스템은 <b>Model-as-a-Service(MaaS)</b>를 선택한다. 병원 EMR이 환자 임상값을 "
        "중앙 추론 API로 전송하면 위험 점수를 반환하는 구조다.", BODY))
    serve_tbl = [
        ["기준", "MaaS 선택 근거"],
        ["지연 시간", "심장병 선별은 실시간(ms) 요구가 아닌 진료 중 수초 내 응답으로 충분 → 서버 추론 적합"],
        ["개인정보(PHI)", "모델·로그를 병원 통제하의 보안 서버에 중앙 집중 → 감사·접근통제·규정 준수 용이. "
                      "단, 전송 구간 암호화와 PHI 최소 수집은 필수"],
        ["업데이트 주기", "재학습 모델을 단일 엔드포인트에 즉시 배포 → 전 사용자가 동일 최신 버전 사용, "
                      "기기별 배포 불필요"],
    ]
    s.append(make_table(serve_tbl, [3.0 * cm, 12.0 * cm]))
    s.append(Spacer(1, 3 * mm))
    s.append(P(
        "On-Device는 오프라인 동작과 데이터 미전송이라는 장점이 있으나, 모델 갱신 배포가 "
        "기기마다 분산되고 버전 파편화·드리프트 모니터링이 어렵다. 병원 환경에서는 중앙 통제와 "
        "감사 가능성이 더 중요하므로 MaaS가 적합하다.", BODY))

    # ── 8. 한계 · 윤리 ──
    s.append(P("8. 한계, 윤리적 고려, 추가 작업", H1))
    s.append(P("<b>한계</b>", H2))
    s.append(P("• 표본이 920행으로 작고 4개 기관에 한정 → 다른 인구집단으로의 일반화가 제한적이다.", BULLET))
    s.append(P("• slope·ca·thal의 높은 결측을 대치로 메웠으므로, 해당 특성의 신호가 약화됐을 수 있다.", BULLET))
    s.append(P("• 1999년 전후 데이터로, 현재 임상 분포와 시점 드리프트가 존재할 수 있다.", BULLET))
    s.append(P("<b>윤리적 고려</b>", H2))
    s.append(P(
        "CardioCare는 <b>의사를 대체하지 않고 보조</b>한다(inform, not decide). 오분류의 비용이 "
        "환자 안전과 직결되므로, 예측은 항상 의료 전문가의 검토를 거쳐야 한다. 성별·연령 등 "
        "민감 변수에 따른 성능 편향 점검과, 환자에게 'AI 보조 사용' 사실을 고지하는 절차가 "
        "운영 단계에서 필요하다. recall을 우선했어도 놓치는 환자(FN)는 여전히 존재하므로, "
        "음성 결과가 곧 안전을 보장하지 않음을 사용자에게 명확히 전달해야 한다.", BODY))
    s.append(P("<b>1주가 더 주어진다면</b>", H2))
    s.append(P("• 결정 임계값을 0.5가 아닌 recall 목표(예: ≥0.90)에 맞춰 보정하고 PR 곡선으로 운영점 선택.", BULLET))
    s.append(P("• SHAP 기반 설명을 추가해 의사가 예측 근거를 검토할 수 있게 함.", BULLET))
    s.append(P("• 성별·연령 하위 그룹별 공정성 지표(subgroup recall)를 측정하고 보고.", BULLET))
    s.append(P("• MLflow Model Registry와 실제 드리프트 대시보드를 연동.", BULLET))

    # ── 부록: AI 사용 공개 ──
    s.append(P("부록 A. AI 도구 사용 공개", H1))
    s.append(P(
        "본 프로젝트에서 AI 코딩 도우미(Claude)를 <b>보일러플레이트 작성과 디버깅 보조</b> 용도로 "
        "사용했다. 구체적으로는 sklearn Pipeline·MLflow·unittest·Dockerfile의 표준 구조 초안 작성, "
        "reportlab 보고서 레이아웃, pandas 버전 호환 오류 디버깅에 활용했다. "
        "데이터 전처리·모델 선택·지표 해석·드리프트 전략 등 모든 핵심 의사결정과 결과 해석은 "
        "직접 검토·검증했으며, 제출한 모든 코드와 결과에 대해 본인이 책임진다.", BODY))

    doc.build(s)
    print(f"생성 완료: {OUT_PATH}")


if __name__ == "__main__":
    build()

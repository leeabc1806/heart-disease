FROM python:3.13-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# 소스 코드 복사
COPY src/ ./src/

# 학습된 모델 복사
COPY models/ ./models/

# 샘플 입력 데이터 복사
COPY data/sample_input.csv ./data/sample_input.csv

# 로그 디렉토리 생성
RUN mkdir -p logs

# 비루트 사용자로 추론 실행
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# 추론 엔트리포인트
ENTRYPOINT ["python", "src/inference.py"]
CMD ["--input", "data/sample_input.csv"]

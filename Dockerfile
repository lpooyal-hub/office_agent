FROM python:3.9-slim

# 시스템 라이브러리 설치 (ffmpeg 및 빌드 도구)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 업로드 폴더 생성
RUN mkdir -p uploads

EXPOSE 5001

# .py 확장자를 제거한 모듈 형태로 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
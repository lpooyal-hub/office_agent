# Office-Agent

RAG(Retrieval-Augmented Generation) 기반의 기업형 업무 지원 시스템

Office-Agent는 두 가지 핵심 기능을 제공합니다.

1. 회의나 통화 음성 파일을 업로드하면 STT로 텍스트를 추출하고, 핵심 요약과 Action Item을 정리합니다.
2. 회사 내규와 가이드라인 문서를 업로드하면 신입사원과 직장인을 위한 맞춤형 Q&A 챗봇으로 활용합니다.

---

## Architecture

Office-Agent는 로컬 임베딩 모델과 OpenAI API를 조합한 하이브리드 AI 파이프라인 구조입니다.

### 1. STT Processing
- OpenAI Whisper 기반 음성 텍스트 변환
- 기본값은 CPU 서버에서도 비교적 가볍게 동작하는 `base` 모델입니다.
- 화면에서 일반 모드(로컬 Whisper)와 고성능 모드(OpenAI STT API)를 선택할 수 있습니다.

### 2. RAG Retrieval
- Ko-SRoBERTa(`jhgan/ko-sroberta-multitask`) 기반 문서 임베딩
- 코사인 유사도 기반 내규 문서 검색
- PDF / DOCX / TXT / MD 문서 업로드 기반 RAG 지식 추가

### 3. LLM Generation / Chatbot
- OpenAI API 기반 회의록 및 Action Item 생성
- 검색된 내규 정보를 Prompt에 주입하여 생성 정확도 향상
- 신입사원용 회사 내규 Q&A 챗봇 제공

---

## Features

- Whisper 기반 STT 처리
- OpenAI STT 기반 고성능 모드
- Ko-SRoBERTa 기반 의미론적 문서 검색
- PDF / DOCX / TXT / MD 문서 업로드
- RAG 기반 회의록 생성
- 신입사원용 회사 내규 챗봇
- 로컬 기반 임베딩 처리 구조
- OCI ARM 서버 기반 운영 환경
- Docker 기반 서비스 실행 환경

---

## Tech Stack

### Backend
- Python
- FastAPI

### AI / NLP
- OpenAI Whisper
- Ko-SRoBERTa
- OpenAI API

### Infra
- OCI
- Docker
- Docker Compose

### Frontend
- Vanilla JavaScript
- Jinja2
- CSS3

---

## Project Structure

```text
office-agent/
├── main.py
├── templates/
├── uploads/
├── documents/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Run

### Install

```bash
pip install -r requirements.txt
```

### Environment

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
WHISPER_MODEL=base
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
```

`.env.example` 파일을 참고해 `.env`를 생성합니다.

### Start

```bash
python main.py
```

### Docker

```bash
docker compose up -d --build
```

서비스는 기본적으로 `http://localhost:5001`에서 실행됩니다.

### Health Check

```bash
curl http://localhost:5001/health
```

---

## Future Improvements

- Vector DB(FAISS / ChromaDB) 기반 검색 최적화
- 사용자 인증 및 권한 관리 기능 추가

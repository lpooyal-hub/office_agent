# Office-Agent

RAG(Retrieval-Augmented Generation) 기반의 사내 업무 지원 AI 서비스

Office-Agent는 회의 음성, 사내 문서, 반복 질문 응대를 한곳에서 다루기 위해 만든 업무용 AI 도구입니다.
단순히 모델을 붙이는 데서 끝내지 않고, 실제 데모 서비스처럼 문서 관리, 근거 제시, 결과 저장, 공개 사용 보호까지 고려해 다듬었습니다.
현재 업로드 문서 검색 저장소는 `ChromaDB`를 사용해, 서버 재시작 이후에도 문서 임베딩과 검색 인덱스를 유지하도록 구성했습니다.

이 프로젝트는 아래 세 가지 핵심 흐름을 제공합니다.

1. 회의나 통화 음성 파일을 업로드하면 STT로 텍스트를 추출하고, 핵심 요약과 Action Item을 정리합니다.
2. 회사 내규와 가이드라인 문서를 업로드하고 보관함처럼 관리하며, 이를 바탕으로 신입사원용 Q&A 챗봇을 제공합니다.
3. 문서를 일회성으로 업로드하면 핵심 요약, 주요 포인트, Action Item을 생성하고 결과를 복사/저장할 수 있습니다.

---

## Why This Project

- 회의록 정리, 사내 규정 검색, 문서 요약은 많은 팀에서 반복되지만 여전히 수작업 비중이 큽니다.
- 특히 데모 수준의 AI 프로젝트는 "질문에 답한다"에서 멈추는 경우가 많아, 실제 사용 흐름인 문서 축적, 근거 확인, 결과 재사용이 약한 편입니다.
- Office-Agent는 이런 간극을 줄이기 위해 음성 처리, RAG 검색, 문서형 결과물 생성을 하나의 업무 경험으로 연결하는 데 초점을 맞췄습니다.

---

## What I Improved

- 문서 업로드를 일회성 입력이 아니라 보관함 형태로 관리할 수 있도록 개선했습니다.
- RAG 결과를 단순 텍스트가 아니라 근거 문서 하이라이트 카드로 보여주도록 바꿨습니다.
- 요약/답변/스크립트를 바로 복사하거나 TXT로 저장할 수 있게 해 결과 활용성을 높였습니다.
- 업로드 검증, 용량 제한, 접속 코드, 요청 제한을 정리해 공개 데모 환경에서도 비교적 안전하게 사용할 수 있게 했습니다.
- `main.py`에 몰려 있던 로직을 `services` 단위로 분리해 이후 기능 확장 시 변경 영향 범위를 줄였습니다.
- 문서 검색 저장소를 메모리 기반에서 `ChromaDB` 기반으로 바꿔 문서 추가/삭제와 벡터 저장소를 함께 관리하도록 확장했습니다.

---

## Architecture

Office-Agent는 로컬 임베딩 모델, `ChromaDB`, OpenAI API를 조합한 하이브리드 AI 파이프라인 구조입니다.

### 1. STT Processing
- OpenAI Whisper 기반 음성 텍스트 변환
- 기본값은 CPU 서버에서도 비교적 가볍게 동작하는 `base` 모델입니다.
- 화면에서 일반 모드(로컬 Whisper)와 고성능 모드(OpenAI STT API)를 선택할 수 있습니다.

### 2. RAG Retrieval
- Ko-SRoBERTa(`jhgan/ko-sroberta-multitask`) 기반 문서 임베딩
- `ChromaDB` 기반 문서 chunk 저장 및 벡터 검색
- PDF / DOCX / TXT / MD 문서 업로드 기반 RAG 지식 추가
- 업로드 문서 보관함과 검색 근거 카드 제공
- 문서 추가/삭제 시 벡터 저장소 동기화

### 3. LLM Generation / Chatbot
- OpenAI API 기반 회의록 및 Action Item 생성
- 검색된 내규 정보를 Prompt에 주입하여 생성 정확도 향상
- 신입사원용 회사 내규 Q&A 챗봇 제공
- 문서 요약 결과 및 답변 결과 저장 지원

---

## Features

- Whisper 기반 STT 처리
- OpenAI STT 기반 고성능 모드
- Ko-SRoBERTa 기반 의미론적 문서 검색
- ChromaDB 기반 벡터 저장소 연동
- PDF / DOCX / TXT / MD 문서 업로드
- 업로드 문서 보관함 조회 / 개별 삭제 / 전체 초기화
- RAG 기반 회의록 생성
- 신입사원용 회사 내규 챗봇
- 근거 문서 하이라이트 카드 제공
- AI 문서 요약 (최대 5개 파일, 업로드 후 자동 삭제)
- 결과 복사 / TXT 저장 지원
- 로컬 기반 임베딩 처리 구조
- 접속 코드, 요청 제한, 업로드 용량 제한 기반 공개 데모 보호
- OCI ARM 서버 기반 운영 환경
- Docker 기반 서비스 실행 환경

---

## Portfolio Points

- 단일 기능 데모가 아니라 "문서 업로드 → 지식 축적 → 검색 → 생성 → 결과 활용" 흐름을 하나의 서비스 경험으로 설계했습니다.
- RAG 시스템에서 자주 빠지는 근거 가시성과 문서 관리 UX를 함께 다뤘습니다.
- 공개 시연을 고려해 접근 제어와 요청 제한을 포함한 운영 관점도 반영했습니다.
- 기능 추가 시 충돌을 줄이기 위해 AI 호출, 문서 관리, 검색 상태를 서비스 단위로 분리했습니다.
- 운영 중인 별도 `ChromaDB` 컨테이너를 활용해 벡터 저장소를 애플리케이션과 분리했습니다.

---

## Tech Stack

### Backend
- Python
- FastAPI

### AI / NLP
- OpenAI Whisper
- Ko-SRoBERTa
- OpenAI API
- ChromaDB

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
├── frontend/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   └── vite.config.mjs
├── main.py
├── services/
│   ├── ai_client.py
│   ├── chroma_store.py
│   ├── document_library.py
│   └── rag_service.py
├── static/
│   └── app/
├── templates/
│   └── index.html
├── uploads/
├── documents/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── tests/
│   └── test_document_summary.py
└── requirements.txt
```

---

## Run

### Install

```bash
pip install -r requirements.txt
```

```bash
cd frontend
npm install
```

### Frontend Build

```bash
cd frontend
npm run build
```

### Environment

```env
OPENAI_API_KEY=your_api_key
CHROMA_URL=http://localhost:9001
CHROMA_TENANT=default_tenant
CHROMA_DATABASE=default_database
CHROMA_COLLECTION=office_agent_documents
CHROMA_TIMEOUT_SECONDS=15
CHROMA_TOKEN=
OPENAI_MODEL=gpt-4o-mini
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
WHISPER_MODEL=base
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
ACCESS_CODE=change_this_demo_password
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=3600
MAX_AUDIO_UPLOAD_MB=25
MAX_DOCUMENT_UPLOAD_MB=10
MAX_SUMMARY_DOCUMENTS=5
```

`.env.example` 파일을 참고해 `.env`를 생성합니다.

`ACCESS_CODE`는 공개 데모 페이지에서 OpenAI API 비용이 발생하는 기능을 보호하기 위한 접속 코드입니다.
Docker 환경에서는 `CHROMA_URL`을 `http://host.docker.internal:9001`로 두면, 현재 호스트에서 운영 중인 ChromaDB 컨테이너에 연결할 수 있습니다.

### Start

```bash
python main.py
```

### Test

```bash
python -m unittest tests/test_document_summary.py
```

또는 `pytest`를 사용하는 경우 다음 명령으로 실행할 수 있습니다.

```bash
pytest
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

- ChromaDB 메타데이터 필터와 재랭킹 로직 고도화
- 사용자 인증 및 권한 관리 기능 추가

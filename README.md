# Office-Agent

RAG(Retrieval-Augmented Generation) 기반의 기업형 회의록 분석 시스템

업로드된 음성 데이터를 STT로 변환한 뒤,
사내 내규 및 가이드라인 문서를 실시간으로 참조하여
회의 내용 분석 및 Action Item 생성을 수행하는 AI 서비스입니다.

로컬 시맨틱 검색 기반 구조를 활용하여
LLM의 Hallucination 문제와 정보 최신성 한계를 보완하였습니다.

---

## Architecture

Office-Agent는 로컬 AI 모델과 클라우드 LLM을 조합한
하이브리드 AI 파이프라인 구조로 구성되어 있습니다.

### 1. STT Processing
- OpenAI Whisper(Small) 기반 음성 텍스트 변환

### 2. RAG Retrieval
- KR-SBERT 기반 문서 임베딩
- 코사인 유사도 기반 내규 문서 검색

### 3. LLM Generation
- Gemini 1.5 Flash 기반 회의록 및 Action Item 생성
- 검색된 내규 정보를 Prompt에 주입하여 생성 정확도 향상

---

## Features

- Whisper 기반 STT 처리
- KR-SBERT 기반 의미론적 문서 검색
- RAG 기반 회의록 생성
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
- KR-SBERT
- Gemini 1.5 Flash API

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
├── rag/
├── stt/
├── prompts/
├── templates/
├── static/
├── docker-compose.yml
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
GEMINI_API_KEY=your_api_key
```

### Start

```bash
python main.py
```

---

## Future Improvements

- Vector DB(FAISS / ChromaDB) 기반 검색 최적화
- PDF / DOCX 자동 문서 파싱
- 사용자 인증 및 권한 관리 기능 추가

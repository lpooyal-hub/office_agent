🏢 Office-Agent: RAG-Integrated Meeting Intelligence

STT(Speech-to-Text)와 로컬 RAG 기반의 지능형 기업 회의록 분석 시스템

📌 Project Overview

Office-Agent는 단순한 음성 요약을 넘어, **사내 내규 및 가이드라인 문서를 실시간으로 참조(RAG)**하여 회의 내용의 준수 여부를 판단하고 실행 가능한 인사이트(Action Items)를 도출하는 기업형 솔루션입니다.

단순 LLM 호출 방식의 한계인 '정보의 최신성 부재'와 '할루시네이션(Hallucination)' 문제를 로컬 시맨틱 검색 엔진을 통해 해결했습니다.

🏗 System Architecture

본 프로젝트는 효율적인 자원 관리와 데이터 보안을 위해 하이브리드 AI 파이프라인을 채택했습니다.

Transcription (STT): OpenAI Whisper (Small) 모델을 활용하여 업로드된 음성 파일을 텍스트 데이터로 정밀 변환합니다.

Retrieval (RAG): 변환된 텍스트를 KR-SBERT 임베딩 모델을 통해 벡터화한 후, 로컬 내규 데이터베이스와의 **코사인 유사도(Cosine Similarity)**를 계산하여 가장 관련성이 높은 규정을 추출합니다.

Augmentation & Generation: 추출된 내규 정보와 전체 스크립트를 프롬프트에 주입(Prompt Engineering)하여 Google Gemini 1.5 Flash 모델이 내규 기반의 전문적인 회의록을 생성합니다.

✨ Key Features

Hybrid AI Pipeline: 로컬 인스턴스(Whisper, SBERT)와 클라우드 LLM(Gemini)의 최적화된 조합.

Semantic Search Engine: 키워드 매칭이 아닌 의미론적 유사도 기반의 내규 검색으로 정확한 근거 제시.

Privacy-First Design: 민감한 음성 데이터와 검색 임베딩 과정을 로컬 서버 내에서 처리하여 보안성 강화.

Cost-Effective Architecture: OCI(Oracle Cloud) ARM 기반의 무료 티어 환경(4 vCPU, 24GB RAM)에서 원활하게 동작하도록 경량화 모델 최적화.

🛠 Tech Stack

Category

Technology

Backend

Python, FastAPI, Uvicorn

STT Model

OpenAI Whisper (Small)

Embedding

Sentence-Transformers (snunlp/KR-SBERT-V40K-klueNLI-aug)

LLM

Google Gemini 1.5 Flash API

Infrastructure

Oracle Cloud Infrastructure (OCI), Docker, Docker Compose

Frontend

Vanilla JS, CSS3, Jinja2 Templates

🚀 Getting Started

1. Prerequisites

Python 3.10+

FFmpeg (Whisper 오디오 디코딩용)

Google Gemini API Key

2. Environment Variables

.env 파일을 생성하고 아래 내용을 설정합니다.

GEMINI_API_KEY=your_google_gemini_api_key


3. Installation & Run

# Repository Clone
git clone [https://github.com/your-username/office-agent.git](https://github.com/your-username/office-agent.git)
cd office-agent

# Install Dependencies
pip install -r requirements.txt

# Run Server
python main.py


📈 Future Roadmap

[ ] Vector Database: FAISS 또는 ChromaDB 도입으로 수만 건 이상의 문서 검색 성능 최적화.

[ ] Auto-Parsing: PDF/Docx 사내 문서를 자동으로 읽어 벡터 DB에 업데이트하는 파이프라인 구축.

[ ] Multi-User Auth: 사용자별 문서 권한 관리 및 히스토리 저장 기능 추가.

# 🎙️ AI 회의록 요약 서비스 (AI Minutes Summarizer)

**AI 회의록 요약 서비스**는 음성 파일(mp3, wav 등)을 업로드하면 OpenAI Whisper를 통해 텍스트로 변환(STT)하고, Google Gemini API를 사용하여 회의 내용을 주제별로 요약해주는 웹 애플리케이션입니다.



## ✨ 주요 기능
- **음성-텍스트 변환 (STT):** OpenAI Whisper (Small 모델)를 사용하여 정확한 한국어 인식 지원.
- **AI 요약:** Google Gemini 2.0 Flash 모델을 활용한 핵심 요약 및 Action Items 추출.
- **사용자 친화적 인터페이스:** 간단한 파일 업로드와 실시간 분석 로딩 UI.
- **도커 지원:** Docker 및 Docker Compose를 통한 간편한 서버 배포.

## 🛠 기술 스택
- **Backend:** FastAPI (Python 3.10+)
- **AI Models:** OpenAI Whisper, Google Gemini 2.0 Flash
- **Infrastructure:** Docker, Docker Compose
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 🚀 시작하기

### 1. 환경 변수 설정
프로젝트 루트 폴더에 `.env` 파일을 생성하고 본인의 Gemini API 키를 입력합니다.
```env
GEMINI_API_KEY=your_google_gemini_api_key_here

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import whisper  # openai-whisper 사용
from google import genai
import os
import shutil
import logging
import uuid
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. Gemini 클라이언트 초기화
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# 3. OpenAI Whisper 모델 로드
# CPU 환경에서는 'base' 또는 'small' 모델을 추천합니다.
whisper_model = whisper.load_model("small")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{audio_file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    full_script = ""
    
    # 임시 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        logging.info(f"STT 변환 시작: {audio_file.filename}")
        
        # 4. OpenAI Whisper로 텍스트 추출
        # fp16=False: CPU 환경에서 실행 시 필수 옵션 (에러 방지 및 속도)
        # language="ko": 한국어 명시 시 정확도 향상
        result = whisper_model.transcribe(file_path, language="ko", fp16=False)
        full_script = result["text"].strip()
        
        if not full_script:
            return {"script": "인식된 음성이 없습니다.", "summary": "내용이 비어있어 요약할 수 없습니다."}

        logging.info("Gemini API 요약 요청 중...")

        # 5. Gemini API 요약
        # 모델명은 실제 출시된 버전(gemini-2.0-flash 또는 1.5-flash)을 권장합니다.
        prompt = f"""
        너는 전문적인 회의록 요약가야. 
        제공된 대화 내용을 바탕으로 다음 형식에 맞춰 한국어로 작성해줘:
        1. 주제별 핵심 요약
        2. 주요 결정 사항 및 할 일(Action Items)
        3. 전체적인 결론
        
        내용: {full_script}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )

        summary = response.text

        logging.info("전체 프로세스 완료")
        return {"script": full_script, "summary": summary}

    except Exception as e:
        logging.error(f"프로세스 오류: {str(e)}")
        return {
            "script": full_script if full_script else "스크립트 추출 실패", 
            "summary": f"오류가 발생했습니다: {str(e)}"
        }
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
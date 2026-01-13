from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import whisper
from ibm_watsonx_ai.foundation_models import ModelInference
import os
import shutil
import logging
import uuid
from dotenv import load_dotenv

# 1. 환경 변수 로드
# 1. 환경 변수 로드
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. WatsonX 환경 변수 읽기
# os.getenv()를 사용하여 .env 파일의 값을 가져옵니다.
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# 필수 환경 변수가 없는 경우 에러 발생 (디버깅용)
if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
    logging.error("환경 변수(WATSONX_API_KEY 또는 WATSONX_PROJECT_ID)가 설정되지 않았습니다.")

credentials = {
    "apikey": WATSONX_API_KEY,
    "url": WATSONX_URL
}

llm_model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID
)

# 3. OpenAI Whisper 모델 로드 (서버 시작 시 한 번만 실행)
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
        
        # 4. Whisper로 텍스트 추출 (CPU 최적화 옵션)
        result = whisper_model.transcribe(file_path, language="ko", fp16=False)
        full_script = result["text"].strip()
        
        if not full_script:
            return {"script": "인식된 음성이 없습니다.", "summary": "내용이 비어있어 요약할 수 없습니다."}

        logging.info("WatsonX Llama 3.3 요약 요청 중...")

        # 5. WatsonX Llama 3.3 프롬프트 구성 (Llama 3 전용 태그 사용)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
너는 전문적인 회의록 요약가야. 
제공된 대화 내용을 바탕으로 다음 형식에 맞춰 한국어로 작성해줘:
1. 주제별 핵심 요약
2. 주요 결정 사항 및 할 일(Action Items)
3. 전체적인 결론<|eot_id|><|start_header_id|>user<|end_header_id|>

내용: {full_script}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # 텍스트 생성
        summary = llm_model.generate_text(prompt=prompt)

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
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

        # 5. 프롬프트 정교화 (가이드라인 명시)
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
너는 전문적인 기업 회의록 정리 전문가야. 
제공된 대화 내용을 분석하여, 읽는 사람이 회의 내용을 완벽히 파악할 수 있도록 상세하고 구조적으로 작성해줘.
다음 지침을 엄격히 준수해:
- 말투는 정중한 비즈니스 문어체(~합니다)를 사용한다.
- 중요한 숫지, 시간, 장소, 인물 등 구체적 정보는 생략하지 않는다.
- 각 섹션은 충분한 설명을 포함해야 한다.

[형식]
1. 회의 안건 및 핵심 요약 (안건별로 상세히 기술)
2. 주요 결정 사항 및 담당자별 할 일 (Action Items)
3. 향후 일정 및 결론<|eot_id|><|start_header_id|>user<|end_header_id|>

내용: {full_script}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # 6. 답변 안정성을 위한 파라미터 추가
        # max_new_tokens를 늘려 답변이 중간에 끊기지 않게 합니다.
        summary = llm_model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 1000,   # 답변 길이 제한 확대
                "temperature": 0.3,       # 낮을수록 일관성 있고 차분한 답변
                "repetition_penalty": 1.1 # 같은 말 반복 방지
            }
        )

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

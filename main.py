import os
import uuid
import shutil
import logging
import whisper
import google.generativeai as genai  # Gemini 라이브러리 추가
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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

# 2. Gemini 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logging.error("환경 변수(GEMINI_API_KEY)가 설정되지 않았습니다.")

# Gemini SDK 초기화
genai.configure(api_key=GEMINI_API_KEY)

# 모델 인스턴스 생성 (system_instruction 기능을 사용하여 역할 부여)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", # 또는 성능 중심의 "gemini-1.5-pro"
    system_instruction=(
        "너는 전문적인 기업 회의록 정리 전문가야. "
        "제공된 대화 내용을 분석하여, 읽는 사람이 회의 내용을 완벽히 파악할 수 있도록 상세하고 구조적으로 작성해줘. "
        "말투는 정중한 비즈니스 문어체(~합니다)를 사용하며, 숫자, 시간, 장소 등 구체적 정보를 엄격히 준수해."
    )
)

# 3. Whisper 모델 로드
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
        
        # 4. Whisper로 텍스트 추출
        result = whisper_model.transcribe(file_path, language="ko", fp16=False)
        full_script = result["text"].strip()
        
        if not full_script:
            return {"script": "인식된 음성이 없습니다.", "summary": "내용이 비어있어 요약할 수 없습니다."}

        logging.info("Gemini 요약 요청 중...")

        # 5. Gemini 프롬프트 구성 및 실행
        # 시스템 지침은 모델 생성 시 이미 정의했으므로, 사용자 입력만 전달합니다.
        user_prompt = f"""
내용: {full_script}

[형식]
1. 회의 안건 및 핵심 요약 (안건별로 상세히 기술)
2. 주요 결정 사항 및 담당자별 할 일 (Action Items)
3. 향후 일정 및 결론
"""

        # 답변 생성 파라미터 설정
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2048, # 충분한 답변 길이를 위해 설정
                temperature=0.3,       # 일관성 있는 답변을 위해 낮게 설정
            )
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

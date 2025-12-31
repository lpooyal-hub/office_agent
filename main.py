from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
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

# 2. Gemini 클라이언트 초기화 (보안 강화)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

# 3. Faster-Whisper 모델 로드 (ARM CPU/오라클 클라우드 최적화)
# 4 OCPU 기준 cpu_threads=4 설정
whisper_model = WhisperModel(
    "small", 
    device="cpu", 
    compute_type="int8", 
    cpu_threads=4, 
    num_workers=1
)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...)):
    # 파일명 중복 및 보안을 위해 UUID 사용
    unique_filename = f"{uuid.uuid4()}_{audio_file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    full_script = ""
    
    # 임시 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        logging.info(f"STT 변환 시작: {audio_file.filename}")
        
        # 4. Faster-Whisper로 텍스트 추출
        segments, info = whisper_model.transcribe(
            file_path, 
            beam_size=5,        # 1에서 5로 상향 조정 (정확도 향상)
            language="ko", 
            vad_filter=True,
            condition_on_previous_text=True # 문맥 유지를 위해 추가 권장
        
        )
        
        full_script = " ".join([segment.text for segment in segments]).strip()
        
        if not full_script:
            return {"script": "인식된 음성이 없습니다.", "summary": "내용이 비어있어 요약할 수 없습니다."}

        logging.info("Gemini API 요약 요청 중...")

        # 5. 공식 SDK를 이용한 모델 호출 (모델 지정 방식)
        # 현재 가장 성능이 좋은 gemini-2.0-flash 또는 gemini-1.5-flash 권장
        prompt = f"""
        너는 전문적인 회의록 요약가야. 
        제공된 대화 내용을 바탕으로 다음 형식에 맞춰 한국어로 작성해줘:
        1. 주제별 핵심 요약
        2. 주요 결정 사항 및 할 일(Action Items)
        3. 전체적인 결론
        4. 모든 대화 내용
        내용: {full_script}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )

        # 6. 결과 추출 (파싱이 매우 간편해졌습니다)
        summary = response.text

        logging.info("전체 프로세스 완료")
        return {"script": full_script, "summary": summary}

    except Exception as e:
        logging.error(f"프로세스 오류: {str(e)}")
        # 에러 발생 시에도 추출된 스크립트가 있다면 최대한 반환
        return {
            "script": full_script if full_script else "스크립트 추출 실패", 
            "summary": f"오류가 발생했습니다: {str(e)}"
        }
        
    finally:
        # 서버 용량 관리를 위해 사용한 오디오 파일 즉시 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    # 포트 5001번으로 실행
    uvicorn.run(app, host="0.0.0.0", port=5001)
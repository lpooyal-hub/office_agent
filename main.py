from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import requests
import os
import shutil
import logging
import sys

# 로그를 표준 출력으로 강제 설정
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 메모리 절약을 위해 base 모델 + 1개의 워커만 사용
try:
    logger.info("모델 로딩 시작...")
    whisper_model = WhisperModel(
        "base", 
        device="cpu", 
        compute_type="int8", 
        cpu_threads=4, 
        num_workers=1
    )
    logger.info("모델 로딩 완료")
except Exception as e:
    logger.error(f"모델 로딩 실패: {e}")

OLLAMA_URL = "http://ollama:11434/api/generate"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        logger.info(f"변환 시작: {file_path}")
        
        # beam_size=1은 정확도는 약간 낮아지지만 메모리와 속도에 가장 유리합니다.
        segments, info = whisper_model.transcribe(
            file_path, beam_size=1, language="ko", vad_filter=False
        )
        
        text_list = []
        for segment in segments:
            logger.info(f"추출 중: {segment.text}") # 진행상황 실시간 출력
            text_list.append(segment.text)
        
        full_script = " ".join(text_list).strip()
        
        if not full_script:
            return {"script": "추출 실패", "summary": "내용 없음"}

        # 요약 요청
        logger.info("요약 요청 시작...")
        payload = {
            "model": "llama3",
            "prompt": f"너는 신입사원을 위한 회의록 요약 전문가야. 주제별로 요약하고 해야할일 정리 및 결론을 알려줘. 꼭 한국어로 답변해.\n  {full_script}",
            "stream": False
        }
        
        # timeout=None으로 무제한 대기
        response = requests.post(OLLAMA_URL, json=payload, timeout=None)
        summary = response.json().get('response', '요약 실패')
        
        return {"script": full_script, "summary": summary}

    except Exception as e:
        logger.error(f"처리 중 치명적 오류: {str(e)}", exc_info=True)
        return {"script": "에러 발생", "summary": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
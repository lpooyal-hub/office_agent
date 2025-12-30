from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import requests
import os
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ARM CPU 최적화 로드 (4 OCPU 활용)
model_size = "small"
whisper_model = WhisperModel(
    model_size, 
    device="cpu", 
    compute_type="int8", 
    cpu_threads=4, 
    num_workers=2
)

OLLAMA_URL = "http://ollama:11434/api/generate"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # 1. Faster-Whisper 변환
        segments, info = whisper_model.transcribe(
            file_path, beam_size=5, language="ko", vad_filter=True
        )
        full_script = "".join([segment.text for segment in segments])

        # 2. Ollama 요약
        system_prompt = "너는 신입사원을 위한 회의록 요약 전문가야. 주제별로 요약하고 해야할일 정리 및 결론을 알려줘. 한국어로 답변해."
        payload = {
            "model": "llama3",
            "prompt": f"{system_prompt}\n\n[내용]:\n{full_script}",
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=200)
        summary = response.json().get('response', '요약 생성 실패')

        return {"script": full_script, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
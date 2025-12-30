from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import whisper
import torch
import requests
import os
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 임시 파일 저장소
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로드 (서버 실행 시 1회)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small").to(device)

# 컨테이너 환경에서는 'ollama' 이름으로 접근
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
        # 1. Whisper로 음성 -> 텍스트 변환
        result = whisper_model.transcribe(file_path, language="ko")
        full_script = result["text"]

        # 2. Ollama(Llama 3)로 요약 요청
        system_prompt = "너는 신입사원을 위한 회의록 요약 전문가야. 주제별로 요약하고 해야할일 정리 및 결론을 알려줘. 한국어로 답변해."
        payload = {
            "model": "llama3",
            "prompt": f"{system_prompt}\n\n[내용]:\n{full_script}",
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
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
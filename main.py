import os
import uuid
import shutil
import logging
import whisper
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 환경 변수 및 초기 설정
load_dotenv()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 2. 모델 로드 (서버 시작 시 한 번만 실행)
# Gemini 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="너는 전문적인 기업 회의록 정리 및 사내 내규 안내 전문가야. 공손한 비즈니스 문어체를 사용해줘."
)

# STT 모델 (Whisper)
whisper_model = whisper.load_model("small")

# 임베딩 모델 (한국어 성능이 우수한 SBERT)
embedder = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-aug')

# 3. 사내 내규 데이터베이스 (데모용 데이터)
# 실제 서비스 시에는 PDF나 텍스트 파일에서 읽어와 리스트로 저장합니다.
COMPANY_RULES = [
    "연차 휴가는 1년 미만 근로자의 경우 1개월 개근 시 1일이 발생하며, 총 11일 한도입니다.",
    "경조사 휴가의 경우 본인 결혼은 5일, 부모상 및 배우자 부모상은 5일의 유급 휴가를 부여합니다.",
    "식대 지원은 연봉 외 별도로 매월 20만 원을 지급하며, 복지카드로 결제합니다.",
    "재택근무는 매주 수요일을 권장하며, 부서장과의 사전 협의가 필요합니다.",
    "업무용 도서 구입비는 월 5만 원 한도 내에서 실비 정산이 가능합니다."
]
# 서버 시작 시 미리 임베딩하여 메모리에 적재
RULE_EMBEDDINGS = embedder.encode(COMPANY_RULES)

# --- 유틸리티 함수 ---
def get_relevant_rule(query: str, threshold=0.4):
    """질문과 가장 유사한 내규를 검색합니다."""
    query_vector = embedder.encode([query])
    distances = cosine_similarity(query_vector, RULE_EMBEDDINGS)
    best_idx = np.argmax(distances)
    score = distances[0][best_idx]
    
    if score >= threshold:
        return COMPANY_RULES[best_idx], score
    return "관련된 내규를 찾을 수 없습니다.", score

# --- API 엔드포인트 ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...)):
    unique_filename = f"{uuid.uuid4()}_{audio_file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # 파일 임시 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # 1. STT: 음성 -> 텍스트
        logging.info("STT 변환 시작")
        result = whisper_model.transcribe(file_path, language="ko", fp16=False)
        full_script = result["text"].strip()
        
        if not full_script:
            return {"script": "인식된 음성 없음", "summary": "내용 없음", "relevant_rule": "N/A"}

        # 2. RAG: 스크립트 내용 중 내규와 관련된 키워드가 있는지 검색
        # (예: 회의 중에 '연차'나 '식대' 이야기가 나왔을 경우를 가정)
        # 여기서는 전체 스크립트를 기반으로 가장 유사한 내규 하나를 가져옵니다.
        related_rule, score = get_relevant_rule(full_script)

        # 3. LLM: Gemini 요약 (검색된 내규 정보를 참고 정보로 전달)
        logging.info("Gemini 요약 요청")
        user_prompt = f"""
[회의 내용]
{full_script}

[참고 사내 내규]
{related_rule} (유사도: {score:.2f})

위 회의 내용을 바탕으로 회의록을 작성해줘. 
만약 회의 내용이 사내 내규와 충돌하거나 관련이 있다면 '참고 내규'를 바탕으로 조언도 포함해줘.

[형식]
1. 회의 핵심 요약
2. 결정 사항 및 Action Items
3. 내규 검토 의견 (필요 시)
"""
        response = gemini_model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048
            )
        )

        return {
            "script": full_script,
            "summary": response.text,
            "retrieved_rule": related_rule
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)

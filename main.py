import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from threading import RLock
from typing import List

import numpy as np
import requests
import whisper
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = Path("uploads")
DOCUMENT_FOLDER = Path("documents")
UPLOAD_FOLDER.mkdir(exist_ok=True)
DOCUMENT_FOLDER.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

DEFAULT_COMPANY_RULES = [
    "연차 휴가는 1년 미만 근로자의 경우 1개월 개근 시 1일이 발생하며, 총 11일 한도입니다.",
    "경조사 휴가의 경우 본인 결혼은 5일, 부모상 및 배우자 부모상은 5일의 유급 휴가를 부여합니다.",
    "식대 지원은 연봉 외 별도로 매월 20만 원을 지급하며, 복지카드로 결제합니다.",
    "재택근무는 매주 수요일을 권장하며, 부서장과의 사전 협의가 필요합니다.",
    "업무용 도서 구입비는 월 5만 원 한도 내에서 실비 정산이 가능합니다.",
]

_whisper_model = None
_embedder = None
_rule_chunks = None
_rule_embeddings = None
_model_lock = RLock()


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                logging.info("Whisper model loading: %s", WHISPER_MODEL)
                _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def get_embedder():
    global _embedder
    if _embedder is None:
        with _model_lock:
            if _embedder is None:
                logging.info("Embedding model loading: %s", EMBEDDING_MODEL)
                _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder


def extract_response_text(data):
    if data.get("output_text"):
        return data["output_text"]

    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "\n".join(texts)


def generate_minutes(script, related_rules):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    prompt = f"""
당신은 전문적인 기업 회의록 정리 및 사내 내규 검토 전문가입니다.
아래 회의 내용과 참고 내규만 근거로 공손한 비즈니스 문어체의 회의록을 작성하세요.
근거가 부족한 내용은 추측하지 마세요.

[회의 내용]
{script}

[참고 사내 내규]
{related_rules}

[출력 형식]
1. 회의 핵심 요약
2. 결정 사항 및 Action Items
3. 내규 검토 의견
"""
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": prompt,
            "temperature": 0.2,
            "max_output_tokens": 1200,
            "store": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return extract_response_text(response.json()).strip()


def generate_policy_answer(question, related_rules):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    prompt = f"""
당신은 회사 내규와 업무 가이드라인을 안내하는 신입사원 온보딩 챗봇입니다.
아래 참고 문서만 근거로 질문에 답하세요.
근거가 부족하면 모른다고 말하고, 인사/총무 담당자에게 확인하라고 안내하세요.
답변은 친절하고 간결한 한국어 존댓말로 작성하세요.

[참고 문서]
{related_rules}

[질문]
{question}

[답변 형식]
- 핵심 답변
- 참고할 점
- 추가 확인이 필요한 경우
"""
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": prompt,
            "temperature": 0.2,
            "max_output_tokens": 700,
            "store": False,
        },
        timeout=45,
    )
    response.raise_for_status()
    return extract_response_text(response.json()).strip()


def transcribe_with_openai(file_path):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    with file_path.open("rb") as audio_file:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (file_path.name, audio_file)},
            data={
                "model": OPENAI_STT_MODEL,
                "language": "ko",
            },
            timeout=180,
        )

    response.raise_for_status()
    data = response.json()
    return data.get("text", "").strip()


def transcribe_audio(file_path, stt_mode):
    if stt_mode == "openai":
        logging.info("OpenAI STT start: %s", OPENAI_STT_MODEL)
        return transcribe_with_openai(file_path)

    logging.info("Local Whisper STT start: %s", WHISPER_MODEL)
    result = get_whisper_model().transcribe(str(file_path), language="ko", fp16=False)
    return result["text"].strip()


def split_text(text, max_chars=900):
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}".strip()

    if current:
        chunks.append(current)
    return chunks


def read_document_text(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"지원하지 않는 문서 형식입니다: {suffix}")


def load_rule_chunks():
    chunks = list(DEFAULT_COMPANY_RULES)
    for path in sorted(DOCUMENT_FOLDER.iterdir()):
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        try:
            text = read_document_text(path)
        except Exception as exc:
            logging.warning("Document load failed: %s (%s)", path.name, exc)
            continue

        for chunk in split_text(text):
            chunks.append(f"[{path.name}]\n{chunk}")
    return chunks


def get_rule_chunks():
    global _rule_chunks
    if _rule_chunks is None:
        with _model_lock:
            if _rule_chunks is None:
                _rule_chunks = load_rule_chunks()
    return _rule_chunks


def get_rule_embeddings():
    global _rule_embeddings
    if _rule_embeddings is None:
        with _model_lock:
            if _rule_embeddings is None:
                _rule_embeddings = get_embedder().encode(get_rule_chunks(), normalize_embeddings=True)
    return _rule_embeddings


def invalidate_rag_index():
    global _rule_chunks, _rule_embeddings
    with _model_lock:
        _rule_chunks = None
        _rule_embeddings = None


def get_relevant_rules(query, threshold=0.35, top_k=3):
    chunks = get_rule_chunks()
    if not chunks:
        return "관련된 내규를 찾을 수 없습니다.", []

    query_vector = get_embedder().encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_vector, get_rule_embeddings())[0]
    lexical_scores = get_lexical_scores(query, chunks)
    combined_scores = scores + lexical_scores
    ranked_indexes = np.argsort(combined_scores)[::-1][:top_k]

    matches = []
    for index in ranked_indexes:
        score = float(scores[index])
        if score >= threshold or not matches:
            matches.append((chunks[index], score))

    rendered = "\n\n".join(
        f"- {document}\n  유사도: {score:.2f}" for document, score in matches
    )
    return rendered, matches


def get_lexical_scores(query, chunks):
    keywords = expand_query_terms(query)
    if not keywords:
        return np.zeros(len(chunks))

    scores = []
    for chunk in chunks:
        lowered = chunk.lower()
        matched = sum(1 for keyword in keywords if keyword in lowered)
        scores.append(min(matched * 0.35, 1.05))
    return np.array(scores)


def expand_query_terms(query):
    particles = (
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "도",
        "만",
        "과",
        "와",
        "에서",
        "에게",
        "으로",
        "로",
        "입니다",
        "인가요",
        "나요",
        "요",
    )
    terms = set()

    for token in re.findall(r"[가-힣A-Za-z0-9]+", query.lower()):
        if len(token) < 2:
            continue

        terms.add(token)
        for particle in particles:
            if token.endswith(particle) and len(token) > len(particle) + 1:
                terms.add(token[: -len(particle)])

        if re.search(r"[가-힣]", token) and len(token) >= 3:
            terms.add(token[:2])

    return terms


def save_upload_file(upload_file, target_folder):
    safe_filename = Path(upload_file.filename or "uploaded_file").name
    target_path = target_folder / f"{uuid.uuid4()}_{safe_filename}"

    with target_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return target_path


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "document_count": len(get_rule_chunks()),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": OPENAI_MODEL,
            "openai_stt_model": OPENAI_STT_MODEL,
            "local_stt_model": WHISPER_MODEL,
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OPENAI_MODEL,
        "openai_stt_model": OPENAI_STT_MODEL,
        "local_stt_model": WHISPER_MODEL,
        "document_chunks": len(get_rule_chunks()),
    }


@app.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    saved = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            return {"error": f"지원하지 않는 문서 형식입니다: {file.filename}"}

        saved_path = save_upload_file(file, DOCUMENT_FOLDER)
        saved.append(saved_path.name)

    invalidate_rag_index()
    return {
        "message": "문서 업로드 및 RAG 인덱스 갱신 준비 완료",
        "files": saved,
        "document_chunks": len(get_rule_chunks()),
    }


@app.post("/chat")
async def chat_policy(question: str = Form(...)):
    question = question.strip()
    if not question:
        return {"error": "질문을 입력해주세요."}

    try:
        related_rules, _ = get_relevant_rules(question, top_k=4)
        answer = generate_policy_answer(question, related_rules)
        return {
            "answer": answer,
            "retrieved_rule": related_rules,
        }
    except Exception as exc:
        logging.error("Chat error: %s", str(exc))
        return {"error": str(exc)}


@app.post("/process")
async def process_audio(audio_file: UploadFile = File(...), stt_mode: str = Form("local")):
    file_path = save_upload_file(audio_file, UPLOAD_FOLDER)

    try:
        if stt_mode not in {"local", "openai"}:
            return {"error": "지원하지 않는 STT 모드입니다."}

        full_script = transcribe_audio(file_path, stt_mode)

        if not full_script:
            return {"script": "인식된 음성 없음", "summary": "내용 없음", "retrieved_rule": "N/A"}

        related_rules, _ = get_relevant_rules(full_script)

        logging.info("OpenAI minutes generation start")
        summary = generate_minutes(full_script, related_rules)

        return {
            "script": full_script,
            "summary": summary,
            "retrieved_rule": related_rules,
            "stt_mode": stt_mode,
        }

    except Exception as exc:
        logging.error("Error: %s", str(exc))
        return {"error": str(exc)}

    finally:
        if file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)

import logging
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    import whisper
except ImportError:  # pragma: no cover - optional dependency
    whisper = None

from services.ai_client import (
    generate_minutes,
    generate_policy_answer,
    summarize_document_text,
    transcribe_with_openai,
)
from services.chroma_store import ChromaStore
from services.document_library import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    clear_document_library,
    delete_document_file,
    get_display_filename,
    get_document_path,
    get_safe_upload_filename,
    list_stored_documents,
    read_document_text,
    save_upload_file,
)
from services.rag_service import DocumentRetriever, parse_rule_match

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = Path("uploads")
DOCUMENT_FOLDER = Path("documents")
STATIC_FOLDER = Path("static")
UPLOAD_FOLDER.mkdir(exist_ok=True)
DOCUMENT_FOLDER.mkdir(exist_ok=True)
STATIC_FOLDER.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
MAX_AUDIO_UPLOAD_MB = int(os.getenv("MAX_AUDIO_UPLOAD_MB", "25"))
MAX_DOCUMENT_UPLOAD_MB = int(os.getenv("MAX_DOCUMENT_UPLOAD_MB", "10"))
MAX_SUMMARY_DOCUMENTS = int(os.getenv("MAX_SUMMARY_DOCUMENTS", "5"))
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:9001")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "default_tenant")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default_database")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "office_agent_documents")
CHROMA_TIMEOUT_SECONDS = int(os.getenv("CHROMA_TIMEOUT_SECONDS", "15"))
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN", "")

DEFAULT_COMPANY_RULES = [
    "연차 휴가는 1년 미만 근로자의 경우 1개월 개근 시 1일이 발생하며, 총 11일 한도입니다.",
    "경조사 휴가의 경우 본인 결혼은 5일, 부모상 및 배우자 부모상은 5일의 유급 휴가를 부여합니다.",
    "식대 지원은 연봉 외 별도로 매월 20만 원을 지급하며, 복지카드로 결제합니다.",
    "재택근무는 매주 수요일을 권장하며, 부서장과의 사전 협의가 필요합니다.",
    "업무용 도서 구입비는 월 5만 원 한도 내에서 실비 정산이 가능합니다.",
]

_whisper_model = None
_model_lock = RLock()
_request_history = defaultdict(deque)
chroma_store = ChromaStore(
    base_url=CHROMA_URL,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    collection_name=CHROMA_COLLECTION,
    timeout_seconds=CHROMA_TIMEOUT_SECONDS,
    token=CHROMA_TOKEN,
)
retriever = DocumentRetriever(
    DOCUMENT_FOLDER,
    EMBEDDING_MODEL,
    DEFAULT_COMPANY_RULES,
    chroma_store,
)


def get_whisper_model():
    global _whisper_model
    if whisper is None:
        raise RuntimeError("whisper 패키지가 설치되지 않았습니다.")

    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                logging.info("Whisper model loading: %s", WHISPER_MODEL)
                _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_audio(file_path, stt_mode):
    if stt_mode == "openai":
        logging.info("OpenAI STT start: %s", OPENAI_STT_MODEL)
        return transcribe_with_openai(OPENAI_API_KEY, OPENAI_STT_MODEL, file_path)

    logging.info("Local Whisper STT start: %s", WHISPER_MODEL)
    result = get_whisper_model().transcribe(str(file_path), language="ko", fp16=False)
    return result["text"].strip()


def get_client_id(request):
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def require_access_code(access_code):
    if not ACCESS_CODE:
        raise HTTPException(
            status_code=503,
            detail="서비스 접속 코드가 설정되어 있지 않습니다.",
        )
    if not secrets.compare_digest(access_code or "", ACCESS_CODE):
        raise HTTPException(status_code=403, detail="접속 코드가 올바르지 않습니다.")


def enforce_rate_limit(client_id):
    now = time.time()
    history = _request_history[client_id]

    while history and now - history[0] > RATE_LIMIT_WINDOW_SECONDS:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
        )

    history.append(now)


def protect_request(request, access_code):
    require_access_code(access_code)
    enforce_rate_limit(get_client_id(request))


def raise_upload_error(message, status_code=400):
    raise HTTPException(status_code=status_code, detail=message)


def validate_document_uploads(files):
    if not files:
        raise_upload_error("문서를 선택해주세요.")

    for file in files:
        original_filename = get_safe_upload_filename(file.filename)
        suffix = Path(original_filename).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise_upload_error(f"지원하지 않는 문서 형식입니다: {original_filename}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "document_count": len(list_stored_documents(DOCUMENT_FOLDER)),
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
        "document_chunks": retriever.get_rule_chunk_count(),
        "vector_store": "chroma",
        "chroma_collection": CHROMA_COLLECTION,
    }


@app.get("/documents")
async def get_documents():
    documents = list_stored_documents(DOCUMENT_FOLDER)
    return {
        "documents": documents,
        "count": len(documents),
    }


@app.post("/documents")
async def upload_documents(
    request: Request,
    access_code: str = Form(""),
    files: List[UploadFile] = File(...),
):
    protect_request(request, access_code)
    validate_document_uploads(files)

    saved_paths = []
    try:
        for file in files:
            saved_path = save_upload_file(
                file,
                DOCUMENT_FOLDER,
                MAX_DOCUMENT_UPLOAD_MB,
                preserve_name=True,
            )
            saved_paths.append(saved_path)

            text = read_document_text(saved_path)
            if not text.strip():
                raise_upload_error(
                    f"문서에서 읽을 수 있는 내용이 없습니다: {get_display_filename(saved_path.name)}"
                )
    except ValueError as exc:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise_upload_error(str(exc), status_code=413)
    except HTTPException:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        logging.error("Document upload error: %s", str(exc))
        raise HTTPException(status_code=500, detail="문서 업로드 중 오류가 발생했습니다.")

    try:
        retriever.index_documents(saved_paths)
    except Exception as exc:
        for path in saved_paths:
            try:
                retriever.remove_document(path.name)
            except Exception:
                logging.warning("Document index rollback skipped: %s", path.name)
        for path in saved_paths:
            path.unlink(missing_ok=True)
        logging.error("Document index error: %s", str(exc))
        raise HTTPException(status_code=502, detail="문서 검색 인덱스 갱신 중 오류가 발생했습니다.") from exc

    retriever.invalidate()
    documents = list_stored_documents(DOCUMENT_FOLDER)
    return {
        "message": "문서 업로드와 지식 베이스 갱신이 완료되었습니다.",
        "files": [
            {
                "stored_name": path.name,
                "display_name": get_display_filename(path.name),
            }
            for path in saved_paths
        ],
        "document_chunks": retriever.get_rule_chunk_count(),
        "documents": documents,
        "count": len(documents),
    }


@app.delete("/documents/{stored_name}")
async def remove_document(
    request: Request,
    stored_name: str,
    access_code: str,
):
    protect_request(request, access_code)
    try:
        target_path = get_document_path(DOCUMENT_FOLDER, stored_name)
        retriever.remove_document(target_path.name)
        display_name = delete_document_file(DOCUMENT_FOLDER, stored_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logging.error("Document delete error: %s", str(exc))
        raise HTTPException(status_code=502, detail="문서 인덱스 삭제 중 오류가 발생했습니다.") from exc

    retriever.invalidate()
    documents = list_stored_documents(DOCUMENT_FOLDER)
    return {
        "message": f"{display_name} 문서를 삭제했습니다.",
        "documents": documents,
        "count": len(documents),
    }


@app.delete("/documents")
async def clear_documents(
    request: Request,
    access_code: str,
):
    protect_request(request, access_code)
    try:
        retriever.clear_documents()
        deleted = clear_document_library(DOCUMENT_FOLDER)
    except Exception as exc:
        logging.error("Document clear error: %s", str(exc))
        raise HTTPException(status_code=502, detail="문서 인덱스 초기화 중 오류가 발생했습니다.") from exc

    retriever.invalidate()
    return {
        "message": "문서 보관함을 비웠습니다.",
        "deleted": deleted,
        "count": 0,
        "documents": [],
    }


@app.post("/summarize")
async def summarize_documents(
    request: Request,
    access_code: str = Form(""),
    files: List[UploadFile] = File(...),
):
    protect_request(request, access_code)
    validate_document_uploads(files)
    if len(files) > MAX_SUMMARY_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"한 번에 최대 {MAX_SUMMARY_DOCUMENTS}개 문서까지 요약할 수 있습니다.",
        )

    saved_paths = []
    try:
        documents = []
        for file in files:
            original_filename = get_safe_upload_filename(file.filename)
            suffix = Path(original_filename).suffix.lower()
            if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"지원하지 않는 문서 형식입니다: {original_filename}",
                )

            temp_path = save_upload_file(file, UPLOAD_FOLDER, MAX_DOCUMENT_UPLOAD_MB)
            saved_paths.append(temp_path)
            text = read_document_text(temp_path)
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"문서에서 읽을 수 있는 내용이 없습니다: {original_filename}",
                )
            documents.append((original_filename, text))

        summaries = [
            {
                "filename": filename,
                "summary": summarize_document_text(OPENAI_API_KEY, OPENAI_MODEL, filename, text),
            }
            for filename, text in documents
        ]

        combined_summary = "\n\n".join(
            f"### {item['filename']}\n{item['summary']}" for item in summaries
        )
        return {
            "message": "문서 요약이 완료되었습니다.",
            "summaries": summaries,
            "combined_summary": combined_summary,
        }
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("Document summary error: %s", str(exc))
        raise HTTPException(status_code=502, detail="문서 요약 중 오류가 발생했습니다.") from exc
    finally:
        for path in saved_paths:
            if path.exists():
                path.unlink(missing_ok=True)


@app.post("/chat")
async def chat_policy(
    request: Request,
    question: str = Form(...),
    access_code: str = Form(""),
):
    protect_request(request, access_code)
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    try:
        related_rules, matches = retriever.get_relevant_rules(question, top_k=4)
        answer = generate_policy_answer(OPENAI_API_KEY, OPENAI_MODEL, question, related_rules)
        return {
            "answer": answer,
            "retrieved_rule": related_rules,
            "sources": [parse_rule_match(chunk, score) for chunk, score in matches],
        }
    except Exception as exc:
        logging.error("Chat error: %s", str(exc))
        raise HTTPException(status_code=502, detail="챗봇 답변 생성 중 오류가 발생했습니다.") from exc


@app.post("/process")
async def process_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    stt_mode: str = Form("local"),
    access_code: str = Form(""),
):
    protect_request(request, access_code)
    if stt_mode not in {"local", "openai"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 STT 모드입니다.")

    try:
        file_path = save_upload_file(audio_file, UPLOAD_FOLDER, MAX_AUDIO_UPLOAD_MB)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    try:
        full_script = transcribe_audio(file_path, stt_mode)

        if not full_script:
            return {"script": "인식된 음성 없음", "summary": "내용 없음", "retrieved_rule": "N/A"}

        related_rules, matches = retriever.get_relevant_rules(full_script)

        logging.info("OpenAI minutes generation start")
        summary = generate_minutes(OPENAI_API_KEY, OPENAI_MODEL, full_script, related_rules)

        return {
            "script": full_script,
            "summary": summary,
            "retrieved_rule": related_rules,
            "stt_mode": stt_mode,
            "sources": [parse_rule_match(chunk, score) for chunk, score in matches],
        }

    except Exception as exc:
        logging.error("Error: %s", str(exc))
        raise HTTPException(status_code=502, detail="음성 분석 중 오류가 발생했습니다.") from exc

    finally:
        if file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)

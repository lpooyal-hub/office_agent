import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from docx import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
UUID_PREFIX_PATTERN = re.compile(r"^[0-9a-f-]{36}_(.+)$")


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


def get_safe_upload_filename(filename):
    return Path(filename or "uploaded_file").name


def get_display_filename(filename):
    safe_name = get_safe_upload_filename(filename)
    matched = UUID_PREFIX_PATTERN.match(safe_name)
    if matched:
        return matched.group(1)
    return safe_name


def build_upload_path(target_folder, original_filename, preserve_name=False):
    safe_filename = get_safe_upload_filename(original_filename)
    if not preserve_name:
        return target_folder / f"{uuid.uuid4()}_{safe_filename}"

    candidate = target_folder / safe_filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        numbered_candidate = target_folder / f"{stem} ({index}){suffix}"
        if not numbered_candidate.exists():
            return numbered_candidate
        index += 1


def save_upload_file(upload_file, target_folder, max_size_mb, preserve_name=False):
    target_path = build_upload_path(
        target_folder,
        upload_file.filename,
        preserve_name=preserve_name,
    )
    max_size_bytes = max_size_mb * 1024 * 1024
    total_size = 0

    with target_path.open("wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > max_size_bytes:
                target_path.unlink(missing_ok=True)
                raise ValueError(f"파일은 {max_size_mb}MB 이하만 업로드할 수 있습니다.")

            buffer.write(chunk)

    return target_path


def read_document_text(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf 패키지가 설치되지 않았습니다.")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("python-docx 패키지가 설치되지 않았습니다.")
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"지원하지 않는 문서 형식입니다: {suffix}")


def list_stored_documents(document_folder):
    documents = []
    for path in sorted(document_folder.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        stat = path.stat()
        uploaded_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        documents.append(
            {
                "stored_name": path.name,
                "display_name": get_display_filename(path.name),
                "size_bytes": stat.st_size,
                "uploaded_at": uploaded_at.isoformat(),
            }
        )
    return documents


def delete_document_file(document_folder, stored_name):
    safe_name = get_safe_upload_filename(stored_name)
    if safe_name != stored_name:
        raise ValueError("잘못된 문서 식별자입니다.")

    target_path = document_folder / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError("삭제할 문서를 찾지 못했습니다.")

    target_path.unlink()
    return get_display_filename(safe_name)


def clear_document_library(document_folder):
    deleted = []
    for document in list_stored_documents(document_folder):
        path = document_folder / document["stored_name"]
        if path.exists():
            path.unlink()
            deleted.append(document["display_name"])
    return deleted

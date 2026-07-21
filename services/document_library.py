import json
import re
import uuid
from dataclasses import dataclass
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
METADATA_SUFFIX = ".metadata.json"
UUID_PREFIX_PATTERN = re.compile(r"^[0-9a-f-]{36}_(.+)$")
INDEX_STATUS_FILENAME = ".index_status.json"
INDEX_STATUS_INDEXED = "indexed"
INDEX_STATUS_FAILED = "failed"
INDEX_STATUS_UNKNOWN = "unknown"


def get_metadata_path(path):
    return path.with_name(f"{path.name}{METADATA_SUFFIX}")


def normalize_document_metadata(owner="system", visibility="public", allowed_roles=None):
    allowed_roles = allowed_roles or ["viewer", "editor", "admin"]
    return {
        "owner": owner or "system",
        "visibility": visibility or "public",
        "allowed_roles": list(allowed_roles),
    }


def save_document_metadata(path, owner="system", visibility="public", allowed_roles=None):
    metadata = normalize_document_metadata(owner, visibility, allowed_roles)
    get_metadata_path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def read_document_metadata(path):
    metadata_path = get_metadata_path(path)
    if not metadata_path.exists():
        return normalize_document_metadata()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return normalize_document_metadata()
    return normalize_document_metadata(
        owner=metadata.get("owner", "system"),
        visibility=metadata.get("visibility", "public"),
        allowed_roles=metadata.get("allowed_roles") or ["viewer", "editor", "admin"],
    )


def split_text(text, max_chars=900):
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
def get_index_status_path(document_folder):
    return document_folder / INDEX_STATUS_FILENAME


def load_index_statuses(document_folder):
    status_path = get_index_status_path(document_folder)
    if not status_path.exists():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_index_statuses(document_folder, statuses):
    status_path = get_index_status_path(document_folder)
    status_path.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def set_document_index_status(document_folder, stored_name, status, error=""):
    statuses = load_index_statuses(document_folder)
    statuses[stored_name] = {
        "status": status,
        "error": str(error or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_index_statuses(document_folder, statuses)


def remove_document_index_status(document_folder, stored_name):
    statuses = load_index_statuses(document_folder)
    if stored_name in statuses:
        statuses.pop(stored_name, None)
        save_index_statuses(document_folder, statuses)


def summarize_index_statuses(document_folder):
    summary = {INDEX_STATUS_INDEXED: 0, INDEX_STATUS_FAILED: 0, INDEX_STATUS_UNKNOWN: 0}
    for document in list_stored_documents(document_folder):
        status = document.get("index_status") or INDEX_STATUS_UNKNOWN
        summary[status] = summary.get(status, 0) + 1
    return summary
SECTION_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
ARTICLE_HEADING_PATTERN = re.compile(r"^(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\([^)]*\))?(?:\s+.+)?|\d+\.\s+.+|-\s+.+)$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？다요함됨임음)])\s+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int


def split_text(text, max_chars=900, overlap_chars=120):
    """Split text into metadata-rich chunks for vector search."""
    if max_chars < 1:
        raise ValueError("max_chars must be greater than 0")
    overlap_chars = max(0, min(overlap_chars, max_chars - 1))

    paragraphs = _extract_paragraphs(text)
    chunks = []
    current_parts = []
    current_start = None
    current_section = ""
    active_section = ""

    def flush_current():
        nonlocal current_parts, current_start, current_section
        if not current_parts:
            return
        chunk_text = "\n".join(part["text"] for part in current_parts).strip()
        if chunk_text:
            chunks.append(
                _make_chunk(
                    chunk_text,
                    current_section,
                    len(chunks),
                    current_start,
                    current_parts[-1]["end"],
                )
            )
        current_parts = []
        current_start = None
        current_section = active_section

    for paragraph in paragraphs:
        heading = _detect_section_title(paragraph["text"])
        if heading:
            flush_current()
            active_section = heading

        for segment in _split_long_paragraph(paragraph, max_chars):
            segment_section = active_section
            candidate_length = _candidate_length(current_parts, segment["text"])
            if current_parts and candidate_length > max_chars:
                previous_text = "\n".join(part["text"] for part in current_parts).strip()
                previous_end = current_parts[-1]["end"]
                flush_current()
                overlap_text = _get_overlap_text(previous_text, overlap_chars)
                if overlap_text and len(overlap_text) + len(segment["text"]) + 1 <= max_chars:
                    current_parts.append({"text": overlap_text, "start": max(previous_end - len(overlap_text), 0), "end": previous_end})
                    current_start = current_parts[0]["start"]
                    current_section = segment_section

            if not current_parts:
                current_start = segment["start"]
                current_section = segment_section
            current_parts.append(segment)

    flush_current()
    return chunks


def _extract_paragraphs(text):
    paragraphs = []
    for match in re.finditer(r"\S.*(?:\n|$)", text):
        raw = match.group(0)
        stripped = raw.strip()
        if stripped:
            leading = len(raw) - len(raw.lstrip())
            paragraphs.append({"text": stripped, "start": match.start() + leading, "end": match.start() + leading + len(stripped)})
    return paragraphs


def _detect_section_title(paragraph):
    markdown_match = SECTION_HEADING_PATTERN.match(paragraph)
    if markdown_match:
        return markdown_match.group(2).strip()
    article_match = ARTICLE_HEADING_PATTERN.match(paragraph)
    if article_match:
        return article_match.group(1).strip()
    return ""


def _split_long_paragraph(paragraph, max_chars):
    if len(paragraph["text"]) <= max_chars:
        return [paragraph]

    segments = []
    cursor = 0
    for sentence in SENTENCE_SPLIT_PATTERN.split(paragraph["text"]):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_start = paragraph["text"].find(sentence, cursor)
        if sentence_start < 0:
            sentence_start = cursor
        cursor = sentence_start + len(sentence)
        absolute_start = paragraph["start"] + sentence_start
        if len(sentence) > max_chars:
            segments.extend(_split_fixed_length(sentence, absolute_start, max_chars))
            continue
        if segments and len(segments[-1]["text"]) + len(sentence) + 1 <= max_chars:
            segments[-1]["text"] = f'{segments[-1]["text"]} {sentence}'
            segments[-1]["end"] = absolute_start + len(sentence)
        else:
            segments.append({"text": sentence, "start": absolute_start, "end": absolute_start + len(sentence)})
    return segments


def _split_fixed_length(text, absolute_start, max_chars):
    return [
        {"text": text[index:index + max_chars], "start": absolute_start + index, "end": absolute_start + index + len(text[index:index + max_chars])}
        for index in range(0, len(text), max_chars)
    ]


def _candidate_length(parts, text):
    return len(text) if not parts else len("\n".join(part["text"] for part in parts)) + 1 + len(text)


def _get_overlap_text(text, overlap_chars):
    if overlap_chars <= 0 or not text:
        return ""
    return text[-overlap_chars:]


def _make_chunk(text, section_title, chunk_index, char_start, char_end):
    return TextChunk(
        text=text,
        section_title=section_title or "",
        chunk_index=chunk_index,
        char_start=char_start or 0,
        char_end=char_end or 0,
    )


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
    index_statuses = load_index_statuses(document_folder)
    for path in sorted(document_folder.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        stat = path.stat()
        uploaded_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        metadata = read_document_metadata(path)
        index_status = index_statuses.get(path.name, {})
        documents.append(
            {
                "stored_name": path.name,
                "display_name": get_display_filename(path.name),
                "size_bytes": stat.st_size,
                "uploaded_at": uploaded_at.isoformat(),
                "owner": metadata["owner"],
                "visibility": metadata["visibility"],
                "allowed_roles": metadata["allowed_roles"],
                "index_status": index_status.get("status", INDEX_STATUS_UNKNOWN),
                "index_error": index_status.get("error", ""),
                "index_updated_at": index_status.get("updated_at", ""),
            }
        )
    return documents


def get_document_path(document_folder, stored_name):
    safe_name = get_safe_upload_filename(stored_name)
    if safe_name != stored_name:
        raise ValueError("잘못된 문서 식별자입니다.")

    target_path = document_folder / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError("삭제할 문서를 찾지 못했습니다.")
    return target_path


def delete_document_file(document_folder, stored_name):
    target_path = get_document_path(document_folder, stored_name)
    target_path.unlink()
    get_metadata_path(target_path).unlink(missing_ok=True)
    remove_document_index_status(document_folder, stored_name)
    return get_display_filename(target_path.name)


def clear_document_library(document_folder):
    deleted = []
    for document in list_stored_documents(document_folder):
        path = document_folder / document["stored_name"]
        if path.exists():
            path.unlink()
            get_metadata_path(path).unlink(missing_ok=True)
            deleted.append(document["display_name"])
    return deleted

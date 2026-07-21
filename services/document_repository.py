import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock


class DocumentRepository:
    """JSONL-backed repository for uploaded document metadata."""

    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _read_all_unlocked(self):
        if not self.metadata_path.exists():
            return []

        records = []
        with self.metadata_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _write_all_unlocked(self, records):
        temp_path = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                file.write("\n")
        temp_path.replace(self.metadata_path)

    def list_documents(self, existing_names=None):
        with self._lock:
            records = self._read_all_unlocked()

        if existing_names is not None:
            existing_names = set(existing_names)
            records = [record for record in records if record.get("stored_name") in existing_names]

        return sorted(records, key=lambda record: record.get("uploaded_at") or "", reverse=True)

    def create_document(
        self,
        *,
        stored_name,
        display_name,
        content_type,
        size_bytes,
        tags=None,
        owner="",
        description="",
    ):
        record = {
            "stored_name": stored_name,
            "display_name": display_name,
            "content_type": content_type or "application/octet-stream",
            "size_bytes": int(size_bytes),
            "uploaded_at": self._now(),
            "indexed_at": "",
            "index_status": "pending",
            "tags": list(tags or []),
            "owner": owner or "",
            "description": description or "",
        }
        with self._lock:
            records = [item for item in self._read_all_unlocked() if item.get("stored_name") != stored_name]
            records.append(record)
            self._write_all_unlocked(records)
        return record

    def update_index_status(self, stored_name, index_status, indexed_at=None):
        with self._lock:
            records = self._read_all_unlocked()
            updated = None
            for record in records:
                if record.get("stored_name") == stored_name:
                    record["index_status"] = index_status
                    record["indexed_at"] = indexed_at if indexed_at is not None else self._now()
                    updated = record
                    break
            if updated is None:
                raise KeyError(f"문서 메타데이터를 찾지 못했습니다: {stored_name}")
            self._write_all_unlocked(records)
            return updated

    def delete_document(self, stored_name):
        with self._lock:
            records = self._read_all_unlocked()
            kept = [record for record in records if record.get("stored_name") != stored_name]
            deleted = len(kept) != len(records)
            if deleted:
                self._write_all_unlocked(kept)
            return deleted

    def clear_documents(self):
        with self._lock:
            deleted = self._read_all_unlocked()
            self._write_all_unlocked([])
            return deleted

    def parse_tags(self, tags_text):
        if not tags_text:
            return []
        return [tag.strip() for tag in tags_text.split(",") if tag.strip()]

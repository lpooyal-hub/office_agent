from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    step: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        data = {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            data["result"] = self.result
        if self.error:
            data["error"] = self.error
        if self.step:
            data["step"] = self.step
        return data


class JobStore:
    def __init__(self):
        self._jobs: dict[str, JobResult] = {}
        self._lock = RLock()

    def create(self) -> JobResult:
        job = JobResult(job_id=str(uuid4()), status=JobStatus.PENDING)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobResult | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str, step: str | None = None) -> JobResult:
        return self._replace(job_id, status=JobStatus.RUNNING, step=step)

    def mark_succeeded(self, job_id: str, result: dict[str, Any]) -> JobResult:
        return self._replace(job_id, status=JobStatus.SUCCEEDED, result=result, error=None, step=None)

    def mark_failed(self, job_id: str, error: str, step: str | None = None) -> JobResult:
        return self._replace(job_id, status=JobStatus.FAILED, error=error, step=step)

    def _replace(self, job_id: str, **changes) -> JobResult:
        with self._lock:
            current = self._jobs[job_id]
            updated = JobResult(
                job_id=current.job_id,
                status=changes.get("status", current.status),
                result=changes.get("result", current.result),
                error=changes.get("error", current.error),
                step=changes.get("step", current.step),
                created_at=current.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._jobs[job_id] = updated
            return updated

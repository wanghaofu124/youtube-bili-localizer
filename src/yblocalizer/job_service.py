from __future__ import annotations

from typing import Protocol


class MutableJob(Protocol):
    status: str
    stage: str
    progress: int
    error: str | None


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"preparing", "ready", "running", "cancelled", "failed"},
    "preparing": {"draft", "ready", "failed"},
    "ready": {"draft", "running", "completed"},
    "queued": {"running", "cancelling", "cancelled", "failed"},
    "running": {"cancelling", "cancelled", "completed", "failed"},
    "cancelling": {"cancelled", "failed"},
    "completed": {"draft", "ready", "running"},
    "failed": {"draft", "ready", "running"},
    "cancelled": {"draft", "ready", "running"},
    "interrupted": {"draft", "ready", "running"},
}


def transition_job(
    job: MutableJob,
    status: str,
    stage: str,
    *,
    progress: int | None = None,
    error: str | None = None,
) -> None:
    """Apply one explicit lifecycle transition or reject an impossible state."""
    if status != job.status and status not in ALLOWED_TRANSITIONS.get(job.status, set()):
        raise RuntimeError(f"Invalid job transition: {job.status} -> {status}")
    job.status = status
    job.stage = stage
    if progress is not None:
        job.progress = max(0, min(100, progress))
    job.error = error

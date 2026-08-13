from dataclasses import dataclass

import pytest

from yblocalizer.job_service import transition_job


@dataclass
class Job:
    status: str = "queued"
    stage: str = "等待"
    progress: int = 0
    error: str | None = None


def test_job_lifecycle_accepts_normal_path() -> None:
    job = Job()
    transition_job(job, "running", "处理", progress=20)
    transition_job(job, "completed", "完成", progress=100)
    assert (job.status, job.progress) == ("completed", 100)


def test_job_lifecycle_rejects_impossible_path() -> None:
    with pytest.raises(RuntimeError, match="queued -> completed"):
        transition_job(Job(), "completed", "完成")

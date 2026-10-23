import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from yblocalizer import download, workbench_api
from yblocalizer.workbench_api import WorkbenchJobs


def _wait_idle(jobs: WorkbenchJobs) -> None:
    for _ in range(200):
        if not jobs.batch_snapshot()["active"]:
            return
        time.sleep(0.02)
    raise AssertionError("batch did not finish")


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    jobs = WorkbenchJobs()
    calls = {"create": [], "running": 0, "max_running": 0, "publish": 0}

    def create_url(url, device, compute_type, authorized, options):
        calls["create"].append((url, dict(options)))
        if "invalid" in url:
            raise ValueError("bad link")
        job = SimpleNamespace(
            id=f"job-{len(calls['create'])}", material=SimpleNamespace(name=url, source_url=url),
            options=dict(options), status="draft", stage="", progress=0, error=None, checks=[],
            stages={"render": {"status": "pending"}}, url=url,
        )
        jobs._jobs[job.id] = job
        return job

    def prepare(job_id):
        job = jobs._jobs[job_id]
        job.checks, job.status = [{"status": "passed"}], "ready"
        return job

    def run_all(job_id):
        job = jobs._jobs[job_id]
        calls["running"] += 1
        calls["max_running"] = max(calls["max_running"], calls["running"])
        time.sleep(0.01)
        calls["running"] -= 1
        if "needs-consent" in job.url and job.options.get("quality_consent") is not True:
            raise RuntimeError("quality consent required")
        if "fail" in job.url and not job.options.get("fixed"):
            job.status, job.error = "failed", "translate failed"
            return job
        job.stages["render"]["status"] = "completed"
        job.status = "completed"
        return job

    def run_stage(job_id, stage):
        calls["publish"] += stage == "publish"

    monkeypatch.setattr(jobs, "create_url", create_url)
    monkeypatch.setattr(jobs, "prepare", prepare)
    monkeypatch.setattr(jobs, "run_all", run_all)
    monkeypatch.setattr(jobs, "run_stage", run_stage)
    jobs.calls = calls
    return jobs


def test_batch_runs_serially_continues_after_failures_and_never_publishes(jobs, tmp_path):
    entries = [{"url": u} for u in ("https://a/1", "https://a/invalid", "https://a/fail", "https://a/needs-consent", "https://a/2")]
    jobs.batch_add(entries, "cpu", "int8", {"publish_to_bilibili": True}, False)
    _wait_idle(jobs)
    items = jobs.batch_snapshot()["items"]
    assert [item["status"] for item in items] == ["done", "failed", "failed", "failed", "done"]
    assert items[1]["error"] == "bad link"
    assert items[2]["error"] == "translate failed"
    assert "quality consent" in items[3]["error"]
    assert jobs.calls["max_running"] == 1
    assert jobs.calls["publish"] == 0
    assert all(options["publish_to_bilibili"] is False for _, options in jobs.calls["create"])
    saved = json.loads((tmp_path / "batch_queue.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in saved] == ["done", "failed", "failed", "failed", "done"]


def test_batch_quality_consent_is_applied_to_items(jobs):
    jobs.batch_add([{"url": "https://a/needs-consent"}], "cpu", "int8", {}, True)
    _wait_idle(jobs)
    assert jobs.batch_snapshot()["items"][0]["status"] == "done"


def test_retry_failed_reuses_job(jobs):
    jobs.batch_add([{"url": "https://a/fail"}], "cpu", "int8", {}, False)
    _wait_idle(jobs)
    item = jobs.batch_snapshot()["items"][0]
    assert item["status"] == "failed"
    jobs._jobs[item["job_id"]].options["fixed"] = True
    jobs.batch_retry()
    _wait_idle(jobs)
    after = jobs.batch_snapshot()["items"][0]
    assert after["status"] == "done" and after["job_id"] == item["job_id"]
    assert len(jobs.calls["create"]) == 1


def test_cancelled_item_is_cancelled_and_queue_moves_on(jobs, monkeypatch):
    original = jobs.run_all

    def run_all(job_id):
        job = jobs._jobs[job_id]
        if "cancel" in job.url:
            job.status = "cancelled"
            return job
        return original(job_id)

    monkeypatch.setattr(jobs, "run_all", run_all)
    jobs.batch_add([{"url": "https://a/cancel"}, {"url": "https://a/ok"}], "cpu", "int8", {}, False)
    _wait_idle(jobs)
    assert [item["status"] for item in jobs.batch_snapshot()["items"]] == ["cancelled", "done"]


def test_batch_waits_for_running_single_job(jobs):
    jobs._running_id = "single"
    jobs.batch_add([{"url": "https://a/1"}], "cpu", "int8", {}, False)
    time.sleep(0.2)
    assert jobs.batch_snapshot()["items"][0]["status"] == "pending"
    jobs._running_id = None
    _wait_idle(jobs)
    assert jobs.batch_snapshot()["items"][0]["status"] == "done"


def test_single_job_rejected_while_batch_item_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    jobs = WorkbenchJobs()
    jobs._batch_job_id = "batch-job"
    other = SimpleNamespace(id="other", material=SimpleNamespace(source_url=None))
    with pytest.raises(RuntimeError):
        jobs._start_stage_sequence(other, ["extract"], auto_run=True)


def test_restart_resets_running_items_to_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    (tmp_path / "batch_queue.json").write_text(json.dumps([
        {"id": "a", "url": "https://a/1", "title": "", "job_id": "job-a", "status": "running", "error": None, "device": "cpu", "compute_type": "int8", "options": {}, "quality_consent": False},
        {"id": "b", "url": "https://a/2", "title": "", "job_id": "job-b", "status": "done", "error": None, "device": "cpu", "compute_type": "int8", "options": {}, "quality_consent": False},
    ]), encoding="utf-8")
    jobs = WorkbenchJobs()
    monkeypatch.setattr(jobs, "_start_batch_worker", lambda: None)
    jobs._load_batch()
    assert [item["status"] for item in jobs.batch_snapshot()["items"]] == ["pending", "done"]
    saved = json.loads((tmp_path / "batch_queue.json").read_text(encoding="utf-8"))
    assert saved[0]["status"] == "pending" and saved[0]["job_id"] == "job-a"


def test_list_playlist_entries_uses_flat_extraction(monkeypatch):
    seen = {}

    class FakeYDL:
        def __init__(self, opts):
            seen.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return {"entries": [
                {"url": "abc", "ie_key": "Youtube", "title": "One", "duration": 10},
                None,
                {"url": "https://www.youtube.com/watch?v=def", "title": "Two"},
            ]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    entries = download.list_playlist_entries("https://www.youtube.com/playlist?list=PL1", playlist_items="1-3")
    assert seen["extract_flat"] == "in_playlist" and seen["playlist_items"] == "1-3"
    assert "noplaylist" not in seen
    assert entries == [
        {"url": "https://www.youtube.com/watch?v=abc", "title": "One", "duration": 10},
        {"url": "https://www.youtube.com/watch?v=def", "title": "Two", "duration": None},
    ]


def test_corrupt_batch_file_is_moved_aside(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    (tmp_path / "batch_queue.json").write_text("{not json", encoding="utf-8")
    jobs = WorkbenchJobs()
    jobs._load_batch()
    assert jobs.batch_snapshot()["items"] == []
    assert (tmp_path / "batch_queue.json.bad").exists()
    assert not (tmp_path / "batch_queue.json").exists()


def test_single_job_rejected_while_batch_item_is_being_created(jobs, monkeypatch):
    original = jobs.create_url
    rejected = []

    def create_url(*args):
        other = SimpleNamespace(id="other", material=SimpleNamespace(source_url=None))
        with pytest.raises(RuntimeError):
            jobs._start_stage_sequence(other, ["extract"], auto_run=True)
        rejected.append(True)
        return original(*args)

    monkeypatch.setattr(jobs, "create_url", create_url)
    monkeypatch.setattr(jobs, "_require_quality_consent", lambda job, stages: None)
    jobs.batch_add([{"url": "https://a/1"}], "cpu", "int8", {}, False)
    _wait_idle(jobs)
    assert rejected == [True]
    assert jobs.batch_snapshot()["items"][0]["status"] == "done"


def test_cancel_queue_stops_running_and_pending(jobs, monkeypatch):
    started, release, cancelled = threading.Event(), threading.Event(), []

    def run_all(job_id):
        started.set()
        release.wait(2)
        job = jobs._jobs[job_id]
        job.status = "cancelled"
        return job

    monkeypatch.setattr(jobs, "run_all", run_all)
    monkeypatch.setattr(jobs, "cancel", lambda job_id: cancelled.append(job_id))
    jobs.batch_add([{"url": "https://a/1"}, {"url": "https://a/2"}], "cpu", "int8", {}, False)
    assert started.wait(2)
    jobs.batch_cancel()
    release.set()
    _wait_idle(jobs)
    assert [item["status"] for item in jobs.batch_snapshot()["items"]] == ["cancelled", "cancelled"]
    assert cancelled == ["job-1"]
    assert len(jobs.calls["create"]) == 1


def test_restart_resumes_restored_job_without_recreating(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    (tmp_path / "batch_queue.json").write_text(json.dumps([
        {"id": "a", "url": "https://a/1", "title": "t", "job_id": "job-a", "status": "running", "error": None, "device": "cpu", "compute_type": "int8", "options": {}, "quality_consent": False},
    ]), encoding="utf-8")
    restored = SimpleNamespace(id="job-a", options={}, status="interrupted", error=None, stage="", progress=0, checks=[{"status": "passed"}],
                               stages={"extract": {"status": "completed"}, "render": {"status": "pending"}})
    calls = []

    def restore_jobs(self):
        self._jobs["job-a"] = restored

    def run_all(self, job_id):
        calls.append(job_id)
        self._jobs[job_id].stages["render"]["status"] = "completed"

    monkeypatch.setattr(WorkbenchJobs, "_restore_jobs", restore_jobs)
    monkeypatch.setattr(WorkbenchJobs, "create_url", lambda self, *a: calls.append("create"))
    monkeypatch.setattr(WorkbenchJobs, "prepare", lambda self, job_id: self._jobs[job_id])
    monkeypatch.setattr(WorkbenchJobs, "run_all", run_all)
    jobs = WorkbenchJobs(restore=True)
    _wait_idle(jobs)
    assert calls == ["job-a"]
    assert jobs.batch_snapshot()["items"][0]["status"] == "done"
    assert restored.stages["extract"]["status"] == "completed"


def test_last_ffmpeg_stderr_line_is_reported():
    import tempfile
    with tempfile.TemporaryFile() as f:
        f.write(b"[https] HTTP error 403 Forbidden\nError opening input files: Server returned 403 Forbidden (access denied)\n\n")
        assert "403 Forbidden" in download._last_ffmpeg_stderr_line([f])
    assert download._last_ffmpeg_stderr_line([]) == ""


def test_non_retryable_download_error_keeps_ffmpeg_detail(tmp_path, monkeypatch):
    import contextlib
    import tempfile

    class DownloadError(Exception):
        pass

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            raise DownloadError("ERROR: ffmpeg exited with code 1")

    stderr = tempfile.TemporaryFile()
    stderr.write(b"Error opening input files: Server returned 404 Not Found\n")

    @contextlib.contextmanager
    def fake_ffmpeg(check, profile):
        yield [stderr]

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setattr(download, "_cancellable_external_ffmpeg", fake_ffmpeg)
    with pytest.raises(Exception) as caught:
        download.download_with_ytdlp("https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)
    stderr.close()
    assert "404 Not Found" in str(caught.value)


def test_restore_removes_next_candidates_only(tmp_path, monkeypatch):
    work = tmp_path / "job"
    work.mkdir()
    for name in ("subtitle.next-ab12.ass", "subtitle.next-ab12.srt", "rendered.next-cd34.mp4", "rendered.mp4", "subtitle.ass", "translated.srt"):
        (work / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(workbench_api, "USER_DATA_ROOT", tmp_path)
    monkeypatch.setattr(workbench_api.job_db, "list_jobs", lambda status: [{"id": "j1", "status": "completed", "output_dir": str(work), "source_url": "https://a/1"}])
    WorkbenchJobs(restore=True)
    assert sorted(path.name for path in work.iterdir() if "next-" in path.name) == []
    assert {"rendered.mp4", "subtitle.ass", "translated.srt"} <= {path.name for path in work.iterdir()}


def test_batch_snapshot_includes_job_stage_and_progress(jobs):
    jobs.batch_add([{"url": "https://a/1"}], "cpu", "int8", {}, False)
    _wait_idle(jobs)
    item = jobs.batch_snapshot()["items"][0]
    job = jobs._jobs[item["job_id"]]
    job.stage, job.progress = "render", 42
    item = jobs.batch_snapshot()["items"][0]
    assert (item["stage"], item["progress"]) == ("render", 42)
    jobs.batch_add([{"url": "https://a/invalid"}], "cpu", "int8", {}, False)
    _wait_idle(jobs)
    assert (jobs.batch_snapshot()["items"][1]["stage"], jobs.batch_snapshot()["items"][1]["progress"]) == (None, None)

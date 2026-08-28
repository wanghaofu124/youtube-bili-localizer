from pathlib import Path
import sys
import threading
import time

import pytest

from yblocalizer.dependencies import (
    DependencyManager,
    DependencyInstallCancelled,
    InstallJob,
    SPECS,
    dependency_statuses,
    _cached_package_matches,
    clear_dependency_cache,
    resolve_whisper_model,
)


def _row(dependency_id: str, *, available: bool = False, installable: bool = True) -> dict:
    return {
        "id": dependency_id,
        "label": dependency_id,
        "purpose": "test",
        "required": dependency_id == "ffmpeg",
        "available": available,
        "installable": installable,
        "size_hint": "1 MB",
        "action_url": None,
        "message": "ready" if available else "install",
    }


def test_dependency_statuses_keep_optional_components_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: Path(name) if name in {"ffmpeg", "winget"} else None)
    monkeypatch.setattr("yblocalizer.dependencies.webview2_available", lambda: True)
    monkeypatch.setattr("yblocalizer.dependencies._playwright_available", lambda: False)
    rows = dependency_statuses(tmp_path)
    assert next(item for item in rows if item["id"] == "ffmpeg")["available"] is True
    assert next(item for item in rows if item["id"] == "tesseract")["required"] is False
    assert next(item for item in rows if item["id"] == "whisper-small")["installable"] is True


def test_managed_whisper_model_is_used_only_after_validation(tmp_path: Path) -> None:
    assert resolve_whisper_model("small", tmp_path) == "small"
    model = tmp_path / "models" / "faster-whisper-small"
    model.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (model / name).write_bytes(b"ok")
    assert resolve_whisper_model("small", tmp_path) == str(model)


def test_dependency_install_requires_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="确认"):
        DependencyManager(tmp_path).start("ffmpeg", confirmed=False)


def test_dependency_manager_serializes_install_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row("ffmpeg"), _row("node")]
    monkeypatch.setattr("yblocalizer.dependencies.dependency_statuses", lambda root=None: rows)
    monkeypatch.setattr("yblocalizer.dependencies.threading.Thread.start", lambda self: None)
    manager = DependencyManager(tmp_path)
    first = manager.start("ffmpeg", confirmed=True)
    assert first.status == "queued"
    with pytest.raises(RuntimeError, match="正在安装"):
        manager.start("node", confirmed=True)


def test_dependency_failure_is_retryable_and_keeps_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DependencyManager(tmp_path)
    job = InstallJob("job", "ffmpeg")
    monkeypatch.setattr(manager, "_install_winget", lambda current, dependency_id: (_ for _ in ()).throw(RuntimeError("network unavailable")))
    manager._run(job, next(spec for spec in SPECS if spec.id == "ffmpeg"))
    assert job.status == "failed"
    assert job.finished_at is not None
    assert "network unavailable" in (job.error or "")


def test_installed_dependency_returns_completed_without_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.dependencies.dependency_statuses", lambda root=None: [_row("ffmpeg", available=True, installable=False)])
    job = DependencyManager(tmp_path).start("ffmpeg", confirmed=True)
    assert job.status == "completed"
    assert job.progress == 100


def test_dependency_cancel_marks_job_and_terminates_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def poll(self): return None

    manager = DependencyManager(tmp_path)
    job = InstallJob("job", "node", status="running")
    job._process = FakeProcess()  # type: ignore[assignment]
    manager._jobs[job.id] = job
    manager._active_id = job.id
    terminated = []
    monkeypatch.setattr("yblocalizer.dependencies._terminate_process_tree", lambda process: terminated.append(process))
    result = manager.cancel(job.id)
    assert result is job
    assert job.status == "cancelling"
    assert job._cancel_event.is_set()
    assert terminated == [job._process]


def test_cancelled_job_finishes_as_cancelled(tmp_path: Path) -> None:
    manager = DependencyManager(tmp_path)
    job = InstallJob("job", "ffmpeg")
    job._cancel_event.set()
    manager._run(job, next(spec for spec in SPECS if spec.id == "ffmpeg"))
    assert job.status == "cancelled"
    assert job.error is None
    assert job.finished_at is not None


def test_winget_recursive_scan_is_cached_until_invalidated(tmp_path: Path) -> None:
    packages = tmp_path / "packages"
    first = packages / "one" / "tool.exe"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    clear_dependency_cache()
    assert _cached_package_matches(packages, "tool.exe") == [first]
    second = packages / "two" / "tool.exe"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"two")
    assert _cached_package_matches(packages, "tool.exe") == [first]
    clear_dependency_cache()
    assert set(_cached_package_matches(packages, "tool.exe")) == {first, second}


def test_manager_shutdown_terminates_real_child_process(tmp_path: Path) -> None:
    manager = DependencyManager(tmp_path)
    job = InstallJob("job", "playwright-chromium", status="running")
    manager._jobs[job.id] = job
    manager._active_id = job.id
    errors: list[Exception] = []

    def run_child() -> None:
        try:
            manager._stream_command(job, [sys.executable, "-c", "import time; time.sleep(30)"])
        except Exception as exc:
            errors.append(exc)

    job._thread = threading.Thread(target=run_child)
    job._thread.start()
    deadline = time.time() + 5
    while job._process is None and time.time() < deadline:
        time.sleep(0.02)
    assert job._process is not None
    child = job._process
    manager.shutdown(timeout=5)
    job._thread.join(timeout=5)
    assert not job._thread.is_alive()
    assert child.poll() is not None
    assert errors and isinstance(errors[0], DependencyInstallCancelled)

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
    ollama_local_status,
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


def test_dependency_snapshot_detects_new_whisper_model_without_recreating_manager(tmp_path: Path) -> None:
    manager = DependencyManager(tmp_path)
    assert next(item for item in manager.snapshot()["dependencies"] if item["id"] == "whisper-large-v3")["available"] is False

    model = tmp_path / "models" / "faster-whisper-large-v3"
    model.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (model / name).write_bytes(b"model")

    assert next(item for item in manager.snapshot()["dependencies"] if item["id"] == "whisper-large-v3")["available"] is True


def test_ollama_probe_matches_exact_local_model_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_json(path: str, *, timeout: float = 2.0):
        if path == "/api/version":
            return {"version": "test"}
        return {"models": [{"name": "qwen3.5:4b"}, {"name": "qwen3.5:4b-custom"}, {"name": "qwen3.5:9b"}]}

    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: None)
    monkeypatch.setattr("yblocalizer.dependencies._ollama_json", fake_json)
    status = ollama_local_status()
    assert status["runtime_available"] is True
    assert status["models"] == {
        "ollama-qwen35-4b": True,
        "ollama-qwen35-9b": True,
        "ollama-translategemma-12b": False,
    }


def test_ollama_model_dependency_waits_for_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: None)
    monkeypatch.setattr("yblocalizer.dependencies.webview2_available", lambda: True)
    monkeypatch.setattr("yblocalizer.dependencies._playwright_available", lambda: False)
    monkeypatch.setattr("yblocalizer.dependencies.ollama_local_status", lambda: {
        "runtime_state": "stopped", "runtime_available": False,
        "runtime_message": "start Ollama", "models": {
            "ollama-qwen35-4b": False, "ollama-qwen35-9b": False,
            "ollama-translategemma-12b": False,
        },
    })
    dependency_rows = dependency_statuses(tmp_path)
    runtime = next(item for item in dependency_rows if item["id"] == "ollama-runtime")
    model = next(item for item in dependency_rows if item["id"] == "ollama-qwen35-4b")
    assert runtime["state"] == "stopped" and not runtime["installable"]
    assert model["state"] == "blocked" and not model["installable"]
    assert model["depends_on"] == "ollama-runtime"


def test_start_ollama_uses_installed_cli_once_and_accepts_running_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = DependencyManager(tmp_path)
    runtime_state = {"state": "stopped"}
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "existing-models"))
    monkeypatch.setattr("yblocalizer.dependencies.ollama_local_status", lambda: {
        "runtime_state": runtime_state["state"],
        "runtime_available": runtime_state["state"] == "ready",
        "runtime_message": "Ollama is stopped",
    })
    executable = tmp_path / "ollama.exe"
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: executable)
    monkeypatch.setattr("yblocalizer.dependencies.time.sleep", lambda _seconds: None)
    launches: list[tuple[list[str], dict]] = []

    class RunningProcess:
        def poll(self) -> None:
            return None

    def launch(command: list[str], **options):
        launches.append((command, options))
        return RunningProcess()

    monkeypatch.setattr("yblocalizer.dependencies.subprocess.Popen", launch)

    assert manager.start_ollama()["state"] == "starting"
    assert manager.start_ollama()["state"] == "starting"
    assert len(launches) == 1
    assert launches[0][0] == [str(executable), "serve"]
    assert launches[0][1]["stdin"] is not None
    assert launches[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert launches[0][1]["env"]["OLLAMA_MODELS"] == str(tmp_path / "existing-models")

    runtime_state["state"] = "ready"
    assert manager.start_ollama()["state"] == "ready"
    assert len(launches) == 1
    terminated: list[object] = []
    monkeypatch.setattr("yblocalizer.dependencies._terminate_process_tree", lambda process: terminated.append(process))
    manager.shutdown()
    assert len(terminated) == 1


def test_start_ollama_prefers_windows_desktop_app_and_does_not_stop_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if sys.platform != "win32":
        pytest.skip("The Ollama desktop app is a Windows installation detail.")
    app_dir = tmp_path / "Ollama"
    app_dir.mkdir()
    executable = app_dir / "ollama.exe"
    desktop_app = app_dir / "ollama app.exe"
    desktop_app.write_bytes(b"app")
    monkeypatch.setattr("yblocalizer.dependencies.ollama_local_status", lambda: {
        "runtime_state": "stopped", "runtime_available": False,
    })
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: executable)
    launched: list[list[str]] = []

    class RunningProcess:
        def poll(self) -> None:
            return None

    monkeypatch.setattr("yblocalizer.dependencies.subprocess.Popen", lambda command, **_options: (launched.append(command), RunningProcess())[1])
    stopped: list[object] = []
    monkeypatch.setattr("yblocalizer.dependencies._terminate_process_tree", lambda process: stopped.append(process))

    manager = DependencyManager(tmp_path)
    assert manager.start_ollama()["state"] == "starting"
    assert launched == [[str(desktop_app)]]
    manager.shutdown()
    assert stopped == []


def test_start_ollama_reports_cli_exit_with_log_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = DependencyManager(tmp_path)
    monkeypatch.setattr("yblocalizer.dependencies.dependency_statuses", lambda _root=None: [
        {"id": "ollama-runtime", "available": False, "required": False, "message": "stopped"},
    ])
    monkeypatch.setattr("yblocalizer.dependencies.ollama_local_status", lambda: {
        "runtime_state": "stopped", "runtime_available": False,
    })
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: tmp_path / "ollama.exe")
    monkeypatch.setattr("yblocalizer.dependencies.time.sleep", lambda _seconds: None)

    class ExitedProcess:
        returncode = 1

        def poll(self) -> int:
            return 1

    def fail_serve(_command: list[str], **options):
        options["stdout"].write(b"listen tcp 127.0.0.1:11434: bind failed")
        options["stdout"].flush()
        return ExitedProcess()

    monkeypatch.setattr("yblocalizer.dependencies.subprocess.Popen", fail_serve)
    with pytest.raises(RuntimeError, match="bind failed"):
        manager.start_ollama()
    runtime = next(row for row in manager.snapshot()["dependencies"] if row["id"] == "ollama-runtime")
    assert "bind failed" in runtime["message"]


def test_ollama_model_download_uses_only_registered_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DependencyManager(tmp_path)
    job = InstallJob("pull", "ollama-qwen35-4b")
    monkeypatch.setenv("OLLAMA_HOST", "192.0.2.1:9999")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "existing-models"))
    monkeypatch.setattr("yblocalizer.dependencies.ollama_local_status", lambda: {"runtime_available": True})
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: Path("ollama.exe"))
    commands: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(manager, "_stream_command", lambda current, command, env=None: commands.append((command, env)))
    manager._install_ollama_model(job, "qwen3.5:4b")
    assert commands[0][0] == ["ollama.exe", "pull", "qwen3.5:4b"]
    assert commands[0][1]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert commands[0][1]["OLLAMA_MODELS"] == str(tmp_path / "existing-models")
    manager._install_ollama_model(InstallJob("pull-gemma", "ollama-translategemma-12b"), "translategemma:12b")
    assert commands[-1][0] == ["ollama.exe", "pull", "translategemma:12b"]
    with pytest.raises(RuntimeError, match="不允许下载"):
        manager._install_ollama_model(job, "untrusted/model:latest")


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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yblocalizer.models import Segment, save_segments
from yblocalizer.runtime import CancellationRequested
from yblocalizer.subtitle import write_srt
from yblocalizer.transcribe import WhisperCudaOutOfMemory, _run_whisper_worker, _transcribe_audio_isolated
from yblocalizer.whisper_worker import main as whisper_worker_main


def test_worker_failure_preserves_existing_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segments = tmp_path / "segments.json"
    subtitles = tmp_path / "source.srt"
    segments.write_text("old-json", encoding="utf-8")
    subtitles.write_text("old-srt", encoding="utf-8")
    monkeypatch.setattr(
        "yblocalizer.transcribe._run_whisper_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native crash")),
    )

    with pytest.raises(RuntimeError, match="native crash"):
        _transcribe_audio_isolated(
            tmp_path / "audio.wav", segments, subtitles, "small", None,
            "cpu", "int8", None, 5, None, None, "balanced",
        )

    assert segments.read_text(encoding="utf-8") == "old-json"
    assert subtitles.read_text(encoding="utf-8") == "old-srt"


def test_auto_device_worker_crash_retries_once_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_worker(control_dir: Path, _audio: Path, _model: str, _language: str | None,
                    device: str, compute_type: str, *_args):
        calls.append((device, compute_type))
        if device == "auto":
            raise RuntimeError("access violation")
        result_json = control_dir / "result.json"
        result_srt = control_dir / "result.srt"
        items = [Segment(0, 1, "hello")]
        save_segments(result_json, items)
        write_srt(result_srt, items, display_mode="source")
        return items, result_json, result_srt

    monkeypatch.setattr("yblocalizer.transcribe._run_whisper_worker", fake_worker)
    logs: list[str] = []
    items = _transcribe_audio_isolated(
        tmp_path / "audio.wav", tmp_path / "segments.json", tmp_path / "source.srt",
        "small", None, "auto", "float16", None, 5, logs.append, None, "balanced",
    )

    assert calls == [("auto", "float16"), ("cpu", "int8")]
    assert items[0].text == "hello"
    assert any("主程序仍在运行" in line for line in logs)


def test_explicit_cuda_failure_never_silently_retries_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fail(_control_dir: Path, _audio: Path, _model: str, _language: str | None,
             device: str, compute_type: str, *_args):
        calls.append((device, compute_type))
        raise RuntimeError("CUDA runtime failure")

    monkeypatch.setattr("yblocalizer.transcribe._run_whisper_worker", fail)
    with pytest.raises(RuntimeError, match="CUDA runtime failure"):
        _transcribe_audio_isolated(
            tmp_path / "audio.wav", tmp_path / "segments.json", tmp_path / "source.srt",
            "small", None, "cuda", "float16", None, 5, None, None, "balanced",
        )
    assert calls == [("cuda", "float16")]


def test_explicit_cuda_oom_stops_and_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fail(_control_dir: Path, _audio: Path, _model: str, _language: str | None,
             device: str, compute_type: str, *_args):
        calls.append((device, compute_type))
        raise WhisperCudaOutOfMemory("Whisper CUDA 显存不足")

    monkeypatch.setattr("yblocalizer.transcribe._run_whisper_worker", fail)
    with pytest.raises(WhisperCudaOutOfMemory):
        _transcribe_audio_isolated(
            tmp_path / "audio.wav", tmp_path / "segments.json", tmp_path / "source.srt",
            "large-v3", None, "cuda", "float16", None, 5, None, None, "balanced",
        )
    assert calls == [("cuda", "float16")]


def test_whisper_worker_returns_stable_cuda_oom_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps({
        "audio_path": str(tmp_path / "audio.wav"),
        "segments_json": str(tmp_path / "segments.json"),
        "srt_path": str(tmp_path / "source.srt"),
        "result_path": str(result_path),
        "events_path": str(tmp_path / "events.jsonl"),
        "model_size": "large-v3",
        "language": "en",
        "device": "cuda",
        "compute_type": "float16",
        "initial_prompt": None,
        "beam_size": 5,
        "resource_profile": "balanced",
    }), encoding="utf-8")
    monkeypatch.setenv("YBLOCALIZER_WHISPER_WORKER", "0")
    monkeypatch.setattr(
        "yblocalizer.transcribe._transcribe_audio_in_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(WhisperCudaOutOfMemory("CUDA out of memory")),
    )

    assert whisper_worker_main([str(request_path)]) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["error_code"] == "cuda_oom"


def test_cpu_worker_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("worker failed")

    monkeypatch.setattr("yblocalizer.transcribe._run_whisper_worker", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        _transcribe_audio_isolated(
            tmp_path / "audio.wav", tmp_path / "segments.json", tmp_path / "source.srt",
            "small", None, "cpu", "int8", None, 5, None, None, "balanced",
        )
    assert calls == 1


def test_cancellation_terminates_whisper_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Process:
        returncode: int | None = None
        terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: int) -> int:
            return self.returncode or 0

    process = Process()
    monkeypatch.setattr("yblocalizer.transcribe.subprocess.Popen", lambda *_args, **_kwargs: process)

    def cancel() -> None:
        raise CancellationRequested("cancelled")

    with pytest.raises(CancellationRequested):
        _run_whisper_worker(
            tmp_path, tmp_path / "audio.wav", "small", None, "cpu", "int8",
            None, 5, lambda _message: None, cancel, "balanced",
        )
    assert process.terminated is True

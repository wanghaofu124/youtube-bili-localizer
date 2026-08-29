from __future__ import annotations

from pathlib import Path

import pytest

from yblocalizer.models import Segment, save_segments
from yblocalizer.runtime import CancellationRequested
from yblocalizer.subtitle import write_srt
from yblocalizer.transcribe import _run_whisper_worker, _transcribe_audio_isolated


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


def test_cuda_worker_crash_retries_once_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_worker(control_dir: Path, _audio: Path, _model: str, _language: str | None,
                    device: str, compute_type: str, *_args):
        calls.append((device, compute_type))
        if device == "cuda":
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
        "small", None, "cuda", "float16", None, 5, logs.append, None, "balanced",
    )

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert items[0].text == "hello"
    assert any("主程序仍在运行" in line for line in logs)


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

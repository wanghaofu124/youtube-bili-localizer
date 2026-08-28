from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from yblocalizer.download import download_with_ytdlp
from yblocalizer.performance import cpu_thread_limit, download_rate_limit, ffmpeg_thread_args
from yblocalizer.render import burn_subtitles
from yblocalizer.transcribe import transcribe_audio
from yblocalizer.workbench_api import DemoJob, Material, WorkbenchJobs


def test_long_jobs_send_only_incremental_bounded_logs(tmp_path: Path) -> None:
    job = DemoJob(
        id="perf-job",
        material=Material("m", tmp_path / "in.mp4", "in.mp4", 10, 1280, 720, True),
        device="cpu",
        compute_type="int8",
    )
    for index in range(20_000):
        job._append_log(f"line {index}")

    assert len(job.logs) == 2_000
    initial = job.snapshot(log_limit=100)
    idle_poll = job.snapshot(log_after=initial["log_cursor"], log_limit=100)
    assert len(initial["logs"]) == 100
    assert idle_poll["logs"] == []
    assert len(json.dumps(idle_poll, ensure_ascii=False)) < 20_000


def test_resource_profiles_have_explicit_cpu_and_network_limits() -> None:
    assert download_rate_limit("background") == 2 * 1024 * 1024
    assert download_rate_limit("balanced") == 6 * 1024 * 1024
    assert download_rate_limit("maximum") is None
    assert cpu_thread_limit("background") <= cpu_thread_limit("balanced") <= cpu_thread_limit("maximum")
    assert ffmpeg_thread_args("balanced")[0] == "-threads"
    assert ffmpeg_thread_args("maximum") == []


def test_balanced_download_applies_rate_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, url: str, download: bool):
            assert download is True
            (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "Demo", "webpage_url": url}

    monkeypatch.setattr("yblocalizer.download.po_token_provider_status", lambda: {"available": False, "browser_path": None})
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    download_with_ytdlp("https://example.test/video", tmp_path, resource_profile="balanced")

    assert captured["ratelimit"] == 6 * 1024 * 1024


def test_balanced_render_limits_ffmpeg_threads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr("yblocalizer.render.require_command", lambda _name: None)
    monkeypatch.setattr("yblocalizer.render._probe_video_size", lambda _path: (1920, 1080))
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.render.run", lambda command, **_kwargs: commands.append(command))

    burn_subtitles(
        tmp_path / "in.mp4",
        tmp_path / "zh.srt",
        tmp_path / "out.mp4",
        encoder="cpu",
        resource_profile="balanced",
    )

    assert "-threads" in commands[0]


def test_transcription_passes_profile_thread_limit_to_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict = {}

    class FakeWhisperModel:
        def __init__(self, _model: str, **kwargs):
            captured.update(kwargs)

        def transcribe(self, _audio: str, **_kwargs):
            return [SimpleNamespace(start=0.0, end=1.0, text="hello")], {}

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr("yblocalizer.transcribe.require_whisper_model", lambda _size: tmp_path / "model")
    monkeypatch.setattr("yblocalizer.transcribe._align_segment_starts", lambda segments, _audio: segments)

    transcribe_audio(
        tmp_path / "audio.wav",
        tmp_path / "segments.json",
        tmp_path / "source.srt",
        device="cpu",
        compute_type="int8",
        resource_profile="background",
    )

    assert captured["cpu_threads"] == cpu_thread_limit("background")
    assert captured["num_workers"] == 1


def test_resource_profile_change_preserves_completed_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    for row in job.stages.values():
        row["status"], row["progress"] = "completed", 100
    job.status = "completed"
    job.stage = "处理完成"
    job.checks = [{"status": "passed"}]

    updated, stale = jobs.update_options(job.id, {"options": {"resource_profile": "background"}})

    assert stale == []
    assert updated.status == "completed"
    assert all(row["status"] == "completed" for row in updated.stages.values())
    assert updated.checks == [{"status": "passed"}]

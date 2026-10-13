from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yblocalizer.download import (
    DownloadProposalMismatch,
    DownloadResolutionMismatch,
    _quality_proposal,
    _raise_with_youtube_hint,
    download_with_ytdlp,
)
from yblocalizer.runtime import CancellationRequested, PipelineContext
from yblocalizer.translate import _OllamaProcessClient
from yblocalizer import ollama_worker
from yblocalizer.workflow import WorkflowArtifacts, _acquire_cache_dir, run_acquire, run_render
from yblocalizer.workflow import atomic_write_manifest
from yblocalizer.models import Segment, save_segments
from yblocalizer.workbench_api import WorkbenchJobs, _carry_quality_baseline


def test_quality_proposal_reads_ytdlp_selected_muxed_stream() -> None:
    mux4k = {"format_id": "mux4k", "width": 3840, "height": 2160, "vcodec": "av01", "acodec": "opus"}
    proposal = _quality_proposal({"requested_formats": [mux4k], "formats": [
        mux4k,
        {"format_id": "v360", "width": 640, "height": 360, "vcodec": "avc1", "acodec": "none"},
        {"format_id": "a360", "abr": 96, "vcodec": "none", "acodec": "mp4a"},
    ]}, "original")
    assert proposal["video_format_id"] == "mux4k"
    assert proposal["audio_format_id"] == ""
    assert proposal["height"] == 2160
    assert proposal["requires_consent"] is False


def test_quality_proposal_does_not_override_ytdlp_muxed_choice() -> None:
    mux4k = {"format_id": "mux4k", "width": 3840, "height": 2160, "vcodec": "av01", "acodec": "opus"}
    proposal = _quality_proposal({"requested_formats": [mux4k], "formats": [
        {"format_id": "silent8k", "width": 7680, "height": 4320, "vcodec": "av01", "acodec": "none"},
        mux4k,
    ]}, "original")
    assert proposal["video_format_id"] == "mux4k"
    assert proposal["has_audio"] is True


def test_quality_proposal_uses_ytdlp_selected_fps() -> None:
    high_fps = {"format_id": "60fps", "width": 3840, "height": 2160, "fps": 60, "vbr": 800, "vcodec": "av01", "acodec": "none"}
    proposal = _quality_proposal({"requested_formats": [high_fps], "formats": [
        {"format_id": "24fps", "width": 3840, "height": 2160, "fps": 24, "vbr": 1200, "vcodec": "av01", "acodec": "none"},
        high_fps,
    ]}, "original")
    assert proposal["video_format_id"] == "60fps"


def test_same_4k_fps_uses_ytdlp_selection_not_raw_bitrate() -> None:
    av1 = {"format_id": "401", "width": 3840, "height": 2160, "fps": 50, "vbr": 13212, "vcodec": "av01", "acodec": "none"}
    audio = {"format_id": "251", "abr": 160, "vcodec": "none", "acodec": "opus"}
    proposal = _quality_proposal({"requested_formats": [av1, audio], "formats": [
        {"format_id": "315", "width": 3840, "height": 2160, "fps": 50, "vbr": 26325, "vcodec": "vp9", "acodec": "none"},
        av1,
        audio,
    ]}, "original")
    assert (proposal["video_format_id"], proposal["audio_format_id"], proposal["height"]) == ("401", "251", 2160)


def test_acquire_cache_changes_only_with_download_contract(tmp_path: Path) -> None:
    options = SimpleNamespace(source="https://example.test/v", download_quality="original", max_seconds=None,
                              source_language="en", prefer_platform_subtitles=False,
                              quality_proposal={"video_format_id": "401", "audio_format_id": "251"})
    first = _acquire_cache_dir(tmp_path, options)
    assert first == _acquire_cache_dir(tmp_path, options)
    options.quality_proposal = {"video_format_id": "315", "audio_format_id": "251"}
    assert first != _acquire_cache_dir(tmp_path, options)
    options.quality_proposal = {"video_format_id": "401", "audio_format_id": "251"}
    options.max_seconds = 30
    assert first != _acquire_cache_dir(tmp_path, options)


def test_quality_proposal_separate_low_stream_requires_consent() -> None:
    video = {"format_id": "v360", "width": 640, "height": 360, "vcodec": "avc1", "acodec": "none"}
    audio = {"format_id": "a", "abr": 128, "vcodec": "none", "acodec": "mp4a"}
    proposal = _quality_proposal({"requested_formats": [video, audio]}, "720p")
    assert proposal["video_format_id"] == "v360"
    assert proposal["audio_format_id"] == "a"
    assert proposal["requires_consent"] is True


def test_original_quality_baseline_survives_repeated_preparation() -> None:
    original = {"quality": "original", "video_format_id": "old4k", "width": 3840, "height": 2160,
                "has_audio": True, "requires_consent": False}
    lower = {"quality": "original", "video_format_id": "new1080", "width": 1920, "height": 1080,
             "has_audio": True, "requires_consent": False}
    first = _carry_quality_baseline(original, None)
    second = _carry_quality_baseline(lower, first)
    third = _carry_quality_baseline(lower, second)
    assert (third["baseline_width"], third["baseline_height"]) == (3840, 2160)
    assert second["requires_consent"] is True
    assert third["requires_consent"] is True


def test_quality_fingerprint_uses_resolution_not_ephemeral_format_ids(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = jobs.create_url("https://example.test/v", "cpu", "int8", True, {"output_dir": str(tmp_path)})
    proposal = {"video_format_id": "old4k", "audio_format_id": "oldaudio", "width": 3840,
                "height": 2160, "has_audio": True, "baseline_width": 3840, "baseline_height": 2160}
    changed_ids = {**proposal, "video_format_id": "new4k", "audio_format_id": "newaudio"}
    assert jobs._quality_fingerprint(job, proposal) == jobs._quality_fingerprint(job, changed_ids)
    assert jobs._quality_fingerprint(job, proposal) != jobs._quality_fingerprint(job, {**proposal, "height": 1080})


def test_missing_proposed_resolution_is_rejected_after_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    constructed = False

    class FakeYDL:
        def __init__(self, options):
            nonlocal constructed
            constructed = True
            self.options = options
        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def extract_info(self, _url, download):
            (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "demo", "width": 1280, "height": 720}
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (1280, 720))
    with pytest.raises(DownloadProposalMismatch, match="画质记录不完整"):
        download_with_ytdlp("https://example.test/v", tmp_path, quality_proposal={"video_format_id": ""})
    assert constructed is False


def test_proposal_resolution_is_checked_when_ytdlp_selected_dims_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def extract_info(self, _url, download):
            (tmp_path / "demo.mp4").write_bytes(b"video")
            return {"id": "demo", "title": "demo", "requested_formats": []}
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    monkeypatch.setattr("yblocalizer.download._probe_video_dimensions", lambda _path: (640, 360))
    proposal = {"video_format_id": "v1080", "audio_format_id": "", "width": 1920, "height": 1080}
    with pytest.raises(Exception, match="1920×1080.*640×360"):
        download_with_ytdlp("https://example.test/v", tmp_path, quality_proposal=proposal)


def test_cancelled_error_never_becomes_quality_retry() -> None:
    with pytest.raises(CancellationRequested):
        _raise_with_youtube_hint(CancellationRequested("cancelled"), None)


def test_cancel_during_download_is_not_converted_to_network_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, options): self.options = options
        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def extract_info(self, _url, download):
            assert download is True
            (tmp_path / "video.part").write_bytes(b"partial")
            self.options["progress_hooks"][0]({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2})
            raise AssertionError("cancel should stop the download")
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    calls = [0]
    def cancel():
        calls[0] += 1
        if calls[0] >= 2:
            raise CancellationRequested("cancelled during download")
    with pytest.raises(CancellationRequested):
        download_with_ytdlp("https://example.test/v", tmp_path, download_quality="1080p", cancel_check=cancel)
    assert calls[0] >= 2


def test_acquire_failure_keeps_old_video_hash_and_pointer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    old = tmp_path / "old.mp4"
    old.write_bytes(b"known-good")
    before = hashlib.sha256(old.read_bytes()).hexdigest()
    artifacts = WorkflowArtifacts(source="https://example.test/v", source_kind="url", raw_video=str(old), original_video=str(old))
    options = SimpleNamespace(
        source="https://example.test/v", source_kind="url", title=None, i_have_rights=True,
        max_seconds=None, require_reuse_allowed=False, cookies_from_browser=None, cookies_file=None,
        youtube_po_token_mode="off", youtube_proxy=None, download_quality="1080p", resource_profile="balanced",
        source_language=None, prefer_platform_subtitles=False,
    )
    def fail(*_args, **_kwargs):
        raise RuntimeError("network failed")
    monkeypatch.setattr("yblocalizer.workflow.download_with_ytdlp", fail)
    with pytest.raises(RuntimeError):
        run_acquire(options, tmp_path / "work", artifacts, PipelineContext(), lambda _m: None)
    assert artifacts.raw_video == str(old)
    assert hashlib.sha256(old.read_bytes()).hexdigest() == before


def test_ollama_worker_cancel_terminates_child(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 4242
        returncode = None
        def communicate(self, *_a, **_k):
            raise __import__("subprocess").TimeoutExpired("worker", 0.2)
        def poll(self): return None
        def wait(self, **_k): self.returncode = -9
        def kill(self): self.returncode = -9
        def terminate(self): self.returncode = -15
    process = FakeProcess()
    monkeypatch.setattr("yblocalizer.translate.subprocess.Popen", lambda *_a, **_k: process)
    monkeypatch.setattr("yblocalizer.translate._terminate_child", lambda p: setattr(p, "returncode", -9))
    calls = [0]
    def cancel():
        calls[0] += 1
        raise CancellationRequested("stop")
    with pytest.raises(CancellationRequested):
        _OllamaProcessClient(cancel).create(model="qwen3.5:4b", messages=[{"role": "user", "content": "x"}], temperature=0.1)
    assert calls[0] >= 1 and process.returncode == -9


def test_ollama_worker_pipe_preserves_unicode_across_windows_code_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "¡Hola, niño!"
    output = "你好，世界"
    captured: dict[str, str] = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, payload: str, **_kwargs):
            captured["payload"] = payload
            return json.dumps({"status": "completed", "content": output}, ensure_ascii=True), ""

        def poll(self):
            return 0

    monkeypatch.setattr("yblocalizer.translate.subprocess.Popen", lambda *_args, **_kwargs: FakeProcess())
    response = _OllamaProcessClient(lambda: None).create(model="translategemma:12b", messages=[{"role": "user", "content": source}])
    assert captured["payload"].isascii()
    assert json.loads(captured["payload"])["kwargs"]["messages"][0]["content"] == source
    assert response.choices[0].message.content == output

    raw_output = io.BytesIO()
    encoded_output = io.TextIOWrapper(raw_output, encoding="gbk", write_through=True)
    monkeypatch.setattr(
        ollama_worker, "sys", SimpleNamespace(stdin=io.StringIO(captured["payload"]), stdout=encoded_output)
    )
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=lambda **_kwargs: object(), Timeout=lambda *_args, **_kwargs: object()))

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=output))]
            )))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    assert ollama_worker.main() == 0
    assert raw_output.getvalue().isascii()
    assert json.loads(raw_output.getvalue().decode("utf-8"))["content"] == output


def _render_options() -> SimpleNamespace:
    return SimpleNamespace(
        subtitle_display_mode="bilingual-source-first", smart_subtitle_layout=False, font_name="Arial", font_size=24,
        subtitle_color="&H00FFFFFF", subtitle_outline_color="&H00000000", subtitle_outline=1, subtitle_shadow=0,
        render_crf=20, render_encoder="cpu", resource_profile="balanced", subtitle_margin_ratio=0.055,
    )


def test_render_rebuilds_bilingual_srt_and_failure_keeps_old_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "raw.mp4"; video.write_bytes(b"video")
    old_render = tmp_path / "rendered.mp4"; old_render.write_bytes(b"old-render")
    translated = tmp_path / "segments.translated.json"
    save_segments(translated, [Segment(0, 1, "Hello", "你好")])
    artifacts = WorkflowArtifacts(raw_video=str(video), translated_segments=str(translated), translated_srt=str(tmp_path / "stale.srt"), rendered_video=str(old_render))
    (tmp_path / "stale.srt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_a: True)
    def fake_burn(_v, subtitle, output, **_kwargs):
        assert "Hello" in subtitle.read_text(encoding="utf-8") and "你好" in subtitle.read_text(encoding="utf-8")
        subtitle.with_suffix(".ass").write_text("temporary style", encoding="utf-8")
        output.write_bytes(b"new-render"); return output
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", fake_burn)
    result = run_render(_render_options(), tmp_path, artifacts, PipelineContext(), lambda _m: None)
    assert Path(result.rendered_video).read_bytes() == b"new-render"
    assert old_render.read_bytes() == b"old-render"
    assert not list(tmp_path.glob("subtitle.next-*.ass"))
    old_pointer = result.rendered_video
    def failed_burn(_v, subtitle, _output, **_kwargs):
        subtitle.with_suffix(".ass").write_text("temporary style", encoding="utf-8")
        raise RuntimeError("render failed")
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", failed_burn)
    with pytest.raises(RuntimeError):
        run_render(_render_options(), tmp_path, result, PipelineContext(), lambda _m: None)
    assert Path(old_pointer).read_bytes() == b"new-render"
    assert result.rendered_video == old_pointer
    assert not list(tmp_path.glob("subtitle.next-*.ass"))


def test_quality_consent_blocks_stage_entry_until_confirmed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = jobs.create_url("https://example.test/v", "cpu", "int8", True, {"output_dir": str(tmp_path)})
    proposal = {"video_format_id": "v360", "audio_format_id": "a", "width": 640, "height": 360, "has_audio": True, "requires_consent": True}
    proposal["fingerprint"] = jobs._quality_fingerprint(job, proposal)
    job.options["quality_proposal"] = proposal
    job.options["quality_consent"] = False
    job.checks = [{"status": "passed"}]
    monkeypatch.setattr(jobs, "_start_stage_sequence", lambda *_a, **_k: job)
    with pytest.raises(RuntimeError, match="尚未确认"):
        jobs.run_stage(job.id, "acquire")
    jobs.confirm_quality(job.id, proposal["fingerprint"])
    assert jobs.run_stage(job.id, "acquire") is job


def test_local_reimport_failure_keeps_registered_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"; source.write_bytes(b"source")
    old = tmp_path / "work" / "raw.mp4"; old.parent.mkdir(); old.write_bytes(b"old")
    artifacts = WorkflowArtifacts(source=str(source), source_kind="file", raw_video=str(old), original_video=str(old))
    options = SimpleNamespace(source=str(source), source_kind="file", title=None, i_have_rights=True)
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_a: False)
    with pytest.raises(RuntimeError, match="ffprobe"):
        run_acquire(options, tmp_path / "work", artifacts, PipelineContext(), lambda _m: None)
    assert artifacts.raw_video == str(old) and old.read_bytes() == b"old"


def test_versioned_artifact_pointers_survive_manifest_reload(tmp_path: Path) -> None:
    raw = tmp_path / "raw.v2.mp4"; source_srt = tmp_path / "source.v2.srt"; rendered = tmp_path / "rendered.v3.mp4"
    for path in (raw, source_srt, rendered): path.write_bytes(b"ok")
    artifacts = WorkflowArtifacts(raw_video=str(raw), original_video=str(raw), source_srt=str(source_srt), rendered_video=str(rendered))
    atomic_write_manifest(tmp_path / "job_manifest.json", {"artifacts": artifacts.to_dict(), "workflow_version": 2})
    restored = json.loads((tmp_path / "job_manifest.json").read_text(encoding="utf-8"))["artifacts"]
    assert restored["raw_video"] == str(raw) and restored["source_srt"] == str(source_srt) and restored["rendered_video"] == str(rendered)

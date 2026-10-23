from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import pytest

from yblocalizer import dub, workflow
from yblocalizer.models import Segment
from yblocalizer.runtime import CancellationToken, PipelineContext
from yblocalizer.workbench_config import default_options, normalize_options


def _seg(start, end, text="你好"):
    return Segment(start=start, end=end, text=text, translated_text=text)


def _wav(path: Path, seconds: float, sr: int = 100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sr)
        handle.writeframes(b"\x01\x00" * int(seconds * sr))


def test_core_stages_constant():
    assert workflow.CORE_STAGES == ("acquire", "extract", "translate", "render")
    assert workflow.WORKFLOW_STAGES == (*workflow.CORE_STAGES, "dub", "publish")
    assert workflow.STAGE_RUNNERS["dub"] is workflow.run_dub


def test_target_durations_borrow_gap_capped():
    segments = [_seg(0, 2), _seg(2.2, 4), _seg(6, 7)]
    assert dub.target_durations(segments, 7.3) == pytest.approx([2.2, 2.3, 1.3])


def test_target_durations_tiny_line_is_not_zero():
    assert dub.target_durations([_seg(1.0, 1.0002)], 1.0002)[0] > 0


def test_tempo_rules():
    assert dub.tempo_for(1.0, 2.0) == 1.0
    assert dub.tempo_for(2.4, 2.0) == pytest.approx(1.2)
    assert dub.tempo_for(5.0, 2.0) == dub.MAX_TEMPO


def test_align_command():
    plain = dub.align_command("ffmpeg", Path("a.wav"), Path("b.wav"), 2.0, 1.0, 48000)
    assert "-af" not in plain and plain[plain.index("-t") + 1] == "2.000"
    fast = dub.align_command("ffmpeg", Path("a.wav"), Path("b.wav"), 2.0, 1.35, 48000)
    assert fast[fast.index("-af") + 1] == "atempo=1.3500"


def test_mix_commands():
    duck = dub.mix_command("ffmpeg", Path("v.mp4"), Path("t.wav"), Path("o.mp4"), "duck")
    graph = duck[duck.index("-filter_complex") + 1]
    assert "volume=0.25" in graph and "sidechaincompress" in graph and "amix" in graph
    assert duck[duck.index("-c:v") + 1] == "copy"
    only = dub.mix_command("ffmpeg", Path("v.mp4"), Path("t.wav"), Path("o.mp4"), "dub_only")
    assert "-filter_complex" not in only and "1:a:0" in only and only[only.index("-c:v") + 1] == "copy"


def test_build_track_places_clips(tmp_path):
    clip = tmp_path / "c.wav"; _wav(clip, 0.5)
    out = tmp_path / "t.wav"
    dub.build_track([(1.0, clip)], 2.0, 100, out)
    with wave.open(str(out)) as handle:
        frames = handle.readframes(handle.getnframes())
    assert len(frames) == 400 and frames[:200] == bytes(200) and frames[200:202] == b"\x01\x00"


def test_dub_video_with_fake_worker(tmp_path, monkeypatch):
    video = tmp_path / "rendered.mp4"; video.write_bytes(b"v")
    calls, commands = [], []

    def fake_tts(python, model_dir, items, control_dir, log, cancel_check):
        calls.append([item["text"] for item in items])
        rows = []
        for item in items:
            seconds = 5.0 if item["text"] == "很长很长很长的一句" else 1.0
            _wav(Path(item["out"]), seconds)
            rows.append({"index": item["index"], "out": item["out"], "duration": seconds, "seconds": 0.1})
        return {"status": "completed", "sample_rate": 100, "items": rows, "peak_vram_bytes": 7}

    def fake_run(command, **kwargs):
        commands.append(command)
        out = Path(command[-1])
        if out.suffix == ".wav":
            _wav(out, 0.5)
        else:
            out.write_bytes(b"mix")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(dub, "probe_duration", lambda path: 6.0)
    monkeypatch.setattr(dub.subprocess, "run", fake_run)
    monkeypatch.setattr(dub.dependencies, "resolve_command", lambda name: name)
    segments = [_seg(0, 2, "第一句"), _seg(3, 4, "很长很长很长的一句")]
    output, report = dub.dub_video(
        video, segments, tmp_path, python="py", model_dir="m", mix_mode="duck",
        log=lambda m: None, cancel_check=lambda: False, rewrite=lambda text, ratio: "短句", tts=fake_tts,
    )
    assert output == tmp_path / "rendered_dub.mp4" and output.read_bytes() == b"mix"
    assert calls == [["第一句", "很长很长很长的一句"], ["短句"]]
    lines = json.loads(report.read_text(encoding="utf-8"))["lines"]
    assert lines[1]["rewritten"] and lines[1]["text"] == "短句" and not lines[1]["truncated"]
    assert video.read_bytes() == b"v"



def test_dub_video_rewrite_error_keeps_original(tmp_path, monkeypatch):
    video = tmp_path / "rendered.mp4"; video.write_bytes(b"v")

    def fake_tts(python, model_dir, items, control_dir, log, cancel_check):
        for item in items:
            _wav(Path(item["out"]), 5.0)
        return {"status": "completed", "sample_rate": 100,
                "items": [{"index": item["index"], "out": item["out"], "duration": 5.0} for item in items]}

    def fake_run(command, **kwargs):
        out = Path(command[-1])
        _wav(out, 0.5) if out.suffix == ".wav" else out.write_bytes(b"mix")
        return subprocess.CompletedProcess(command, 0, "", "")

    def broken_rewrite(text, ratio):
        raise RuntimeError("llm down")

    monkeypatch.setattr(dub, "probe_duration", lambda path: 2.0)
    monkeypatch.setattr(dub.subprocess, "run", fake_run)
    monkeypatch.setattr(dub.dependencies, "resolve_command", lambda name: name)
    output, report = dub.dub_video(
        video, [_seg(0, 1, "原文")], tmp_path, python="py", model_dir="m", mix_mode="duck",
        log=lambda m: None, cancel_check=lambda: False, rewrite=broken_rewrite, tts=fake_tts,
    )
    lines = json.loads(report.read_text(encoding="utf-8"))["lines"]
    assert output.read_bytes() == b"mix" and lines[0]["text"] == "原文" and not lines[0]["rewritten"]


def test_dub_video_without_text_skips_worker(tmp_path, monkeypatch):
    video = tmp_path / "rendered.mp4"; video.write_bytes(b"v")
    monkeypatch.setattr(dub, "probe_duration", lambda path: 2.0)
    monkeypatch.setattr(dub.dependencies, "resolve_command", lambda name: name)
    output, report = dub.dub_video(
        video, [_seg(0, 1, " ")], tmp_path, python="py", model_dir="m", mix_mode="duck",
        log=lambda m: None, cancel_check=lambda: False, rewrite=lambda t, r: None, tts=lambda *a: pytest.fail("worker started"),
    )
    assert output is None and json.loads(report.read_text(encoding="utf-8"))["lines"] == []


def test_run_dub_skipped_when_disabled(tmp_path):
    artifacts = workflow.WorkflowArtifacts(rendered_video=str(tmp_path / "r.mp4"))
    options = type("O", (), {"dub_enabled": False})()
    messages = []
    result = workflow.run_dub(options, tmp_path, artifacts, PipelineContext(CancellationToken(), lambda e: None), messages.append)
    assert result is artifacts and result.dub_video is None and "跳过" in messages[0]


def test_options_defaults_and_normalize():
    options = default_options()
    assert options["dub_enabled"] is False and options["dub_mix"] == "duck"
    normalized = normalize_options({"dub_enabled": True, "dub_mix": "dub_only"}, "out")
    assert normalized["dub_enabled"] is True and normalized["dub_mix"] == "dub_only"
    assert normalize_options({"dub_mix": "bogus"}, "out")["dub_mix"] == "duck"


def test_target_durations_zero_length_has_floor():
    assert dub.target_durations([_seg(1.0, 1.0)], 1.0) == [dub.MIN_TARGET]


def test_forward_events_skips_broken_lines(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text('{"message": "a"}\n{broken\n{"message": "b"}\n', encoding="utf-8")
    messages = []
    assert dub._forward_events(events, 0, messages.append) == events.stat().st_size
    assert messages == ["a", "b"]


def test_run_tts_worker_clears_old_events(tmp_path, monkeypatch):
    control = tmp_path / "ctl"; control.mkdir()
    (control / "events.jsonl").write_text('{"message": "old"}\n', encoding="utf-8")

    class Done:
        returncode = 0
        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        (control / "result.json").write_text('{"status": "completed"}', encoding="utf-8")
        return Done()

    monkeypatch.setattr(dub.subprocess, "Popen", fake_popen)
    messages = []
    dub.run_tts_worker("py", "m", [], control, messages.append, lambda: False)
    assert messages == []


def test_dub_video_reports_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(dub.dependencies, "resolve_command", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        dub.dub_video(tmp_path / "v.mp4", [], tmp_path, python="py", model_dir="m", mix_mode="duck",
                      log=print, cancel_check=lambda: False, rewrite=lambda t, r: None)


def test_duck_sidechain_is_keyed_by_dub():
    duck = dub.mix_command("ffmpeg", Path("v.mp4"), Path("t.wav"), Path("o.mp4"), "duck")
    graph = duck[duck.index("-filter_complex") + 1]
    assert "asplit=2[dub][key]" in graph and "[orig][key]sidechaincompress" in graph

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yblocalizer.models import Segment, save_segments
from yblocalizer.ocr_subtitle import (
    OCRCandidate,
    _adaptive_frame_candidate,
    _check_ocr_runtime,
    _find_tesseract_command,
    _frames_to_segments,
    _ocr_region_candidate,
    extract_ocr_subtitles,
    ocr_text_quality,
)
from yblocalizer.runtime import CancellationRequested, PipelineContext
from yblocalizer.subtitle import write_srt
from yblocalizer.workflow import WorkflowArtifacts, run_extract
from yblocalizer.workbench_config import preflight


def test_quality_filter_accepts_general_topics_and_unicode() -> None:
    assert ocr_text_quality("Authorized 10-second demo") >= 0.55
    assert ocr_text_quality("这是画面中的中文字幕") >= 0.55
    assert ocr_text_quality("xqzrt bcdffg nnnnnn") == 0.0


def test_adaptive_search_stops_after_a_strong_preferred_region(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    calls: list[str] = []

    def fake_candidate(_engine, _image, region, *_args, **_kwargs):
        calls.append(region)
        return OCRCandidate("Welcome to the demo", 0.91, region)

    monkeypatch.setattr("yblocalizer.ocr_subtitle._ocr_region_candidate", fake_candidate)
    result, attempted = _adaptive_frame_candidate(
        object(), Image.new("RGB", (1280, 720)), crop_bottom_ratio=0.3, language="eng", preferred_region="bottom"
    )

    assert result and result.region == "bottom"
    assert attempted == 1
    assert calls == ["bottom"]


def test_adaptive_search_expands_and_supports_moving_caption_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    calls: list[str] = []

    def fake_candidate(_engine, _image, region, *_args, **_kwargs):
        calls.append(region)
        if region == "middle":
            return OCRCandidate("A caption moved here", 0.88, region)
        return None

    monkeypatch.setattr("yblocalizer.ocr_subtitle._ocr_region_candidate", fake_candidate)
    result, attempted = _adaptive_frame_candidate(
        object(), Image.new("RGB", (1280, 720)), crop_bottom_ratio=0.3, language="eng", preferred_region="bottom"
    )

    assert result and result.region == "middle"
    assert attempted == 4
    assert calls == ["bottom", "full", "top", "middle"]


def test_frame_sequence_can_switch_regions_without_losing_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from PIL import Image

    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index:06d}.png"
        Image.new("RGB", (640, 360), "black").save(frame)
        frames.append(frame)
    candidates = iter([
        (OCRCandidate("First subtitle line", 0.9, "bottom"), 1),
        (OCRCandidate("First subtitle line", 0.9, "bottom"), 1),
        (OCRCandidate("Second subtitle moved", 0.9, "top"), 3),
    ])
    monkeypatch.setattr("yblocalizer.ocr_subtitle._adaptive_frame_candidate", lambda *_args, **_kwargs: next(candidates))
    stats: dict[str, int] = {}

    segments = _frames_to_segments(frames, 0.5, 3, stats=stats)

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0.0, 1.0, "First subtitle line"),
    ]
    assert stats["ocr_calls"] == 5
    assert stats["expanded_frames"] == 1


def test_single_frame_ocr_noise_is_not_emitted_as_subtitles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from PIL import Image

    frames = []
    for index in range(5):
        frame = tmp_path / f"frame-{index:06d}.png"
        Image.new("RGB", (640, 360), "black").save(frame)
        frames.append(frame)
    candidates = iter([
        (OCRCandidate("Fg ie SSSSS55", 0.72, "bottom"), 2),
        (OCRCandidate("ee ger are SES", 0.75, "bottom"), 2),
        (OCRCandidate("MR oe ete gett", 0.70, "bottom"), 2),
        (OCRCandidate("row ey Grice sem", 0.73, "bottom"), 2),
        (OCRCandidate("Be OSS Se easy", 0.71, "bottom"), 2),
    ])
    monkeypatch.setattr(
        "yblocalizer.ocr_subtitle._adaptive_frame_candidate",
        lambda *_args, **_kwargs: next(candidates),
    )

    assert _frames_to_segments(frames, 0.5, 3) == []


def test_adjacent_ocr_variants_confirm_one_real_subtitle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from PIL import Image

    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index:06d}.png"
        Image.new("RGB", (640, 360), "black").save(frame)
        frames.append(frame)
    candidates = iter([
        (OCRCandidate("Welcome to the demo", 0.88, "bottom"), 1),
        (OCRCandidate("Welcome to the demo", 0.91, "bottom"), 1),
        (None, 4),
    ])
    monkeypatch.setattr(
        "yblocalizer.ocr_subtitle._adaptive_frame_candidate",
        lambda *_args, **_kwargs: next(candidates),
    )

    segments = _frames_to_segments(frames, 0.5, 3)

    assert [(item.start, item.end, item.text) for item in segments] == [
        (0.0, 1.0, "Welcome to the demo"),
    ]


def test_temporary_frames_are_removed_after_ocr_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image

    monkeypatch.setattr("yblocalizer.ocr_subtitle.require_command", lambda _name: None)
    monkeypatch.setattr("yblocalizer.ocr_subtitle._check_ocr_runtime", lambda _language: "eng")

    def fake_extract(*_args, **kwargs):
        Image.new("RGB", (320, 180), "black").save(kwargs["frame_dir"] / "frame-000001.png")

    monkeypatch.setattr("yblocalizer.ocr_subtitle._extract_video_frames", fake_extract)
    monkeypatch.setattr("yblocalizer.ocr_subtitle._frames_to_segments", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="没有找到可信"):
        extract_ocr_subtitles(tmp_path / "input.mp4", tmp_path, tmp_path / "segments.json", tmp_path / "source.srt")

    assert list(tmp_path.glob(".ocr-temp-*")) == []


def test_real_tesseract_recognizes_a_synthetic_bottom_caption() -> None:
    command = _find_tesseract_command()
    if not command:
        pytest.skip("Tesseract is not installed")
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract

    _check_ocr_runtime("eng")
    image = Image.new("RGB", (1280, 720), "black")
    draw = ImageDraw.Draw(image)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font = ImageFont.truetype(str(font_path), 62) if font_path.is_file() else ImageFont.load_default()
    draw.text((240, 590), "Welcome to the demo", font=font, fill="white")

    result = _ocr_region_candidate(pytesseract, image, "bottom", 0.7, 0.3, language="eng")

    assert result is not None
    assert "welcome" in result.text.lower()


def test_missing_ocr_language_is_reported_before_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.ocr_subtitle._find_tesseract_command", lambda: "tesseract.exe")
    monkeypatch.setattr("yblocalizer.ocr_subtitle.available_ocr_languages", lambda _command=None: frozenset({"eng"}))

    with pytest.raises(RuntimeError, match="chi_sim"):
        _check_ocr_runtime("chi_sim")


def test_preflight_blocks_a_missing_ocr_language_before_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "yblocalizer.workbench_config.capabilities",
        lambda: {"ffmpeg": {"available": True}, "tesseract": {"available": True}},
    )
    monkeypatch.setattr("yblocalizer.workbench_config.available_ocr_languages", lambda: frozenset({"eng"}))
    monkeypatch.setattr("yblocalizer.workbench_config.render_encoder_status", lambda: {"cpu": True, "nvidia": False})

    result = preflight({
        "material_id": "local", "authorized": True, "device": "cpu", "compute_type": "int8",
        "options": {
            "subtitle_source": "ocr", "ocr_language": "chi_sim",
            "translator": "none", "target_lang": "en", "output_dir": str(tmp_path),
        },
    }, str(tmp_path))

    assert any(item["code"] == "ocr_language_missing" for item in result["blocking"])


def _ocr_options() -> SimpleNamespace:
    return SimpleNamespace(
        subtitle_source="ocr", ocr_fallback_to_audio=True,
        whisper_model_size="small", source_language=None, device="cpu", compute_type="int8", beam_size=5,
        ocr_interval=0.5, ocr_crop_ratio=0.3, ocr_min_chars=3, ocr_language="eng",
        resource_profile="balanced",
    )


def test_ocr_fallback_is_recorded_and_explained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workflow.extract_ocr_subtitles", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no caption")))

    def fake_audio(_video, output, _ctx):
        output.write_bytes(b"audio")
        return output

    def fake_transcribe(_audio, **kwargs):
        segments = [Segment(0, 1, "Audio fallback result")]
        save_segments(kwargs["segments_json"], segments)
        write_srt(kwargs["srt_path"], segments, display_mode="source")
        return segments

    monkeypatch.setattr("yblocalizer.workflow._extract_audio", fake_audio)
    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", fake_transcribe)
    logs: list[str] = []

    artifacts = run_extract(
        _ocr_options(), tmp_path, WorkflowArtifacts(raw_video=str(raw)), PipelineContext(), logs.append
    )

    assert artifacts.subtitle_extraction_mode == "audio-fallback"
    assert artifacts.ocr_status == "fallback"
    assert artifacts.ocr_message == "no caption"
    assert any("明确切换" in line for line in logs)


def test_ocr_cancellation_never_starts_audio_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr(
        "yblocalizer.workflow.extract_ocr_subtitles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CancellationRequested("stop")),
    )
    monkeypatch.setattr(
        "yblocalizer.workflow._extract_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fallback must not start")),
    )

    with pytest.raises(CancellationRequested):
        run_extract(_ocr_options(), tmp_path, WorkflowArtifacts(raw_video=str(raw)), PipelineContext(), lambda _line: None)

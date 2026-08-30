from pathlib import Path

import pytest

from yblocalizer.models import PublishMetadata, Segment, VideoJob
from yblocalizer.pipeline import PipelineOptions, run_pipeline


def test_pipeline_requires_rights_before_creating_output(tmp_path: Path) -> None:
    options = PipelineOptions(source="video.mp4", source_kind="file", output_dir=tmp_path / "outputs")
    with pytest.raises(SystemExit, match="rights"):
        run_pipeline(options)
    assert not options.output_dir.exists()


def test_pipeline_happy_path_with_mocked_integrations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")

    def fake_import(path: Path, work_dir: Path, title: str | None = None) -> VideoJob:
        return VideoJob("test", str(path), "file", work_dir, title=title, raw_video=raw_video, description="demo")

    def fake_audio(_: Path, output: Path) -> Path:
        output.write_bytes(b"audio")
        return output

    def fake_transcribe(_: Path, **kwargs):
        items = [Segment(0, 1, "hello world")]
        from yblocalizer.models import save_segments
        from yblocalizer.subtitle import write_srt
        save_segments(kwargs["segments_json"], items)
        write_srt(kwargs["srt_path"], items, display_mode="source")
        return items

    def fake_translate(_: Path, *, output_json: Path, output_srt: Path, **kwargs):
        output_json.write_text("[]", encoding="utf-8")
        output_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好世界\n", encoding="utf-8")
        return [Segment(0, 1, "hello world", "你好世界")]

    def fake_render(_: Path, __: Path, output: Path, **kwargs) -> Path:
        output.write_bytes(b"rendered")
        return output

    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-test")
    monkeypatch.setattr("yblocalizer.workflow.import_local_video", fake_import)
    monkeypatch.setattr("yblocalizer.workflow.extract_audio", fake_audio)
    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("yblocalizer.workflow.correct_source_segments", lambda segments, **_: segments)
    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate)
    monkeypatch.setattr("yblocalizer.workflow.generate_publish_metadata", lambda **_: PublishMetadata("测试标题", ["测试"]))
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", fake_render)
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_: True)

    result = run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", title="demo", i_have_rights=True, translator="deepseek"))
    assert result.rendered_video.read_bytes() == b"rendered"
    assert result.translated_srt.read_text(encoding="utf-8").endswith("你好世界\n")
    assert (result.work_dir / "publish_metadata.json").is_file()


def test_non_publish_pipeline_uses_local_metadata_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")
    seen_providers: list[str] = []

    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-fast-translation")
    monkeypatch.setattr(
        "yblocalizer.workflow.import_local_video",
        lambda path, work_dir, title=None: VideoJob(
            "test", str(path), "file", work_dir, title=title, raw_video=raw_video
        ),
    )
    monkeypatch.setattr("yblocalizer.workflow.extract_audio", lambda _video, output: output)

    def fake_transcribe(_audio, **kwargs):
        items = [Segment(0, 1, "hello")]
        from yblocalizer.models import save_segments
        from yblocalizer.subtitle import write_srt
        save_segments(kwargs["segments_json"], items)
        write_srt(kwargs["srt_path"], items, display_mode="source")
        return items

    def fake_translate(_source, *, output_json, output_srt, **_kwargs):
        items = [Segment(0, 1, "hello", "你好")]
        from yblocalizer.models import save_segments
        from yblocalizer.subtitle import write_srt
        save_segments(output_json, items)
        write_srt(output_srt, items)
        return items

    def fake_metadata(**kwargs):
        seen_providers.append(kwargs["provider"])
        return PublishMetadata("测试", ["中文字幕"])

    def fake_render(_video, _subtitle, output, **_kwargs):
        output.write_bytes(b"rendered")
        return output

    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate)
    monkeypatch.setattr("yblocalizer.workflow.generate_publish_metadata", fake_metadata)
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", fake_render)
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_: True)

    run_pipeline(
        PipelineOptions(
            source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs",
            i_have_rights=True, translator="deepseek", publish_to_bilibili=False,
        )
    )

    assert seen_providers == ["none"]


def test_pipeline_errors_when_ocr_and_audio_have_no_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-empty")
    monkeypatch.setattr("yblocalizer.workflow.import_local_video", lambda path, work_dir, title=None: VideoJob("test", str(path), "file", work_dir, raw_video=raw_video))
    monkeypatch.setattr("yblocalizer.workflow.extract_audio", lambda _video, output: output)
    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("yblocalizer.workflow.extract_ocr_subtitles", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_: True)

    with pytest.raises(RuntimeError, match="没有找到可用字幕"):
        run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", i_have_rights=True, translator="deepseek"))


def test_ocr_mode_falls_back_to_audio_when_ocr_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-ocr-fallback")
    monkeypatch.setattr("yblocalizer.workflow.import_local_video", lambda path, work_dir, title=None: VideoJob("test", str(path), "file", work_dir, raw_video=raw_video))
    monkeypatch.setattr("yblocalizer.workflow.extract_audio", lambda _video, output: output)
    monkeypatch.setattr("yblocalizer.workflow.extract_ocr_subtitles", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OCR unavailable")))
    def fallback_transcribe(*_args, **kwargs):
        items = [Segment(0, 1, "fallback")]
        from yblocalizer.models import save_segments
        from yblocalizer.subtitle import write_srt
        save_segments(kwargs["segments_json"], items); write_srt(kwargs["srt_path"], items, display_mode="source")
        return items
    def fallback_translate(*_args, **kwargs):
        items = [Segment(0, 1, "fallback", "回退成功")]
        from yblocalizer.models import save_segments
        from yblocalizer.subtitle import write_srt
        save_segments(kwargs["output_json"], items); write_srt(kwargs["output_srt"], items)
        return items
    def fallback_render(_video, _srt, output, **_kwargs):
        output.write_bytes(b"rendered"); return output
    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", fallback_transcribe)
    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fallback_translate)
    monkeypatch.setattr("yblocalizer.workflow.generate_publish_metadata", lambda **_: PublishMetadata("测试", []))
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", fallback_render)
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_: True)

    result = run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", i_have_rights=True, subtitle_source="ocr", translator="deepseek"))
    assert result.work_dir.name == "job-ocr-fallback"

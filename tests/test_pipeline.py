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
        return [Segment(0, 1, "hello world")]

    def fake_translate(_: Path, *, output_json: Path, output_srt: Path, **kwargs):
        output_json.write_text("[]", encoding="utf-8")
        output_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好世界\n", encoding="utf-8")
        return [Segment(0, 1, "hello world", "你好世界")]

    def fake_render(_: Path, __: Path, output: Path, **kwargs) -> Path:
        output.write_bytes(b"rendered")
        return output

    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-test")
    monkeypatch.setattr("yblocalizer.pipeline.import_local_video", fake_import)
    monkeypatch.setattr("yblocalizer.pipeline.extract_audio", fake_audio)
    monkeypatch.setattr("yblocalizer.pipeline.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("yblocalizer.pipeline.correct_source_segments", lambda segments, **_: segments)
    monkeypatch.setattr("yblocalizer.pipeline.translate_segments_file", fake_translate)
    monkeypatch.setattr("yblocalizer.pipeline.generate_publish_metadata", lambda **_: PublishMetadata("测试标题", ["测试"]))
    monkeypatch.setattr("yblocalizer.pipeline.burn_subtitles", fake_render)

    result = run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", title="demo", i_have_rights=True, translator="deepseek"))
    assert result.rendered_video.read_bytes() == b"rendered"
    assert result.translated_srt.read_text(encoding="utf-8").endswith("你好世界\n")
    assert (result.work_dir / "publish_metadata.json").is_file()


def test_pipeline_errors_when_ocr_and_audio_have_no_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-empty")
    monkeypatch.setattr("yblocalizer.pipeline.import_local_video", lambda path, work_dir, title=None: VideoJob("test", str(path), "file", work_dir, raw_video=raw_video))
    monkeypatch.setattr("yblocalizer.pipeline.extract_audio", lambda _video, output: output)
    monkeypatch.setattr("yblocalizer.pipeline.transcribe_audio", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("yblocalizer.pipeline.extract_ocr_subtitles", lambda *_args, **_kwargs: [])

    with pytest.raises(RuntimeError, match="No readable subtitle text"):
        run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", i_have_rights=True, translator="deepseek"))


def test_ocr_mode_falls_back_to_audio_when_ocr_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw_video = tmp_path / "input.mp4"
    raw_video.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.pipeline.timestamp_id", lambda: "job-ocr-fallback")
    monkeypatch.setattr("yblocalizer.pipeline.import_local_video", lambda path, work_dir, title=None: VideoJob("test", str(path), "file", work_dir, raw_video=raw_video))
    monkeypatch.setattr("yblocalizer.pipeline.extract_audio", lambda _video, output: output)
    monkeypatch.setattr("yblocalizer.pipeline.extract_ocr_subtitles", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OCR unavailable")))
    monkeypatch.setattr("yblocalizer.pipeline.transcribe_audio", lambda *_args, **_kwargs: [Segment(0, 1, "fallback")])
    monkeypatch.setattr("yblocalizer.pipeline.translate_segments_file", lambda *_args, **kwargs: [Segment(0, 1, "fallback", "回退成功")])
    monkeypatch.setattr("yblocalizer.pipeline.generate_publish_metadata", lambda **_: PublishMetadata("测试", []))
    monkeypatch.setattr("yblocalizer.pipeline.burn_subtitles", lambda _video, _srt, output, **_kwargs: output)

    result = run_pipeline(PipelineOptions(source=str(raw_video), source_kind="file", output_dir=tmp_path / "outputs", i_have_rights=True, subtitle_source="ocr", translator="deepseek"))
    assert result.work_dir.name == "job-ocr-fallback"

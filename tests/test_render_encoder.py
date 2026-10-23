from __future__ import annotations

from pathlib import Path

import pytest

from yblocalizer.render import burn_subtitles


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("yblocalizer.render.require_command", lambda _name: "ffmpeg")
    monkeypatch.setattr("yblocalizer.render._probe_video_size", lambda _path: (1920, 1080))
    (tmp_path / "zh.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n测试字幕\n",
        encoding="utf-8",
    )


def test_auto_render_prefers_nvidia(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    logs: list[str] = []
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": True})
    monkeypatch.setattr("yblocalizer.render.run", lambda command, **_kwargs: commands.append(command))

    burn_subtitles(tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4", log=logs.append)

    assert "h264_nvenc" in commands[0]
    assert "ass=zh.ass" in commands[0]
    assert any("NVIDIA" in line for line in logs)


def test_auto_render_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    logs: list[str] = []
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": True})

    def fake_run(command: list[str], **_kwargs) -> None:
        commands.append(command)
        if len(commands) == 1:
            raise RuntimeError("NVENC unavailable at runtime")

    monkeypatch.setattr("yblocalizer.render.run", fake_run)
    burn_subtitles(tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4", log=logs.append)

    assert "h264_nvenc" in commands[0]
    assert "libx264" in commands[1]
    assert any("降级" in line for line in logs)


def test_explicit_nvidia_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    with pytest.raises(RuntimeError, match="NVENC"):
        burn_subtitles(tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4", encoder="nvidia")


def test_render_writes_resolution_aware_ass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.render.run", lambda *_args, **_kwargs: None)

    burn_subtitles(
        tmp_path / "in.mp4",
        tmp_path / "zh.srt",
        tmp_path / "out.mp4",
        font_name="Microsoft YaHei",
        font_size=24,
        primary_color="&H0000FFFF",
        margin_v=60,
    )

    generated = (tmp_path / "zh.ass").read_text(encoding="utf-8-sig")
    assert "PlayResX: 1920" in generated
    assert "PlayResY: 1080" in generated
    assert "Style: Default,Microsoft YaHei,48" in generated
    assert ",60," in generated


def test_invalid_cue_font_plan_warns_and_uses_default_style(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.render.run", lambda *_args, **_kwargs: None)
    logs: list[str] = []

    burn_subtitles(
        tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4",
        encoder="cpu", cue_font_sizes={2: 12}, expected_cue_count=1, log=logs.append,
    )

    generated = (tmp_path / "zh.ass").read_text(encoding="utf-8-sig")
    assert "YBCueFont" not in generated
    assert any("默认字号继续渲染" in line for line in logs)


def test_style_bilingual_source_lines_uses_position_not_script() -> None:
    from yblocalizer.render import _style_bilingual_source_lines

    secondary = "{\\rSecondary}"
    # Target language English: the source line is Chinese.
    assert _style_bilingual_source_lines("你好世界\\NHello world", "bilingual-source-first", 1) == secondary + "你好世界{\\r}\\NHello world"
    assert _style_bilingual_source_lines("Hello world\\N你好世界", "bilingual-translation-first", 1) == "Hello world\\N" + secondary + "你好世界{\\r}"
    # Two lines per side.
    assert _style_bilingual_source_lines("a\\Nb\\N中\\N文", "bilingual-translation-first", 2) == "a\\Nb\\N" + secondary + "中{\\r}\\N" + secondary + "文{\\r}"
    assert _style_bilingual_source_lines("single", "bilingual-source-first", 0) == "single"


def test_bilingual_source_style_follows_source_line_count_not_half() -> None:
    from yblocalizer.render import _style_bilingual_source_lines

    secondary = "{\\rSecondary}"
    # One translated line + two source lines: halving would style the wrong line.
    assert _style_bilingual_source_lines("译文\\NLine one\\NLine two", "bilingual-translation-first", 2) == (
        "译文\\N" + secondary + "Line one{\\r}\\N" + secondary + "Line two{\\r}"
    )
    assert _style_bilingual_source_lines("Line one\\NLine two\\N译文", "bilingual-source-first", 2) == (
        secondary + "Line one{\\r}\\N" + secondary + "Line two{\\r}\\N译文"
    )


def test_cue_source_line_counts_matches_bilingual_srt_layout() -> None:
    from yblocalizer.models import Segment
    from yblocalizer.subtitle import cue_source_line_counts, format_segment_text

    segments = [
        Segment(0, 2, "Line one\nLine two", "译文"),
        Segment(2, 4, "same", "same"),
        Segment(4, 6, "", "只有译文"),
    ]
    counts = cue_source_line_counts(segments, "bilingual-translation-first", smart_layout=False)
    assert counts == {1: 2, 2: 0, 3: 0}
    assert format_segment_text(segments[0], "bilingual-translation-first", smart_layout=False) == "译文\nLine one\nLine two"


def test_prepare_ass_styles_only_source_lines_from_counts(tmp_path) -> None:
    import pysubs2
    from yblocalizer.render import _prepare_ass

    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n译文\nLine one\nLine two\n", encoding="utf-8")
    path = _prepare_ass(srt, size=(1920, 1080), font_name="Microsoft YaHei", font_size=24, primary_color="&H00FFFFFF",
                        outline_color="&H00000000", outline=2, shadow=0, margin_v=None, raised_margin=False,
                        margin_ratio=0.05, display_mode="bilingual-translation-first", cue_source_lines={1: 2})
    text = pysubs2.load(str(path)).events[0].text
    assert text.startswith("译文\\N{\\rSecondary}Line one")
    assert "{\\rSecondary}Line two" in text

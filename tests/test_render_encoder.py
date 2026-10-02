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

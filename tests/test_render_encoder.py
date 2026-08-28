from __future__ import annotations

from pathlib import Path

import pytest

from yblocalizer.render import burn_subtitles


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.render.require_command", lambda _name: "ffmpeg")
    monkeypatch.setattr("yblocalizer.render._probe_video_size", lambda _path: (1920, 1080))


def test_auto_render_prefers_nvidia(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch)
    commands: list[list[str]] = []
    logs: list[str] = []
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": True})
    monkeypatch.setattr("yblocalizer.render.run", lambda command, **_kwargs: commands.append(command))

    burn_subtitles(tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4", log=logs.append)

    assert "h264_nvenc" in commands[0]
    assert any("NVIDIA" in line for line in logs)


def test_auto_render_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _prepare(monkeypatch)
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
    _prepare(monkeypatch)
    monkeypatch.setattr("yblocalizer.render.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    with pytest.raises(RuntimeError, match="NVENC"):
        burn_subtitles(tmp_path / "in.mp4", tmp_path / "zh.srt", tmp_path / "out.mp4", encoder="nvidia")

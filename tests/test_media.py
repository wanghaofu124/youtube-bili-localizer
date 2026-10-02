from pathlib import Path

import pytest

from yblocalizer.media import extract_audio


def test_extract_audio_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("yblocalizer.media.require_command", lambda _: (_ for _ in ()).throw(RuntimeError("ffmpeg is required")))
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        extract_audio(tmp_path / "input.mp4")


def test_extract_audio_preserves_media_timestamps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    command: list[str] = []
    monkeypatch.setattr("yblocalizer.media.require_command", lambda _name: "ffmpeg")
    monkeypatch.setattr("yblocalizer.media.run", lambda value, **_kwargs: command.extend(value))

    extract_audio(tmp_path / "input.mp4", tmp_path / "audio.wav")

    assert command[command.index("-i") - 2 : command.index("-i")] == ["-copyts", "-start_at_zero"]
    assert command[command.index("-af") + 1] == "aresample=async=1:first_pts=0"

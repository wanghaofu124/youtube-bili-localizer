from pathlib import Path

import pytest

from yblocalizer.media import extract_audio


def test_extract_audio_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("yblocalizer.media.require_command", lambda _: (_ for _ in ()).throw(RuntimeError("ffmpeg is required")))
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        extract_audio(tmp_path / "input.mp4")

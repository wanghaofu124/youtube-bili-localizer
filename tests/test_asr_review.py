from pathlib import Path

from yblocalizer.asr_review import prepare_session


def test_truncated_wav_is_unknown_duration(tmp_path: Path) -> None:
    """A cut-off WAV raises EOFError; that is the same as an unreadable duration."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF\x24\x00")

    session = prepare_session(
        work_dir=tmp_path,
        job_id="job",
        audio_path=audio,
        baseline_words_path=None,
        baseline_kind="legacy_partial",
        model="small",
        language="en",
        beam_size=1,
        channel="mix",
        subtitle_revision=0,
        subtitle_fingerprint="abc",
    )

    assert session["audio_duration"] is None
    assert session["candidates"] == []

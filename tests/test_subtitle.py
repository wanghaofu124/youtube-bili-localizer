from pathlib import Path

import pytest

from yblocalizer.models import Segment, save_segments
from yblocalizer.subtitle import format_segment_text, format_srt_timestamp, segments_to_srt, write_srt


def test_format_srt_timestamp_clamps_and_rounds() -> None:
    assert format_srt_timestamp(-0.1) == "00:00:00,000"
    assert format_srt_timestamp(3661.2346) == "01:01:01,235"


def test_bilingual_translation_first_and_cjk_layout() -> None:
    segment = Segment(0, 2, "A short English source", "这是一个用于测试字幕自动换行效果的中文句子。")
    text = format_segment_text(segment, display_mode="bilingual-translation-first", max_chars_per_line=12)
    assert text.splitlines()[-1] == "A short English source"
    assert "\n" in text


def test_srt_keeps_sequence_and_time_alignment() -> None:
    output = segments_to_srt([Segment(0, 1.2, "hello", "你好")], smart_layout=False)
    assert output == "1\n00:00:00,000 --> 00:00:01,200\n你好\n"


def test_atomic_subtitle_writes_preserve_old_files_and_remove_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments_path = tmp_path / "segments.json"
    subtitle_path = tmp_path / "subtitle.srt"
    segments_path.write_text("old-json", encoding="utf-8")
    subtitle_path.write_text("old-srt", encoding="utf-8")
    monkeypatch.setattr(Path, "replace", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("locked")))

    with pytest.raises(OSError, match="locked"):
        save_segments(segments_path, [Segment(0, 1, "hello")])
    with pytest.raises(OSError, match="locked"):
        write_srt(subtitle_path, [Segment(0, 1, "hello", "你好")])

    assert segments_path.read_text(encoding="utf-8") == "old-json"
    assert subtitle_path.read_text(encoding="utf-8") == "old-srt"
    assert not segments_path.with_suffix(".json.tmp").exists()
    assert not subtitle_path.with_suffix(".srt.tmp").exists()

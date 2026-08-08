from yblocalizer.models import Segment
from yblocalizer.subtitle import format_segment_text, format_srt_timestamp, segments_to_srt


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

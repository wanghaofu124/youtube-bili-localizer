from yblocalizer.models import Segment
from yblocalizer.subtitle_merge import merge_audio_ocr_segments


def test_merge_preserves_useful_visual_label() -> None:
    merged = merge_audio_ocr_segments(
        [Segment(0, 2, "Welcome to the race")],
        [Segment(0.4, 1.8, "F1 world champion Alex Morgan")],
    )
    assert len(merged) == 1
    assert merged[0].text == "Speech: Welcome to the race | On-screen text: F1 world champion Alex Morgan"


def test_merge_ignores_empty_ocr_and_returns_audio() -> None:
    audio = [Segment(0, 1, "hello")]
    assert merge_audio_ocr_segments(audio, [Segment(0, 1, "...")]) == audio


def test_merge_keeps_standalone_cjk_ocr_text() -> None:
    merged = merge_audio_ocr_segments(
        [Segment(3, 4, "Later audio line")],
        [Segment(0, 1.5, "这是画面中的中文字幕")],
    )
    assert [item.text for item in merged] == ["这是画面中的中文字幕", "Later audio line"]

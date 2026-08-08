import pytest

from yblocalizer.models import Segment
from yblocalizer.translate import _parse_numbered_lines
from yblocalizer.validation import validate_target_language


def test_parse_numbered_lines_repairs_extra_short_line() -> None:
    parsed = _parse_numbered_lines("1. 你好\n2. 世界\n3. ！", expected=2)
    assert parsed == ["你好世界", "！"]


def test_parse_numbered_lines_splits_joined_sentences() -> None:
    parsed = _parse_numbered_lines("1. 第一句。第二句。", expected=2)
    assert parsed == ["第一句。", "第二句。"]


@pytest.mark.parametrize(
    ("segments", "provider", "message"),
    [([], "deepseek", "empty subtitles"), ([Segment(0, 1, "source", "English only")], "deepseek", "does not contain Chinese"), ([Segment(0, 1, "source", "中文")], "none", "only for debugging")],
)
def test_chinese_validation_rejects_invalid_results(segments, provider, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_target_language(segments, target_lang="zh-Hans", provider=provider)

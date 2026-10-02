from types import SimpleNamespace

from yblocalizer.transcribe import _segment_from_whisper, _segments_from_whisper, _timed_words_with_fallback


def test_whisper_uses_first_and_last_word_times() -> None:
    item = SimpleNamespace(
        start=99.0,
        end=154.0,
        text="I love it.",
        words=[
            SimpleNamespace(start=99.01, end=99.18, word="I"),
            SimpleNamespace(start=99.18, end=99.40, word="love"),
            SimpleNamespace(start=99.40, end=99.72, word="it."),
        ],
    )
    segment = _segment_from_whisper(item)
    assert segment is not None
    assert segment.start == 99.01
    assert segment.end == 99.72


def test_mixed_whisper_items_keep_speech_without_word_timestamps() -> None:
    items = [
        SimpleNamespace(
            start=1.0, end=1.3, text="Hello",
            words=[SimpleNamespace(start=1.0, end=1.3, word="Hello")],
        ),
        SimpleNamespace(start=3.0, end=4.0, text="More speech.", words=None),
    ]
    words = _timed_words_with_fallback(items)
    assert [(word.start, word.end, word.word) for word in words] == [
        (1.0, 1.3, "Hello"), (3.0, 4.0, "More speech."),
    ]


def test_whisper_splits_long_broll_gap_using_word_times() -> None:
    item = SimpleNamespace(
        start=54.61,
        end=70.46,
        text="Best hope is to trap it against the rocks, but it finds a way out.",
        words=[
            SimpleNamespace(start=54.61, end=54.87, word=" Best"),
            SimpleNamespace(start=54.87, end=55.15, word=" hope"),
            SimpleNamespace(start=55.15, end=55.63, word=" is"),
            SimpleNamespace(start=55.63, end=55.75, word=" to"),
            SimpleNamespace(start=55.75, end=55.99, word=" trap"),
            SimpleNamespace(start=55.99, end=56.11, word=" it"),
            SimpleNamespace(start=56.11, end=56.37, word=" against"),
            SimpleNamespace(start=56.37, end=56.61, word=" the"),
            SimpleNamespace(start=56.61, end=56.97, word=" rocks,"),
            SimpleNamespace(start=67.68, end=67.86, word=" but"),
            SimpleNamespace(start=67.86, end=67.98, word=" as"),
            SimpleNamespace(start=67.98, end=68.24, word=" soon"),
            SimpleNamespace(start=68.24, end=68.34, word=" as"),
            SimpleNamespace(start=68.34, end=68.54, word=" it's"),
            SimpleNamespace(start=68.54, end=69.14, word=" cornered,"),
            SimpleNamespace(start=69.56, end=69.66, word=" it"),
            SimpleNamespace(start=69.66, end=69.90, word=" finds"),
            SimpleNamespace(start=69.90, end=70.06, word=" a"),
            SimpleNamespace(start=70.06, end=70.20, word=" way"),
            SimpleNamespace(start=70.20, end=70.46, word=" out."),
        ],
    )

    cues = _segments_from_whisper(item)

    assert [cue.text for cue in cues] == [
        "Best hope is to trap it against the rocks,",
        "but as soon as it's cornered, it finds a way out.",
    ]
    assert cues[0].start == 54.61
    assert cues[0].end == 57.09
    assert cues[1].start == 67.68
    assert cues[1].end == 70.58


def test_orphan_article_does_not_bridge_long_silence() -> None:
    item = SimpleNamespace(
        start=167.02,
        end=175.68,
        text="The overview captured above.",
        words=[
            SimpleNamespace(start=167.02, end=167.44, word=" The"),
            SimpleNamespace(start=174.24, end=174.68, word=" overview"),
            SimpleNamespace(start=174.68, end=175.24, word=" captured"),
            SimpleNamespace(start=175.24, end=175.68, word=" above."),
        ],
    )

    cues = _segments_from_whisper(item)

    assert [cue.text for cue in cues] == ["The", "overview captured above."]
    assert cues[0].start == 167.02
    assert cues[0].end == 167.56
    assert cues[1].start == 174.24
    assert cues[1].end == 175.8

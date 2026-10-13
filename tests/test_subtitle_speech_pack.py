from types import SimpleNamespace
from pathlib import Path
import pytest
from yblocalizer.models import Segment

from yblocalizer.subtitle_speech_pack import (
    TimedWord,
    load_words,
    normalize_word_stream,
    pack_cues_from_words,
    pack_cues_from_words_detailed,
    save_words,
    timed_words_from_whisper_item,
    words_path_for_segments,
    group_speech_translation_units,
)
from yblocalizer.transcribe import _uncovered_speech_gaps


def test_grouping_keeps_numeric_spanish_quantity_boundary_even_at_hard_cut() -> None:
    cues = [
        Segment(0, 4, "Tengo 1000"),
        Segment(4.7, 11, "de esos"),
    ]
    grouped = group_speech_translation_units(cues)
    assert len(grouped) == 1
    assert grouped[0].text == "Tengo 1000 de esos"


def test_grouping_propagates_review_flag() -> None:
    grouped = group_speech_translation_units([
        Segment(0, 1, "hello", None, False),
        Segment(1, 2, "world", None, True),
    ])
    assert grouped[0].needs_review is True


def test_words_path_for_segments() -> None:
    assert words_path_for_segments(Path("segments.source.json")).name == "words.source.json"
    assert words_path_for_segments(Path("segments.audio.json")).name == "words.audio.json"


def test_pack_keeps_handed_me_with_box_across_whisper_chunks() -> None:
    """Whisper often ends a chunk after 'me' and starts the next on 'this box'.

    Global word packing must ignore those chunk boundaries and keep the phrase
    on one cue timed to the spoken words.
    """
    first = SimpleNamespace(
        start=30.5,
        end=38.0,
        text="It's just a normal warehouse and they handed me",
        words=[
            SimpleNamespace(start=30.50, end=30.70, word="It's"),
            SimpleNamespace(start=30.72, end=30.90, word=" just"),
            SimpleNamespace(start=30.92, end=31.10, word=" a"),
            SimpleNamespace(start=31.12, end=31.40, word=" normal"),
            SimpleNamespace(start=31.42, end=31.90, word=" warehouse"),
            SimpleNamespace(start=32.00, end=32.20, word=" and"),
            SimpleNamespace(start=32.22, end=32.40, word=" they"),
            SimpleNamespace(start=32.42, end=32.80, word=" handed"),
            SimpleNamespace(start=32.82, end=33.00, word=" me"),
        ],
    )
    second = SimpleNamespace(
        start=33.2,
        end=35.4,
        text="this sealed box.",
        words=[
            SimpleNamespace(start=33.20, end=33.40, word=" this"),
            SimpleNamespace(start=33.42, end=33.80, word=" sealed"),
            SimpleNamespace(start=33.82, end=34.20, word=" box."),
        ],
    )

    words = normalize_word_stream(
        timed_words_from_whisper_item(first) + timed_words_from_whisper_item(second)
    )
    cues = pack_cues_from_words(words)
    joined = " ".join(cue.text for cue in cues)
    assert "handed me this sealed box." in joined
    assert not any(cue.text.rstrip(".").endswith("handed me") for cue in cues)


def test_pack_splits_on_real_speech_pause() -> None:
    words = [
        TimedWord(1.0, 1.2, "Hello"),
        TimedWord(1.25, 1.5, "there."),
        TimedWord(3.2, 3.5, "Later"),
        TimedWord(3.55, 3.9, "on."),
    ]
    cues = pack_cues_from_words(words)
    assert [cue.text for cue in cues] == ["Hello there.", "Later on."]
    assert cues[0].end == 1.62
    assert cues[1].start == 3.2


def test_unfinished_phrase_cannot_hold_caption_across_one_minute_of_silence() -> None:
    words = [
        TimedWord(442.38, 442.56, "We've"),
        TimedWord(504.95, 505.13, " got"),
        TimedWord(505.14, 505.46, " ourselves"),
        TimedWord(505.50, 507.0, " a beautiful yellow fish."),
    ]
    cues = pack_cues_from_words(words)
    assert [cue.text for cue in cues] == ["We've", "got ourselves a beautiful yellow fish."]
    assert cues[0].end < 443
    assert cues[1].start == 504.95


def test_pack_does_not_cut_after_incomplete_tail_on_soft_duration() -> None:
    # Build a long cue ending in "the" then a noun 0.2s later — must stay together.
    words = [TimedWord(0.0, 0.3, "We")]
    t = 0.35
    filler = "kept walking through another empty hallway near"
    for token in filler.split():
        words.append(TimedWord(t, t + 0.28, f" {token}"))
        t += 0.30
    words.append(TimedWord(t, t + 0.25, " the"))
    t += 0.30
    words.append(TimedWord(t, t + 0.35, " glass."))
    cues = pack_cues_from_words(words)
    assert any("the glass." in cue.text for cue in cues)


def test_save_and_load_words_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "words.source.json"
    original = [TimedWord(1.0, 1.2, "Hi"), TimedWord(1.3, 1.6, "there.")]
    save_words(path, original)
    loaded = load_words(path)
    assert [(w.start, w.end, w.word) for w in loaded] == [(1.0, 1.2, "Hi"), (1.3, 1.6, "there.")]


def test_sustained_speech_gap_is_flagged_without_extending_neighboring_words() -> None:
    gaps = _uncovered_speech_gaps(
        [(430.0, 442.0)],
        [TimedWord(430.0, 431.0, "earlier"), TimedWord(437.0, 442.0, "later")],
    )
    assert gaps == [(431.0, 437.0)]


def test_short_packing_prefers_clause_start_to_split_inside_noun_phrase() -> None:
    tokens = [
        (504.52, 504.96, "got"), (504.96, 505.26, " ourselves"),
        (505.26, 505.66, " a"), (505.66, 506.04, " beautiful"),
        (506.04, 506.56, " yellow"), (506.56, 506.98, " fish"),
        (506.98, 507.32, " this"), (507.32, 507.60, " is"),
        (507.60, 507.84, " what"), (507.84, 507.96, " we"),
        (507.96, 508.20, " call"), (508.20, 508.40, " a"),
        (508.40, 508.82, " painted"), (508.82, 509.18, " sweet"),
        (509.18, 509.38, " lip"),
    ]
    cues = pack_cues_from_words([TimedWord(*token) for token in tokens])
    assert [cue.text for cue in cues] == [
        "got ourselves a beautiful yellow fish",
        "this is what we call a painted sweet lip",
    ]


def test_duration_limit_does_not_cut_predatory_fish() -> None:
    tokens = (
        "they use it as a way of camouflage so that they can pretty much "
        "stay safe from all the predatory fish that follows"
    ).split()
    words = [
        TimedWord(index * 0.22, index * 0.22 + 0.22, f" {token}")
        for index, token in enumerate(tokens)
    ]
    cues = pack_cues_from_words(words)
    assert any("predatory fish" in cue.text for cue in cues)
    assert not any(cue.text.endswith("predatory") for cue in cues)


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("and you can see inside of its mouth right there is bright red now", "bright red"),
        ("we are going to meet some of the most crazy animals ever", "crazy animals"),
        ("this thing is black and we are not eating it today", "eating it"),
        ("I want to have a sunset cook up a little bit later on", "bit later on"),
    ],
)
def test_continuous_speech_keeps_short_grammatical_units(
    phrase: str, expected: str,
) -> None:
    words = [
        TimedWord(index * 0.25, index * 0.25 + 0.25, f" {word}")
        for index, word in enumerate(phrase.split())
    ]
    cues = pack_cues_from_words(words)
    assert any(expected in cue.text for cue in cues)


def test_duration_limit_keeps_sea_cucumber_and_starts_next_clause() -> None:
    tokens = [
        (902.92, 903.12, "this"), (903.12, 903.24, "is"),
        (903.24, 903.36, "the"), (903.36, 903.62, "first"),
        (903.62, 903.80, "thing"), (903.80, 903.94, "that"),
        (903.94, 904.04, "we"), (904.04, 904.32, "find"),
        (904.32, 905.04, "guys"), (905.04, 905.30, "look"),
        (905.30, 905.42, "at"), (905.42, 905.54, "it"),
        (905.54, 906.34, "we"), (906.91, 907.11, "got"),
        (907.11, 907.25, "the"), (907.25, 907.51, "sea"),
        (907.51, 907.93, "cucumber"), (907.93, 908.55, "baby"),
        (909.03, 909.25, "it"), (909.25, 909.43, "looks"),
    ]
    cues = pack_cues_from_words([TimedWord(start, end, f" {word}") for start, end, word in tokens])
    assert any("sea cucumber" in cue.text for cue in cues)
    assert any(cue.text.startswith("it looks") for cue in cues)


def test_long_silence_keeps_fragments_separate_and_flags_review() -> None:
    words = [
        TimedWord(1.0, 1.2, "I"), TimedWord(181.0, 181.2, "did"),
        TimedWord(181.2, 181.5, " it"),
        TimedWord(200.0, 200.4, "Spanish"),
        TimedWord(320.0, 320.4, "mackerel"),
    ]
    packed = pack_cues_from_words_detailed(words)
    assert [cue.text for cue in packed.cues] == ["did it", "mackerel"]
    assert [fragment.text for fragment in packed.detached_fragments] == ["I", "Spanish"]
    assert [(fragment.start, fragment.end) for fragment in packed.detached_fragments] == [
        (1.0, 1.2),
        (200.0, 200.4),
    ]
    assert all("未放入成片字幕" in fragment.reason for fragment in packed.detached_fragments)

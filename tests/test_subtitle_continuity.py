from __future__ import annotations

from yblocalizer.models import Segment
from yblocalizer.subtitle_continuity import (
    coalesce_incomplete_source_cues,
    source_cue_requires_continuation,
)
from yblocalizer.transcribe import _coalesce_micro_segments
from yblocalizer.translate import _stitch_continuation_translations


def test_coalesce_moves_glass_onto_unfinished_article_cue() -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 4.0, "cristal entonces seguimos"),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert [item.text for item in merged] == [
        "puede rayar el cristal",
        "entonces seguimos",
    ]
    assert merged[0].end <= merged[1].start + 0.001
    assert merged[0].end > 2.2
    assert merged[1].start >= merged[0].end - 0.001


def test_coalesce_accepts_capitalized_noun() -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 3.8, "Cristal entonces"),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert merged[0].text == "puede rayar el Cristal"
    assert merged[1].text == "entonces"


def test_coalesce_does_not_join_discourse_starter() -> None:
    segments = [
        Segment(1.0, 2.5, "limpiamos bien"),
        Segment(2.55, 4.0, "entonces seguimos"),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert [item.text for item in merged] == [
        "limpiamos bien",
        "entonces seguimos",
    ]


def test_coalesce_is_idempotent() -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 4.0, "cristal entonces seguimos"),
    ]
    once = coalesce_incomplete_source_cues(segments)
    twice = coalesce_incomplete_source_cues(once)
    assert [item.text for item in once] == [item.text for item in twice]
    assert [round(item.start, 3) for item in once] == [round(item.start, 3) for item in twice]
    assert [round(item.end, 3) for item in once] == [round(item.end, 3) for item in twice]


def test_source_requires_continuation_for_article_tail() -> None:
    assert source_cue_requires_continuation("puede rayar el", "cristal entonces")
    assert not source_cue_requires_continuation("limpiamos bien", "entonces seguimos")
    assert not source_cue_requires_continuation("así que", "si vamos")


def test_stitch_moves_glass_with_remainder() -> None:
    source = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 4.0, "cristal entonces seguimos"),
    ]
    translated = [
        Segment(1.0, 2.2, "puede rayar el", "可能会划伤"),
        Segment(2.25, 4.0, "cristal entonces seguimos", "玻璃。所以我们继续"),
    ]
    out = _stitch_continuation_translations(source, translated)
    assert out[0].translated_text == "可能会划伤玻璃。"
    assert out[1].translated_text == "所以我们继续"


def test_stitch_moves_glass_only_cue() -> None:
    source = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 3.0, "cristal."),
    ]
    translated = [
        Segment(1.0, 2.2, "puede rayar el", "可能会划伤"),
        Segment(2.25, 3.0, "cristal.", "玻璃。"),
    ]
    out = _stitch_continuation_translations(source, translated)
    assert len(out) == 1
    assert out[0].translated_text == "可能会划伤玻璃。"


def test_stitch_accepts_capitalized_source() -> None:
    source = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 4.0, "Cristal entonces seguimos"),
    ]
    translated = [
        Segment(1.0, 2.2, "puede rayar el", "可能会划伤"),
        Segment(2.25, 4.0, "Cristal entonces seguimos", "玻璃。所以我们继续"),
    ]
    out = _stitch_continuation_translations(source, translated)
    assert out[0].translated_text == "可能会划伤玻璃。"
    assert out[1].translated_text == "所以我们继续"


def test_stitch_does_not_glue_discourse_markers() -> None:
    source = [
        Segment(1.0, 2.0, "listo"),
        Segment(2.05, 3.0, "si"),
    ]
    translated = [
        Segment(1.0, 2.0, "listo", "所以"),
        Segment(2.05, 3.0, "si", "是的"),
    ]
    out = _stitch_continuation_translations(source, translated)
    assert [item.translated_text for item in out] == ["所以", "是的"]


def test_micro_coalesce_still_joins_yo_fragment() -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.22, 2.5, "yo"),
    ]
    merged = _coalesce_micro_segments(segments)
    assert len(merged) == 1
    assert merged[0].text == "puede rayar el yo"


def test_coalesce_joins_infinitive_trail() -> None:
    segments = [
        Segment(93.58, 98.14, "So I put some in each of their setups and then a piece of an Indian almond leaf to give them a place"),
        Segment(98.14, 98.56, "to hide."),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert len(merged) == 1
    assert merged[0].text.endswith("place to hide.")


def test_coalesce_joins_poured_the_across_half_second_gap() -> None:
    segments = [
        Segment(406.13, 411.78, "poured the"),
        Segment(412.26, 415.86, "springtails in, and they took off looking for food."),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert merged[0].text.startswith("poured the springtails")
    assert "took off" in merged[0].text or (len(merged) > 1 and "took off" in merged[1].text)


def test_coalesce_merges_orphan_i_forward() -> None:
    segments = [
        Segment(404.11, 404.67, "I"),
        Segment(406.13, 411.78, "poured the springtails in."),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert len(merged) == 1
    assert merged[0].text.startswith("I poured")


def test_coalesce_peels_dangling_to_onto_next_cue() -> None:
    segments = [
        Segment(14.26, 16.18, "Let's go back to the beginning. To"),
        Segment(17.80, 21.16, "raise these tadpoles, we're going to have to overcome all kinds of challenges,"),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert merged[0].text == "Let's go back to the beginning."
    assert merged[1].text.startswith("To raise these tadpoles")


def test_coalesce_joins_object_pronoun_to_noun_phrase() -> None:
    segments = [
        Segment(30.50, 38.16, "looks sketchy but it should be fine. It's just a normal warehouse and they handed me"),
        Segment(38.04, 40.20, "this sealed box."),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert len(merged) == 1
    assert "handed me this sealed box." in merged[0].text


def test_coalesce_joins_quantifier_few_months() -> None:
    segments = [
        Segment(60.08, 61.10, "But over the next few"),
        Segment(60.98, 66.58, "months these tadpoles are going to grow some ridiculous colors,"),
    ]
    merged = coalesce_incomplete_source_cues(segments)
    assert "few months" in merged[0].text

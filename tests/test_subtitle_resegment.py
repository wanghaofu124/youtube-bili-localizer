from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yblocalizer.models import Segment
from yblocalizer.subtitle_resegment import (
    _materialize_groups,
    _parse_join_after,
    resegment_source_segments,
)


def test_materialize_joins_adjacent_cues_and_preserves_span() -> None:
    segments = [
        Segment(1.0, 2.0, "they handed me"),
        Segment(2.05, 3.0, "this sealed box."),
        Segment(3.5, 5.0, "Then we left."),
    ]
    grouped = _materialize_groups(segments, {0: True, 1: False})
    assert [item.text for item in grouped] == [
        "they handed me this sealed box.",
        "Then we left.",
    ]
    assert grouped[0].start == 1.0
    assert grouped[0].end == 3.0


def test_materialize_rejects_overlong_join() -> None:
    segments = [
        Segment(0.0, 8.0, "one long unfinished phrase about cleaning water carefully today"),
        Segment(8.1, 16.5, "and then another long unfinished phrase continues without a short tail"),
    ]
    grouped = _materialize_groups(segments, {0: True})
    # Far beyond the relaxed duration budget, so the join is refused.
    assert len(grouped) == 2


def test_materialize_allows_short_box_completer_on_long_left_cue() -> None:
    segments = [
        Segment(30.50, 38.16, "It was just a random warehouse and they handed me"),
        Segment(38.04, 40.20, "this box sealed very well."),
    ]
    grouped = _materialize_groups(segments, {0: True})
    assert len(grouped) == 1
    assert "handed me this box sealed very well." in grouped[0].text


def test_parse_join_after_reads_json_object() -> None:
    assert _parse_join_after('{"join_after": [1, 3, 4]}') == {1, 3, 4}


def test_resegment_uses_model_decisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segments = [
        Segment(1.0, 2.0, "poured the"),
        Segment(2.1, 3.5, "springtails in,"),
        Segment(4.0, 5.0, "and they explored."),
    ]

    class _FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"join_after":[1]}'))]
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(
        "yblocalizer.subtitle_resegment._provider_client",
        lambda _provider: (_FakeClient(), "DEEPSEEK_API_KEY", None),
    )
    out = resegment_source_segments(
        segments,
        provider="deepseek",
        model="dummy",
        batch_size=40,
        checkpoint_path=tmp_path / "semantic_resegmentation_checkpoint.json",
    )
    assert [item.text for item in out] == [
        "poured the springtails in,",
        "and they explored.",
    ]
    assert (tmp_path / "semantic_resegmentation_checkpoint.json").is_file()


def test_resegment_falls_back_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 3.0, "cristal."),
    ]

    def _boom(_provider):
        raise RuntimeError("network down")

    monkeypatch.setattr("yblocalizer.subtitle_resegment._provider_client", _boom)
    out = resegment_source_segments(segments, provider="deepseek", model="dummy")
    assert len(out) == 1
    assert "cristal" in out[0].text.lower()


def test_resegment_skips_llm_for_none_provider() -> None:
    segments = [
        Segment(1.0, 2.2, "puede rayar el"),
        Segment(2.25, 3.0, "cristal."),
    ]
    out = resegment_source_segments(segments, provider="none")
    assert len(out) == 1

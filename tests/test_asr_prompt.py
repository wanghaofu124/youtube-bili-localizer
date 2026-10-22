from yblocalizer.asr_prompt import build_initial_prompt
from yblocalizer.transcribe import _vad_profile
from yblocalizer.workflow import invalidation_stage
from yblocalizer.workbench_config import InputValidationError, normalize_options

import pytest


def test_title_keeps_place_name_and_drops_generic_words() -> None:
    prompt = build_initial_prompt(
        "2 Day Palm Beach Fishing Adventure - Clear Water, Snorkeling, Tropical Fish",
    )
    assert prompt == "Keywords: Palm Beach"


def test_user_hint_is_appended_without_repeating_auto_terms() -> None:
    prompt = build_initial_prompt(
        "Palm Beach Fishing",
        user_hint="tarpon, ballyhoo, Palm Beach",
    )
    assert prompt == "tarpon, ballyhoo, Palm Beach"


def test_user_hint_alone_is_enough() -> None:
    assert build_initial_prompt(user_hint="  pompano jig  ") == "pompano jig"


def test_empty_metadata_makes_no_prompt() -> None:
    assert build_initial_prompt("Fishing Adventure", "clear water", ["fish"]) is None
    assert build_initial_prompt() is None


def test_tag_keeps_a_species_the_title_does_not_capitalize() -> None:
    prompt = build_initial_prompt("Fishing", tags=["tarpon"])
    assert prompt == "Keywords: tarpon"


def test_vad_profiles_keep_the_current_standard_behavior() -> None:
    standard = _vad_profile("standard")
    sensitive = _vad_profile("sensitive")
    assert standard["threshold"] is None
    assert standard["speech_pad_ms"] == 400
    assert standard["recovery_threshold"] == 0.35
    assert sensitive["threshold"] == 0.35
    assert sensitive["speech_pad_ms"] == 500
    assert sensitive["recovery_threshold"] == 0.30
    assert _vad_profile("unknown") == standard


def test_asr_options_default_and_reject_bad_values() -> None:
    options = normalize_options({}, "outputs/test")
    assert options["asr_hint"] == ""
    assert options["asr_vad_mode"] == "standard"
    saved = normalize_options({"asr_hint": "tarpon", "asr_vad_mode": "sensitive"}, "outputs/test")
    assert saved["asr_hint"] == "tarpon"
    assert saved["asr_vad_mode"] == "sensitive"
    with pytest.raises(InputValidationError) as captured:
        normalize_options({"asr_vad_mode": "aggressive"}, "outputs/test")
    assert captured.value.field == "asr_vad_mode"


def test_asr_settings_invalidate_extract() -> None:
    assert invalidation_stage({"asr_hint"}) == "extract"
    assert invalidation_stage({"asr_vad_mode"}) == "extract"

from types import SimpleNamespace
import json
from pathlib import Path

import pysubs2

from yblocalizer.asr_prompt import build_initial_prompt
from yblocalizer.glossary import DEFAULT_GLOSSARY_PATH, glossary_hash, glossary_hits, load_glossary, read_glossary_file, save_glossary
from yblocalizer.models import Segment
from yblocalizer.render import _prepare_ass
from yblocalizer.translate import correct_source_segments, translate_segments
from yblocalizer.workflow import invalidate_downstream, new_stage_states

GLOSSARY = [{"source": "Super GT", "target": "Super GT", "note": "Japanese racing series", "aliases": ["SuperGT"]},
            {"source": "downforce", "target": "下压力", "note": "", "aliases": []}]


def _capture(monkeypatch, reply=lambda prompt: "1. 中文"):
    prompts: list[str] = []

    class Completions:
        def create(self, **kwargs):
            prompt = "\n".join(message["content"] for message in kwargs["messages"])
            prompts.append(prompt)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply(prompt)))])

    monkeypatch.setattr("yblocalizer.translate._chat_client", lambda **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr("yblocalizer.translate._require_ollama_model", lambda model: model)
    return prompts


def test_hits_are_case_insensitive_and_use_aliases() -> None:
    assert [item["source"] for item in glossary_hits(GLOSSARY, "the supergt car has DOWNFORCE")] == ["Super GT", "downforce"]
    assert glossary_hits(GLOSSARY, "nothing here") == []
    assert glossary_hits(None, "Super GT") == []


def test_global_default_and_job_override(tmp_path, monkeypatch) -> None:
    import yblocalizer.glossary as glossary_module

    monkeypatch.setattr(glossary_module, "GLOBAL_ROOT", None)
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    assert load_glossary(job_dir) == read_glossary_file(DEFAULT_GLOSSARY_PATH)
    save_glossary(tmp_path / "glossary.json", GLOSSARY)
    (job_dir / "glossary.json").write_text(json.dumps([{"source": "DOWNFORCE", "target": "气动下压力"}]), encoding="utf-8")
    loaded = load_glossary(job_dir)
    assert [item["target"] for item in loaded] == ["Super GT", "气动下压力"]
    assert glossary_hash(loaded) != glossary_hash(GLOSSARY)


def test_strict_prompt_contains_glossary_only_on_hit(monkeypatch) -> None:
    prompts = _capture(monkeypatch)
    source = [Segment(0, 1, "Lots of downforce here")]
    translate_segments(source, provider="deepseek", smart_translation=False, strict_alignment=True, glossary=GLOSSARY)
    translate_segments(source, provider="deepseek", smart_translation=False, strict_alignment=True, glossary=[])
    translate_segments(source, provider="deepseek", smart_translation=False, strict_alignment=True)
    assert "downforce = 下压力" in prompts[0]
    assert "Glossary" not in prompts[1] and prompts[1] == prompts[2]


def test_smart_prompt_contains_glossary(monkeypatch) -> None:
    prompts = _capture(monkeypatch)
    source = [Segment(0, 1, "Super GT is fast")]
    translate_segments(source, provider="deepseek", smart_translation=True, glossary=GLOSSARY)
    translate_segments(source, provider="deepseek", smart_translation=True, glossary=[])
    assert any("- Super GT: Super GT. (Japanese racing series)" in prompt for prompt in prompts[:len(prompts) // 2])
    assert prompts[: len(prompts) // 2] != prompts[len(prompts) // 2:]
    assert not any("Super GT: Super GT" in prompt for prompt in prompts[len(prompts) // 2:])


def test_translategemma_prompt_contains_glossary(monkeypatch) -> None:
    prompts = _capture(monkeypatch, reply=lambda _prompt: "中文")
    source = [Segment(0, 1, "More downforce.")]
    translate_segments(source, provider="ollama", model="translategemma:12b", source_lang="en", glossary=GLOSSARY)
    translate_segments(source, provider="ollama", model="translategemma:12b", source_lang="en", glossary=[])
    translate_segments(source, provider="ollama", model="translategemma:12b", source_lang="en")
    assert "downforce = 下压力" in prompts[0]
    assert prompts[1] == prompts[2] and "Glossary" not in prompts[1]


def test_source_correction_prompt_contains_glossary_terms(monkeypatch) -> None:
    prompts = _capture(monkeypatch, reply=lambda _prompt: "1. The SuperGT car.")
    source = [Segment(0, 1, "The SuperGT car.")]
    correct_source_segments(source, provider="deepseek", force_review=True, glossary=GLOSSARY)
    correct_source_segments(source, provider="deepseek", force_review=True, glossary=[])
    assert "Glossary terms (keep the source spelling): Super GT." in prompts[0]
    assert "Glossary terms" not in prompts[-1]


def test_asr_prompt_puts_glossary_terms_first() -> None:
    plain = build_initial_prompt(title="Super GT onboard lap", tags=["racing"])
    with_glossary = build_initial_prompt(title="Super GT onboard lap", tags=["racing"], glossary=GLOSSARY)
    assert build_initial_prompt(title="Super GT onboard lap", tags=["racing"], glossary=[]) == plain
    assert with_glossary.startswith("Keywords: Super GT")


def test_glossary_change_only_invalidates_translate_and_later() -> None:
    states = new_stage_states()
    for row in states.values():
        row["status"] = "completed"
    invalidate_downstream(states, "translate")
    assert states["acquire"]["status"] == "completed" and states["extract"]["status"] == "completed"
    assert states["translate"]["status"] != "completed" and states["render"]["status"] != "completed"


def _ass(tmp_path, display_mode, preset="soft"):
    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\nHello\n", encoding="utf-8")
    path = _prepare_ass(srt, size=(1920, 1080), font_name="Microsoft YaHei", font_size=24, primary_color="&H00FFFFFF",
                        outline_color="&H00000000", outline=2, shadow=0, margin_v=None, raised_margin=False,
                        margin_ratio=0.05, display_mode=display_mode, subtitle_preset=preset,
                        cue_source_lines={1: 1})
    return pysubs2.load(str(path))


def test_bilingual_source_line_uses_secondary_style(tmp_path) -> None:
    subs = _ass(tmp_path, "bilingual-translation-first", "compact")
    assert subs.styles["Secondary"].fontsize < subs.styles["Default"].fontsize
    assert "{\\rSecondary}Hello{\\r}" in subs.events[0].text
    equal = _ass(tmp_path, "bilingual-translation-first", "equal")
    assert equal.styles["Secondary"].fontsize == equal.styles["Default"].fontsize


def test_single_language_ass_is_unchanged(tmp_path) -> None:
    subs = _ass(tmp_path, "translated")
    assert "Secondary" not in subs.styles
    assert subs.events[0].text == "你好\\NHello"


def test_glossary_hits_ascii_terms_use_word_boundaries() -> None:
    def hit(term: str, text: str) -> bool:
        return bool(glossary_hits([{"source": term, "target": "", "note": "", "aliases": []}], text))

    assert not hit("AI", "he said so")
    assert hit("AI", "new AI model")
    assert not hit("aero", "aerospace parts")
    assert hit("aero", "the aero package")
    assert not hit("GT300", "GT3000 class")
    assert hit("GT300", "GT300 class")
    assert not hit("lap time", "lap timer")
    assert hit("lap time", "best Lap Time today")
    assert hit("下压力", "这个下压力很大")


def test_load_glossary_reads_global_root_for_custom_output_dir(tmp_path: Path, monkeypatch) -> None:
    import yblocalizer.glossary as glossary_module

    global_root = tmp_path / "workbench"
    save_glossary(global_root / "glossary.json", [{"source": "aero", "target": "空力"}, {"source": "AI", "target": "人工智能"}])
    job_dir = tmp_path / "custom_output" / "job1"
    save_glossary(job_dir / "glossary.json", [{"source": "AI", "target": "AI"}])
    monkeypatch.setattr(glossary_module, "GLOBAL_ROOT", global_root)
    loaded = {item["source"]: item["target"] for item in load_glossary(job_dir)}
    assert loaded == {"aero": "空力", "AI": "AI"}

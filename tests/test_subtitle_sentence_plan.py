from yblocalizer.models import Segment
from yblocalizer.subtitle import _wrap_cjk_line
from yblocalizer.subtitle_sentence_plan import (
    SentenceUnit,
    build_sentence_plan,
    normalize_dense_translation_cues,
    redistribute_sentence_translations,
)
from yblocalizer.subtitle_speech_pack import (
    TimedWord,
    pack_cues_from_words,
    pack_cues_from_words_detailed,
)
import re


def _to_raise_words() -> list[TimedWord]:
    return [
        TimedWord(14.26, 14.50, "Let's"),
        TimedWord(14.52, 14.70, " go"),
        TimedWord(14.72, 14.90, " back"),
        TimedWord(14.92, 15.10, " to"),
        TimedWord(15.12, 15.30, " the"),
        TimedWord(15.32, 15.70, " beginning."),
        TimedWord(15.88, 16.18, " To"),
        TimedWord(17.80, 18.10, " raise"),
        TimedWord(18.12, 18.40, " these"),
        TimedWord(18.42, 18.90, " tadpoles,"),
        TimedWord(19.00, 19.30, " we're"),
        TimedWord(19.32, 19.50, " going"),
        TimedWord(19.52, 19.70, " to"),
        TimedWord(19.72, 19.90, " have"),
        TimedWord(19.92, 20.10, " to"),
        TimedWord(20.12, 20.50, " overcome"),
        TimedWord(20.52, 20.80, " challenges."),
    ]


def test_sentence_plan_keeps_to_raise_together() -> None:
    plan = build_sentence_plan(_to_raise_words())
    texts = [unit.source.text for unit in plan]
    assert any("To raise these tadpoles" in text for text in texts)
    assert not any(unit.source.text.strip() in {"To", "to"} for unit in plan)
    for unit in plan:
        assert not any(slot.text.strip() in {"To", "to"} for slot in unit.display_slots)


def test_pack_no_longer_leaves_to_orphan() -> None:
    cues = pack_cues_from_words(_to_raise_words())
    assert not any(cue.text.strip() in {"To", "to"} for cue in cues)


def test_redistribute_avoids_weile_orphan() -> None:
    plan = build_sentence_plan(_to_raise_words())
    unit = next(item for item in plan if "To raise" in item.source.text)
    translated = [
        Segment(u.source.start, u.source.end, u.source.text, None)
        for u in plan
    ]
    for index, item in enumerate(plan):
        if item is unit:
            translated[index].translated_text = "为了养大这些蝌蚪，我们得克服各种挑战。"
        else:
            translated[index].translated_text = "让我们回到最开始。"
    display = redistribute_sentence_translations(plan, translated)
    zh_texts = [(item.translated_text or "").strip() for item in display]
    assert "为了" not in zh_texts
    assert any("为了养大这些蝌蚪" in text for text in zh_texts)


def test_redistribute_does_not_split_mid_word() -> None:
    unit_slots = [
        Segment(46.0, 50.9, "After a long drive home, I was worried the tadpoles might be stressed out, but when I cut into the box,"),
        Segment(50.9, 52.7, "they seemed like they were packed super well."),
    ]
    source_text = (
        "After a long drive home, I was worried the tadpoles might be stressed out, "
        "but when I cut into the box, they seemed like they were packed super well."
    )
    plan = [
        SentenceUnit(
            source=Segment(46.0, 52.7, source_text),
            display_slots=unit_slots,
        )
    ]
    translated = [
        Segment(
            46.0,
            52.7,
            source_text,
            "开回家花了很久，我担心蝌蚪会紧张，但剪开箱子后，它们看起来包装得特别好。",
        )
    ]
    display = redistribute_sentence_translations(plan, translated)
    texts = [(item.translated_text or "") for item in display]
    assert "它们" in "".join(texts)
    assert not any(t.endswith("它") and nxt.startswith("们") for t, nxt in zip(texts, texts[1:]))


def test_cjk_wrap_keeps_piaoliang_together() -> None:
    text = "我想看看自己能不能养出这颗星球上最漂亮的青蛙。"
    lines = _wrap_cjk_line(text, max_chars=18)
    assert "".join(lines) == text
    for left, right in zip(lines, lines[1:]):
        assert not (left.endswith("漂") and right.startswith("亮"))


def test_dense_translation_fallback_splits_inside_original_time_and_preserves_content() -> None:
    source = "From mock battles to prey animals, then a thousand ants fight a hornet army."
    translated = "从模拟子弹对战致命猎物，到让一千只蚂蚁对抗杀手大黄蜂大军，再到意外在我家引发全面战争——这些就是我2026年最精彩的视频。"
    original = Segment(20.0, 28.0, source, translated)

    result = normalize_dense_translation_cues([original])

    assert len(result) == 2
    assert "".join(item.translated_text or "" for item in result) == translated
    assert "".join(item.text for item in result) == source
    assert result[0].start == original.start
    assert result[-1].end == original.end
    assert all(item.end - item.start >= 0.8 for item in result)
    assert all(
        (following.start - current.end) == 0
        for current, following in zip(result, result[1:])
    )


def test_dense_unpunctuated_translation_splits_losslessly() -> None:
    translated = "这是一个没有标点但内容很长的字幕句子需要在合理位置拆开继续表达完整意思而不能全部堆在同一个画面里面影响阅读体验"
    original = Segment(1.0, 7.0, "This is a long unpunctuated subtitle sentence that must remain complete.", translated)

    result = normalize_dense_translation_cues([original])

    assert len(result) == 2
    assert "".join(item.translated_text or "" for item in result) == translated
    assert "".join(item.text for item in result) == original.text
    assert result[0].start == original.start
    assert result[-1].end == original.end


def test_dense_translation_is_not_extended_beyond_a_short_source_cue() -> None:
    translated = "这是一条超长字幕，不应为了排版而延长原始语音时间。"
    original = Segment(2.0, 2.7, "A short cue.", translated)

    result = normalize_dense_translation_cues([original])

    assert result == [original]


def test_redistribute_coalesces_cross_sentence_incomplete_zh() -> None:
    """Adjacent sentence units ending on discourse/pronoun stubs must merge."""
    plan = [
        SentenceUnit(
            source=Segment(10.0, 12.0, "I'll keep watching him in case"),
            display_slots=[Segment(10.0, 12.0, "I'll keep watching him in case")],
        ),
        SentenceUnit(
            source=Segment(12.05, 14.0, "something happens."),
            display_slots=[Segment(12.05, 14.0, "something happens.")],
        ),
    ]
    translated = [
        Segment(10.0, 12.0, plan[0].source.text, "我会一直盯着他，万一"),
        Segment(12.05, 14.0, plan[1].source.text, "有情况，就能马上试着救他。"),
    ]
    display = redistribute_sentence_translations(plan, translated)
    texts = [(item.translated_text or "").strip() for item in display]
    assert len(texts) == 1
    assert texts[0].startswith("我会一直盯着他，万一有情况")


def test_coalesce_does_not_glue_finished_le_ending() -> None:
    """Natural 「…了」 endings must not merge with the next unrelated cue."""
    plan = [
        SentenceUnit(
            source=Segment(1.0, 2.0, "I got out of the car."),
            display_slots=[Segment(1.0, 2.0, "I got out of the car.")],
        ),
        SentenceUnit(
            source=Segment(2.1, 3.5, "Then I opened the box."),
            display_slots=[Segment(2.1, 3.5, "Then I opened the box.")],
        ),
    ]
    translated = [
        Segment(1.0, 2.0, plan[0].source.text, "我下车了"),
        Segment(2.1, 3.5, plan[1].source.text, "然后打开了箱子。"),
    ]
    display = redistribute_sentence_translations(plan, translated)
    texts = [(item.translated_text or "").strip() for item in display]
    assert texts == ["我下车了", "然后打开了箱子。"]


def test_dont_know_stays_one_sentence_unit() -> None:
    # Soft-cap would otherwise cut after ~8s right before "know".
    words = [
        TimedWord(100.0, 100.3, "bigger"),
        TimedWord(100.3, 100.5, " than"),
        TimedWord(100.5, 100.8, " basically"),
        TimedWord(100.8, 101.0, " some"),
        TimedWord(101.0, 101.3, " little"),
        TimedWord(101.3, 101.6, " small"),
        TimedWord(101.6, 102.0, " minnows"),
        TimedWord(102.0, 102.3, " maybe"),
        TimedWord(102.3, 102.5, " that"),
        TimedWord(102.5, 102.8, " long"),
        TimedWord(102.8, 103.0, " that's"),
        TimedWord(103.0, 103.2, " all"),
        TimedWord(103.2, 103.4, " I"),
        TimedWord(103.4, 103.6, " put"),
        TimedWord(103.6, 103.8, " in"),
        TimedWord(103.8, 104.0, " a"),
        TimedWord(104.0, 104.2, " tank"),
        TimedWord(104.2, 104.4, " like"),
        TimedWord(104.4, 104.7, " this"),
        TimedWord(104.7, 105.0, " maybe"),
        TimedWord(105.0, 105.2, " a"),
        TimedWord(105.2, 105.5, " betta"),
        TimedWord(105.5, 105.8, " fish"),
        TimedWord(105.8, 106.0, " but"),
        TimedWord(106.0, 106.2, " again"),
        TimedWord(106.2, 106.35, " I"),
        TimedWord(106.35, 106.55, " don't"),
        TimedWord(106.55, 106.8, " know"),
        TimedWord(106.8, 107.0, " it's"),
        TimedWord(107.0, 107.2, " still"),
        TimedWord(107.2, 107.5, " small."),
    ]
    plan = build_sentence_plan(words)
    blob = " ".join(unit.source.text for unit in plan).lower()
    assert "don't know" in blob or "dont know" in blob.replace("'", "")
    assert not any(
        unit.source.text.strip().lower().endswith("don't")
        or unit.source.text.strip().lower().endswith("i don't")
        for unit in plan
    )
    for unit in plan:
        for slot in unit.display_slots:
            st = slot.text.strip().lower()
            assert not st.endswith("don't")
            assert not st.endswith("i don't")


def test_pack_keeps_dont_know_across_soft_cap() -> None:
    words = [
        TimedWord(0.0, 0.3, "bigger"),
        TimedWord(0.3, 0.5, " than"),
        TimedWord(0.5, 0.8, " basically"),
        TimedWord(0.8, 1.1, " some"),
        TimedWord(1.1, 1.4, " little"),
        TimedWord(1.4, 1.7, " small"),
        TimedWord(1.7, 2.1, " minnows"),
        TimedWord(2.1, 2.4, " maybe"),
        TimedWord(2.4, 2.6, " that"),
        TimedWord(2.6, 2.9, " long"),
        TimedWord(2.9, 3.1, " that's"),
        TimedWord(3.1, 3.3, " all"),
        TimedWord(3.3, 3.5, " I"),
        TimedWord(3.5, 3.7, " put"),
        TimedWord(3.7, 3.9, " in"),
        TimedWord(3.9, 4.1, " a"),
        TimedWord(4.1, 4.3, " tank"),
        TimedWord(4.3, 4.5, " like"),
        TimedWord(4.5, 4.8, " this"),
        TimedWord(4.8, 5.1, " maybe"),
        TimedWord(5.1, 5.3, " a"),
        TimedWord(5.3, 5.6, " betta"),
        TimedWord(5.6, 5.9, " fish"),
        TimedWord(5.9, 6.1, " but"),
        TimedWord(6.1, 6.3, " again"),
        TimedWord(6.3, 6.45, " I"),
        TimedWord(6.45, 6.7, " don't"),
        TimedWord(6.7, 7.0, " know"),
        TimedWord(7.0, 7.3, " it's"),
        TimedWord(7.3, 7.6, " small."),
    ]
    cues = pack_cues_from_words(words)
    assert any("don't know" in cue.text.lower() for cue in cues)
    assert not any(cue.text.strip().lower().endswith("don't") for cue in cues)



def test_zh_coalesce_merges_mid_phrase_bu() -> None:
    plan = [
        SentenceUnit(
            source=Segment(1.0, 2.0, "I don't"),
            display_slots=[Segment(1.0, 2.0, "I don't")],
        ),
        SentenceUnit(
            source=Segment(2.05, 3.5, "know for sure."),
            display_slots=[Segment(2.05, 3.5, "know for sure.")],
        ),
    ]
    translated = [
        Segment(1.0, 2.0, plan[0].source.text, "但再说一次，我并不"),
        Segment(2.05, 3.5, plan[1].source.text, "确定。这毕竟还是很小的缸。"),
    ]
    display = redistribute_sentence_translations(plan, translated)
    texts = [(item.translated_text or "").strip() for item in display]
    assert len(texts) == 1
    assert "我并不确定" in texts[0]

def _unpunctuated_release_tail() -> list[TimedWord]:
    """Mimic the caterpillar-video ending: no punctuation, long editing pauses."""
    return [
        TimedWord(499.59, 499.99, "Next"),
        TimedWord(500.74, 501.08, " up,"),
        TimedWord(501.08, 501.22, " were"),
        TimedWord(501.22, 501.36, " the"),
        TimedWord(501.36, 501.90, " swallowtails"),
        TimedWord(501.90, 502.34, " all"),
        TimedWord(502.34, 502.56, " right"),
        TimedWord(503.92, 504.10, " they're"),
        TimedWord(504.10, 504.28, " ready"),
        TimedWord(504.28, 504.46, " to"),
        TimedWord(504.46, 504.78, " go"),
        TimedWord(507.67, 507.95, " there"),
        TimedWord(510.30, 510.42, " goes"),
        TimedWord(510.42, 510.62, " one"),
        TimedWord(517.81, 517.97, " he"),
        TimedWord(517.97, 518.21, " goes"),
        TimedWord(527.14, 527.62, " it's"),
        TimedWord(539.54, 539.74, " the"),
        TimedWord(539.74, 539.96, " last"),
        TimedWord(539.96, 540.30, " one"),
        TimedWord(540.30, 540.40, " all"),
        TimedWord(546.43, 546.55, " right"),
        TimedWord(546.55, 546.75, " guys"),
        TimedWord(563.70, 563.96, " here"),
        TimedWord(567.43, 567.57, " he"),
        TimedWord(567.57, 567.77, " goes"),
        TimedWord(576.51, 577.11, " there"),
        TimedWord(581.54, 581.64, " he"),
        TimedWord(581.64, 581.90, " goes"),
        TimedWord(598.00, 598.22, " all"),
        TimedWord(606.02, 606.14, " right"),
        TimedWord(606.14, 606.26, " guys"),
        TimedWord(606.84, 607.04, " end"),
        TimedWord(607.26, 607.40, " thank"),
        TimedWord(607.40, 607.54, " you"),
    ]


def test_unpunctuated_tail_does_not_become_mega_unit() -> None:
    plan = build_sentence_plan(_unpunctuated_release_tail())
    assert len(plan) >= 4
    assert max(unit.source.end - unit.source.start for unit in plan) <= 40
    assert max(len(unit.display_slots) for unit in plan) <= 3
    joined = " ".join(unit.source.text for unit in plan)
    assert "all right" in joined
    assert "there goes" in joined or "there he goes" in joined
    for unit in plan:
        for slot in unit.display_slots:
            st = slot.text.strip().lower()
            assert not st.endswith(" all")
            assert not st.endswith(" there he")
    assert "all right" in joined


def test_pack_does_not_keep_phrases_across_scene_cuts() -> None:
    words = [
        TimedWord(0.0, 0.3, "it's"),
        TimedWord(0.3, 0.5, " the"),
        TimedWord(0.5, 0.8, " last"),
        TimedWord(0.8, 1.1, " one"),
        TimedWord(1.1, 1.3, " all"),
        TimedWord(7.5, 7.7, " right"),
        TimedWord(7.7, 8.0, " guys"),
        TimedWord(8.0, 8.3, " there"),
        TimedWord(12.0, 12.2, " he"),
        TimedWord(12.2, 12.5, " goes"),
    ]
    packed = pack_cues_from_words_detailed(words)
    cues = packed.cues
    assert [cue.text for cue in cues] == ["it's the last one", "right guys there", "he goes"]
    assert [fragment.text for fragment in packed.detached_fragments] == ["all"]
    assert cues[0].end < cues[1].start
    assert cues[1].end < cues[2].start


def test_long_pause_all_right_merges_into_following_slot() -> None:
    """A clipped ``all`` is audited instead of becoming an orphan subtitle."""
    words = [
        TimedWord(597.86, 598.22, "all"),
        TimedWord(606.02, 606.14, " right"),
        TimedWord(606.14, 606.26, " guys"),
        TimedWord(606.26, 606.40, " if"),
        TimedWord(606.40, 606.44, " you"),
        TimedWord(606.44, 606.58, " made"),
        TimedWord(606.58, 606.70, " it"),
        TimedWord(606.70, 606.76, " to"),
        TimedWord(606.76, 606.84, " the"),
        TimedWord(606.84, 607.04, " end"),
        TimedWord(607.26, 607.40, " thank"),
        TimedWord(607.40, 607.54, " you"),
        TimedWord(615.61, 615.75, " for"),
        TimedWord(615.75, 615.97, " watching"),
        TimedWord(617.13, 617.63, " peace"),
    ]
    packed = pack_cues_from_words_detailed(words)
    assert packed.cues
    assert packed.detached_fragments[0].text.lower() == "all"
    assert packed.detached_fragments[0].start == 597.86
    assert any(cue.text.lower().startswith("right guys") for cue in packed.cues)
    assert not any(cue.text.lower().startswith("all right") for cue in packed.cues)


def test_oh_yeah_stays_one_unit_across_editing_pause() -> None:
    words = [
        TimedWord(457.46, 457.78, "Oh,"),
        TimedWord(464.54, 464.84, " yeah."),
        TimedWord(465.84, 466.24, " He's"),
        TimedWord(466.24, 466.44, " ready"),
        TimedWord(466.44, 466.68, " now."),
    ]
    plan = build_sentence_plan(words)
    texts = [unit.source.text for unit in plan]
    assert any("Oh" in text and "yeah" in text.lower() for text in texts)
    assert not any(unit.source.text.strip() in {"Oh,", "Oh"} for unit in plan)

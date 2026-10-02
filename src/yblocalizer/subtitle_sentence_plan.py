"""Sentence-unit translation plan with timed display redistribution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unicodedata

from .models import Segment
from .subtitle_speech_pack import (
    TimedWord,
    END_PAD_SEC,
    _clean_token,
    _cue_from_group,
    _is_incomplete_tail,
    _is_sentence_end,
    _join_word_texts,
    is_phrase_glued,
    normalize_word_stream,
    pack_cues_from_words,
)


SENTENCE_PLAN_VERSION = 1
MAX_SENTENCE_DURATION_SEC = 18.0
MAX_SENTENCE_WORDS = 45
EMERGENCY_SENTENCE_DURATION_SEC = 25.0
MAX_MEGA_DURATION_SEC = 36.0
MAX_SLOTS_PER_TRANSLATION_UNIT = 3
LONG_PAUSE_SEC = 1.15
VERY_LONG_PAUSE_SEC = 2.4
EXTREME_PAUSE_SEC = 5.0
MIN_STANDALONE_WORDS = 3
MIN_STANDALONE_DURATION = 0.9
MIN_CJK_CHARS = 4
MAX_CJK_DISPLAY_UNITS_PER_CUE = 36
MIN_DENSITY_SLOT_DURATION_SEC = 1.0
MIN_WORDS_PER_DENSITY_SLOT = 2
MIN_ESTIMATED_SLOT_DURATION_SEC = 0.8
_ZH_CLAUSE_CONNECTORS = (
    "与此同时", "也就是说", "接下来", "尽管", "即使", "即便", "但是", "不过",
    "然而", "因此", "所以", "而且", "并且", "或者", "然后", "接着", "于是",
    "如果", "因为", "虽然", "同时", "结果", "随后", "不然", "否则", "尤其",
    "比如", "例如", "只要", "除非", "最终", "此外", "另外", "总之",
)
_ZH_ATTACHED_PREFIXES = frozenset("的地得了着过和与及并把被让给从对向到在是有没不很更最一这那可但而又或就都也还再才只要会能想")
BARE_ARTICLES = frozenset({"a", "an", "the", "el", "la", "un", "una", "los", "las"})
# Verbs/adjectives that commonly need a complement; duration caps must not cut after these.
_NEEDS_COMPLEMENT = frozenset(
    {
        "put",
        "give",
        "make",
        "take",
        "get",
        "got",
        "keep",
        "let",
        "help",
        "look",
        "looks",
        "looking",
        "start",
        "started",
        "starting",
        "begin",
        "beginning",
        "going",
        "wanna",
        "gonna",
        "proper",
        "full",
        "own",
        "more",
        "much",
        "about",
        "into",
        "onto",
        "from",
        "with",
        "without",
        "like",
        "than",
        "perfect",
        "floating",
        "all",
        "there",
        "here",
        # Apostrophes are stripped by _clean_token; keep bare forms only.
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "wont",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "havent",
        "hasnt",
        "hadnt",
        "thats",
        "whats",
        "lets",
    }
)


@dataclass(slots=True)
class SentenceUnit:
    source: Segment
    display_slots: list[Segment]
    timing_words: list[TimedWord] = field(default_factory=list)


def build_sentence_plan(words: list[TimedWord]) -> list[SentenceUnit]:
    """Build complete sentence translation units and speech-timed display slots."""
    stream = normalize_word_stream(words)
    if not stream:
        return []

    sentence_groups = _group_sentence_words(stream)
    units: list[SentenceUnit] = []
    for group in sentence_groups:
        source = _cue_from_group(group)
        if source is None:
            continue
        slots = pack_cues_from_words(group)
        slots = _merge_fragile_slots(slots)
        if not slots:
            slots = [Segment(source.start, source.end, source.text)]
        chunks = _chunk_translation_slots(slots)
        word_cursor = 0
        for chunk_index, chunk in enumerate(chunks):
            joined = _join_slot_texts(chunk)
            if not joined:
                continue
            chunk_source = Segment(chunk[0].start, chunk[-1].end, joined)
            # Keep the exact word-clock atoms for this translation window. The
            # display planner may later add a few safe cuts without changing
            # the full-sentence translation request.
            final_chunk = chunk_index == len(chunks) - 1
            raw_end = chunk[-1].end - END_PAD_SEC
            chunk_words: list[TimedWord] = []
            while word_cursor < len(group) and (
                final_chunk or group[word_cursor].end <= raw_end + 0.01
            ):
                chunk_words.append(group[word_cursor])
                word_cursor += 1
            units.append(
                SentenceUnit(
                    source=chunk_source,
                    display_slots=chunk,
                    timing_words=chunk_words,
                )
            )
    return units


def _chunk_translation_slots(slots: list[Segment]) -> list[list[Segment]]:
    """Keep redistribute windows small so one ZH sentence isn't painted across many cues."""
    if len(slots) <= MAX_SLOTS_PER_TRANSLATION_UNIT:
        duration = slots[-1].end - slots[0].start if slots else 0.0
        if duration <= MAX_MEGA_DURATION_SEC:
            return [slots]
    chunks: list[list[Segment]] = []
    current: list[Segment] = []
    for slot in slots:
        trial = current + [slot]
        duration = trial[-1].end - trial[0].start
        if current and (
            len(trial) > MAX_SLOTS_PER_TRANSLATION_UNIT or duration > MAX_MEGA_DURATION_SEC
        ):
            # Never open a new translate unit after unfinished EN ("I don't" | "know").
            tail_word = _last_en_word(current[-1].text)
            unfinished = _is_incomplete_tail(tail_word) or _needs_complement(tail_word)
            if unfinished and len(current) >= 2:
                chunks.append(current[:-1])
                current = [current[-1], slot]
            elif unfinished:
                current = trial
            else:
                chunks.append(current)
                current = [slot]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks or [slots]


def _last_en_word(text: str) -> str:
    parts = (text or "").strip().split()
    return parts[-1] if parts else ""


def redistribute_sentence_translations(
    units: list[SentenceUnit],
    translated_sentences: list[Segment],
) -> list[Segment]:
    """Map each sentence translation onto its speech-timed display slots."""
    if len(units) != len(translated_sentences):
        raise RuntimeError(
            f"sentence plan size mismatch: {len(units)} units vs {len(translated_sentences)} translations"
        )
    output: list[Segment] = []
    for unit, translated in zip(units, translated_sentences):
        zh = (translated.translated_text or "").strip()
        slots = list(unit.display_slots)
        if not slots:
            continue
        if not zh:
            # Keep English timing even if translation is empty.
            for slot in slots:
                output.append(Segment(slot.start, slot.end, slot.text, ""))
            continue
        mapped = _map_translation_to_slots(zh, slots, timing_words=unit.timing_words)
        output.extend(mapped)
    merged = _coalesce_incomplete_zh_cues(output)
    merged = _reflow_dense_segments(merged, units)
    return merged


def redistribute_dense_translations_to_existing_slots(
    translated_segments: list[Segment], source_slots: list[Segment]
) -> list[Segment]:
    """Split dense translations only across exact, existing source cue slots."""
    output: list[Segment] = []
    ordered_slots = sorted(source_slots, key=lambda item: (item.start, item.end))
    for segment in translated_segments:
        translated = (segment.translated_text or "").strip()
        if not _has_cjk(translated) or _display_units(translated) <= MAX_CJK_DISPLAY_UNITS_PER_CUE:
            output.append(segment)
            continue
        slots = [
            slot for slot in ordered_slots
            if slot.start >= segment.start - 0.05 and slot.end <= segment.end + 0.05
        ]
        if (
            len(slots) < 2
            or abs(slots[0].start - segment.start) > 0.1
            or abs(slots[-1].end - segment.end) > 0.15
            or any(
                not 0 <= following.start - current.end <= 0.55
                for current, following in zip(slots, slots[1:])
            )
            or _source_signature(_join_slot_texts(slots)) != _source_signature(segment.text)
        ):
            output.append(segment)
            continue
        mapped = _map_translation_to_slots(translated, slots)
        if len(mapped) < 2 or "".join(item.translated_text or "" for item in mapped) != translated:
            output.append(segment)
            continue
        output.extend(mapped)
    return output


def normalize_dense_translation_cues(segments: list[Segment]) -> list[Segment]:
    """Split remaining dense Chinese cues inside their existing time windows.

    Exact word-clock and existing-slot mapping are attempted first by callers.
    This final fallback uses safe Chinese clause boundaries when available, then
    a lossless balanced cut as a last resort. It never moves text outside the
    original cue's interval or extends the media timeline.
    """
    normalized: list[Segment] = []
    for segment in segments:
        text = segment.translated_text or ""
        width = _display_units(text)
        span = segment.end - segment.start
        if not _has_cjk(text) or width <= MAX_CJK_DISPLAY_UNITS_PER_CUE or span <= 0:
            normalized.append(segment)
            continue

        requested = math.ceil(width / MAX_CJK_DISPLAY_UNITS_PER_CUE)
        available = math.floor((span + 1e-9) / MIN_ESTIMATED_SLOT_DURATION_SEC)
        count = min(requested, available)
        if count < 2:
            normalized.append(segment)
            continue

        parts = _partition_dense_chinese(text, count)
        if not parts or "".join(parts) != text:
            normalized.append(segment)
            continue

        weights = [max(1.0, _display_units(part)) for part in parts]
        source_parts = _partition_source_text(segment.text, weights)
        durations = _allocate_dense_durations(span, weights)
        cursor = segment.start
        children: list[Segment] = []
        for index, (part, source_part, duration) in enumerate(zip(parts, source_parts, durations)):
            end = segment.end if index == len(parts) - 1 else round(cursor + duration, 3)
            if end <= cursor:
                children = []
                break
            children.append(Segment(cursor, end, source_part, part))
            cursor = end

        if len(children) != len(parts) or "".join(item.translated_text or "" for item in children) != text:
            normalized.append(segment)
            continue
        normalized.extend(children)

    return normalized


def _partition_dense_chinese(text: str, count: int) -> list[str] | None:
    # Prefer punctuation and discourse connectors, as in the exact timing path.
    equal_slots = [Segment(float(index), float(index + 1), "x") for index in range(count)]
    natural = _partition_zh_for_display(text, count, equal_slots)
    if natural and "".join(natural) == text:
        return natural

    # No clean clause boundary exists. Keep every character and select cuts
    # near equal reading units, biasing toward punctuation/space/connector edges.
    parts: list[str] = []
    start = 0
    remaining_width = _display_units(text)
    for part_index in range(count - 1):
        remaining_parts = count - part_index
        target = remaining_width / remaining_parts
        min_cut = start + 1
        max_cut = len(text) - (remaining_parts - 1)
        candidates: list[tuple[float, int]] = []
        for cut in range(min_cut, max_cut + 1):
            left = text[start:cut]
            right = text[cut:]
            left_width = _display_units(left)
            right_width = _display_units(right)
            if left_width < MIN_CJK_CHARS or right_width < MIN_CJK_CHARS:
                continue
            penalty = abs(left_width - target)
            if text[cut - 1] in "，。！？；、,.!?;:：":
                penalty -= 8.0
            elif text[cut - 1].isspace():
                penalty -= 3.0
            if any(right.startswith(connector) for connector in _ZH_CLAUSE_CONNECTORS):
                penalty -= 5.0
            if left[-1] in _ZH_ATTACHED_PREFIXES and not any(
                right.startswith(connector) for connector in _ZH_CLAUSE_CONNECTORS
            ):
                penalty += 4.0
            if right[0] in _ZH_ATTACHED_PREFIXES and not any(
                right.startswith(connector) for connector in _ZH_CLAUSE_CONNECTORS
            ):
                penalty += 2.0
            if _is_weak_zh_part(left) or _zh_ends_incomplete(left):
                penalty += 6.0
            candidates.append((penalty, cut))
        if not candidates:
            return None
        _, cut = min(candidates)
        parts.append(text[start:cut])
        start = cut
        remaining_width = _display_units(text[start:])
    parts.append(text[start:])
    if len(parts) != count or any(not part for part in parts):
        return None
    return parts


def _partition_source_text(source: str, weights: list[float]) -> list[str]:
    """Partition source words proportionally without splitting a Latin token."""
    count = len(weights)
    if count <= 1:
        return [source]
    tokens = list(re.finditer(r"\S+", source))
    if len(tokens) < count:
        return [source, *([""] * (count - 1))]

    total_weight = sum(weights) or float(count)
    total_chars = len(source)
    target_positions: list[float] = []
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        target_positions.append(total_chars * running / total_weight)

    cuts: list[int] = []
    token_start = 0
    for index, target in enumerate(target_positions):
        remaining_chunks = len(target_positions) - index
        candidates = tokens[token_start + 1 : len(tokens) - remaining_chunks]
        if not candidates:
            return [source, *([""] * (count - 1))]
        chosen = min(candidates, key=lambda match: abs(match.end() - target))
        cuts.append(chosen.end())
        token_start = tokens.index(chosen) + 1

    boundaries = [0, *cuts, len(source)]
    return [source[left:right] for left, right in zip(boundaries, boundaries[1:])]


def _allocate_dense_durations(span: float, weights: list[float]) -> list[float]:
    count = len(weights)
    total_weight = sum(weights) or float(count)
    extra = max(0.0, span - count * MIN_ESTIMATED_SLOT_DURATION_SEC)
    return [MIN_ESTIMATED_SLOT_DURATION_SEC + extra * weight / total_weight for weight in weights]


def _coalesce_incomplete_zh_cues(segments: list[Segment]) -> list[Segment]:
    """Join abutting cues when the left Chinese side is clearly unfinished.

    Uses a narrower predicate than in-sentence incomplete detection so natural
    endings like 「我下车了」 do not glue unrelated next cues.
    """
    if len(segments) <= 1:
        return segments
    output = [
        Segment(item.start, item.end, item.text, item.translated_text)
        for item in segments
    ]
    merges: list[dict[str, object]] = []
    index = 0
    while index < len(output) - 1:
        current = output[index]
        following = output[index + 1]
        left = (current.translated_text or "").strip()
        right = (following.translated_text or "").strip()
        gap = following.start - current.end
        if left and right and gap <= 0.45 and _zh_needs_cross_cue_merge(left):
            output[index] = Segment(
                current.start,
                max(current.end, following.end),
                _join_slot_texts([current, following]),
                f"{left}{right}",
            )
            output.pop(index + 1)
            continue
        index += 1
    return output


def _zh_needs_cross_cue_merge(text: str) -> bool:
    """True when the left Chinese cue is unfinished and should join the next cue."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if re.search(r"[。！？]$", compact):
        return False
    if re.search(
        r"(不过|所以|因为|如果|虽然|万一|但是|然后|而且|于是|可是|因此|并且)$",
        compact,
    ):
        return True
    # 「……。我」 / 「……，我」 — pronoun starting the next clause inside this cue.
    if re.search(r"[。！？，、](我|你|他|她|它|我们|你们|他们|它们)$", compact):
        return True
    if re.fullmatch(r"(我|你|他|她|它|我们|你们|他们|它们|为了|To|to)", compact):
        return True
    # Mid-phrase cuts from multi-slot redistribute / EN soft-caps (「我并不|确定」).
    # Intentionally omit 了/着/过 so natural endings like「我下车了」 stay separate.
    if re.search(
        r"(的|地|得|和|与|及|并|把|被|让|给|从|对|向|到|在|是|有|没|不|很|更|最|"
        r"这|那|可|但|而|又|或|就|都|也|还|再|才|只|要|会|能|想|觉得|准备|开始|"
        r"不过|所以|因为|如果|虽然|但是|然后|而且|于是|像|个|们|为)$",
        compact,
    ):
        return True
    return False


def _map_translation_to_slots(
    zh: str,
    slots: list[Segment],
    *,
    timing_words: list[TimedWord] | None = None,
) -> list[Segment]:
    """Attach ZH to slots without cutting inside Chinese words.

    Prefer clause boundaries (，。！？). If a clean split is impossible, keep one
    cue for the whole sentence so text is never orphaned mid-word.
    """
    slot_in = len(slots)
    if slot_in == 1:
        result = [Segment(slots[0].start, slots[0].end, slots[0].text, zh)]
    else:
        parts = _partition_by_clauses(zh, slot_in, slots=slots)
        mode = "punct" if parts is not None else "merge_sentence"
        if parts is None:
            result = [
                Segment(
                    slots[0].start,
                    slots[-1].end,
                    _join_slot_texts(slots),
                    zh,
                )
            ]
        else:
            mapped_slots = slots
            if len(parts) != len(mapped_slots):
                mapped_slots = _merge_slots_to_count(mapped_slots, len(parts))
            result = [
                Segment(slot.start, slot.end, slot.text, part)
                for slot, part in zip(mapped_slots, parts)
                if part.strip()
            ]

    # First keep the existing speech-slot mapping whenever it already fits.
    # Only dense Chinese captions ask the raw word clock for a few more slots.
    if timing_words and _has_cjk(zh) and any(
        _display_units(item.translated_text or "") > MAX_CJK_DISPLAY_UNITS_PER_CUE
        for item in result
    ):
        adaptive = _split_dense_translation(zh, timing_words)
        if adaptive:
            result = adaptive
            mode = "speech_density"
    return result


def _reflow_dense_segments(
    segments: list[Segment], units: list[SentenceUnit]
) -> list[Segment]:
    """Re-plan captions made dense by the incomplete-phrase coalescing pass."""
    timed_words = sorted(
        (word for unit in units for word in unit.timing_words),
        key=lambda word: (word.start, word.end),
    )
    if not timed_words:
        return segments
    output: list[Segment] = []
    for segment in segments:
        translated = (segment.translated_text or "").strip()
        if not translated or not _has_cjk(translated):
            output.append(segment)
            continue
        if _display_units(translated) <= MAX_CJK_DISPLAY_UNITS_PER_CUE:
            output.append(segment)
            continue
        # Word midpoints avoid counting the previous slot's small end padding as
        # part of the next spoken phrase. Verify source coverage before using
        # these atoms so a timing guess can never silently lose source text.
        raw_end = segment.end - END_PAD_SEC * 0.75
        candidates = [
            word for word in timed_words
            if word.start >= segment.start - 0.02
            and (word.start + word.end) / 2 < raw_end + 0.001
        ]
        if _source_signature(_join_word_texts(candidates)) != _source_signature(segment.text):
            output.append(segment)
            continue
        split = _split_dense_translation(translated, candidates)
        output.extend(split or [segment])
    return output


def _split_dense_translation(text: str, timing_words: list[TimedWord]) -> list[Segment] | None:
    """Split only at natural ZH phrase boundaries and safe speech-word edges."""
    width = _display_units(text)
    if width <= MAX_CJK_DISPLAY_UNITS_PER_CUE or not timing_words:
        return None
    span = timing_words[-1].end - timing_words[0].start
    requested = math.ceil(width / MAX_CJK_DISPLAY_UNITS_PER_CUE)
    available = min(
        len(timing_words) // MIN_WORDS_PER_DENSITY_SLOT,
        int(span / MIN_DENSITY_SLOT_DURATION_SEC),
    )
    for count in range(min(requested, available), 1, -1):
        word_groups = _partition_timing_words(timing_words, count)
        if not word_groups:
            continue
        slots = [_cue_from_group(group) for group in word_groups]
        if any(slot is None for slot in slots):
            continue
        valid_slots = [slot for slot in slots if slot is not None]
        parts = _partition_zh_for_display(text, count, valid_slots)
        if not parts or len(parts) != count:
            continue
        if "".join(parts) != text:
            continue
        if any(_display_units(part) > MAX_CJK_DISPLAY_UNITS_PER_CUE for part in parts):
            continue
        return [
            Segment(slot.start, slot.end, slot.text, part)
            for slot, part in zip(valid_slots, parts)
        ]
    return None


def _partition_timing_words(
    words: list[TimedWord], count: int
) -> list[list[TimedWord]] | None:
    """Select a small number of safe, approximately speech-balanced word cuts."""
    if count < 2 or len(words) < count * MIN_WORDS_PER_DENSITY_SLOT:
        return None
    if words[-1].end - words[0].start < count * MIN_DENSITY_SLOT_DURATION_SEC:
        return None
    total_mass = sum(max(0.01, word.end - word.start) for word in words)
    if total_mass <= 0:
        return None
    boundaries: list[int] = []
    start_index = 0
    for part_index in range(count - 1):
        remaining_parts = count - part_index
        minimum = start_index + MIN_WORDS_PER_DENSITY_SLOT
        maximum = len(words) - MIN_WORDS_PER_DENSITY_SLOT * (remaining_parts - 1)
        target_ratio = (part_index + 1) / count
        scored: list[tuple[float, int, float]] = []
        prefix_mass = sum(max(0.01, item.end - item.start) for item in words[:start_index])
        for split_at in range(minimum, maximum + 1):
            left, right = words[split_at - 1], words[split_at]
            if (
                _is_incomplete_tail(left.word)
                or _needs_complement(left.word)
                or is_phrase_glued(left.word, right.word)
            ):
                continue
            gap = max(0.0, right.start - left.end)
            clear_source_boundary = (
                gap >= 0.18
                or re.search(r"[,;:.!?]$", left.word.strip()) is not None
                or _looks_like_sentence_start(right.word)
            )
            if not clear_source_boundary:
                continue
            left_duration = left.end - words[start_index].start
            right_min_duration = words[-1].end - right.start
            if (
                left_duration < MIN_DENSITY_SLOT_DURATION_SEC
                or right_min_duration < (remaining_parts - 1) * MIN_DENSITY_SLOT_DURATION_SEC
            ):
                continue
            mass_ratio = (
                prefix_mass
                + sum(max(0.01, item.end - item.start) for item in words[start_index:split_at])
            ) / total_mass
            score = abs(mass_ratio - target_ratio) * 3.0
            if gap >= 0.25:
                score -= min(gap, 1.0) * 0.2
            if re.search(r"[,;:.!?]$", left.word.strip()):
                score -= 0.5
            if _looks_like_sentence_start(right.word):
                score -= 0.1
            scored.append((score, split_at, mass_ratio))
        if not scored:
            return None
        _, chosen, _ = min(scored)
        boundaries.append(chosen)
        start_index = chosen
    indices = [0, *boundaries, len(words)]
    groups = [words[left:right] for left, right in zip(indices, indices[1:])]
    if any(len(group) < MIN_WORDS_PER_DENSITY_SLOT for group in groups):
        return None
    return groups


def _partition_zh_for_display(
    text: str, count: int, slots: list[Segment]
) -> list[str] | None:
    """Group existing punctuation/conjunction phrases; never cut inside a word."""
    if count < 2:
        return None
    boundaries: set[int] = set()
    for match in re.finditer(r"[，。！？；、,.!?;:：]", text):
        boundaries.add(match.end())
    connector_pattern = re.compile(
        "|".join(
            re.escape(value)
            for value in sorted(_ZH_CLAUSE_CONNECTORS, key=len, reverse=True)
        )
    )
    for match in connector_pattern.finditer(text):
        if match.start() > 0:
            boundaries.add(match.start())
    for match in re.finditer(r"\s+", text):
        boundaries.add(match.end())

    safe: list[int] = []
    for boundary in sorted(boundaries):
        left = text[:boundary]
        right = text[boundary:]
        if _display_units(left) < MIN_CJK_CHARS or _display_units(right) < MIN_CJK_CHARS:
            continue
        left_clean = re.sub(r"\s+", "", left)
        right_clean = re.sub(r"\s+", "", right)
        if not left_clean or not right_clean:
            continue
        if _zh_ends_incomplete(left_clean) and left_clean[-1] not in "，。！？；、,.!?;:：":
            continue
        starts_connector = any(
            right_clean.startswith(connector) for connector in _ZH_CLAUSE_CONNECTORS
        )
        if right_clean[0] in _ZH_ATTACHED_PREFIXES and not starts_connector:
            continue
        safe.append(boundary)
    if len(safe) < count - 1:
        return None

    # Greedily choose cuts closest to source-slot speech mass. Natural clause
    # boundaries are the only candidates, so this stays lossless and word-safe.
    weights = [_slot_weight(slot) for slot in slots]
    total_weight = sum(weights) or float(count)
    target_positions: list[float] = []
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        target_positions.append(len(text) * running / total_weight)
    selected: list[int] = []
    last = 0
    remaining = list(safe)
    for index, target in enumerate(target_positions):
        candidates = []
        for boundary_index, boundary in enumerate(remaining):
            if (
                boundary > last
                and len(remaining) - boundary_index - 1
                >= len(target_positions) - index - 1
            ):
                candidates.append(boundary)
        if not candidates:
            return None
        chosen = min(candidates, key=lambda boundary: abs(boundary - target))
        selected.append(chosen)
        last = chosen
        remaining = [boundary for boundary in remaining if boundary > chosen]
    indices = [0, *selected, len(text)]
    parts = [text[left:right] for left, right in zip(indices, indices[1:])]
    if any(_display_units(part) > MAX_CJK_DISPLAY_UNITS_PER_CUE for part in parts):
        return None
    if any(_is_weak_zh_part(part) or _zh_ends_incomplete(part) for part in parts[:-1]):
        return None
    return parts if "".join(parts) == text else None


def _display_units(text: str) -> float:
    width = 0.0
    for char in text:
        if char.isspace():
            continue
        width += 1.0 if unicodedata.east_asian_width(char) in {"W", "F"} else 0.55
    return width


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def _source_signature(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text or "", flags=re.UNICODE).casefold()


def subtitle_density_overflow_count(segments: list[Segment]) -> int:
    """Count captions that still exceed the two-line Chinese display budget."""
    return sum(
        1
        for segment in segments
        if _has_cjk(segment.translated_text or "")
        and _display_units(segment.translated_text or "") > MAX_CJK_DISPLAY_UNITS_PER_CUE
    )


def subtitle_density_overflow_indices(segments: list[Segment]) -> list[int]:
    """Return zero-based cue indexes that exceed the two-line display budget."""
    return [
        index
        for index, segment in enumerate(segments)
        if _has_cjk(segment.translated_text or "")
        and _display_units(segment.translated_text or "") > MAX_CJK_DISPLAY_UNITS_PER_CUE
    ]


def subtitle_high_reading_speed_count(segments: list[Segment]) -> int:
    """Count captions whose spoken timing exceeds a conservative hard reading rate."""
    return sum(
        1
        for segment in segments
        if _has_cjk(segment.translated_text or "")
        and segment.end > segment.start
        and _display_units(segment.translated_text or "") / (segment.end - segment.start) > 12.0
    )


def _partition_by_clauses(
    zh: str,
    slot_count: int,
    slots: list[Segment] | None = None,
) -> list[str] | None:
    clauses = _split_zh_clauses(zh)
    if len(clauses) <= 1:
        return None
    target_count = min(slot_count, len(clauses))
    if target_count <= 1:
        return None
    if slots is not None and len(slots) == target_count:
        # Bias clause packing toward spoken EN mass so cuts track speech better.
        grouped = _group_clauses_by_slot_weights(
            clauses, [_slot_weight(slot) for slot in slots]
        )
    else:
        grouped = _group_clauses(clauses, target_count)
    if any(_is_weak_zh_part(part) or _zh_ends_incomplete(part) for part in grouped[:-1]):
        return None
    if any(_is_weak_zh_part(part) for part in grouped):
        return None
    if re.sub(r"\s+", "", "".join(grouped)) != re.sub(r"\s+", "", zh):
        return None
    return grouped


def _zh_ends_incomplete(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if compact[-1] in "，。！？；、,.!?":
        return False
    return bool(
        re.search(
            r"(的|地|得|了|着|过|和|与|及|并|把|被|让|给|从|对|向|到|在|是|有|没|不|很|更|最|一|这|那|可|但|而|又|或|就|都|也|还|再|才|只|要|会|能|想|觉得|准备|开始|看起来|像样|不过|所以|因为|如果|虽然|万一|但是|然后|而且|于是|我|你|他|她|它|我们|你们|他们|它们)$",
            compact,
        )
    )


def _split_zh_clauses(zh: str) -> list[str]:
    pieces = re.findall(r"[^，。！？；、,.!?]+[，。！？；、,.!?]?", zh)
    return [piece for piece in pieces if piece.strip()]


def _group_clauses(clauses: list[str], count: int) -> list[str]:
    if count >= len(clauses):
        return list(clauses)
    weights = [max(1, len(re.sub(r"\s+", "", clause))) for clause in clauses]
    total = sum(weights) or 1
    targets = [total * ((i + 1) / count) for i in range(count)]
    groups: list[list[str]] = [[] for _ in range(count)]
    cumulative = 0
    group_index = 0
    for clause, weight in zip(clauses, weights):
        if group_index < count - 1 and cumulative >= targets[group_index] and groups[group_index]:
            group_index += 1
        groups[group_index].append(clause)
        cumulative += weight
    return ["".join(group).strip() for group in groups]


def _group_clauses_by_slot_weights(
    clauses: list[str],
    slot_weights: list[float],
) -> list[str]:
    """Pack ZH clauses into N buckets sized like the spoken English slots."""
    count = len(slot_weights)
    if count <= 1:
        return ["".join(clauses)]
    if count >= len(clauses):
        return list(clauses)
    clause_weights = [max(1, len(re.sub(r"\s+", "", clause))) for clause in clauses]
    zh_total = sum(clause_weights) or 1
    slot_total = sum(max(1.0, weight) for weight in slot_weights) or 1.0
    targets: list[float] = []
    running = 0.0
    for weight in slot_weights:
        running += max(1.0, weight)
        targets.append(zh_total * (running / slot_total))
    groups: list[list[str]] = [[] for _ in range(count)]
    cumulative = 0
    group_index = 0
    for clause, weight in zip(clauses, clause_weights):
        if (
            group_index < count - 1
            and cumulative >= targets[group_index]
            and groups[group_index]
        ):
            group_index += 1
        groups[group_index].append(clause)
        cumulative += weight
    return ["".join(group).strip() for group in groups]


def _merge_slots_to_count(slots: list[Segment], count: int) -> list[Segment]:
    if count >= len(slots):
        return slots
    if count <= 1:
        return [Segment(slots[0].start, slots[-1].end, _join_slot_texts(slots))]
    weights = [_slot_weight(slot) for slot in slots]
    total = sum(weights) or 1.0
    targets = [total * ((i + 1) / count) for i in range(count)]
    groups: list[list[Segment]] = [[] for _ in range(count)]
    cumulative = 0.0
    group_index = 0
    for slot, weight in zip(slots, weights):
        if group_index < count - 1 and cumulative >= targets[group_index] and groups[group_index]:
            group_index += 1
        groups[group_index].append(slot)
        cumulative += weight
    merged: list[Segment] = []
    for group in groups:
        if not group:
            continue
        merged.append(Segment(group[0].start, group[-1].end, _join_slot_texts(group)))
    return merged


def _group_sentence_words(stream: list[TimedWord]) -> list[list[TimedWord]]:
    groups: list[list[TimedWord]] = []
    current: list[TimedWord] = [stream[0]]
    for word in stream[1:]:
        previous = current[-1]
        gap = word.start - previous.end
        duration = previous.end - current[0].start
        word_count = len(current)
        sentence_end = _is_sentence_end(previous.word)
        phrase_glued = is_phrase_glued(previous.word, word.word)
        incomplete = (
            _is_incomplete_tail(previous.word)
            or _needs_complement(previous.word)
            or phrase_glued
        )

        split = False
        split_at: int | None = None
        reason = ""
        if sentence_end and gap >= 0.12:
            split = True
            reason = "punct"
        elif phrase_glued:
            # Keep "all right" / "there he goes" intact across editing pauses.
            split = False
        elif (
            gap >= VERY_LONG_PAUSE_SEC
            and _clean_token(previous.word) not in BARE_ARTICLES
            and not _is_incomplete_tail(previous.word)
            and not _needs_complement(previous.word)
        ):
            # Unpunctuated ASR still has scene cuts; split on long silence.
            # Keep true unfinished tails (been/to/the/don’t/…) attached to the next word.
            split = True
            reason = "very_long_pause"
        elif (
            gap >= LONG_PAUSE_SEC
            and not incomplete
            and word_count >= MIN_STANDALONE_WORDS
            and _looks_like_sentence_start(word.word)
        ):
            split = True
            reason = "pause"
        elif duration >= MAX_SENTENCE_DURATION_SEC or word_count >= MAX_SENTENCE_WORDS:
            # Never cut mid-phrase for a soft/hard cap. Prefer an earlier safe
            # boundary (punct / comma / complete+sentence-start); under emergency
            # duration, fall back to the largest internal pause.
            split_at = _last_safe_sentence_break(current)
            if split_at is not None:
                split = True
                reason = "cap_safe"
            elif duration >= EMERGENCY_SENTENCE_DURATION_SEC:
                split_at = _largest_pause_break(current, min_gap=0.85)
                if split_at is not None:
                    split = True
                    reason = "cap_pause"
                elif duration >= MAX_MEGA_DURATION_SEC:
                    split_at = _largest_pause_break(
                        current, min_gap=EXTREME_PAUSE_SEC, allow_extreme=True
                    )
                    if split_at is not None:
                        split = True
                        reason = "cap_mega"
                    elif not incomplete:
                        split = True
                        reason = "cap_hard"
                elif not incomplete:
                    split = True
                    reason = "cap_hard"

        if split:
            if split_at is not None and 1 <= split_at < len(current):
                groups.append(current[:split_at])
                current = current[split_at:] + [word]
            else:
                groups.append(current)
                current = [word]
        else:
            current.append(word)
    groups.append(current)
    return _coalesce_incomplete_en_groups(groups)


def _coalesce_incomplete_en_groups(groups: list[list[TimedWord]]) -> list[list[TimedWord]]:
    """Join adjacent sentence groups when the left English side is unfinished."""
    if len(groups) <= 1:
        return groups
    output = [list(group) for group in groups if group]
    index = 0
    while index < len(output) - 1:
        left = output[index]
        right = output[index + 1]
        if not left or not right:
            index += 1
            continue
        tail = left[-1].word
        head = right[0].word
        gap = right[0].start - left[-1].end
        if is_phrase_glued(tail, head) and gap <= EXTREME_PAUSE_SEC:
            output[index] = left + right
            output.pop(index + 1)
            continue
        if (
            gap <= 0.85
            and not _is_sentence_end(tail)
            and (
                _is_incomplete_tail(tail)
                or _needs_complement(tail)
                or (
                    not _looks_like_sentence_start(head)
                    and _clean_token(head) not in {"i", "okay", "ok"}
                    and (head or "").lstrip()[:1].islower()
                )
            )
        ):
            output[index] = left + right
            output.pop(index + 1)
            continue
        index += 1
    return output


def _needs_complement(token: str) -> bool:
    clean = _clean_token(token)
    if clean in _NEEDS_COMPLEMENT:
        return True
    # "we're" / "I'm" clean to were/im; only treat as unfinished when apostrophe present.
    raw = (token or "").strip().lower()
    if "'" in raw or "’" in raw:
        return clean in {"were", "im", "youre", "theyre", "hes", "shes", "ill", "weve", "youve", "theyve"}
    return False


def _last_safe_sentence_break(group: list[TimedWord]) -> int | None:
    """Return index after which a soft-cap split is safe (exclusive end of left group)."""
    min_left = max(3, len(group) // 5)
    for index in range(len(group) - 1, min_left - 1, -1):
        if _is_sentence_end(group[index - 1].word):
            return index
    for index in range(len(group) - 1, min_left - 1, -1):
        left = group[index - 1].word
        right = group[index].word
        if is_phrase_glued(left, right):
            continue
        if re.search(r"[,;:]$", (left or "").strip()) and not (
            _is_incomplete_tail(left) or _needs_complement(left)
        ):
            return index
    best_index: int | None = None
    best_gap = -1.0
    for index in range(min_left, len(group)):
        left = group[index - 1].word
        right = group[index].word
        if is_phrase_glued(left, right):
            continue
        if _is_incomplete_tail(left) or _needs_complement(left):
            continue
        if not _looks_like_sentence_start(right):
            continue
        gap = group[index].start - group[index - 1].end
        if gap >= 0.32 and gap >= best_gap:
            best_gap = gap
            best_index = index
    return best_index


def _largest_pause_break(
    group: list[TimedWord],
    *,
    min_gap: float,
    allow_extreme: bool = False,
) -> int | None:
    """Emergency split at the largest internal pause that is not mid-phrase."""
    if len(group) < MIN_STANDALONE_WORDS * 2:
        return None
    min_left = MIN_STANDALONE_WORDS
    max_index = len(group) - MIN_STANDALONE_WORDS
    best_index: int | None = None
    best_gap = min_gap - 0.001
    for index in range(min_left, max_index + 1):
        left = group[index - 1].word
        right = group[index].word
        gap = group[index].start - group[index - 1].end
        glued = is_phrase_glued(left, right)
        unfinished = (
            _is_incomplete_tail(left)
            or _needs_complement(left)
            or _clean_token(left) in BARE_ARTICLES
        )
        if glued or unfinished:
            if not allow_extreme or gap < EXTREME_PAUSE_SEC:
                continue
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return best_index


def _looks_like_sentence_start(token: str) -> bool:
    clean = _clean_token(token)
    if not clean:
        return False
    stripped = (token or "").lstrip()
    if stripped[:1].isupper() and clean not in {"i"}:
        return True
    return clean in {"so", "then", "but", "and", "okay", "ok", "well", "now", "all"}


def _merge_fragile_slots(slots: list[Segment]) -> list[Segment]:
    if len(slots) <= 1:
        return slots
    output = [
        Segment(float(item.start), float(item.end), str(item.text or "").strip())
        for item in slots
        if str(item.text or "").strip()
    ]
    index = 0
    while index < len(output) - 1:
        current = output[index]
        following = output[index + 1]
        left_words = re.findall(r"[A-Za-zÀ-ÿ']+", current.text or "")
        right_words = re.findall(r"[A-Za-zÀ-ÿ']+", following.text or "")
        phrase_cut = bool(
            left_words
            and right_words
            and is_phrase_glued(left_words[-1], right_words[0])
        )
        thin_lead = _is_thin_discourse_slot(current)
        if _is_fragile_slot(current) or phrase_cut or thin_lead:
            output[index] = Segment(
                current.start,
                max(current.end, following.end),
                _join_slot_texts([current, following]),
            )
            output.pop(index + 1)
            continue
        index += 1
    # Merge trailing fragile into previous.
    if len(output) >= 2 and _is_fragile_slot(output[-1]):
        previous = output[-2]
        last = output[-1]
        output[-2] = Segment(previous.start, max(previous.end, last.end), _join_slot_texts([previous, last]))
        output.pop()
    return output


def _is_fragile_slot(slot: Segment) -> bool:
    text = (slot.text or "").strip()
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    duration = max(0.0, slot.end - slot.start)
    if not words:
        return True
    if len(words) <= 2 and duration <= MIN_STANDALONE_DURATION:
        return True
    if len(words) == 1 and _is_incomplete_tail(words[0]):
        return True
    if words and _is_incomplete_tail(words[-1]) and len(words) <= 3:
        return True
    return False


def _is_thin_discourse_slot(slot: Segment) -> bool:
    """True for filler openers that must ride with the following cue.

    Editing pauses can stretch ``all right`` across many seconds; duration-based
    fragile checks miss those, and ZH redistribution then dumps a whole clause
    onto the filler alone.
    """
    normalized = re.sub(r"\s+", " ", (slot.text or "").strip().lower())
    normalized = re.sub(r"[,.!?，。！？]+$", "", normalized).strip()
    if normalized in {
        "all right",
        "alright",
        "okay",
        "ok",
        "oh",
        "oh yeah",
        "oh yes",
        "well",
        "so",
        "anyway",
        "yeah",
        "yes",
    }:
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ']+", normalized)
    return len(words) <= 2 and " ".join(words) in {
        "all right",
        "oh yeah",
        "oh yes",
    }


def _join_slot_texts(slots: list[Segment]) -> str:
    pieces: list[str] = []
    for slot in slots:
        raw = (slot.text or "").strip()
        if not raw:
            continue
        if not pieces:
            pieces.append(raw)
            continue
        if raw[:1] in ",.;:!?":
            pieces.append(raw)
        else:
            pieces.append(" " + raw)
    return re.sub(r"\s+([,.;:!?])", r"\1", "".join(pieces)).strip()


def _is_weak_zh_part(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if len(compact) < MIN_CJK_CHARS and not re.search(r"[。！？]", compact):
        return True
    if compact in {"为了", "结果", "还有", "然后", "所以", "但是", "因为", "如果", "好", "好，", "在", "并", "及"}:
        return True
    return False


def _slot_weight(slot: Segment) -> float:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9']+", slot.text or "")
    chars = sum(len(word) for word in words)
    return float(max(1, chars))

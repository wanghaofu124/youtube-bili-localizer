"""Pack subtitle cues from a continuous word clock (speech-timed cuts)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
import json
import re

from .models import Segment


# Cut on real pauses / sentence ends. Soft duration only after a readable boundary.
SPEECH_PAUSE_SEC = 0.50
# A grammatical fragment must not keep a subtitle visible across a scene cut.
# This is an absolute boundary, including articles and glued phrases.
MAX_INTERNAL_SILENCE_SEC = 2.0
SOFT_DURATION_SEC = 2.8
HARD_DURATION_SEC = 4.2
HARD_INCOMPLETE_EXTRA_SEC = 0.5
# An unfinished function word cannot form a readable cue after a mandatory
# speech gap. A content-bearing modifier is withheld only at a much longer
# edit, where carrying it forward would be especially misleading.
CLIPPED_FUNCTION_GAP_SEC = MAX_INTERNAL_SILENCE_SEC
CLIPPED_MODIFIER_GAP_SEC = 8.0
SOFT_CHARS = 42
END_PAD_SEC = 0.12
MIN_SOFT_GAP_SEC = 0.18
MAX_TRANSLATION_UNIT_DURATION_SEC = 8.0
MAX_TRANSLATION_UNIT_HARD_SEC = 10.0
MAX_TRANSLATION_UNIT_GAP_SEC = 0.55
_CLAUSE_OPENERS = frozenset({"and", "but", "so", "then", "because", "however", "while", "pero", "y"})
_SPOKEN_CLAUSE_STARTERS = frozenset({
    "we", "well", "were", "i", "im", "you", "they", "he", "she", "it",
    "this", "that", "these", "those", "there", "here", "lets", "guys",
    "to", "catch", "rank", "and", "but", "so", "then", "because", "however", "when", "if",
})
_INDEPENDENT_CLAUSE_STARTERS = frozenset({
    "i", "we", "you", "he", "she", "they", "it", "this", "that",
    "there", "here", "lets", "let", "look", "now", "guys",
})

# Keep common spoken phrases on one cue even across editing pauses.
_PHRASE_CONTINUATIONS = {
    "all": frozenset({"right"}),
    "bunch": frozenset({"of"}),
    "there": frozenset({"goes", "he", "she", "it"}),
    "here": frozenset({"goes", "he", "she", "it", "comes", "is"}),
    "right": frozenset({"here", "there"}),
    "underneath": frozenset({"here", "there"}),
    "out": frozenset({"here", "there"}),
    "where": frozenset({"we", "you", "they", "he", "she", "it"}),
    "show": frozenset({"me", "you", "us", "them", "him", "her"}),
    "look": frozenset({"at", "up"}),
    "starting": frozenset({"this", "that", "the", "a", "an"}),
    "bit": frozenset({"later"}),
    "later": frozenset({"on"}),
    "he": frozenset({"goes"}),
    "she": frozenset({"goes"}),
    "it": frozenset({"goes"}),
    "bye": frozenset({"buddy"}),
    "oh": frozenset({"yeah", "yes", "ok", "okay"}),
    "ah": frozenset({"yeah", "yes", "ok", "okay"}),
    "lets": frozenset({"go", "see", "do", "get"}),
    "heaps": frozenset({"of"}),
    "start": frozenset({"to"}),
    "need": frozenset({"to"}),
    "happens": frozenset({"when"}),
    "like": frozenset({"that"}),
    # Contractions / auxiliaries that must keep their complement on the same cue.
    "dont": frozenset({"know", "think", "want", "like", "have", "need", "get", "go"}),
    "doesnt": frozenset({"know", "think", "want", "like", "have", "need", "get", "go"}),
    "didnt": frozenset({"know", "think", "want", "like", "have", "need", "get", "go"}),
    "cant": frozenset({"know", "see", "tell", "wait", "believe", "help", "get", "go"}),
    "wont": frozenset({"know", "work", "fit", "hold", "get", "go"}),
}

# Word timestamps are not grammatical boundaries. In uninterrupted speech,
# keep likely English modifiers with the following head word even when the
# display-duration soft cap has been reached.
_MODIFIER_WORDS = frozenset({
    "sea", "coastal", "predatory", "beautiful", "dangerous", "giant",
    "bright", "crazy",
    "little", "small", "big", "wild", "painted", "spanish", "rock",
    "clown", "sweet", "job", "red", "yellow", "green", "blue", "purple",
    "orange", "black", "white", "brown", "gray", "grey",
})
_MODIFIER_SUFFIXES = ("ary", "ory", "ous", "ful", "less", "ive", "ical", "ic")
_NON_HEAD_WORDS = frozenset({
    "and", "but", "or", "so", "then", "because", "if", "when", "that",
    "this", "these", "those", "there", "here",
    "we", "i", "you", "they", "he", "she", "it", "is", "are", "was",
    "a", "an", "the", "to", "of", "for", "from", "with", "in", "on",
})
_SUBJECT_STARTERS = frozenset({"we", "i", "you", "they", "he", "she"})
_LONG_GAP_DANGLING_WORDS = frozenset({
    "a", "an", "the", "to", "of", "for", "and", "or", "but", "with",
    "in", "on", "at", "from", "into", "by", "as", "if", "when",
    "than", "so", "my", "your", "his", "her", "their", "our",
    "is", "are", "was", "were", "be", "have", "has", "had",
    "will", "would", "could", "should", "can", "may",
    "dont", "doesnt", "didnt", "cant", "wont", "lets", "all",
    "weve", "youve", "theyve", "im", "ive", "youre", "hes", "shes", "its",
})

_INCOMPLETE_TAILS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "for",
        "and",
        "or",
        "but",
        "with",
        "in",
        "on",
        "at",
        "from",
        "into",
        "onto",
        "by",
        "as",
        "if",
        "when",
        "while",
        "than",
        "so",
        "because",
        "since",
        "although",
        "though",
        "just",
        "very",
        "most",
        "more",
        "too",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
        "this",
        "that",
        "these",
        "those",
        "few",
        "many",
        "some",
        "any",
        "all",
        "each",
        "every",
        "me",
        "him",
        "us",
        "them",
        "i",
        "we",
        "you",
        "he",
        "she",
        "it",
        "they",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "de",
        "del",
        "y",
        "o",
        "pero",
        "en",
        "al",
        "lo",
        "le",
        "les",
        "mi",
        "su",
        "que",
        "se",
        "me",
        "te",
        "nos",
        # Apostrophes are stripped by _clean_token ("don't" → "dont").
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "wont",
        "youre",
        "theyre",
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
        "where",
        "starting",
        "show",
        "underneath",
    }
)


@dataclass(slots=True)
class TimedWord:
    start: float
    end: float
    word: str


@dataclass(slots=True)
class SpeechFragment:
    start: float
    end: float
    text: str
    reason: str


@dataclass(slots=True)
class PackedSpeech:
    cues: list[Segment]
    detached_fragments: list[SpeechFragment]


def words_path_for_segments(segments_json: Path) -> Path:
    name = segments_json.name
    if name.startswith("segments."):
        return segments_json.with_name("words." + name[len("segments.") :])
    return segments_json.with_name(f"{segments_json.stem}.words.json")


def save_words(path: Path, words: list[TimedWord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps([asdict(word) for word in words], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_words(path: Path) -> list[TimedWord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    words: list[TimedWord] = []
    for item in raw:
        text = str(item.get("word", "") or "")
        if not text.strip():
            continue
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            continue
        words.append(TimedWord(start=start, end=end, word=text))
    return normalize_word_stream(words)


def timed_words_from_whisper_item(item: object) -> list[TimedWord]:
    words: list[TimedWord] = []
    for entry in getattr(item, "words", None) or []:
        if getattr(entry, "start", None) is None or getattr(entry, "end", None) is None:
            continue
        # Keep Whisper's leading spaces; they are the join glue for Latin text.
        text = str(getattr(entry, "word", "") or "")
        if not text.strip():
            continue
        start = float(entry.start)
        end = float(entry.end)
        if end <= start:
            continue
        words.append(TimedWord(start=start, end=end, word=text))
    return words


def normalize_word_stream(words: list[TimedWord]) -> list[TimedWord]:
    """Sort, drop exact duplicates, and keep a stable left-to-right clock."""
    ordered = sorted(words, key=lambda item: (item.start, item.end, item.word.lower()))
    output: list[TimedWord] = []
    for word in ordered:
        if output:
            previous = output[-1]
            same_text = _clean_token(previous.word) == _clean_token(word.word)
            overlap = word.start < previous.end - 0.02
            near_dup = same_text and abs(word.start - previous.start) <= 0.12
            if near_dup or (same_text and overlap):
                if word.end > previous.end:
                    output[-1] = TimedWord(previous.start, word.end, previous.word)
                continue
            if word.start < previous.end:
                # Keep text; nudge start so the stream stays monotonic.
                word = TimedWord(previous.end, max(previous.end + 0.01, word.end), word.word)
        output.append(word)
    return output


def pack_cues_from_words(words: list[TimedWord]) -> list[Segment]:
    """Compatibility entry point for the renderable part of the word clock."""
    return pack_cues_from_words_detailed(words).cues


def pack_cues_from_words_detailed(words: list[TimedWord]) -> PackedSpeech:
    """Keep word timing while quarantining clipped speech at pause boundaries."""
    stream = normalize_word_stream(words)
    if not stream:
        return PackedSpeech([], [])

    groups: list[list[TimedWord]] = []
    detached_fragments: list[SpeechFragment] = []
    current: list[TimedWord] = [stream[0]]
    for word in stream[1:]:
        previous = current[-1]
        gap = word.start - previous.end
        duration = previous.end - _effective_group_start(current)
        chars = _compact_len(current)
        sentence_end = _is_sentence_end(previous.word)
        phrase_glued = is_phrase_glued(previous.word, word.word)
        modifier_head = _is_modifier_head_edge(previous.word, word.word)
        incomplete = _is_incomplete_tail(previous.word) or phrase_glued or modifier_head
        clause_break = (
            duration >= 2.4
            and chars >= 42
            and not incomplete
            and (
                bool(re.search(r"[,;:，；：]$", previous.word.strip()))
                or _clean_token(word.word) in _CLAUSE_OPENERS
            )
        )
        independent_clause = (
            duration >= 2.25
            and len(current) >= 5
            and not incomplete
            and _clean_token(word.word) in _INDEPENDENT_CLAUSE_STARTERS
        )
        must_split = sentence_end or gap >= MAX_INTERNAL_SILENCE_SEC or clause_break or independent_clause or (
            gap >= SPEECH_PAUSE_SEC and not incomplete
        )
        if incomplete and not sentence_end and not clause_break and gap < MAX_INTERNAL_SILENCE_SEC:
            must_split = False
        soft_ok = (not incomplete) and (
            (duration >= SOFT_DURATION_SEC and gap >= MIN_SOFT_GAP_SEC)
            or (duration >= SOFT_DURATION_SEC * 0.85 and chars >= SOFT_CHARS and gap >= 0.12)
            or duration >= HARD_DURATION_SEC
        )
        if incomplete and duration >= HARD_DURATION_SEC:
            soft_ok = gap >= 0.25 or duration >= HARD_DURATION_SEC + HARD_INCOMPLETE_EXTRA_SEC
        if (phrase_glued or modifier_head) and not sentence_end and gap < MAX_INTERNAL_SILENCE_SEC:
            must_split = False
            soft_ok = False

        if must_split or soft_ok:
            shift_clause_onset = independent_clause and gap < MIN_SOFT_GAP_SEC
            prior_cut = _preferred_prior_cut(current) if not sentence_end and (
                shift_clause_onset or (
                    soft_ok and not must_split and gap < MIN_SOFT_GAP_SEC
                    and _clean_token(word.word) not in _SPOKEN_CLAUSE_STARTERS
                )
            ) else None
            if prior_cut is not None:
                groups.append(current[:prior_cut])
                current = current[prior_cut:] + [word]
            else:
                _append_group_at_boundary(groups, detached_fragments, current, word, gap)
                current = [word]
        else:
            current.append(word)
    groups.append(current)
    groups = _attach_orphan_articles(groups)
    cues: list[Segment] = []
    for index, group in enumerate(groups):
        cue = _cue_from_group(group)
        if cue is None:
            continue
        if index + 1 < len(groups):
            cue.end = min(cue.end, round(groups[index + 1][0].start, 3))
        cues.append(cue)
    return PackedSpeech(cues, detached_fragments)


def _append_group_at_boundary(
    groups: list[list[TimedWord]],
    detached_fragments: list[SpeechFragment],
    group: list[TimedWord],
    following: TimedWord,
    gap: float,
) -> None:
    if gap >= CLIPPED_FUNCTION_GAP_SEC:
        tail_count = _clipped_tail_word_count(group, following, gap)
        if tail_count:
            kept, detached = group[:-tail_count], group[-tail_count:]
            if kept:
                groups.append(kept)
            detached_fragments.append(SpeechFragment(
                start=round(detached[0].start, 3),
                end=round(detached[-1].end, 3),
                text=_join_word_texts(detached),
                reason="停顿前的不完整词句，未放入成片字幕",
            ))
            return
    groups.append(group)


def _clipped_tail_word_count(group: list[TimedWord], following: TimedWord, gap: float) -> int:
    """Detach only clear one/two-word debris, never a substantial utterance."""
    if not group or _is_sentence_end(group[-1].word):
        return 0
    tail = group[-1].word
    if len(group) >= 2 and (
        _clean_token(group[-2].word) in {"weve", "youve", "theyve", "ive"}
        and _clean_token(tail) in {"got", "been", "had"}
    ):
        return 2
    if _is_incomplete_tail(tail) or (
        gap >= CLIPPED_MODIFIER_GAP_SEC
        and _is_modifier_head_edge(tail, following.word)
    ):
        return 1
    return 0


def _preferred_prior_cut(group: list[TimedWord]) -> int | None:
    """Move an imminent duration cut back to a nearby clause onset."""
    candidates: list[tuple[float, int]] = []
    for index in range(2, len(group) - 1):
        previous, following = group[index - 1], group[index]
        left_duration = previous.end - group[0].start
        right_duration = group[-1].end - following.start
        if left_duration < 2.0 or right_duration < 0.3:
            continue
        if (
            _is_incomplete_tail(previous.word)
            and not _is_complete_object_before_new_subject(previous.word, following.word)
        ) or is_phrase_glued(previous.word, following.word) or _is_modifier_head_edge(previous.word, following.word):
            continue
        starter = _clean_token(following.word)
        if starter not in _SPOKEN_CLAUSE_STARTERS:
            continue
        score = -abs(left_duration - SOFT_DURATION_SEC)
        score += min(0.5, max(0.0, following.start - previous.end))
        candidates.append((score, index))
    return max(candidates)[1] if candidates else None


def _effective_group_start(group: list[TimedWord]) -> float:
    if len(group) >= 2:
        clean = _clean_token(group[0].word)
        first_gap = group[1].start - group[0].end
        if clean in {"a", "an", "the", "el", "la", "un", "una"} and first_gap >= 0.8:
            return group[1].start
    return group[0].start


def _attach_orphan_articles(groups: list[list[TimedWord]]) -> list[list[TimedWord]]:
    index = 0
    while index + 1 < len(groups):
        group = groups[index]
        clean = _clean_token(group[0].word) if len(group) == 1 else ""
        gap = groups[index + 1][0].start - group[-1].end
        if clean in {"a", "an", "the", "el", "la", "un", "una"} and gap < MAX_INTERNAL_SILENCE_SEC:
            groups[index + 1] = group + groups[index + 1]
            groups.pop(index)
            continue
        index += 1
    return groups


def _cue_from_group(group: list[TimedWord]) -> Segment | None:
    if not group:
        return None
    start = group[0].start
    end = group[-1].end + END_PAD_SEC
    cue_text = _join_word_texts(group)
    if not cue_text or end <= start:
        return None
    return Segment(round(start, 3), round(end, 3), cue_text)


def _join_word_texts(group: list[TimedWord]) -> str:
    pieces: list[str] = []
    for word in group:
        raw = word.word
        if not pieces:
            pieces.append(raw)
            continue
        if raw[:1].isspace() or (raw[:1] in ",.;:!?"):
            pieces.append(raw)
        else:
            pieces.append(" " + raw)
    return re.sub(r"\s+([,.;:!?])", r"\1", "".join(pieces)).strip()


def _clean_token(text: str) -> str:
    return re.sub(r"[^a-zà-ÿ]", "", (text or "").lower())


def _is_sentence_end(text: str) -> bool:
    stripped = (text or "").strip()
    if re.search(r"\b(?:mr|mrs|ms|dr|prof|st|vs|etc)\.[\"'”’)]?$", stripped, flags=re.I):
        return False
    if re.search(r"\b(?:[A-Z]\.){2,}[\"'”’)]?$", stripped):
        return False
    return bool(re.search(r"[.!?。！？][\"'”’)]?$", stripped))


def _is_incomplete_tail(text: str) -> bool:
    return _clean_token(text) in _INCOMPLETE_TAILS


def _is_modifier_head_edge(left: str, right: str) -> bool:
    """Protect likely modifier + noun pairs from a duration-only cue cut."""
    if re.search(r"[.!?。！？,;:]$", (left or "").strip()):
        return False
    modifier, head = _clean_token(left), _clean_token(right)
    if not head or head in _NON_HEAD_WORDS:
        return False
    return modifier in _MODIFIER_WORDS or (
        len(modifier) >= 6 and modifier.endswith(_MODIFIER_SUFFIXES)
    )


def _is_complete_object_before_new_subject(left: str, right: str) -> bool:
    """A pronoun such as 'it' can finish one clause before 'we' starts another."""
    return (
        _clean_token(left) in {"it", "him", "her", "them", "us", "me", "you"}
        and _clean_token(right) in _SUBJECT_STARTERS
    )


def long_gap_fragment_warnings(cues: list[Segment]) -> list[str]:
    """Flag unfinished phrases that cannot safely bridge an edited silence."""
    warnings: list[str] = []
    for left, right in zip(cues, cues[1:]):
        if right.start - left.end < MAX_INTERNAL_SILENCE_SEC:
            continue
        left_word = left.text.split()[-1] if left.text.split() else ""
        right_word = right.text.split()[0] if right.text.split() else ""
        if _is_sentence_end(left_word):
            continue
        dangling = _clean_token(left_word) in _LONG_GAP_DANGLING_WORDS or (
            len(left.text.split()) == 1
            and _clean_token(left_word) in {"i", "we", "you", "he", "she", "it", "they"}
        )
        if dangling or _is_modifier_head_edge(left_word, right_word):
            warnings.append(
                f"{left.start:.1f}s “{left.text[-35:]}” → "
                f"{right.start:.1f}s “{right.text[:35]}”"
            )
    return warnings


def split_corrected_punctuation_cues(
    original: list[Segment], corrected: list[Segment], words: list[TimedWord],
    *, log: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Anchor restored sentence and clause punctuation to timed words.

    An edited word with no reliable clock match is left in its original cue; a
    guessed timestamp would be worse than a reviewable multi-sentence cue.
    """
    if len(original) != len(corrected):
        raise ValueError("源文标点校对返回的字幕数量与语音时间片不一致。")
    output: list[Segment] = []
    unresolved = 0
    for source, edited in zip(original, corrected):
        text = edited.text.strip()
        marks = [
            match for match in re.finditer(r"[.!?。！？,;:，；：][\"'”’)]*(?=\s+\S|[\u3400-\u9fff])", text)
            if match.group()[0] not in ".。" or not _is_abbreviation_before_stop(text, match.start())
        ]
        if not marks:
            output.append(edited)
            continue
        clock = [
            word for word in words
            if word.start >= source.start - 0.02 and word.end <= source.end + 0.02
        ]
        tokens = list(re.finditer(r"[^\W_]+(?:['’\-][^\W_]+)*", text, flags=re.UNICODE))
        source_keys = [_alignment_key(word.word) for word in clock]
        edited_keys = [_alignment_key(token.group()) for token in tokens]
        alignment: dict[int, int] = {}
        if clock and tokens:
            matcher = SequenceMatcher(None, source_keys, edited_keys, autojunk=False)
            for operation, source_start, source_end, edited_start, edited_end in matcher.get_opcodes():
                if operation == "equal":
                    for offset in range(source_end - source_start):
                        alignment[edited_start + offset] = source_start + offset
                elif operation == "replace" and source_end - source_start == edited_end - edited_start:
                    for source_index, edited_index in zip(
                        range(source_start, source_end), range(edited_start, edited_end)
                    ):
                        if SequenceMatcher(None, source_keys[source_index], edited_keys[edited_index]).ratio() >= 0.65:
                            alignment[edited_index] = source_index
                elif operation == "replace" and source_end - source_start == 2 and edited_end - edited_start == 1:
                    joined = "".join(source_keys[source_start:source_end])
                    if joined and SequenceMatcher(None, joined, edited_keys[edited_start]).ratio() >= 0.85:
                        alignment[edited_start] = source_end - 1
        cut_points: list[tuple[int, int]] = []
        for mark in marks:
            preceding = [index for index, token in enumerate(tokens) if token.end() <= mark.start()]
            token_index = preceding[-1] if preceding else -1
            clock_index = alignment.get(token_index)
            if clock_index is None or clock_index + 1 >= len(clock):
                unresolved += 1
                continue
            if cut_points and (mark.end() <= cut_points[-1][0] or clock_index + 1 <= cut_points[-1][1]):
                unresolved += 1
                continue
            cut_points.append((mark.end(), clock_index + 1))
        if not cut_points:
            output.append(edited)
            continue
        text_offsets = [0, *(position for position, _ in cut_points), len(text)]
        clock_offsets = [0, *(index for _, index in cut_points), len(clock)]
        pieces: list[Segment] = []
        for part in range(len(text_offsets) - 1):
            chunk = text[text_offsets[part]:text_offsets[part + 1]].strip()
            timed = clock[clock_offsets[part]:clock_offsets[part + 1]]
            if not chunk or not timed:
                break
            start = source.start if part == 0 else timed[0].start
            end = source.end if part == len(text_offsets) - 2 else min(
                timed[-1].end + END_PAD_SEC, clock[clock_offsets[part + 1]].start,
            )
            if end <= start or (pieces and pieces[-1].end > start + 0.001):
                break
            pieces.append(Segment(round(start, 3), round(end, 3), chunk))
        if (len(pieces) == len(text_offsets) - 1
                and re.sub(r"\s+", "", "".join(item.text for item in pieces)) == re.sub(r"\s+", "", text)):
            output.extend(pieces)
        else:
            unresolved += 1
            output.append(edited)
    if unresolved and log:
        log(f"字幕标点提醒：{unresolved} 处新增标点无法可靠对应词级时间，已保留原时间片供复核。")
    return output


def _alignment_key(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def _is_abbreviation_before_stop(text: str, position: int) -> bool:
    match = re.search(r"([A-Za-z]+)$", text[:position])
    return bool(match and match.group(1).casefold() in {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc"})


def group_speech_translation_units(
    cues: list[Segment], *, log: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Respect completed sentences, then choose safe fallback cuts by speech time."""
    units: list[Segment] = []
    group: list[Segment] = []

    def append_group(items: list[Segment]) -> None:
        text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(item.text.strip() for item in items)).strip()
        if text:
            units.append(Segment(items[0].start, items[-1].end, text))

    for cue in cues:
        if not group:
            group = [cue]
            continue

        previous = group[-1]
        gap = cue.start - previous.end
        left_word = previous.text.split()[-1] if previous.text.split() else ""
        right_word = cue.text.split()[0] if cue.text.split() else ""
        crosses_edit = gap >= MAX_INTERNAL_SILENCE_SEC and not (
            gap <= 3.0 and (
                _is_incomplete_tail(left_word)
                or is_phrase_glued(left_word, right_word)
                or _is_modifier_head_edge(left_word, right_word)
            )
        )
        if _is_sentence_end(previous.text) or crosses_edit:
            append_group(group)
            group = [cue]
        else:
            group.append(cue)
            while len(group) > 1 and group[-1].end - group[0].start > MAX_TRANSLATION_UNIT_DURATION_SEC:
                cut = _best_translation_unit_cut(group, allow_unsafe=False)
                if cut is None and group[-1].end - group[0].start <= MAX_TRANSLATION_UNIT_HARD_SEC:
                    break
                if cut is None:
                    cut = _best_translation_unit_cut(group, allow_unsafe=True)
                    if cut is not None and log:
                        log(
                            f"字幕切句提醒：{group[cut - 1].end:.1f}s 附近缺少可靠标点或完整短语边界；"
                            "为避免超长字幕，已按最接近的语音边界切分，请人工复核。"
                        )
                if cut is None:
                    break
                append_group(group[:cut])
                group = group[cut:]

    if group:
        append_group(group)
    return units


def _best_translation_unit_cut(cues: list[Segment], *, allow_unsafe: bool) -> int | None:
    """Prefer punctuation and complete phrases over a blind duration boundary."""
    candidates: list[tuple[float, int]] = []
    for index in range(1, len(cues)):
        left, right = cues[index - 1], cues[index]
        left_word = left.text.split()[-1] if left.text.split() else ""
        right_word = right.text.split()[0] if right.text.split() else ""
        gap = max(0.0, right.start - left.end)
        unsafe = (
            _is_incomplete_tail(left_word)
            or is_phrase_glued(left_word, right_word)
            or _is_modifier_head_edge(left_word, right_word)
            or (
                gap < 0.12
                and _clean_token(right_word) in {"i", "we", "you", "he", "she", "they"}
                and not re.search(r"[,;:，；：]$", left_word.strip())
            )
        )
        if unsafe and not allow_unsafe:
            continue
        left_duration = left.end - cues[0].start
        score = -abs(left_duration - MAX_TRANSLATION_UNIT_DURATION_SEC) * 0.5
        if re.search(r"[,;:，；：]$", left_word.strip()):
            score += 3.0
        if gap >= MAX_TRANSLATION_UNIT_GAP_SEC:
            score += 3.0
        elif gap >= MIN_SOFT_GAP_SEC:
            score += 1.0
        if (gap >= MIN_SOFT_GAP_SEC
                and _clean_token(right_word) in _INDEPENDENT_CLAUSE_STARTERS):
            score += 2.0
        if left_duration < 1.0:
            score -= 2.0
        if unsafe:
            score -= 10.0
        candidates.append((score, index))
    return max(candidates)[1] if candidates else None


def is_phrase_glued(left: str, right: str) -> bool:
    """True when left+right form a spoken collocation that must stay together."""
    previous, following = _clean_token(left), _clean_token(right)
    if following == "ever" and not _is_sentence_end(left):
        return True
    if previous.endswith("ing") and following in {
        "it", "them", "him", "her", "us", "me", "this", "that", "these", "those", "the", "a", "an",
    }:
        return True
    if previous in {"catching", "watching", "using", "building", "making", "getting", "taking", "finding", "eating"} and following not in _NON_HEAD_WORDS:
        return True
    if previous.endswith("ed") and following in {"the", "a", "an", "this", "that", "these", "those"}:
        return True
    if previous in {"eat", "eats", "ate", "try", "tries", "tried", "got", "see", "saw", "mean"} and following in {
        "it", "them", "him", "her", "us", "me", "you", "this", "that", "these", "those", "the", "a", "an", "here",
    }:
        return True
    if previous == "not" and not _is_sentence_end(left):
        return True
    if following in {"that", "slash"} and not _is_sentence_end(left):
        return True
    if previous == "what" and following in {"i", "im", "we", "you", "he", "she", "they", "it"}:
        return True
    if previous == "theres" and following in {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    }:
        return True
    if previous in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"} or previous.isdigit():
        if following in {"inch", "inches", "foot", "feet", "cm", "mm", "meter", "meters", "kg", "gram", "grams"}:
            return True
    if "-" in left.strip() and following and following not in _NON_HEAD_WORDS and not _is_sentence_end(left):
        return True
    return following in _PHRASE_CONTINUATIONS.get(previous, frozenset())


def _compact_len(group: list[TimedWord]) -> int:
    return len(re.sub(r"\s+", "", "".join(word.word for word in group)))

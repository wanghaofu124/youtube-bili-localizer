"""Shared source-cue continuation rules for extract and translate."""

from __future__ import annotations

import re

from .models import Segment


CONTINUATION_TAILS = {
    # Spanish
    "a", "al", "con", "como", "de", "del", "el", "en", "entre", "la",
    "las", "lo", "los", "para", "por", "que", "sin", "sobre", "un", "una",
    "y", "o",
    # English articles / prepositions / conjunctions
    "an", "and", "at", "but", "for", "from", "in", "into", "of", "on",
    "or", "the", "to", "with", "about", "after", "before", "by", "over",
    "under", "up", "down", "out", "off", "as", "than", "via", "per",
    # Determiners / quantifiers / possessives / pronouns that need a complement
    "a", "this", "that", "these", "those", "some", "any", "many", "much",
    "few", "several", "more", "most", "less", "least", "each", "every",
    "both", "all", "half", "such", "no", "other", "another",
    "my", "your", "his", "her", "its", "our", "their", "whose",
    "me", "him", "us", "them",
    # Light verbs / auxiliaries often left hanging mid-phrase
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "get", "got", "give", "gave", "put", "make", "made", "take", "took",
    "let", "lets",
}

DISCOURSE_STARTERS = {
    "ahora", "aunque", "así", "bueno", "como", "cuando", "entonces", "menos",
    "mientras", "o", "pero", "porque", "que", "si", "también", "y", "ya",
    # English
    "but", "so", "then", "and", "or", "because", "however", "although",
    "anyway", "meanwhile", "next", "later", "finally", "suddenly",
}

_CJK_DISCOURSE = {
    "所以", "因此", "但是", "而且", "然后", "是的", "好的", "现在", "不过", "当然",
    "可是", "然而", "接着", "于是",
}

_MAX_MOVED_WORDS = 8
_NEAR_GAP = 0.55
_NEAR_OVERLAP = -0.35
_MICRO_FORWARD_GAP = 1.6
_SHORT_TRAIL_WORDS = 4


def source_cue_requires_continuation(previous_text: str, current_text: str) -> bool:
    previous_words = re.findall(r"[A-Za-zÀ-ÿ]+", (previous_text or "").lower())
    current_words = re.findall(r"[A-Za-zÀ-ÿ]+", (current_text or "").lower())
    if not previous_words or not current_words:
        return False
    current_first = current_words[0]
    if current_first in DISCOURSE_STARTERS:
        return False
    tail = previous_words[-1]
    leading = (current_text or "").lstrip()[:1]
    # Object pronouns at a clause end often meet a new capitalized sentence when
    # Whisper omitted the period ("blame him" / "He's right under...").
    if (
        leading.isupper()
        and current_first != "i"
        and tail in {"me", "him", "her", "us", "them", "you"}
    ):
        return False
    if tail == "que":
        return len(previous_words) >= 2 and previous_words[-2] in {
            "para", "por", "lo", "la", "el", "las", "los", "de",
        }
    return tail in CONTINUATION_TAILS


def segments_are_near(left: Segment, right: Segment, gap_limit: float = _NEAR_GAP) -> bool:
    gap = right.start - left.end
    return gap <= gap_limit and gap >= _NEAR_OVERLAP


def coalesce_incomplete_source_cues(segments: list[Segment]) -> list[Segment]:
    """Repair unfinished Whisper/platform cue boundaries before translation.

    Handles:
    - article/preposition/quantifier tails (``poured the`` + ``springtails``)
    - short trailing fragments (``...place`` + ``to hide.``)
    - orphan micro cues merged forward (``I`` + ``poured the...``)
    """
    if not segments:
        return []
    pending = [
        Segment(float(item.start), float(item.end), str(item.text or "").strip(), item.translated_text)
        for item in segments
        if str(item.text or "").strip() and float(item.end) > float(item.start)
    ]
    if not pending:
        return []

    output: list[Segment] = []
    index = 0
    while index < len(pending):
        current = pending[index]
        if index + 1 >= len(pending):
            output.append(current)
            break
        following = pending[index + 1]

        # 0) "...beginning. To" → keep the finished sentence, push "To" onto next.
        peeled = _peel_dangling_sentence_opener(current, following)
        if peeled is not None:
            pending[index], pending[index + 1] = peeled
            current, following = pending[index], pending[index + 1]

        # 1) Orphan / micro cue → absorb into the next cue.
        if _should_merge_micro_into_following(current, following):
            pending[index + 1] = Segment(
                min(current.start, following.start),
                max(current.end, following.end),
                _join_segment_text(current.text, following.text),
                following.translated_text,
            )
            pending.pop(index)
            continue

        # 2) Short trailing fragment / infinitive phrase → absorb into previous.
        if _should_absorb_following_entirely(current, following):
            pending[index] = Segment(
                current.start,
                max(current.end, following.end),
                _join_segment_text(current.text, following.text),
                current.translated_text,
            )
            pending.pop(index + 1)
            continue

        # 3) Borrow leading complement words from the next cue.
        if _can_borrow_from_following(current, following):
            moved_text, remainder_text, ratio = _borrow_leading_words(current.text, following.text)
            if moved_text:
                split_at = _split_time(following.start, following.end, ratio)
                merged = Segment(
                    current.start,
                    max(current.end, split_at),
                    _join_segment_text(current.text, moved_text),
                    current.translated_text,
                )
                if remainder_text:
                    pending[index + 1] = Segment(
                        max(split_at, following.start),
                        max(following.end, split_at + 0.05),
                        remainder_text,
                        following.translated_text,
                    )
                    output.append(merged)
                    index += 1
                else:
                    pending[index] = Segment(
                        merged.start,
                        max(merged.end, following.end),
                        merged.text,
                        merged.translated_text,
                    )
                    pending.pop(index + 1)
                continue

        output.append(current)
        index += 1
    return output


def _sentence_closed(text: str) -> bool:
    return bool(re.search(r"[.!?。！？][\"'”’)]?\s*$", (text or "").strip()))


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÿ]+", text or ""))


def _first_word(text: str) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", (text or "").lower())
    return words[0] if words else ""


_DANGLING_OPENERS = {"to", "the", "a", "an", "and", "or", "but", "of", "for", "with", "i"}


def _peel_dangling_sentence_opener(
    previous: Segment, following: Segment
) -> tuple[Segment, Segment] | None:
    """Split ``...end. To`` so the finished sentence stays and ``To`` joins the next cue."""
    match = re.search(
        r"^(?P<body>.*?[.!?…][\"'”’)]?)\s+(?P<dangling>[A-Za-zÀ-ÿ]{1,6})$",
        previous.text.strip(),
        flags=re.DOTALL,
    )
    if not match:
        return None
    dangling = match.group("dangling")
    if dangling.lower() not in _DANGLING_OPENERS:
        return None
    body = match.group("body").strip()
    if not body:
        return None
    # Keep a tiny tail of the previous cue's time for the dangling word when the
    # next cue is far; otherwise attach using the existing gap.
    peel_at = max(previous.start + 0.2, previous.end - 0.25)
    left = Segment(previous.start, peel_at, body, previous.translated_text)
    right = Segment(
        peel_at,
        max(following.end, peel_at + 0.05),
        _join_segment_text(dangling, following.text),
        following.translated_text,
    )
    # Prefer bridging into the real next cue start when it is reasonably close.
    if following.start - peel_at <= _MICRO_FORWARD_GAP:
        right = Segment(
            min(peel_at, following.start),
            following.end,
            _join_segment_text(dangling, following.text),
            following.translated_text,
        )
    return left, right


def _should_merge_micro_into_following(previous: Segment, following: Segment) -> bool:
    if _sentence_closed(previous.text):
        return False
    words = _word_count(previous.text)
    duration = previous.end - previous.start
    if words == 0 or words > 2 or duration > 0.85:
        return False
    if _first_word(previous.text) in DISCOURSE_STARTERS:
        return False
    if not segments_are_near(previous, following, gap_limit=_MICRO_FORWARD_GAP):
        return False
    # Avoid gluing a lone conjunction onto an unrelated next sentence.
    if words == 1 and _first_word(previous.text) in {"and", "or", "but", "so", "y", "o", "pero"}:
        return False
    return True


def _should_absorb_following_entirely(previous: Segment, following: Segment) -> bool:
    if _sentence_closed(previous.text):
        return False
    if not segments_are_near(previous, following):
        return False
    following_words = re.findall(r"[A-Za-zÀ-ÿ]+", following.text.lower())
    if not following_words or len(following_words) > _SHORT_TRAIL_WORDS:
        return False
    first = following_words[0]
    if first in DISCOURSE_STARTERS:
        return False
    # Infinitive / purpose fragment: "...a place" + "to hide."
    if first == "to" and len(following_words) <= 3:
        return True
    # Short trailing completer when previous is unfinished by grammar.
    # Do not swallow a discourse pivot such as "cristal entonces seguimos".
    if source_cue_requires_continuation(previous.text, following.text):
        if not any(word in DISCOURSE_STARTERS for word in following_words[1:]):
            return True
    # Very short leftover cue without its own sentence force (e.g. "guys now.")
    if len(following_words) <= 2 and (following.end - following.start) <= 1.0:
        prev_words = re.findall(r"[A-Za-zÀ-ÿ]+", previous.text.lower())
        if prev_words and prev_words[-1] not in DISCOURSE_STARTERS:
            return True
    return False


def _can_borrow_from_following(previous: Segment, following: Segment) -> bool:
    if _sentence_closed(previous.text):
        return False
    if not segments_are_near(previous, following):
        return False
    return source_cue_requires_continuation(previous.text, following.text)


def _borrow_leading_words(previous_text: str, following_text: str) -> tuple[str, str, float]:
    tokens = _tokenize_keep_punct(following_text)
    if not tokens:
        return "", following_text.strip(), 0.0
    word_indexes = [index for index, token in enumerate(tokens) if _is_word_token(token)]
    if not word_indexes:
        return "", following_text.strip(), 0.0

    moved_word_count = 0
    cut_after = -1
    built_previous = previous_text.strip()
    for word_pos, token_index in enumerate(word_indexes):
        if moved_word_count >= _MAX_MOVED_WORDS:
            break
        word = re.sub(r"[^A-Za-zÀ-ÿ]", "", tokens[token_index]).lower()
        if not word:
            continue
        remainder_words = [
            re.sub(r"[^A-Za-zÀ-ÿ]", "", tokens[idx]).lower()
            for idx in word_indexes[word_pos:]
            if re.sub(r"[^A-Za-zÀ-ÿ]", "", tokens[idx])
        ]
        remainder_probe = " ".join(remainder_words)
        if moved_word_count and word in DISCOURSE_STARTERS:
            break
        if not source_cue_requires_continuation(built_previous, remainder_probe):
            break
        if word in DISCOURSE_STARTERS:
            break
        moved_word_count += 1
        cut_after = token_index
        piece = tokens[token_index]
        built_previous = _join_segment_text(built_previous, piece)

    if cut_after < 0:
        return "", following_text.strip(), 0.0

    moved_tokens = tokens[: cut_after + 1]
    rest_tokens = tokens[cut_after + 1 :]
    if rest_tokens and re.fullmatch(r"[,:;，、]+", rest_tokens[0].strip() or ""):
        moved_tokens.append(rest_tokens.pop(0))
    moved_text = _detokenize(moved_tokens).strip()
    remainder_text = _detokenize(rest_tokens).strip()
    if not moved_text:
        return "", following_text.strip(), 0.0
    total_chars = max(1, len(re.sub(r"\s+", "", following_text)))
    moved_chars = len(re.sub(r"\s+", "", moved_text))
    ratio = min(1.0, max(0.05, moved_chars / total_chars))
    if not remainder_text:
        ratio = 1.0
    return moved_text, remainder_text, ratio


def _tokenize_keep_punct(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:'[A-Za-zÀ-ÿ]+)?|[^\s]", text.strip())


def _is_word_token(token: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ]", token))


def _detokenize(tokens: list[str]) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    for token in tokens:
        if not parts:
            parts.append(token)
            continue
        if token in {",", ".", "!", "?", ";", ":", "，", "。", "！", "？", "；", "：", "、"} or token.startswith("'"):
            parts[-1] = parts[-1] + token
        else:
            parts.append(token)
    return " ".join(parts)


def _join_segment_text(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?，。！？、；：)]}」』”’":
        return left + right
    return f"{left} {right}"


def _split_time(start: float, end: float, ratio: float) -> float:
    span = max(0.05, end - start)
    split_at = start + span * ratio
    return round(min(end, max(start + 0.05, split_at)), 3)


def segments_content_equal(left: list[Segment], right: list[Segment]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if (
            round(a.start, 3) != round(b.start, 3)
            or round(a.end, 3) != round(b.end, 3)
            or (a.text or "").strip() != (b.text or "").strip()
        ):
            return False
    return True


def cjk_discourse_markers() -> set[str]:
    return set(_CJK_DISCOURSE)

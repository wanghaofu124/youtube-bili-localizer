"""Build a short Whisper initial prompt from metadata the user already has.

The prompt only biases proper nouns. It never requires listening to the video,
and an empty result means transcription runs exactly as before.
"""
from __future__ import annotations

import re

from .glossary import glossary_hits

_STOPWORDS = frozenset({
    "a", "an", "and", "at", "adventure", "animal", "animals", "best", "by",
    "caught", "clear", "cooked", "day", "days", "double", "eating", "episode",
    "every", "exotic", "fish", "fishing", "for", "from", "full", "hd", "how",
    "i", "in", "lure", "my", "new", "ocean", "of", "official", "on", "only",
    "or", "part", "reef", "snorkeling", "species", "tank", "the", "time",
    "times", "tiny", "to", "tropical", "video", "water", "what", "when",
    "where", "why", "wild", "with",
})
_PHRASE = re.compile(
    r"\b[A-Z][A-Za-z0-9'’-]{1,}(?:\s+[A-Z][A-Za-z0-9'’-]{1,}){1,2}\b"
)
_SINGLE = re.compile(r"\b[A-Z][A-Za-z0-9'’-]{4,}\b")
_MAX_TERMS = 8
_MAX_PROMPT = 240


def build_initial_prompt(
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    user_hint: str | None = None,
    glossary: list[dict] | None = None,
) -> str | None:
    """Return a short keyword prompt, or None when nothing usable exists.

    Glossary entries whose source term (or alias) appears in the metadata or
    hint are placed first, spelled the way the glossary spells them.
    """
    hint = " ".join(str(user_hint or "").split())[:160]
    metadata = " ".join([str(title or ""), str(description or "")[:300], " ".join(tags or []), hint])
    glossary_terms = [item["source"] for item in glossary_hits(glossary, metadata)]
    terms = _extract_terms(title, description, tags, hint, glossary_terms)
    parts: list[str] = []
    if terms:
        parts.append("Keywords: " + ", ".join(terms))
    if hint:
        parts.append(hint)
    text = ". ".join(parts).strip(" .")
    return text[:_MAX_PROMPT].strip(" .") or None


def _extract_terms(
    title: str | None,
    description: str | None,
    tags: list[str] | None,
    user_hint: str,
    glossary_terms: list[str] | None = None,
) -> list[str]:
    metadata = " ".join(
        part for part in (str(title or ""), str(description or "")[:300]) if part.strip()
    )
    seen: dict[str, str] = {}
    hinted = {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9'’-]{1,}", user_hint)}

    def add(term: str) -> None:
        cleaned = " ".join(term.split())
        key = cleaned.casefold()
        words = [word.casefold() for word in cleaned.split()]
        if (
            not cleaned
            or key in seen
            or key in hinted
            or (words and all(word in hinted for word in words))
            or len(seen) >= _MAX_TERMS
        ):
            return
        seen[key] = cleaned

    for term in glossary_terms or []:
        add(term)
    for phrase in _PHRASE.findall(metadata):
        kept = [word for word in phrase.split() if word.casefold() not in _STOPWORDS]
        if len(kept) >= 2:
            add(" ".join(kept))
        elif len(kept) == 1 and len(kept[0]) >= 5:
            add(kept[0])
    for word in _SINGLE.findall(metadata):
        if word.casefold() in _STOPWORDS or len(word) < 5:
            continue
        if any(word.casefold() in phrase.casefold().split() for phrase in seen.values()):
            continue
        add(word)
    for tag in tags or []:
        cleaned = " ".join(str(tag).split())
        if not cleaned or len(cleaned) < 3 or len(cleaned) > 30:
            continue
        if cleaned.casefold() in _STOPWORDS:
            continue
        if not re.search(r"[A-Za-z]", cleaned):
            continue
        add(cleaned)
    return list(seen.values())

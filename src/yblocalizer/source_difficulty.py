"""Advisory difficulty hint for extracted English subtitles.

Common words come from the 20,000 most frequent English tokens in the
bundled FrequencyWords subtitle list. Each distinct lemma outside that list
adds 2 points, up to 80. A long sentence with a clause marker and two such lemmas adds 2
points, up to 10. Callers display the band; they must not change the translator.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

MIN_TOKENS = 80
VOCAB_POINT = 2
VOCAB_CAP = 80
SENTENCE_POINT = 2
SENTENCE_CAP = 10
SENTENCE_MIN_WORDS = 18
_WORD_LIST = Path(__file__).with_name("common_en_20k.txt")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']+")
_CLAUSE = re.compile(r"\b(which|although|though|whose|whereas|because|while|unless|despite)\b", re.I)
_SPOKEN = frozenset({
    "bro", "dude", "gosh", "gotta", "gonna", "wanna", "yeah", "yo", "ok", "okay",
    "nah", "yep", "whoa", "wow", "hmm", "uh", "um", "hey", "alright", "dang",
    "biggie", "heck", "crap", "poop", "cannot",
})
_CONTRACTIONS = {
    "i'm": "i", "i'll": "i", "i've": "i", "i'd": "i",
    "you're": "you", "you'll": "you", "you've": "you", "you'd": "you",
    "he's": "he", "he'll": "he", "he'd": "he",
    "she's": "she", "she'll": "she", "she'd": "she",
    "it's": "it", "it'll": "it",
    "we're": "we", "we'll": "we", "we've": "we",
    "they're": "they", "they'll": "they", "they've": "they", "they'd": "they",
    "that's": "that", "there's": "there", "here's": "here", "what's": "what",
    "don't": "do", "doesn't": "does", "didn't": "do",
    "can't": "can", "couldn't": "could", "won't": "will", "wouldn't": "would",
    "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
    "haven't": "have", "hasn't": "has", "hadn't": "have", "let's": "let",
    "that'd": "that", "it'd": "it", "there'd": "there", "who'd": "who",
}
_BANDS: tuple[tuple[int, str, str], ...] = (
    (0, "0–19", "这段大多是常见词。建议用本地部署的开源模型，也可以用 API。"),
    (20, "20–39", "只有少量不常见的词。建议用本地部署的开源模型，要更稳可以改用 API。"),
    (40, "40–59", "专有名词变多了，本地部署的开源模型开始容易译偏。建议用 API。"),
    (60, "60–79", "术语多，本地部署的开源模型容易把专名译错。建议用 API。"),
    (80, "80–100", "术语和书面长句很密，本地部署的开源模型稳不住这些专名。建议用 API。"),
)
_common: frozenset[str] | None = None
_hint_cache: dict[tuple[str, int, int], dict[str, Any] | None] = {}


def common_words() -> frozenset[str]:
    global _common
    if _common is None:
        words = {
            line.strip().lower()
            for line in _WORD_LIST.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        _common = frozenset(words)
    return _common


def _lemma(word: str, common: frozenset[str]) -> str | None:
    text = word.lower().replace("’", "'").replace("‘", "'")
    if text.endswith("'s"):
        text = text[:-2]
    text = _CONTRACTIONS.get(text, text)
    if text in _SPOKEN:
        return None
    candidates = [text]
    for suffix in ("ing", "ed", "es", "s"):
        if len(text) > len(suffix) + 2 and text.endswith(suffix):
            stem = text[: -len(suffix)]
            candidates.append(stem)
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                candidates.append(stem[:-1])
            if suffix in {"ing", "ed"}:
                candidates.append(stem + "e")
    if any(item in _SPOKEN or item in common for item in candidates):
        return None
    return text


def subtitle_prose(raw: str) -> str:
    lines: list[str] = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(stripped)
    return " ".join(lines)


def band_for(score: int) -> int:
    if score >= 80:
        return 80
    if score <= 0:
        return 0
    return (score // 20) * 20


def score_english_text(text: str) -> dict[str, Any] | None:
    """Score prose. Returns None when there are fewer than 80 English tokens."""
    tokens = [token.lower() for token in _TOKEN.findall(text)]
    if len(tokens) < MIN_TOKENS:
        return None
    common = common_words()
    rare = {lemma for token in tokens if (lemma := _lemma(token, common))}
    vocab = min(VOCAB_CAP, len(rare) * VOCAB_POINT)
    hard = 0
    for sentence in re.split(r"[.!?]+", text):
        words = _TOKEN.findall(sentence)
        specialized = sum(1 for word in words if _lemma(word.lower(), common))
        if len(words) >= SENTENCE_MIN_WORDS and _CLAUSE.search(sentence) and specialized >= 2:
            hard += 1
    sentence_points = min(SENTENCE_CAP, hard * SENTENCE_POINT)
    score = vocab + sentence_points
    band = band_for(score)
    label, message = next(item[1:] for item in _BANDS if item[0] == band)
    return {
        "score": score,
        "band": band,
        "title": f"难度 {score} · {label}",
        "message": f"{message}少见词 {len(rare)} 个，重复不计。",
        "rare_count": len(rare),
        "advisory": True,
    }


def hint_for_source(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _hint_cache:
        if len(_hint_cache) > 32:
            _hint_cache.clear()
        raw = path.read_text(encoding="utf-8", errors="replace")
        _hint_cache[key] = score_english_text(subtitle_prose(raw))
    return _hint_cache[key]

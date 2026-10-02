"""Semantic subtitle resegmentation via LLM join decisions."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
from typing import Any, Callable

from .models import Segment
from .subtitle_continuity import coalesce_incomplete_source_cues, _join_segment_text


RESEGMENT_CHECKPOINT_VERSION = 2
MAX_GROUP_DURATION = 10.0
MAX_SHORT_COMPLETER_DURATION = 14.0
MAX_INTERNAL_GAP = 0.9
MAX_LATIN_CHARS = 160
MAX_LATIN_WORDS = 36
MAX_CJK_CHARS = 56
MAX_SHORT_COMPLETER_WORDS = 6
MAX_SHORT_COMPLETER_SPAN = 2.5
CONTEXT_RADIUS = 8


def resegment_source_segments(
    segments: list[Segment],
    *,
    provider: str,
    model: str | None = None,
    batch_size: int = 40,
    checkpoint_path: Path | None = None,
    cancel_check: Callable[[], None] | None = None,
    progress: Callable[[float], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Regroup cues by asking the model which adjacent boundaries to join.

    The model must not rewrite text. Timings are materialised from the first
    start and max end of each accepted contiguous group. On ordinary failure,
    callers should fall back to heuristic coalesce; this function raises on
    cancellation and returns heuristically-safe output only when there is
    nothing to decide.
    """
    logger = log or (lambda _message: None)
    check = cancel_check or (lambda: None)
    items = [
        Segment(float(item.start), float(item.end), str(item.text or "").strip(), None)
        for item in segments
        if str(item.text or "").strip() and float(item.end) > float(item.start)
    ]
    if len(items) <= 1:
        return items

    provider_key = (provider or "").strip().lower()
    if provider_key not in {"openai", "deepseek", "ollama"}:
        return coalesce_incomplete_source_cues(items)

    resolved_model = model or _default_model(provider_key)
    edge_count = len(items) - 1
    core_batch = max(8, min(60, int(batch_size or 40)))
    fingerprint = _fingerprint(items, provider_key, resolved_model, core_batch)
    decisions = _load_decisions(checkpoint_path, fingerprint)

    try:
        if provider_key == "ollama":
            from .translate import _require_ollama_model
            _require_ollama_model(resolved_model)
        client, api_key_env, base_url = _provider_client(provider_key)
        for core_start in range(0, edge_count, core_batch):
            check()
            core_end = min(edge_count, core_start + core_batch)
            missing = [edge for edge in range(core_start, core_end) if edge not in decisions]
            if not missing:
                if progress:
                    progress(core_end / edge_count)
                continue
            owned = list(range(core_start, core_end))
            payload = _build_request_payload(items, owned)
            join_after = _request_join_after(
                client,
                model=resolved_model,
                payload=payload,
                owned_left_ids=[edge + 1 for edge in owned],
            )
            allowed = {edge + 1 for edge in owned}
            for left_id in allowed:
                decisions[left_id - 1] = left_id in join_after
            _save_decisions(checkpoint_path, fingerprint, decisions)
            if progress:
                progress(core_end / edge_count)
        grouped = _materialize_groups(items, decisions)
        _assert_conservation(items, grouped)
        # Attach remaining short grammatical completers that the duration gate
        # may still have blocked (e.g. "...handed me" + "this box...").
        grouped = coalesce_incomplete_source_cues(grouped)
        _assert_conservation(items, grouped)
        logger(f"语义重切完成：{len(items)} 条 → {len(grouped)} 条。")
        return grouped
    except _ResegmentCancelled:
        raise
    except Exception as exc:
        # Propagate explicit cancellation wrappers used by the workflow.
        if _is_cancellation(exc):
            raise
        logger(f"语义重切失败，回退规则合并：{exc}")
        if checkpoint_path and checkpoint_path.exists():
            # Keep partial successful batches; do not mark this run as complete.
            pass
        return coalesce_incomplete_source_cues(items)


class _ResegmentCancelled(Exception):
    pass


def _default_model(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini")
    if provider == "ollama":
        return "qwen3.5:4b"
    return os.getenv("DEEPSEEK_TRANSLATE_MODEL", "deepseek-v4-flash")


def _provider_client(provider: str):
    from .translate import _chat_client

    if provider == "openai":
        return _chat_client("OPENAI_API_KEY", None), "OPENAI_API_KEY", None
    if provider == "ollama":
        from .dependencies import OLLAMA_BASE_URL
        base_url = f"{OLLAMA_BASE_URL}/v1"
        return _chat_client("YBLOCALIZER_OLLAMA", base_url), "YBLOCALIZER_OLLAMA", base_url
    return (
        _chat_client("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        "DEEPSEEK_API_KEY",
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _fingerprint(segments: list[Segment], provider: str, model: str, batch_size: int) -> dict[str, Any]:
    blob = json.dumps(
        [{"start": round(item.start, 3), "end": round(item.end, 3), "text": item.text} for item in segments],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "version": RESEGMENT_CHECKPOINT_VERSION,
        "source_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "provider": provider,
        "model": model,
        "batch_size": batch_size,
        "context_radius": CONTEXT_RADIUS,
        "max_duration": MAX_GROUP_DURATION,
        "max_gap": MAX_INTERNAL_GAP,
        "max_latin_chars": MAX_LATIN_CHARS,
        "max_latin_words": MAX_LATIN_WORDS,
        "max_cjk_chars": MAX_CJK_CHARS,
    }


def _load_decisions(path: Path | None, fingerprint: dict[str, Any]) -> dict[int, bool]:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if raw.get("fingerprint") != fingerprint:
        return {}
    decisions: dict[int, bool] = {}
    for key, value in (raw.get("decisions") or {}).items():
        try:
            decisions[int(key)] = bool(value)
        except (TypeError, ValueError):
            continue
    return decisions


def _save_decisions(path: Path | None, fingerprint: dict[str, Any], decisions: dict[int, bool]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "decisions": {str(key): bool(value) for key, value in sorted(decisions.items())},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_request_payload(segments: list[Segment], owned_edges: list[int]) -> dict[str, Any]:
    if not owned_edges:
        return {"cues": [], "decide_join_after_ids": []}
    left = max(0, owned_edges[0] - CONTEXT_RADIUS)
    right = min(len(segments), owned_edges[-1] + 1 + CONTEXT_RADIUS + 1)
    cues = [
        {
            "id": index + 1,
            "start": round(segments[index].start, 3),
            "end": round(segments[index].end, 3),
            "text": segments[index].text,
        }
        for index in range(left, right)
    ]
    return {
        "cues": cues,
        "decide_join_after_ids": [edge + 1 for edge in owned_edges],
        "rules": {
            "join_after_means": "Join cue id with the immediately following cue into one subtitle.",
            "do_not_rewrite_text": True,
            "prefer_complete_phrases": True,
            "avoid_joining_across_long_pauses": True,
            "max_duration_seconds": MAX_GROUP_DURATION,
        },
    }


def _request_join_after(client, *, model: str, payload: dict[str, Any], owned_left_ids: list[int]) -> set[int]:
    owned = set(owned_left_ids)
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You repair subtitle cue boundaries for localization. "
                    "Decide only which adjacent cues should be joined. "
                    "Never rewrite, translate, delete, reorder, or invent text. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Given numbered subtitle cues, return which left-cue ids should join with the next cue.\n"
                    "Only include ids from decide_join_after_ids when the two cues are one unfinished phrase "
                    "or one readable subtitle unit.\n"
                    "Do not join across a clear sentence end or a long pause.\n"
                    "Output schema: {\"join_after\": [id, ...]}\n\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ],
    )
    text = response.choices[0].message.content or ""
    parsed = _parse_join_after(text)
    return {item for item in parsed if item in owned}


def _parse_join_after(text: str) -> set[int]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"semantic resegment returned invalid JSON: {exc}") from exc
    values = raw.get("join_after") if isinstance(raw, dict) else None
    if values is None and isinstance(raw, list):
        values = raw
    if not isinstance(values, list):
        raise RuntimeError("semantic resegment JSON missing join_after list")
    output: set[int] = set()
    for item in values:
        try:
            output.add(int(item))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"semantic resegment join id is not an integer: {item!r}") from exc
    return output


def _materialize_groups(segments: list[Segment], decisions: dict[int, bool]) -> list[Segment]:
    output: list[Segment] = []
    index = 0
    while index < len(segments):
        end = index
        while end < len(segments) - 1 and decisions.get(end, False):
            candidate = segments[index : end + 2]
            if not _group_within_limits(candidate):
                break
            end += 1
        group = segments[index : end + 1]
        text = group[0].text
        for item in group[1:]:
            text = _join_segment_text(text, item.text)
        output.append(
            Segment(
                group[0].start,
                max(item.end for item in group),
                text,
                None,
            )
        )
        index = end + 1
    return output


def _group_within_limits(group: list[Segment]) -> bool:
    if len(group) <= 1:
        return True
    for left, right in zip(group, group[1:]):
        gap = right.start - left.end
        if gap > MAX_INTERNAL_GAP + 1e-6 or gap < -0.35:
            return False
    duration = max(item.end for item in group) - group[0].start
    text = group[0].text
    for item in group[1:]:
        text = _join_segment_text(text, item.text)
    short_tail = len(group) >= 2 and _is_short_completer(group[-1])
    max_duration = MAX_SHORT_COMPLETER_DURATION if short_tail else MAX_GROUP_DURATION
    if duration > max_duration + 1e-6:
        return False
    return _text_within_budget(text, relaxed=short_tail)


def _is_short_completer(segment: Segment) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", segment.text or "")
    span = float(segment.end) - float(segment.start)
    return 0 < len(words) <= MAX_SHORT_COMPLETER_WORDS and span <= MAX_SHORT_COMPLETER_SPAN


def _text_within_budget(text: str, *, relaxed: bool) -> bool:
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    if cjk_chars:
        return cjk_chars <= (MAX_CJK_CHARS + 12 if relaxed else MAX_CJK_CHARS)
    latin_chars = len(re.sub(r"\s+", "", text))
    latin_words = len(re.findall(r"[A-Za-zÀ-ÿ]+", text))
    char_limit = MAX_LATIN_CHARS + 40 if relaxed else MAX_LATIN_CHARS
    word_limit = MAX_LATIN_WORDS + 10 if relaxed else MAX_LATIN_WORDS
    return latin_chars <= char_limit and latin_words <= word_limit


def _assert_conservation(source: list[Segment], grouped: list[Segment]) -> None:
    source_text = _normalize_conservation_text(" ".join(item.text for item in source))
    grouped_text = _normalize_conservation_text(" ".join(item.text for item in grouped))
    if source_text != grouped_text:
        raise RuntimeError("semantic resegment changed source text coverage")
    if abs(source[0].start - grouped[0].start) > 1e-6:
        raise RuntimeError("semantic resegment changed timeline start")
    if abs(max(item.end for item in source) - max(item.end for item in grouped)) > 1e-6:
        raise RuntimeError("semantic resegment changed timeline end")


def _normalize_conservation_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_cancellation(exc: BaseException) -> bool:
    name = type(exc).__name__
    message = str(exc).lower()
    return name in {"CancellationRequested", "_ResegmentCancelled"} or "cancel" in message

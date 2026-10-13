"""Request accounting without storing subtitle text or credentials."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid

from .llm_cost import add_cost, empty_cost, estimate_cost, origin_of, pricing_snapshot


LOGGER = logging.getLogger(__name__)
_ACTIVE_TRACKER: ContextVar[LlmUsageTracker | None] = ContextVar("llm_usage_tracker", default=None)
_ACTIVE_CALL: ContextVar[dict[str, Any] | None] = ContextVar("llm_http_call", default=None)
_UNSUPPORTED_THINKING: set[tuple[str, str]] = set()
_CAPABILITY_LOCK = threading.Lock()
TOKEN_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens", "reasoning_tokens",
)
COUNTER_FIELDS = (
    "logical_calls", "http_requests", "sdk_retry_requests", "thinking_fallback_requests",
    "successful_calls", "failed_calls", "calls_with_usage", "calls_without_usage",
    "duration_ms_total", "cache_reused_batches", "cache_reused_items", "split_events",
)


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _empty_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "revision": 0,
        **{key: 0 for key in (
            "logical_calls", "http_requests", "sdk_retry_requests", "thinking_fallback_requests",
            "successful_calls", "failed_calls", "calls_with_usage", "calls_without_usage",
            "duration_ms_total", "duration_ms_max", "cache_reused_batches", "cache_reused_items",
            "split_events",
        )},
        **{key: None for key in TOKEN_FIELDS},
        "cost": empty_cost(),
        "token_field_calls": {key: 0 for key in TOKEN_FIELDS},
        "by_kind": {},
    }


def combine_summaries(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum complete session snapshots; callers replace revisions before summing."""
    total = _empty_summary()
    kinds: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        for key in COUNTER_FIELDS:
            total[key] += value.get(key, 0) or 0
        total["duration_ms_max"] = max(total["duration_ms_max"], value.get("duration_ms_max", 0) or 0)
        for key in TOKEN_FIELDS:
            if value.get(key) is not None:
                total[key] = (total[key] or 0) + value[key]
            total["token_field_calls"][key] += value.get("token_field_calls", {}).get(key, 0)
        if isinstance(value.get("cost"), dict):
            add_cost(total["cost"], value["cost"])
        elif value.get("logical_calls"):
            unknown = empty_cost()
            unknown["unpriced_calls"] = value["logical_calls"]
            add_cost(total["cost"], unknown)
        for kind, group in value.get("by_kind", {}).items():
            kinds.setdefault(kind, []).append(group)
    total["by_kind"] = {kind: combine_summaries(groups) for kind, groups in kinds.items()}
    return total


def update_usage_index(index: dict[str, Any], token: str, path: Path, summary: dict[str, Any]) -> bool:
    sessions = index.setdefault("sessions", {})
    previous = sessions.get(token)
    if previous and (previous.get("summary_path") != str(path) or previous.get("revision", 0) >= summary["revision"]):
        return False
    sessions[token] = {"revision": summary["revision"], "summary_path": str(path), "summary": deepcopy(summary)}
    values = [session["summary"] for session in sessions.values()]
    if index.get("baseline"):
        values.append(index["baseline"])
    total = combine_summaries(values)
    total["recorded_since"] = index["recorded_since"]
    total["session_count"] = len(sessions)
    index["revision"] = index.get("revision", 0) + 1
    index["total"] = total
    return True


class LlmUsageTracker:
    """One processing attempt, with incremental counters and best-effort disk writes."""

    def __init__(self, directory: Path, callback: Callable[[dict[str, Any]], None] | None = None, *, scope: str = "workflow", pricing_options: dict[str, Any] | None = None):
        self.directory = directory
        self.summary_path = directory / "llm_usage.json"
        self._callback = callback
        self._lock = threading.Lock()
        self._summary = _empty_summary()
        self._summary["scope"] = scope
        self._warned = False
        self.pricing_options = deepcopy(pricing_options or {})

    def _update(self, kind: str, row: dict[str, Any] | None = None, *, items: int = 0, split: bool = False, publish_only: bool = False) -> None:
        with self._lock:
            summary = self._summary
            summary["revision"] += 1
            targets = ()
            if not publish_only:
                group = summary["by_kind"].setdefault(kind, _empty_summary())
                group.pop("by_kind", None)
                targets = (summary, group)
            for target in targets:
                if row is not None:
                    target["logical_calls"] += 1
                    for key in ("http_requests", "sdk_retry_requests", "thinking_fallback_requests"):
                        target[key] += row[key]
                    target["successful_calls" if row["success"] else "failed_calls"] += 1
                    target["calls_with_usage" if row["usage_status"] == "reported" else "calls_without_usage"] += 1
                    target["duration_ms_total"] += row["duration_ms"]
                    target["duration_ms_max"] = max(target["duration_ms_max"], row["duration_ms"])
                    add_cost(target["cost"], row["cost"])
                    for key, value in row["usage"].items():
                        if value is not None:
                            target[key] = (target[key] or 0) + value
                            target["token_field_calls"][key] += 1
                elif split:
                    target["split_events"] += 1
                else:
                    target["cache_reused_batches"] += 1
                    target["cache_reused_items"] += items
            snapshot = deepcopy(summary)
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if row is not None:
                    with (self.directory / "llm_usage.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                temporary = self.summary_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, self.summary_path)
            except OSError:
                if not self._warned:
                    LOGGER.warning("Could not persist LLM usage; in-memory accounting is retained.")
                    self._warned = True
        # Never acquire a job lock while holding the tracker lock.
        if self._callback:
            try:
                self._callback(snapshot)
            except Exception:
                LOGGER.warning("Could not update LLM usage display.")

    def record(self, row: dict[str, Any]) -> None:
        self._update(row["kind"], row)

    def note_cache_reuse(self, kind: str, items: int) -> None:
        self._update(kind, items=items)

    def note_split(self, kind: str) -> None:
        self._update(kind, split=True)

    def flush(self) -> None:
        self._update("", publish_only=True)


@contextmanager
def usage_session(tracker: LlmUsageTracker):
    token = _ACTIVE_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _ACTIVE_TRACKER.reset(token)


def attach_tracker(client: Any) -> Any:
    # Client creation happens on the workflow thread, before batch workers start.
    # The explicitly attached tracker therefore also reaches thread-pool calls.
    client._yblocalizer_usage_tracker = _ACTIVE_TRACKER.get()
    return client


def note_cache_reuse(client: Any, kind: str, items: int) -> None:
    tracker = getattr(client, "_yblocalizer_usage_tracker", None)
    if tracker:
        tracker.note_cache_reuse(kind, items)


def note_split(client: Any, kind: str) -> None:
    tracker = getattr(client, "_yblocalizer_usage_tracker", None)
    if tracker:
        tracker.note_split(kind)


def http_request_hook(_request: Any) -> None:
    call = _ACTIVE_CALL.get()
    if call is not None:
        if not call["http_requests"]:
            call["first_request_at"] = time.time()
            factory = call.get("pricing_factory")
            if factory:
                try:
                    call["pricing"] = factory(call["first_request_at"])
                except Exception:
                    call["pricing"] = {"source": "pricing_error"}
        call["http_requests"] += 1


def _thinking_unsupported(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 400:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return False
    message = " ".join(str(detail.get(key) or "") for key in ("param", "code", "message", "type")).lower()
    return "thinking" in message and any(word in message for word in (
        "unsupported", "not supported", "unknown", "unrecognized", "not permitted",
        "extra_forbidden", "extra field", "extra inputs", "unexpected keyword",
    ))


def completion_create(client: Any, *, kind: str, provider: str, model: str, **kwargs: Any) -> Any:
    tracker = getattr(client, "_yblocalizer_usage_tracker", None)
    provider = {"DEEPSEEK_API_KEY": "deepseek", "OPENAI_API_KEY": "openai", "YBLOCALIZER_OLLAMA": "ollama"}.get(provider, provider.lower())
    if provider.startswith("ollama"):
        provider = "ollama"
    extra = kwargs.get("extra_body")
    thinking = extra.get("thinking") if isinstance(extra, dict) else None
    requested = kwargs.pop("_thinking_requested", "auto")
    effective = thinking.get("type", "auto") if isinstance(thinking, dict) else "auto"
    endpoint = urlsplit(str(getattr(client, "base_url", "")))
    capability_key = (f"{endpoint.scheme}://{endpoint.netloc}", model)
    with _CAPABILITY_LOCK:
        unsupported = capability_key in _UNSUPPORTED_THINKING
    if provider == "deepseek" and thinking and unsupported:
        extra = dict(extra)
        extra.pop("thinking", None)
        kwargs["extra_body"] = extra or None
        effective = "provider_default_unknown"
    wall_started = time.time()
    def resolve_pricing(timestamp: float) -> dict[str, Any]:
        return pricing_snapshot(provider, model, origin_of(str(getattr(client, "base_url", ""))), timestamp, tracker.pricing_options)
    call: dict[str, Any] = {"http_requests": 0}
    if tracker and provider != "ollama":
        call["pricing_factory"] = resolve_pricing
        try:
            call["pricing"] = resolve_pricing(wall_started)
        except Exception:
            call["pricing"] = {"source": "pricing_error"}
    token = _ACTIVE_CALL.set(call)
    started = time.monotonic()
    response = None
    failure: BaseException | None = None
    fallback = 0
    try:
        try:
            response = client.chat.completions.create(model=model, **kwargs)
        except Exception as exc:
            if provider != "deepseek" or not thinking or unsupported or not _thinking_unsupported(exc):
                raise
            with _CAPABILITY_LOCK:
                _UNSUPPORTED_THINKING.add(capability_key)
            extra = dict(kwargs.get("extra_body") or {})
            extra.pop("thinking", None)
            kwargs["extra_body"] = extra or None
            effective = "provider_default_unknown"
            fallback = 1
            response = client.chat.completions.create(model=model, **kwargs)
        return response
    except BaseException as exc:
        failure = exc
        raise
    finally:
        _ACTIVE_CALL.reset(token)
        if tracker and provider != "ollama":
            usage = _get(response, "usage")
            prompt_details = _get(usage, "prompt_tokens_details")
            completion_details = _get(usage, "completion_tokens_details")
            hit = _number(_get(usage, "prompt_cache_hit_tokens"))
            if hit is None:
                hit = _number(_get(prompt_details, "cached_tokens"))
            counts = {
                "prompt_tokens": _number(_get(usage, "prompt_tokens")),
                "completion_tokens": _number(_get(usage, "completion_tokens")),
                "total_tokens": _number(_get(usage, "total_tokens")),
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": _number(_get(usage, "prompt_cache_miss_tokens")),
                "reasoning_tokens": _number(_get(completion_details, "reasoning_tokens")),
            }
            pricing = call.get("pricing", {"source": "unknown_price"})
            try:
                cost = estimate_cost(counts, pricing, call["http_requests"], failure is None)
            except Exception:
                cost = empty_cost()
                cost.update({"status": "unknown", "unpriced_calls": 1})
                LOGGER.warning("Cost estimation failed; the model response is retained.")
            row = {
                "schema_version": 1, "call_id": uuid.uuid4().hex, "timestamp": time.time(),
                "started_at": call.get("first_request_at", wall_started), "finished_at": time.time(),
                "kind": kind, "provider": provider, "model": model,
                "success": failure is None, "error_type": type(failure).__name__ if failure else None,
                "http_status": getattr(failure, "status_code", None),
                "duration_ms": round((time.monotonic() - started) * 1000),
                "http_requests": call["http_requests"],
                "sdk_retry_requests": max(call["http_requests"] - 1 - fallback, 0),
                "thinking_fallback_requests": fallback,
                "thinking_requested": requested, "thinking_effective": effective,
                "usage_status": "reported" if any(value is not None for value in counts.values()) else ("unavailable" if failure else "missing"),
                "usage": counts,
                "pricing": pricing, "cost": cost,
            }
            try:
                tracker.record(row)
            except Exception:
                LOGGER.warning("LLM accounting failed; the model response is retained.")

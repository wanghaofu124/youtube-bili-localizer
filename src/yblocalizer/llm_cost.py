"""Offline CNY estimates, frozen per request; no billing or model requests."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

PRICE_FIELDS = ("cache_hit_input_cny_per_million", "cache_miss_input_cny_per_million", "output_cny_per_million")
PRICE_SOURCE = "deepseek_official_2026-10-11"
# https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
# https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm
HOLIDAYS_2026 = {(month, day) for month, first, last in (
    (1, 1, 3), (2, 15, 23), (4, 4, 6), (5, 1, 5), (6, 19, 21),
    (9, 25, 27), (10, 1, 7),
) for day in range(first, last + 1)}
MODELS = {
    "deepseek-flash": ("0.04", "2", "8"),
    "deepseek-v4-flash": ("0.04", "2", "8"),
    "deepseek-v4-flash-vision-exp": ("0.04", "2", "8"),
    "deepseek-v4-pro": ("0.30", "9", "27"),
}


def origin_of(value: str) -> str:
    parsed = urlsplit(str(value or "").strip())
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        return ""
    port = parsed.port
    suffix = f":{port}" if port and port != (443 if parsed.scheme == "https" else 80) else ""
    return f"{parsed.scheme}://{parsed.hostname.lower()}{suffix}"


def normalize_custom(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        return {}
    provider = str(value.get("provider") or "").strip().lower()
    origin = origin_of(str(value.get("endpoint_origin") or ""))
    model = str(value.get("model") or "").strip()
    if provider not in {"deepseek", "openai"} or not origin or not model:
        raise ValueError("自定义单价需要选择服务、填写接口地址和准确的模型名称。")
    result = {"provider": provider, "endpoint_origin": origin, "model": model}
    for key in PRICE_FIELDS:
        try:
            price = Decimal(str(value.get(key, "")))
        except InvalidOperation as exc:
            raise ValueError("请填写三项人民币单价（每百万 Token）。") from exc
        if not price.is_finite() or price < 0 or price > 1_000_000:
            raise ValueError("单价必须是 0 到 1000000 之间的有限数值。")
        result[key] = format(price, "f")
    return result


def pricing_snapshot(provider: str, model: str, origin: str, timestamp: float, options: dict[str, Any]) -> dict[str, Any]:
    mode = options.get("llm_pricing_mode", "auto")
    base = {"currency": "CNY", "requested_model": model, "endpoint_origin": origin, "provider": provider}
    if mode == "off":
        return {**base, "source": "disabled"}
    if mode == "custom":
        custom = normalize_custom(options.get("llm_custom_pricing"))
        if not custom or any(custom.get(key) != value for key, value in (
            ("provider", provider), ("model", model), ("endpoint_origin", origin),
        )):
            return {**base, "source": "custom_identity_mismatch"}
        rates = {key: custom[field] for key, field in zip(("hit", "miss", "output"), PRICE_FIELDS)}
        return {**base, "source": "custom", "tariff": "custom", "rates_min": rates, "rates_max": rates}
    if provider != "deepseek" or origin != "https://api.deepseek.com" or model not in MODELS:
        return {**base, "source": "unknown_price"}
    local = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(timezone(timedelta(hours=8)))
    peak = local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)
    if local.year == 2026 and (local.month, local.day) in HOLIDAYS_2026:
        peak = False
    peak_rates = {key: rate for key, rate in zip(("hit", "miss", "output"), MODELS[model])}
    off_rates = {key: format(Decimal(rate) / 2, "f") for key, rate in peak_rates.items()}
    # Weekends and times outside the peak windows do not depend on the holiday calendar.
    uncertain = local.year != 2026 and peak
    low = off_rates if uncertain or not peak else peak_rates
    high = peak_rates if uncertain or peak else off_rates
    return {**base, "source": PRICE_SOURCE, "tariff": "unknown_schedule" if uncertain else ("peak" if peak else "off_peak"),
            "rates_min": low, "rates_max": high}


def empty_cost() -> dict[str, Any]:
    return {"status": "no_requests", "min_cny": "0", "max_cny": "0", "priced_calls": 0,
            "unpriced_calls": 0, "incomplete_calls": 0, "usage_unknown_requests": 0,
            "request_count_unknown_calls": 0, "pricing_sources": []}


def _status(cost: dict[str, Any]) -> None:
    if cost["unpriced_calls"] or cost["incomplete_calls"] or cost["usage_unknown_requests"] or cost["request_count_unknown_calls"]:
        cost["status"] = "partial" if cost["priced_calls"] else "unknown"
    elif not cost["priced_calls"]:
        cost["status"] = "no_requests"
    else:
        cost["status"] = "range" if cost["min_cny"] != cost["max_cny"] else "estimated"


def estimate_cost(usage: dict[str, int | None], pricing: dict[str, Any], requests: int, success: bool) -> dict[str, Any]:
    result = empty_cost()
    reported = any(value is not None for value in usage.values())
    result["usage_unknown_requests"] = max(0, requests - (1 if reported else 0))
    result["request_count_unknown_calls"] = int(success and requests == 0)
    if requests == 0 and not success:
        return result
    result["pricing_sources"] = [pricing["source"]]
    if "rates_min" not in pricing:
        result["unpriced_calls"] = 1
        _status(result)
        return result
    prompt, hit, miss, output = (usage.get(key) for key in (
        "prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens",
    ))
    input_complete = hit is not None and miss is not None
    if input_complete and prompt is not None and hit + miss != prompt:
        input_complete = False
        result["incomplete_calls"] = 1
    input_known = input_complete or prompt is not None
    values: list[Decimal] = []
    for bound in ("rates_min", "rates_max"):
        rates = {key: Decimal(value) for key, value in pricing[bound].items()}
        total = Decimal(output or 0) * rates["output"]
        if input_complete:
            total += Decimal(hit) * rates["hit"] + Decimal(miss) * rates["miss"]
        elif prompt is not None:
            total += Decimal(prompt) * (min(rates["hit"], rates["miss"]) if bound == "rates_min" else max(rates["hit"], rates["miss"]))
        values.append(total / 1_000_000)
    result["priced_calls"] = int(input_known or output is not None)
    result["incomplete_calls"] = max(result["incomplete_calls"], int(not input_known or output is None))
    result["min_cny"], result["max_cny"] = (format(value, "f") for value in values)
    _status(result)
    return result


def add_cost(target: dict[str, Any], value: dict[str, Any]) -> None:
    for key in ("min_cny", "max_cny"):
        target[key] = format(Decimal(target[key]) + Decimal(value[key]), "f")
    for key in ("priced_calls", "unpriced_calls", "incomplete_calls", "usage_unknown_requests", "request_count_unknown_calls"):
        target[key] += value[key]
    target["pricing_sources"] = sorted(set(target["pricing_sources"]) | set(value["pricing_sources"]))
    _status(target)

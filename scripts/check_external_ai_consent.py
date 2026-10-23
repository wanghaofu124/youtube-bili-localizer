"""Offline external AI consent contract check; no network, model, or WebView.

Run from the repository root with: python scripts/check_external_ai_consent.py
"""
from __future__ import annotations

import os
import time
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

from yblocalizer.workbench_api import WorkbenchHandler
from yblocalizer.translate import ai_provider_request_info

class Jobs:
    def __init__(self) -> None:
        self.job = SimpleNamespace(
            options={"translator": "deepseek", "translate_model": ""},
            artifacts=SimpleNamespace(revision="revision-1"),
        )
    def get(self, _job_id: str):
        return self.job

handler = object.__new__(WorkbenchHandler)
handler.jobs = Jobs()
handler.command = "POST"
responses: list[tuple[dict, HTTPStatus]] = []
handler._json = lambda value, status=HTTPStatus.OK: responses.append((value, status))

assertions = 0
def ok(condition: bool) -> None:
    global assertions
    assert condition
    assertions += 1

def round_trip(path: str, payload: dict) -> None:
    responses.clear()
    original = dict(payload)
    ok(not handler._require_external_ai_consent(path, payload))
    first = responses[-1][0]
    ok(first["code"] == "external_ai_consent_required")
    token = first["consent_id"]
    approved = dict(original, external_ai_consent={"consent_id": token, "approved": True})
    ok(handler._require_external_ai_consent(path, approved))
    ok("external_ai_consent" not in approved)
    ok(not handler._require_external_ai_consent(path, dict(original, external_ai_consent={"consent_id": token, "approved": True})))

for route in (
    "/api/jobs/J/run-all", "/api/jobs/J/resume", "/api/jobs/J/stages/translate/run",
    "/api/jobs/J/stages/render/run", "/api/jobs/J/cues/3/retranslate",
    "/api/jobs/J/cues/compact", "/api/jobs/J/voice-review/candidates/c/translate",
):
    round_trip(route, {})
round_trip("/api/publish/copy", {"job_id": "J", "generate": True})

# Draft creation, non-AI translator, missing options, fixed local Ollama, and multipart upload are exempt.
ok(handler._external_ai_binding("/api/jobs", {"options": {"translator": "deepseek"}}) is None)
handler.jobs.job.options = {"translator": "none"}
ok(handler._external_ai_binding("/api/jobs/J/run-all", {}) is None)
handler.jobs.job.options = {}
ok(handler._external_ai_binding("/api/jobs/J/run-all", {}) is None)
ok(handler._external_ai_binding("/api/materials", {}) is None)
handler.jobs.job.options = {"translator": "deepseek", "translate_model": ""}

# Environment-derived endpoint/model agree with the low-level constructor inputs,
# while display strips credentials/path/query and invalid endpoints are generic.
with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://user:secret@custom.example:9443/v1?x=1", "DEEPSEEK_TRANSLATE_MODEL": "custom-model"}, clear=False):
    provider, model, origin, local = ai_provider_request_info("deepseek", None)
    ok((provider, model, origin, local) == ("deepseek", "custom-model", "https://custom.example:9443", False))
    _, disclosure = handler._external_ai_binding("/api/jobs/J/run-all", {})
    ok(disclosure["endpoint_origin"] == "https://custom.example:9443")
    ok(disclosure["model"] == "custom-model")
with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://openai.example/v1", "OPENAI_TRANSLATE_MODEL": "o-model"}, clear=False):
    ok(ai_provider_request_info("openai", None)[1:] == ("o-model", "https://openai.example", False))
with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "https://user:secret@bad.example:bad/v1"}, clear=False):
    ok(ai_provider_request_info("deepseek", None)[2] == "unknown://configured-endpoint")
ok(ai_provider_request_info("ollama", "qwen3.5:4b")[3] is True)

# Payload and current artifact revision changes invalidate a previously issued binding.
first, _ = handler._external_ai_binding("/api/jobs/J/run-all", {"x": 1})
second, _ = handler._external_ai_binding("/api/jobs/J/run-all", {"x": 2})
ok(first != second)
handler.jobs.job.artifacts.revision = "revision-2"
third, _ = handler._external_ai_binding("/api/jobs/J/run-all", {"x": 1})
ok(first != third)

# Expiry is enforced without waiting.
responses.clear()
handler.jobs.job.artifacts.revision = "revision-1"
ok(not handler._require_external_ai_consent("/api/jobs/J/run-all", {}))
token = responses[-1][0]["consent_id"]
handler._external_consents[token] = (time.monotonic() - 1, handler._external_consents[token][1])
expired = {"external_ai_consent": {"consent_id": token, "approved": True}}
ok(not handler._require_external_ai_consent("/api/jobs/J/run-all", expired))

# The guard returns before a route's downstream mutation/worker hook is called.
downstream_calls = 0
def downstream() -> None:
    global downstream_calls
    downstream_calls += 1
responses.clear()
handler._require_external_ai_consent("/api/jobs/J/run-all", {})
ok(downstream_calls == 0)

# Expired entries are pruned and the in-memory map remains bounded.
for _ in range(handler._external_consent_limit + 8):
    handler._require_external_ai_consent("/api/jobs/J/run-all", {})
ok(len(handler._external_consents) <= handler._external_consent_limit)

print(f"external AI consent contract checks passed ({assertions} assertions)")

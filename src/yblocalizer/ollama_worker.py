"""Cancellable child request for Ollama's OpenAI-compatible local API."""
from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.load(sys.stdin)
    try:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key="ollama-local",
            base_url=str(request["base_url"]),
            timeout=float(request.get("timeout", 600.0)),
            max_retries=0,
            http_client=httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(float(request.get("timeout", 600.0)), connect=5.0),
            ),
        )
        response = client.chat.completions.create(**request["kwargs"])
        content = response.choices[0].message.content or ""
        json.dump({"status": "completed", "content": content}, sys.stdout, ensure_ascii=False)
        return 0
    except Exception as exc:
        json.dump({"status": "failed", "error": str(exc)}, sys.stdout, ensure_ascii=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

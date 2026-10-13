"""Debug-mode NDJSON logger for subtitle pipeline investigations."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time


def agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = "pre",
) -> None:
    configured_path = os.environ.get("YBLOCALIZER_AGENT_DEBUG_LOG")
    if not configured_path:
        return
    # #region agent log
    payload = {
        "sessionId": "2adefb",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    try:
        path = Path(configured_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        return
    # #endregion

"""User glossary: one global file under the output root, optionally overridden per job.

Format: a JSON list of {"source", "target", "note", "aliases"}; only "source" is required.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_GLOSSARY_PATH = Path(__file__).with_name("default_glossary.json")
GLOSSARY_FILE = "glossary.json"
# Where the global glossary lives; the workbench points this at OUTPUT_ROOT.
# None means the job directory's parent (CLI layout).
GLOBAL_ROOT: Path | None = None


def global_glossary_path(output_root: Path) -> Path:
    return Path(output_root) / GLOSSARY_FILE


def read_glossary_file(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_clean(item) for item in entries if str(item.get("source") or "").strip()]


def load_glossary(job_dir: Path) -> list[dict[str, Any]]:
    """Global glossary (or the shipped default) merged with job_dir/glossary.json."""
    job_dir = Path(job_dir)
    global_path = global_glossary_path(GLOBAL_ROOT or job_dir.parent)
    entries = read_glossary_file(global_path if global_path.is_file() else DEFAULT_GLOSSARY_PATH)
    job_path = job_dir / GLOSSARY_FILE
    if job_path.is_file():
        merged = {item["source"].casefold(): item for item in entries}
        for item in read_glossary_file(job_path):
            merged[item["source"].casefold()] = item
        entries = list(merged.values())
    return entries


def save_glossary(path: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = [_clean(item) for item in entries if str(item.get("source") or "").strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned


def glossary_hash(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def glossary_hits(entries: list[dict[str, Any]] | None, text: str) -> list[dict[str, Any]]:
    text = text or ""
    return [item for item in entries or [] if any(_term_in(term, text) for term in [item["source"], *item["aliases"]])]


def _term_in(term: str, text: str) -> bool:
    """ASCII terms match on word boundaries; other terms match as substrings."""
    if term.isascii():
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text, re.I) is not None
    return term.casefold() in text.casefold()


def glossary_lines(entries: list[dict[str, Any]]) -> list[str]:
    lines = []
    for item in entries:
        line = f"- {item['source']}: {item['target']}." if item["target"] else f"- {item['source']}"
        if item["note"]:
            line += f" ({item['note']})"
        lines.append(line)
    return lines


def _clean(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": " ".join(str(item.get("source") or "").split()),
        "target": str(item.get("target") or "").strip(),
        "note": str(item.get("note") or "").strip(),
        "aliases": [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()],
    }

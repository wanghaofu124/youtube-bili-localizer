"""Per-term glossary usage report for finished jobs.

Run from the repository root with:
    python scripts/glossary_report.py <job_dir> [<job_dir> ...] [--out report.csv]

For every glossary term, counts the cues whose source text mentions the term
and how many of those translations contain the term's target.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yblocalizer.glossary import glossary_hits, load_glossary  # noqa: E402


def latest_translation(job_dir: Path) -> Path:
    files = sorted(job_dir.rglob("segments.translated*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise SystemExit(f"{job_dir} 下没有找到 segments.translated*.json")
    return files[-1]


def job_rows(job_dir: Path) -> list[dict[str, object]]:
    glossary = load_glossary(job_dir)
    segments = json.loads(latest_translation(job_dir).read_text(encoding="utf-8"))
    rows = []
    for term in glossary:
        source_cues = [segment for segment in segments if glossary_hits([term], str(segment.get("text") or ""))]
        used = [segment for segment in source_cues if term["target"] in str(segment.get("translated_text") or "")]
        rows.append({
            "job": job_dir.name,
            "source": term["source"],
            "target": term["target"],
            "source_cues": len(source_cues),
            "target_used": len(used),
            "usage_rate": f"{len(used) / len(source_cues):.2%}" if source_cues else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="CSV 输出路径；不填则打印到终端")
    args = parser.parse_args()
    rows = [row for job_dir in args.job_dirs for row in job_rows(job_dir)]
    fields = ["job", "source", "target", "source_cues", "target_used", "usage_rate"]
    if args.out:
        with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写入 {args.out}（{len(rows)} 行）")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

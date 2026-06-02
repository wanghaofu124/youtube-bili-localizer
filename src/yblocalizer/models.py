from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    translated_text: str | None = None

    @property
    def final_text(self) -> str:
        return self.translated_text or self.text


@dataclass(slots=True)
class VideoJob:
    job_id: str
    source: str
    source_kind: str
    work_dir: Path
    title: str | None = None
    description: str | None = None
    raw_video: Path | None = None
    audio: Path | None = None
    source_subtitles: Path | None = None
    translated_subtitles: Path | None = None
    rendered_video: Path | None = None
    license: str | None = None
    view_count: int | None = None
    webpage_url: str | None = None
    thumbnail_url: str | None = None
    thumbnail_path: Path | None = None


@dataclass(slots=True)
class PublishMetadata:
    title: str
    tags: list[str]


def save_segments(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(segment) for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(path: Path) -> list[Segment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(**item) for item in raw]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass"}
DATA_EXTENSIONS = {".json", ".wav"}


@dataclass(slots=True)
class FileUsage:
    path: Path
    size: int
    kind: str


@dataclass(slots=True)
class TaskUsage:
    path: Path
    size: int
    files: list[FileUsage]
    modified_at: float

    @property
    def name(self) -> str:
        return self.path.name


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def scan_outputs(root: Path) -> list[TaskUsage]:
    root = root.resolve()
    if not root.exists():
        return []
    tasks: list[TaskUsage] = []
    # A job owns its direct children; immutable stage-attempt directories are
    # implementation details and must not appear as fake library tasks.
    for directory in sorted([path for path in root.iterdir() if path.is_dir()]):
        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            continue
        if not _looks_like_task_dir(directory, files):
            continue
        file_usages = [FileUsage(path=file, size=file.stat().st_size, kind=classify_file(file)) for file in files]
        modified_at = max(file.stat().st_mtime for file in files)
        tasks.append(TaskUsage(path=directory, size=sum(item.size for item in file_usages), files=file_usages, modified_at=modified_at))
    return sorted(tasks, key=lambda item: item.modified_at, reverse=True)


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "rendered.mp4" or name.startswith("rendered."):
        return "成品视频"
    if suffix in VIDEO_EXTENSIONS:
        return "下载/原视频"
    if suffix in SUBTITLE_EXTENSIONS:
        return "字幕"
    if suffix in DATA_EXTENSIONS:
        return "中间文件"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "预览图"
    return "其他"


def resolve_task_title(directory: Path) -> str:
    """Best-effort human title for a task output folder (never the bare hash alone)."""
    publish = _read_json_object(directory / "publish_metadata.json")
    title = str(publish.get("title") or "").strip()
    if title:
        return title
    manifest = _read_json_object(directory / "job_manifest.json")
    options = manifest.get("options") if isinstance(manifest.get("options"), dict) else {}
    title = str(options.get("title") or "").strip()
    if title:
        return title
    for info in sorted(directory.glob("*.info.json")):
        payload = _read_json_object(info)
        title = str(payload.get("title") or "").strip()
        if title:
            return title
    return directory.name


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def delete_paths(paths: list[Path], allowed_root: Path) -> int:
    allowed_root = allowed_root.resolve()
    deleted_bytes = 0
    for path in paths:
        resolved = path.resolve()
        if resolved == allowed_root or not _is_inside(resolved, allowed_root):
            raise RuntimeError(f"Refusing to delete outside output directory: {resolved}")
        if _contains_reparse_point(path, allowed_root):
            raise RuntimeError(f"Refusing to delete symlink or reparse path: {path}")
        if not resolved.exists():
            continue
        if resolved.is_dir():
            deleted_bytes += _directory_size(resolved)
            shutil.rmtree(resolved)
        else:
            deleted_bytes += resolved.stat().st_size
            resolved.unlink()
    return deleted_bytes


def _looks_like_task_dir(directory: Path, files: list[Path]) -> bool:
    names = {path.name.lower() for path in files}
    suffixes = {path.suffix.lower() for path in files}
    return (
        "rendered.mp4" in names
        or "source.srt" in names
        or "zh.srt" in names
        or "segments.source.json" in names
        or bool(suffixes & VIDEO_EXTENSIONS)
    )


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _contains_reparse_point(path: Path, root: Path) -> bool:
    """Reject links/junctions before a recursive delete can follow them."""
    try:
        current = path
        root = root.resolve()
        while True:
            if current.is_symlink():
                return True
            try:
                attributes = int(current.stat(follow_symlinks=False).st_file_attributes)
            except (AttributeError, OSError):
                attributes = 0
            if attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
                return True
            if current.resolve() == root:
                break
            if current.parent == current:
                break
            current = current.parent
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink():
                    return True
                try:
                    attributes = int(child.stat(follow_symlinks=False).st_file_attributes)
                except (AttributeError, OSError):
                    attributes = 0
                if attributes & 0x400:
                    return True
    except OSError:
        return True
    return False

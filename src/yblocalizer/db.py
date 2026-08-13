"""Lightweight SQLite storage for workbench job history.

The workbench keeps live job state in memory; this module persists a
read-only history so users can find previous tasks after a restart.
The database lives in the user data directory and is never touched by
EXE rebuilds. All access opens a fresh connection per call (thread-safe
enough for the workbench's low write volume).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    data_dir = os.environ.get("YBLOCALIZER_DATA_DIR") or (
        Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"
    )
    return Path(data_dir) / "yblocalizer.db"


_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    material_id TEXT,
    source_url TEXT,
    title TEXT,
    status TEXT,
    stage TEXT,
    progress INTEGER,
    error TEXT,
    output_dir TEXT,
    rendered_video TEXT,
    device TEXT,
    compute_type TEXT,
    options TEXT,
    created_at REAL,
    started_at REAL,
    finished_at REAL,
    owner_pid INTEGER
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_db() -> None:
    with _lock:
        connection = _connect()
        try:
            connection.executescript(_SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "owner_pid" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN owner_pid INTEGER")
            connection.commit()
        finally:
            connection.close()


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def recover_interrupted_jobs(now: float) -> int:
    """Turn jobs abandoned by a terminated desktop process into honest history."""
    with _lock:
        connection = _connect()
        try:
            rows = connection.execute(
                "SELECT id, owner_pid FROM jobs WHERE status IN ('queued', 'running', 'cancelling') AND owner_pid IS NOT NULL"
            ).fetchall()
            abandoned = [job_id for job_id, pid in rows if int(pid) != os.getpid() and not _process_is_alive(int(pid))]
            if not abandoned:
                return 0
            placeholders = ",".join("?" for _ in abandoned)
            cursor = connection.execute(
                f"""UPDATE jobs SET status='interrupted', stage='上次启动被中断',
                    error=COALESCE(error, '应用在任务完成前退出。'), finished_at=?
                    WHERE id IN ({placeholders})""",
                [now, *abandoned],
            )
            connection.commit()
            return int(cursor.rowcount)
        finally:
            connection.close()


def record_job(
    job_id: str,
    material_id: str | None,
    source_url: str | None,
    title: str | None,
    status: str,
    stage: str,
    progress: int,
    error: str | None,
    output_dir: str | None,
    rendered_video: str | None,
    device: str,
    compute_type: str,
    options: dict[str, Any],
    created_at: float | None,
    started_at: float | None,
    finished_at: float | None,
) -> None:
    with _lock:
        connection = _connect()
        try:
            connection.execute(
                """
                INSERT INTO jobs (id, material_id, source_url, title, status, stage, progress,
                                  error, output_dir, rendered_video, device, compute_type, options,
                                  created_at, started_at, finished_at, owner_pid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status, stage=excluded.stage, progress=excluded.progress,
                    error=excluded.error, output_dir=excluded.output_dir,
                    rendered_video=excluded.rendered_video,
                    created_at=excluded.created_at, started_at=excluded.started_at,
                    finished_at=excluded.finished_at, owner_pid=excluded.owner_pid
                """,
                (
                    job_id, material_id, source_url, title, status, stage, progress,
                    error, output_dir, rendered_video, device, compute_type,
                    json.dumps(options, ensure_ascii=False),
                    created_at, started_at, finished_at, os.getpid(),
                ),
            )
            connection.commit()
        finally:
            connection.close()


def list_jobs(limit: int | None = 50) -> list[dict[str, Any]]:
    """Return persisted jobs, newest first.

    ``None`` is deliberately reserved for server-side maintenance operations
    such as clearing the visible history scope.  UI callers should keep using
    a bounded limit.
    """
    with _lock:
        connection = _connect()
        try:
            query = (
                "SELECT id, material_id, source_url, title, status, stage, progress, error, "
                "output_dir, rendered_video, device, compute_type, options, "
                "created_at, started_at, finished_at FROM jobs ORDER BY created_at DESC"
            )
            rows = (
                connection.execute(query).fetchall()
                if limit is None
                else connection.execute(f"{query} LIMIT ?", (limit,)).fetchall()
            )
        finally:
            connection.close()
    columns = [
        "id", "material_id", "source_url", "title", "status", "stage", "progress", "error",
        "output_dir", "rendered_video", "device", "compute_type", "options",
        "created_at", "started_at", "finished_at",
    ]
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row))
        try:
            item["options"] = json.loads(item["options"] or "{}")
        except (ValueError, TypeError):
            item["options"] = {}
        output.append(item)
    return output


def delete_jobs(job_ids: list[str]) -> int:
    """Delete persisted history records only; never delete output files."""
    ids = [job_id for job_id in job_ids if isinstance(job_id, str) and job_id]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with _lock:
        connection = _connect()
        try:
            cursor = connection.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", ids)
            connection.commit()
            return int(cursor.rowcount)
        finally:
            connection.close()

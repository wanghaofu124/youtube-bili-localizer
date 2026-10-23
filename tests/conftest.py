from __future__ import annotations

import os
from pathlib import Path
import sqlite3

import pytest


def _real_database() -> Path:
    root = Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"
    return root / "yblocalizer.db"


def _row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    connection = None
    try:
        connection = sqlite3.connect(str(path))
        return int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
    except sqlite3.Error:
        return 0
    finally:
        if connection is not None:
            connection.close()


@pytest.fixture(scope="session", autouse=True)
def protect_real_user_database() -> None:
    """Fail the suite if a test leaks a task into the real desktop database."""
    path = _real_database()
    before = _row_count(path)
    yield
    after = _row_count(path)
    assert after == before, f"pytest changed the real user database: {before} -> {after} rows"


@pytest.fixture(autouse=True)
def isolated_user_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets private settings, logs, outputs and SQLite storage."""
    root = tmp_path / "user-data"
    output = root / "outputs" / "workbench_demo"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("YBLOCALIZER_DATA_DIR", str(root))

    import yblocalizer.workbench_api as api

    monkeypatch.setattr(api, "USER_DATA_ROOT", root)
    monkeypatch.setattr(api, "LOG_ROOT", logs)
    monkeypatch.setattr(api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(api, "UPLOAD_ROOT", root / "outputs" / "workbench_uploads")
    from yblocalizer import db as job_db
    job_db.init_db()
    return root

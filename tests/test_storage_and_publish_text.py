from pathlib import Path

import pytest

from yblocalizer.publish_text import (
    build_bilibili_description,
    ensure_source_link,
    get_all_templates,
    save_custom_template,
    validate_template,
)
from yblocalizer.storage import delete_paths, resolve_task_title, scan_outputs
from yblocalizer import db as job_db


def test_delete_paths_only_allows_output_root(tmp_path: Path) -> None:
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    inside = allowed / "job" / "file.txt"
    inside.parent.mkdir()
    inside.write_text("123", encoding="utf-8")
    outside = tmp_path / "keep.txt"
    outside.write_text("keep", encoding="utf-8")

    assert delete_paths([inside], allowed) == 3
    assert not inside.exists()
    with pytest.raises(RuntimeError, match="outside output directory"):
        delete_paths([outside], allowed)
    assert outside.exists()


def test_delete_paths_rejects_output_root_itself(tmp_path: Path) -> None:
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    with pytest.raises(RuntimeError):
        delete_paths([allowed], allowed)


def test_resolve_task_title_prefers_publish_then_manifest(tmp_path: Path) -> None:
    task = tmp_path / "abcdef123456"
    task.mkdir()
    assert resolve_task_title(task) == "abcdef123456"

    (task / "job_manifest.json").write_text(
        '{"options": {"title": "Source Title From Manifest"}}',
        encoding="utf-8",
    )
    assert resolve_task_title(task) == "Source Title From Manifest"

    (task / "publish_metadata.json").write_text(
        '{"title": "中文投稿标题", "tags": ["标签"]}',
        encoding="utf-8",
    )
    assert resolve_task_title(task) == "中文投稿标题"


def test_scan_outputs_counts_nested_attempt_files_without_fake_tasks(tmp_path: Path) -> None:
    task = tmp_path / "job-1"
    task.mkdir()
    attempt = task / ".attempt-extract-abc123"
    attempt.mkdir()
    (attempt / "segments.source.json").write_text("{}", encoding="utf-8")
    (attempt / "audio.wav").write_bytes(b"audio")
    tasks = scan_outputs(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].path == task
    assert tasks[0].size == 7
    assert {item.path.name for item in tasks[0].files} == {"segments.source.json", "audio.wav"}


def test_delete_job_history_records_does_not_touch_output_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YBLOCALIZER_DATA_DIR", str(tmp_path / "user-data"))
    job_db.init_db()
    output = tmp_path / "output" / "rendered.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"video")
    job_db.record_job(
        "task-1", None, None, "Example", "completed", "Done", 100, None,
        str(output.parent), str(output), "cpu", "int8", {}, 1, 1, 2,
    )
    assert job_db.delete_jobs(["task-1"]) == 1
    assert job_db.list_jobs() == []
    assert output.exists()


def test_publish_description_has_one_source_link() -> None:
    source = "https://example.com/video"
    description = build_bilibili_description("授权本地化", source)
    assert description.count(source) == 1
    assert ensure_source_link(description, source) == description


def test_custom_template_rejects_unknown_variables() -> None:
    with pytest.raises(ValueError, match="模板变量不支持"):
        validate_template("{source} {secret}")


def test_custom_templates_follow_current_user_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YBLOCALIZER_DATA_DIR", str(tmp_path / "user-data"))
    save_custom_template("测试模板", "来源：{source}")
    assert (tmp_path / "user-data" / "custom_templates.json").exists()
    assert get_all_templates()["测试模板"] == "来源：{source}"

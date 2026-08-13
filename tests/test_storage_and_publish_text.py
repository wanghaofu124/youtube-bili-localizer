from pathlib import Path

import pytest

from yblocalizer.publish_text import (
    build_bilibili_description,
    ensure_source_link,
    get_all_templates,
    save_custom_template,
    validate_template,
)
from yblocalizer.storage import delete_paths


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

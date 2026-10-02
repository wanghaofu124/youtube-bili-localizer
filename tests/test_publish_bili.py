from __future__ import annotations

from pathlib import Path

import pytest

from yblocalizer.publish_bili import (
    _format_upload_state,
    _page_blocker_status,
    _publish_form_timeout_seconds,
    _upload_state_acknowledged,
    _wait_for_upload_acknowledgement,
)


class _FakePage:
    def __init__(self, url: str = "", text: str = "") -> None:
        self.url = url
        self._text = text

    def locator(self, _selector: str):
        page = self

        class _Locator:
            def inner_text(self, timeout: int = 1000) -> str:  # noqa: ARG002
                return page._text

        return _Locator()


def test_publish_form_timeout_scales_for_large_renders(tmp_path: Path) -> None:
    small = tmp_path / "small.mp4"
    large = tmp_path / "large.mp4"
    small.write_bytes(b"0" * 1024)
    large.write_bytes(b"0" * (360 * 1024 * 1024))
    assert _publish_form_timeout_seconds(small) == 600
    assert _publish_form_timeout_seconds(large) >= 1200
    assert _publish_form_timeout_seconds(large) <= 3600


def test_page_blocker_status_detects_login_and_draft() -> None:
    assert _page_blocker_status(_FakePage("https://passport.bilibili.com/login", "扫码登录")) == "login"
    assert _page_blocker_status(_FakePage("https://member.bilibili.com/platform/home", "创作中心 首页")) == "home"
    assert _page_blocker_status(
        _FakePage("https://member.bilibili.com/platform/upload/video/frame", "未提交的视频 继续编辑 不用了")
    ) == "draft"
    assert _page_blocker_status(
        _FakePage("https://member.bilibili.com/platform/upload/video/frame", "上传视频 标题 简介")
    ) is None


def test_upload_state_acknowledged_by_progress_markers() -> None:
    assert _upload_state_acknowledged({"marker": "上传中", "percent": None, "filename_seen": False, "form_ready": False})
    assert _upload_state_acknowledged({"marker": None, "percent": 12, "filename_seen": False, "form_ready": False})
    assert _upload_state_acknowledged(
        {"marker": None, "percent": None, "filename_seen": True, "form_ready": False}
    )
    assert _upload_state_acknowledged(
        {"marker": None, "percent": None, "filename_seen": False, "form_ready": True}
    )
    assert not _upload_state_acknowledged(
        {"marker": None, "percent": None, "filename_seen": False, "form_ready": False}
    )
    assert not _upload_state_acknowledged(
        {"marker": "上传失败", "percent": None, "filename_seen": False, "form_ready": False}
    )


def test_format_upload_state_includes_percent() -> None:
    assert "37%" in _format_upload_state({"marker": "上传中", "percent": 37})
    assert _format_upload_state({}) == ""


def test_wait_for_upload_acknowledgement_fails_fast_without_progress(tmp_path: Path) -> None:
    video = tmp_path / "rendered.mp4"
    video.write_bytes(b"0")
    page = _FakePage(
        "https://member.bilibili.com/platform/upload/video/frame",
        "上传视频 拖拽到此处",
    )
    logs: list[str] = []
    with pytest.raises(RuntimeError, match="未开始上传"):
        _wait_for_upload_acknowledgement(page, video, logs.append, timeout_seconds=1)
    assert any("等待 B 站确认上传开始" in item for item in logs)


def test_wait_for_upload_acknowledgement_accepts_filename(tmp_path: Path) -> None:
    video = tmp_path / "rendered.mp4"
    video.write_bytes(b"0")
    page = _FakePage(
        "https://member.bilibili.com/platform/upload/video/frame",
        "rendered.mp4 上传中 12%",
    )
    logs: list[str] = []
    _wait_for_upload_acknowledgement(page, video, logs.append, timeout_seconds=2)
    assert any("已确认开始处理成片" in item for item in logs)

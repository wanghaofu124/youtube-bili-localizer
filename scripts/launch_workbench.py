"""Native WebView launcher for the local workbench."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import webview

from yblocalizer.workbench_api import ASSET_ROOT, OUTPUT_ROOT, build_server

if sys.platform == "win32":
    # 无控制台（windowed）模式下，ffmpeg / node / ffprobe 等控制台子进程
    # 每次运行都会闪出一个黑色控制台窗口。这里给所有子进程默认加上
    # CREATE_NO_WINDOW，让它们静默运行（同时覆盖 yt-dlp 内部的 ffmpeg/node 调用）。
    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen_init = subprocess.Popen.__init__

    def _popen_init_no_window(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
        _orig_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _popen_init_no_window  # type: ignore[method-assign]


def _ensure_playwright_browsers() -> None:
    """校正 Playwright 浏览器路径。

    部分机器设置了 PLAYWRIGHT_BROWSERS_PATH 但指向空目录（例如指向别的盘），
    导致 B 站发布辅助找不到浏览器。这里检测实际安装位置并修正。
    """
    def has_browsers(root: Path) -> bool:
        return root.exists() and bool(list(root.glob("chromium-*")) or list(root.glob("chromium_headless_shell-*")))

    default_root = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    if has_browsers(default_root):
        if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != str(default_root):
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(default_root)
        return
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured and has_browsers(Path(configured)):
        return
    os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)


_ensure_playwright_browsers()


class DesktopBridge:
    """Small, explicit native-only API used by the local workbench.

    IMPORTANT: attributes used by pywebview's js_api introspection must be
    underscore-prefixed (e.g. ``_server``), otherwise ``get_functions``
    recursively walks the whole object graph (HTTP server -> handler ->
    jobs -> ...) and deadlocks the window at startup.
    """

    def __init__(self, server: object) -> None:
        self._server = server
        self._window: webview.Window | None = None

    def choose_publish_video(self) -> dict[str, str] | None:
        if self._window is None:
            return None
        selected = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Video files (*.mp4;*.mov;*.mkv;*.webm;*.avi;*.m4v)",),
        )
        if not selected:
            return None
        token = self._server.RequestHandlerClass.jobs.register_native_publish_file(Path(selected[0]))
        return {"token": token}

    def choose_material_video(self) -> dict[str, str] | None:
        """Pick a local video and register it by its original path (no copy)."""
        if self._window is None:
            return None
        selected = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            directory=str(OUTPUT_ROOT) if OUTPUT_ROOT.exists() else "",
            file_types=("Video files (*.mp4;*.mov;*.mkv;*.webm;*.avi;*.m4v)",),
        )
        if not selected:
            return None
        try:
            material = self._server.RequestHandlerClass.jobs.add_local_file(Path(selected[0]), authorized=True)
        except Exception:
            return None
        return {"material_id": material.id, "name": material.name}

    def choose_output_dir(self) -> dict[str, str] | None:
        if self._window is None:
            return None
        selected = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not selected:
            return None
        return {"path": str(Path(selected[0]))}

    def choose_cookies_file(self) -> dict[str, str] | None:
        if self._window is None:
            return None
        selected = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Cookies files (*.txt)",),
        )
        if not selected:
            return None
        return {"path": str(Path(selected[0]))}


def main() -> int:
    server = build_server(ASSET_ROOT / "frontend" / "dist", "127.0.0.1", 8765)
    worker = threading.Thread(target=server.serve_forever, daemon=True, name="workbench-api")
    worker.start()
    try:
        bridge = DesktopBridge(server)
        bridge._window = webview.create_window(
            "YouTube Bili Localizer",
            "http://127.0.0.1:8765",
            width=1440,
            height=920,
            min_size=(1024, 720),
            js_api=bridge,
            text_select=True,
            confirm_close=True,
            localization={
                "global.quitConfirmation": "确定要关闭窗口吗？正在进行的任务可能会被中断。",
                "global.ok": "确定",
                "global.quit": "退出",
                "global.cancel": "取消",
                "global.saveFile": "保存文件",
                "windows.fileFilter.allFiles": "所有文件",
                "windows.fileFilter.otherFiles": "其他文件类型",
            },
        )
        # 注意：不要启用 debug=True。它会打开 WebView2 的浏览器快捷键/DevTools，
        # 已观察到处理任务时窗口被异常关闭（仅剩 RADAR 内存预警，无崩溃记录）。
        # 右键复制由前端自定义菜单实现。
        webview.start(gui="edgechromium")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

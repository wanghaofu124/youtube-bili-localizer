from __future__ import annotations

from pathlib import Path
from typing import Callable
import re
import time


UPLOAD_URL = "https://member.bilibili.com/platform/upload/video/frame"
LogFn = Callable[[str], None]
FIELD_SCROLL_POSITIONS = [0, 450, 900, 1350, 1800, 2250]
TITLE_KEYWORDS = ["标题", "稿件标题", "title"]
DESCRIPTION_KEYWORDS = ["简介", "介绍", "描述", "稿件简介", "description", "desc"]
TAG_KEYWORDS = ["标签", "稿件标签", "tag", "tags", "回车"]


def open_upload_page(
    profile_dir: Path | None = None,
    browser: str = "chromium",
    headless: bool = False,
    log: LogFn | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is not installed. Run: pip install -r requirements.txt && playwright install chromium") from exc

    logger = log or _print_log
    browser = _normalize_browser(browser)
    profile_dir = profile_dir or _default_profile_dir(browser)
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = _launch_bilibili_context(playwright, profile_dir=profile_dir, browser=browser, headless=headless)
        page = context.new_page()
        logger(f"Opening Bilibili login/check page with {browser} profile {profile_dir}: {UPLOAD_URL}")
        page_load_started = time.monotonic()
        page.goto(UPLOAD_URL, wait_until="domcontentloaded")
        logger(f"Bilibili upload page loaded in {_elapsed(page_load_started)}.")
        logger("This window is only for login/access checking. It will not select or upload a video.")
        logger("After login is confirmed, close this window and start direct upload or the full publish flow.")
        _wait_until_browser_closed(context)
        context.close()


def assist_publish(
    video_path: Path,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    cover_path: Path | None = None,
    profile_dir: Path | None = None,
    browser: str = "chromium",
    screenshot_path: Path | None = None,
    wait_for_review: bool = True,
    headless: bool = False,
    log: LogFn | None = None,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is not installed. Run: pip install -r requirements.txt && playwright install chromium") from exc

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if cover_path is not None:
        cover_path = cover_path.resolve()
        if not cover_path.exists():
            raise FileNotFoundError(cover_path)
    browser = _normalize_browser(browser)
    profile_dir = profile_dir or _default_profile_dir(browser)
    profile_dir.mkdir(parents=True, exist_ok=True)
    logger = log or _print_log

    with sync_playwright() as playwright:
        context = _launch_bilibili_context(playwright, profile_dir=profile_dir, browser=browser, headless=headless)
        page = context.new_page()
        logger(f"正在打开 B 站投稿页（{browser}）…")
        page_load_started = time.monotonic()
        page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60_000)
        logger(f"投稿页已加载（{_elapsed(page_load_started)}）。当前页面：{page.url}")
        logger("若出现登录页，请先在打开的浏览器里完成登录；程序会自动重试投稿页。")
        _dismiss_existing_draft_prompt(page, logger, wait_seconds=6)
        _ensure_upload_page_ready(page, logger)

        select_started = time.monotonic()
        # Prefer click + Playwright file chooser. Direct set_input_files can
        # succeed on a non-active input without Bilibili starting the upload.
        if _select_video_file(page, video_path, logger):
            pass
        elif _set_video_via_hidden_input(page, video_path, logger):
            pass
        else:
            file_input = _wait_for_upload_input(page, logger)
            file_input.set_input_files(str(video_path))
            logger(f"已通过隐藏文件控件选择成片（{_elapsed(select_started)}）。")
        logger(f"已选择成片：{video_path.name}（{_elapsed(select_started)}）")
        try:
            _wait_for_upload_acknowledgement(page, video_path, logger, timeout_seconds=45)
            publish_form_timeout = _publish_form_timeout_seconds(video_path)
            logger(f"正在等待 B 站上传/转码完成并进入发布表单，最长 {_format_seconds(publish_form_timeout)}…")
            _wait_for_publish_form(page, logger, timeout_seconds=publish_form_timeout)
        except Exception:
            if screenshot_path:
                screenshot_path = screenshot_path.resolve()
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path), full_page=True)
                logger(f"已保存诊断截图：{screenshot_path}")
            raise
        if cover_path:
            cover_started = time.monotonic()
            if _upload_cover(page, cover_path, logger=logger):
                logger(f"已尝试上传封面（{_elapsed(cover_started)}）。")
            else:
                logger("未找到封面上传控件，请手动设置封面。")
        if _fill_first_available(
            page,
            _title_selectors(),
            title,
            logger=logger,
            keywords=TITLE_KEYWORDS,
            candidate_selector="input, textarea, [contenteditable='true'], [contenteditable='plaintext-only']",
        ):
            logger("已填写标题。")
        else:
            logger("未找到标题栏，请手动填写。")
        if description:
            if _fill_first_available(
                page,
                _description_selectors(),
                description,
                logger=logger,
                keywords=DESCRIPTION_KEYWORDS,
                candidate_selector="textarea, [contenteditable='true'], [contenteditable='plaintext-only'], input",
            ):
                logger("已填写简介。")
            else:
                logger("未找到简介栏，请手动填写。")
        if tags:
            if _fill_tags(page, tags, logger=logger):
                logger("已填写标签。")
            else:
                logger("未找到标签栏，请手动填写。")
        if screenshot_path:
            screenshot_path = screenshot_path.resolve()
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
            logger(f"已保存投稿页截图：{screenshot_path}")

        logger("Bilibili upload assistance reached the pre-submit step.")
        logger("视频、封面、标题、简介、标签已尽量填好。尚未正式投稿，请人工检查后手动点「立即投稿」。")
        if wait_for_review:
            logger("浏览器会保持打开。检查/投稿结束后请关闭该 B 站浏览器窗口。")
            _wait_until_browser_closed(context)
        context.close()


def _wait_for_upload_input(page, logger: LogFn):
    started = time.monotonic()
    deadline = time.monotonic() + 600
    last_url = ""
    last_heartbeat = 0.0
    login_seen_at: float | None = None
    while time.monotonic() < deadline:
        status = _page_blocker_status(page)
        if status == "login":
            login_seen_at = login_seen_at or time.monotonic()
            if time.monotonic() - login_seen_at >= 120:
                raise RuntimeError(
                    "B 站投稿页仍需要登录。请先在打开的浏览器窗口完成登录，然后重新点击「打开投稿辅助」。"
                )
        else:
            login_seen_at = None
        try:
            file_input = _find_video_upload_input(page, timeout_ms=3_000)
            logger(f"已找到上传控件（{_elapsed(started)}）。")
            return file_input
        except Exception:
            current_url = ""
            try:
                current_url = page.url
            except Exception:
                pass
            now = time.monotonic()
            if current_url and current_url != last_url:
                logger(f"等待上传控件（{_elapsed(started)}）。当前页面：{current_url}")
                last_url = current_url
                last_heartbeat = now
            elif now - last_heartbeat >= 10:
                hint = {
                    "login": "检测到登录页，请在浏览器中登录。",
                    "draft": "检测到未提交草稿弹窗，正在尝试关闭…",
                    "home": "当前不像投稿页，正在重试打开投稿入口…",
                }.get(status or "", "仍在等待投稿上传控件…")
                logger(f"等待上传控件（{_elapsed(started)}）。{hint}")
                last_heartbeat = now
            _dismiss_existing_draft_prompt(page, logger)
            _try_open_upload_entry(page)
            if "/platform/upload/video" not in (current_url or ""):
                try:
                    retry_started = time.monotonic()
                    page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=30_000)
                    logger(f"已重新打开投稿页（{_elapsed(retry_started)}）。")
                except Exception:
                    pass
            time.sleep(2)
    raise RuntimeError(
        "10 分钟内仍未找到 B 站上传控件。请确认浏览器已登录并可打开创作中心投稿页后重试。"
    )


def _normalize_browser(browser: str) -> str:
    value = (browser or "chromium").strip().lower()
    if value in {"edge", "msedge", "microsoft edge", "microsoft-edge"}:
        return "msedge"
    return "chromium"


def _default_profile_dir(browser: str) -> Path:
    if _normalize_browser(browser) == "msedge":
        return Path("data") / "bilibili-edge-profile"
    return Path("data") / "bilibili-profile"


def _launch_bilibili_context(playwright, profile_dir: Path, browser: str, headless: bool):
    options = {
        "user_data_dir": str(profile_dir),
        "headless": headless,
        "viewport": {"width": 1440, "height": 960},
    }
    if _normalize_browser(browser) == "msedge":
        options["channel"] = "msedge"
    return playwright.chromium.launch_persistent_context(**options)


def _page_blocker_status(page) -> str | None:
    """Return a coarse page state used for progress and early failure hints."""
    try:
        url = (page.url or "").lower()
    except Exception:
        url = ""
    try:
        text = page.locator("body").inner_text(timeout=1_000)
    except Exception:
        text = ""
    compact = text.replace(" ", "")
    if any(marker in url for marker in ("passport.bilibili.com", "login", "account")):
        return "login"
    if any(marker in compact for marker in ("扫码登录", "密码登录", "短信登录", "登录后即可")):
        return "login"
    if any(marker in compact for marker in ("未提交的视频", "继续编辑", "不用了")):
        return "draft"
    if "/platform/upload/video" not in url and any(marker in compact for marker in ("创作中心", "数据中心", "首页")):
        return "home"
    return None


def _ensure_upload_page_ready(page, logger: LogFn) -> None:
    status = _page_blocker_status(page)
    if status == "login":
        logger("当前是登录页：请在打开的浏览器完成登录，程序会继续等待投稿页。")
        return
    if status == "home":
        logger("当前不在投稿页，正在尝试进入上传入口…")
        _try_open_upload_entry(page)
        try:
            page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            pass


def _set_video_via_hidden_input(page, video_path: Path, logger: LogFn) -> bool:
    """Fallback: assign files only to a strict video upload input."""
    try:
        file_input = _find_video_upload_input(page, timeout_ms=4_000)
        file_input.set_input_files(str(video_path))
        logger("已向投稿上传控件写入成片（备用路径）。")
        return True
    except Exception:
        return False


def _select_video_file(page, video_path: Path, logger: LogFn) -> bool:
    started = time.monotonic()
    # Click the visible upload affordance first so Bilibili wires the real chooser.
    click_candidates = [
        "button:has-text('上传视频')",
        "div:has-text('上传视频') button",
        ".upload-btn",
        ".bcc-upload-wrapper",
        "text=上传视频",
    ]
    for selector in click_candidates:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=1_500)
            with page.expect_file_chooser(timeout=4_000) as chooser_info:
                locator.click(timeout=2_000)
            chooser_info.value.set_files(str(video_path))
            logger(f"已通过文件选择器选择成片（{_elapsed(started)}）：{selector}")
            return True
        except Exception:
            continue

    input_candidates = [
        "input[type='file'][name='buploader']",
        ".bcc-upload-wrapper input[type='file']",
        "input[type='file'][accept*='.mp4']",
    ]
    for selector in input_candidates:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="attached", timeout=1_500)
            if not _locator_is_video_upload_input(locator):
                continue
            locator.set_input_files(str(video_path))
            logger(f"已通过控件选择成片（{_elapsed(started)}）：{selector}")
            return True
        except Exception:
            continue
    logger(f"点击选择成片未成功（{_elapsed(started)}），改为等待隐藏文件控件。")
    return False


_VIDEO_UPLOAD_INPUT_JS = """element => {
    const accept = (element.getAttribute("accept") || "").toLowerCase();
    const name = (element.getAttribute("name") || "").toLowerCase();
    if (accept.includes("image") && !accept.includes("video") && !accept.includes(".mp4")) {
        return false;
    }
    if (name === "buploader") {
        return true;
    }
    if (accept.includes(".mp4") || accept.includes("video") || accept.includes(".flv") || accept.includes(".mkv")) {
        return true;
    }
    // Empty accept is only trusted inside the known upload wrapper.
    if (!accept) {
        return !!element.closest(".bcc-upload-wrapper");
    }
    return false;
}"""


def _locator_is_video_upload_input(locator) -> bool:
    try:
        return bool(locator.evaluate(_VIDEO_UPLOAD_INPUT_JS))
    except Exception:
        return False


def _find_video_upload_input(page, timeout_ms: int = 5_000):
    deadline = time.monotonic() + (timeout_ms / 1000)
    last_error: Exception | None = None
    selectors = [
        "input[type='file'][name='buploader']",
        "input[type='file'][accept*='.mp4']",
        ".bcc-upload-wrapper input[type='file']",
        "input[type='file']",
    ]
    while time.monotonic() < deadline:
        for selector in selectors:
            locator = page.locator(selector)
            try:
                count = locator.count()
            except Exception as exc:
                last_error = exc
                continue
            visible_match = None
            attached_match = None
            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if not _locator_is_video_upload_input(candidate):
                        continue
                    if attached_match is None:
                        attached_match = candidate
                    if candidate.evaluate(
                        """element => {
                            const rect = element.getBoundingClientRect();
                            const style = window.getComputedStyle(element);
                            return rect.width > 0
                                && rect.height > 0
                                && style.display !== "none"
                                && style.visibility !== "hidden";
                        }"""
                    ):
                        visible_match = candidate
                        break
                except Exception as exc:
                    last_error = exc
                    continue
            if visible_match is not None:
                return visible_match
            if attached_match is not None:
                return attached_match
        time.sleep(0.2)
    if last_error:
        raise last_error
    raise RuntimeError("No Bilibili video upload input was found.")


def _try_open_upload_entry(page) -> None:
    candidates = [
        "text=发布视频",
        "text=上传视频",
        "text=立即投稿",
        "text=投稿",
        "a[href*='/platform/upload/video']",
    ]
    for selector in candidates:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=800)
            locator.click()
            return
        except Exception:
            continue


def _dismiss_existing_draft_prompt(page, logger: LogFn | None = None, wait_seconds: float = 0) -> bool:
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        try:
            body = page.locator("body").inner_text(timeout=1_000)
        except Exception:
            body = ""
        if "未提交的视频" in body or "继续编辑" in body or "不用了" in body:
            for selector in [
                "button:has-text('不用了')",
                "text=不用了",
                "button:has-text('放弃')",
                "text=放弃",
            ]:
                try:
                    locator = page.locator(selector).first
                    locator.wait_for(state="visible", timeout=800)
                    locator.click(timeout=2_000)
                    time.sleep(1)
                    if logger:
                        logger("Dismissed existing unsubmitted Bilibili draft prompt before starting a new upload.")
                    return True
                except Exception:
                    continue
            try:
                clicked = bool(
                    page.evaluate(
                        """() => {
                            const visible = (element) => {
                                const rect = element.getBoundingClientRect();
                                const style = window.getComputedStyle(element);
                                return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
                            };
                            const candidates = Array.from(document.querySelectorAll("button, a, span, div"));
                            const target = candidates.find((element) => visible(element) && (element.innerText || "").trim() === "不用了");
                            if (!target) return false;
                            (target.closest("button, a, [role='button']") || target).click();
                            return true;
                        }"""
                    )
                )
                if clicked:
                    time.sleep(1)
                    if logger:
                        logger("Dismissed existing unsubmitted Bilibili draft prompt before starting a new upload.")
                    return True
            except Exception:
                pass
            if logger:
                logger("Bilibili shows an existing unsubmitted draft prompt, but the discard button was not found.")
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)
    return False


def _wait_until_browser_closed(context) -> None:
    while context.pages:
        if all(page.is_closed() for page in context.pages):
            return
        time.sleep(1)


def _print_log(message: str) -> None:
    print(message, flush=True)


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"


def _format_seconds(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    if rest:
        return f"{minutes}m {rest}s"
    return f"{minutes}m"


def _publish_form_timeout_seconds(video_path: Path) -> int:
    try:
        size_mb = video_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 600
    # Only used after upload start is acknowledged. Keep headroom for slow links.
    return max(600, min(3600, int(240 + size_mb * 3.0)))


def _wait_for_upload_acknowledgement(
    page,
    video_path: Path,
    logger: LogFn,
    timeout_seconds: int = 45,
) -> None:
    """Fail fast if Bilibili never reacts after a file was assigned."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_heartbeat = 0.0
    video_name = video_path.name
    while time.monotonic() < deadline:
        state = _read_upload_state(page, video_name)
        if _upload_state_acknowledged(state):
            logger(f"B 站已确认开始处理成片（{_elapsed(started)}）。{_format_upload_state(state)}")
            return
        now = time.monotonic()
        if now - last_heartbeat >= 5:
            logger(f"等待 B 站确认上传开始（{_elapsed(started)}）…{_format_upload_state(state)}")
            last_heartbeat = now
        time.sleep(1)
    raise RuntimeError(
        "成片已写入控件，但 B 站未开始上传（约 45 秒内无进度/文件名/转码提示）。"
        "这通常是选错了上传控件或投稿页未就绪，请确认已登录后重新点击「打开投稿辅助」。"
    )


def _wait_for_publish_form(page, logger: LogFn, timeout_seconds: int = 180) -> None:
    started = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    last_url = ""
    last_heartbeat = 0.0
    last_state_key = ""
    while time.monotonic() < deadline:
        if _first_visible(page, _title_selectors(), timeout_ms=1_500):
            logger(f"发布表单已就绪（{_elapsed(started)}）。")
            return
        if _find_field_index(
            page,
            "input, textarea, [contenteditable='true'], [contenteditable='plaintext-only']",
            TITLE_KEYWORDS,
        ) is not None:
            logger(f"发布表单已就绪（{_elapsed(started)}）。")
            return
        if _publish_form_markers_present(page):
            logger(f"发布表单已就绪（{_elapsed(started)}）。")
            return
        current_url = ""
        try:
            current_url = page.url
        except Exception:
            pass
        now = time.monotonic()
        state = _read_upload_state(page)
        state_text = _format_upload_state(state)
        state_key = f"{state.get('marker')}|{state.get('percent')}|{current_url}"
        if current_url and current_url != last_url:
            logger(f"等待发布表单（{_elapsed(started)}）。当前页面：{current_url}{state_text}")
            last_url = current_url
            last_heartbeat = now
            last_state_key = state_key
        elif state_key != last_state_key and state_text:
            logger(f"等待发布表单（{_elapsed(started)}）…{state_text}")
            last_heartbeat = now
            last_state_key = state_key
        elif now - last_heartbeat >= 8:
            logger(f"等待发布表单（{_elapsed(started)}）…{state_text or ' 仍在等待上传/转码'}")
            last_heartbeat = now
        time.sleep(2)
    raise RuntimeError(
        "成片已开始上传，但发布表单长时间未出现。可能仍在上传/转码，或页面有弹窗需要你在浏览器里手动处理。"
    )


def _read_upload_state(page, video_name: str = "") -> dict[str, object]:
    try:
        text = page.locator("body").inner_text(timeout=800)
    except Exception:
        text = ""
    marker = None
    for candidate in ("上传完成", "转码中", "上传中", "处理中", "解析中", "上传失败", "重新上传"):
        if candidate in text:
            marker = candidate
            break
    percent = None
    match = re.search(r"(\d{1,3})\s*%", text)
    if match:
        value = int(match.group(1))
        if 0 <= value <= 100:
            percent = value
    filename_seen = False
    if video_name:
        stem = Path(video_name).stem
        filename_seen = video_name in text or (bool(stem) and stem in text)
    return {
        "marker": marker,
        "percent": percent,
        "filename_seen": filename_seen,
        "form_ready": _publish_form_markers_present(page),
        "text_excerpt": text[:240],
    }


def _upload_state_acknowledged(state: dict[str, object]) -> bool:
    marker = state.get("marker")
    if marker in {"上传失败"}:
        return False
    if marker in {"上传完成", "转码中", "上传中", "处理中", "解析中", "重新上传"}:
        return True
    if state.get("percent") is not None:
        return True
    if state.get("filename_seen"):
        return True
    if state.get("form_ready"):
        return True
    return False


def _format_upload_state(state: dict[str, object]) -> str:
    parts: list[str] = []
    marker = state.get("marker")
    percent = state.get("percent")
    if marker:
        parts.append(str(marker))
    if percent is not None:
        parts.append(f"{percent}%")
    if state.get("filename_seen") and not parts:
        parts.append("已显示成片文件名")
    if not parts:
        return ""
    return " 状态：" + " / ".join(parts)


def _publish_form_markers_present(page) -> bool:
    try:
        text = page.locator("body").inner_text(timeout=1_000)
    except Exception:
        return False
    required_groups = [
        ["基本设置", "发布设置"],
        ["标题", "稿件标题"],
        ["简介", "稿件简介", "描述"],
    ]
    if not all(any(marker in text for marker in group) for group in required_groups):
        return False
    upload_markers = ["上传中", "上传完成", "封面", "封面设置", "分区", "创作声明", "立即投稿", "存草稿"]
    return any(marker in text for marker in upload_markers)


def _title_selectors() -> list[str]:
    return [
        "input[placeholder*='标题']",
        "textarea[placeholder*='标题']",
        "input[aria-label*='标题']",
        "textarea[aria-label*='标题']",
        "input[name*='title']",
        "textarea[name*='title']",
        ".title input",
        ".title textarea",
        ".video-title input",
    ]


def _description_selectors() -> list[str]:
    return [
        "textarea[placeholder*='简介']",
        "textarea[placeholder*='介绍']",
        "textarea[placeholder*='描述']",
        "textarea[aria-label*='简介']",
        "textarea[aria-label*='介绍']",
        "textarea[aria-label*='描述']",
        "textarea[name*='desc']",
        "textarea[name*='description']",
        "div[contenteditable='true'][placeholder*='简介']",
        "[contenteditable='true'][data-placeholder*='简介']",
        "[contenteditable='true'][aria-label*='简介']",
        "[contenteditable='plaintext-only'][data-placeholder*='简介']",
        ".ql-editor[contenteditable='true']",
        ".ql-container .ql-editor",
    ]


def _tag_selectors() -> list[str]:
    return [
        "input[placeholder*='标签']",
        "input[aria-label*='标签']",
        "input[name*='tag']",
        "input[name*='tags']",
        "input[placeholder*='tag']",
        "input[placeholder*='Tag']",
        "input[placeholder*='回车']",
        ".tag-input input",
        ".tag-area input",
        ".tag-container input",
    ]


def _fill_first_available(
    page,
    selectors: list[str],
    value: str,
    logger: LogFn | None = None,
    keywords: list[str] | None = None,
    candidate_selector: str = "input, textarea, [contenteditable='true'], [contenteditable='plaintext-only']",
) -> bool:
    if _try_fill_selectors(page, selectors, value, timeout_ms=150):
        return True
    if keywords and _fill_by_keywords(page, value, keywords, candidate_selector, logger=logger):
        return True
    for scroll_y in FIELD_SCROLL_POSITIONS[1:]:
        try:
            page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
            time.sleep(0.25)
        except Exception:
            pass
        if _try_fill_selectors(page, selectors, value, timeout_ms=80):
            return True
        if keywords and _fill_by_keywords(page, value, keywords, candidate_selector, logger=logger):
            return True
    return False


def _try_fill_selectors(page, selectors: list[str], value: str, timeout_ms: int) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=timeout_ms)
            _fill_locator(page, locator, value)
            return True
        except Exception:
            continue
    return False


def _fill_tags(page, tags: list[str], logger: LogFn | None = None) -> bool:
    cleaned_tags = [tag.strip() for tag in tags if tag.strip()]
    if not cleaned_tags:
        return True
    joined_tags = " ".join(cleaned_tags)
    for scroll_y in [600, 1000, 1400, 1800, 2200]:
        try:
            page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
            time.sleep(0.4)
        except Exception:
            pass
        for selector in _tag_selectors():
            locator = page.locator(selector).first
            try:
                locator.wait_for(state="visible", timeout=250)
                locator.scroll_into_view_if_needed(timeout=2_000)
                locator.click()
                _clear_existing_tags(page)
                for tag in cleaned_tags:
                    page.keyboard.type(tag, delay=30)
                    page.keyboard.press("Enter")
                    time.sleep(0.3)
                return True
            except Exception as exc:
                _ = exc
                continue
    if _fill_tag_input_by_keywords(page, cleaned_tags, logger=logger):
        return True
    return _fill_first_available(
        page,
        _tag_selectors(),
        joined_tags,
        logger=logger,
        keywords=TAG_KEYWORDS,
        candidate_selector="input",
    )


def _upload_cover(page, cover_path: Path, logger: LogFn | None = None) -> bool:
    if _set_cover_file_input(page, cover_path, timeout_ms=300):
        _confirm_cover_dialog(page, logger=logger)
        return True
    for scroll_y in [0, 350, 700, 1050, 1400]:
        try:
            page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
            time.sleep(0.25)
        except Exception:
            pass
        if _choose_cover_file(page, cover_path):
            _confirm_cover_dialog(page, logger=logger)
            return True
        if _click_cover_entry(page):
            time.sleep(0.6)
            if _set_cover_file_input(page, cover_path, timeout_ms=2_000):
                _confirm_cover_dialog(page, logger=logger)
                return True
            if _choose_cover_file(page, cover_path):
                _confirm_cover_dialog(page, logger=logger)
                return True
        if _set_cover_file_input(page, cover_path, timeout_ms=200):
            _confirm_cover_dialog(page, logger=logger)
            return True
    return False


def _choose_cover_file(page, cover_path: Path) -> bool:
    try:
        with page.expect_file_chooser(timeout=1_500) as chooser_info:
            if not _click_cover_entry(page):
                return False
        chooser_info.value.set_files(str(cover_path))
        return True
    except Exception:
        return False


def _set_cover_file_input(page, cover_path: Path, timeout_ms: int = 500) -> bool:
    selectors = [
        "input[type='file'][accept*='image']",
        "input[type='file'][accept*='jpg']",
        "input[type='file'][accept*='jpeg']",
        "input[type='file'][accept*='png']",
        "input[type='file'][name*='cover']",
        "input[type='file'][class*='cover']",
    ]
    for selector in selectors:
        locators = page.locator(selector)
        try:
            count = min(locators.count(), 6)
        except Exception:
            count = 1
        for index in range(count):
            locator = locators.nth(index)
            try:
                locator.wait_for(state="attached", timeout=timeout_ms)
                accept = (locator.get_attribute("accept") or "").lower()
                name = (locator.get_attribute("name") or "").lower()
                class_name = (locator.get_attribute("class") or "").lower()
                if "video" in accept or "buploader" in name:
                    continue
                if not any(token in f"{accept} {name} {class_name}" for token in ["image", "jpg", "jpeg", "png", "cover"]):
                    continue
                locator.set_input_files(str(cover_path))
                return True
            except Exception:
                continue
    return False


def _click_cover_entry(page) -> bool:
    selectors = [
        "text=封面设置",
        "text=上传封面",
        "text=更换封面",
        "text=编辑封面",
        "text=选择封面",
        "button:has-text('封面设置')",
        "button:has-text('上传封面')",
        "button:has-text('更换封面')",
        "button:has-text('编辑封面')",
        "button:has-text('选择封面')",
        "div:has-text('封面设置')",
        "span:has-text('封面设置')",
        "[class*='cover']:has-text('封面')",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=250)
            locator.scroll_into_view_if_needed(timeout=1_000)
            locator.click(timeout=1_000)
            return True
        except Exception:
            continue
    try:
        return bool(
            page.evaluate(
                """() => {
                    const keywords = ["封面设置", "上传封面", "更换封面", "编辑封面", "选择封面"];
                    const elements = Array.from(document.querySelectorAll("button, div, span, a, i, svg"));
                    const visible = (element) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
                    };
                    const element = elements.find((item) => visible(item) && keywords.some((keyword) => item.innerText?.includes(keyword) || item.getAttribute("aria-label")?.includes(keyword)));
                    if (!element) return false;
                    const clickable = element.closest("button, a, [role='button'], [class*='cover'], [class*='upload']") || element;
                    clickable.scrollIntoView({ block: "center", inline: "nearest" });
                    clickable.click();
                    return true;
                }"""
            )
        )
    except Exception:
        return False


def _confirm_cover_dialog(page, logger: LogFn | None = None) -> None:
    for _ in range(8):
        time.sleep(0.5)
        for selector in [
            "button:has-text('完成')",
            "button:has-text('确认')",
            "button:has-text('确定')",
            "button:has-text('保存')",
            "text=完成",
            "text=确认",
            "text=确定",
            "text=保存",
        ]:
            try:
                locator = page.locator(selector).last
                locator.wait_for(state="visible", timeout=150)
                locator.click(timeout=1_000)
                return
            except Exception:
                continue
    if logger:
        logger("Cover file was selected; if Bilibili opened a crop dialog, please confirm it manually.")


def _first_visible(page, selectors: list[str], timeout_ms: int = 1_000) -> bool:
    for selector in selectors:
        try:
            page.locator(selector).first.wait_for(state="visible", timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def _fill_locator(page, locator, value: str) -> None:
    locator.scroll_into_view_if_needed(timeout=2_000)
    locator.click()
    time.sleep(0.2)
    tag_name = locator.evaluate("element => element.tagName.toLowerCase()")
    contenteditable = locator.evaluate("element => element.isContentEditable")
    if tag_name in {"input", "textarea"}:
        try:
            locator.fill(value)
            return
        except Exception:
            pass
    if contenteditable:
        locator.evaluate(
            """(element, value) => {
                element.focus();
                element.textContent = value;
                element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
            value,
        )
        return
    page.keyboard.press("Control+a")
    page.keyboard.press("Delete")
    page.keyboard.type(value, delay=20)


def _fill_by_keywords(
    page,
    value: str,
    keywords: list[str],
    candidate_selector: str,
    logger: LogFn | None = None,
) -> bool:
    index = _find_field_index(page, candidate_selector, keywords)
    if index is None:
        return False
    if _fill_field_by_index_js(page, candidate_selector, index, value):
        return True
    try:
        locator = page.locator(candidate_selector).nth(index)
        _fill_locator(page, locator, value)
        return True
    except Exception as exc:
        if logger:
            logger(f"Keyword field fill failed ({type(exc).__name__}).")
        return False


def _fill_field_by_index_js(page, candidate_selector: str, index: int, value: str) -> bool:
    try:
        return bool(
            page.evaluate(
                """({ candidateSelector, index, value }) => {
                    const element = Array.from(document.querySelectorAll(candidateSelector))[index];
                    if (!element) return false;
                    element.scrollIntoView({ block: "center", inline: "nearest" });
                    element.focus();

                    if (element.isContentEditable) {
                        element.textContent = value;
                    } else if (element.tagName === "TEXTAREA") {
                        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
                        setter ? setter.call(element, value) : element.value = value;
                    } else if (element.tagName === "INPUT") {
                        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
                        setter ? setter.call(element, value) : element.value = value;
                    } else {
                        element.textContent = value;
                    }

                    element.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
                    element.dispatchEvent(new Event("change", { bubbles: true }));
                    element.blur();
                    return true;
                }""",
                {"candidateSelector": candidate_selector, "index": index, "value": value},
            )
        )
    except Exception:
        return False


def _fill_tag_input_by_keywords(page, tags: list[str], logger: LogFn | None = None) -> bool:
    index = _find_field_index(page, "input", TAG_KEYWORDS)
    if index is None:
        return False
    try:
        locator = page.locator("input").nth(index)
        locator.scroll_into_view_if_needed(timeout=2_000)
        locator.click()
        _clear_existing_tags(page)
        for tag in tags:
            page.keyboard.type(tag, delay=30)
            page.keyboard.press("Enter")
            time.sleep(0.3)
        return True
    except Exception as exc:
        if logger:
            logger(f"Keyword tag fill failed ({type(exc).__name__}).")
        return False


def _clear_existing_tags(page) -> None:
    try:
        page.keyboard.press("Control+a")
        page.keyboard.press("Delete")
        for _ in range(12):
            page.keyboard.press("Backspace")
            time.sleep(0.05)
    except Exception:
        pass
    try:
        page.evaluate(
            """() => {
                const visible = (element) => {
                    const rect = element.getBoundingClientRect();
                    const style = window.getComputedStyle(element);
                    return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
                };
                const tagLabels = Array.from(document.querySelectorAll("label, div, span"))
                    .filter((element) => visible(element) && /标签|tag/i.test(element.innerText || ""));
                const areas = tagLabels
                    .map((element) => element.closest("section, form, .form-item, .bcc-form-item, .tag, .tag-container, .tag-area") || element.parentElement)
                    .filter(Boolean);
                const candidates = Array.from(document.querySelectorAll("button, i, svg, span, a"))
                    .filter((element) => visible(element) && ["×", "x", "X"].includes((element.innerText || element.getAttribute("aria-label") || "").trim()));
                for (const close of candidates) {
                    const inTagArea = areas.some((area) => area.contains(close));
                    const classText = `${close.className || ""} ${close.parentElement?.className || ""}`.toLowerCase();
                    if (inTagArea || /tag|close|remove/.test(classText)) {
                        (close.closest("button, a, [role='button']") || close).click();
                    }
                }
            }"""
        )
    except Exception:
        pass


def _find_field_index(page, candidate_selector: str, keywords: list[str]) -> int | None:
    try:
        index = page.evaluate(
            """({ candidateSelector, keywords }) => {
                const candidates = Array.from(document.querySelectorAll(candidateSelector));
                const loweredKeywords = keywords.map((keyword) => String(keyword).toLowerCase());

                const isVisible = (element) => {
                    const rect = element.getBoundingClientRect();
                    const style = window.getComputedStyle(element);
                    return rect.width > 0
                        && rect.height > 0
                        && style.display !== "none"
                        && style.visibility !== "hidden"
                        && !element.disabled
                        && element.getAttribute("aria-hidden") !== "true";
                };

                const attrText = (element) => [
                    element.getAttribute("placeholder"),
                    element.getAttribute("aria-label"),
                    element.getAttribute("name"),
                    element.getAttribute("id"),
                    element.getAttribute("class"),
                    element.getAttribute("data-placeholder"),
                    element.getAttribute("title"),
                    element.getAttribute("data-testid"),
                ].filter(Boolean).join(" ");

                const labelText = (element) => {
                    const parts = [];
                    const closestLabel = element.closest("label");
                    if (closestLabel) parts.push(closestLabel.innerText || closestLabel.textContent || "");
                    if (element.labels) {
                        for (const label of Array.from(element.labels)) {
                            parts.push(label.innerText || label.textContent || "");
                        }
                    }
                    const labelledBy = element.getAttribute("aria-labelledby");
                    if (labelledBy) {
                        for (const id of labelledBy.split(/\\s+/)) {
                            const label = document.getElementById(id);
                            if (label) parts.push(label.innerText || label.textContent || "");
                        }
                    }
                    return parts.join(" ");
                };

                const nearbyText = (element) => {
                    const parts = [];
                    let sibling = element.previousElementSibling;
                    for (let i = 0; sibling && i < 3; i += 1) {
                        parts.push(sibling.innerText || sibling.textContent || "");
                        sibling = sibling.previousElementSibling;
                    }
                    let parent = element.parentElement;
                    for (let i = 0; parent && i < 2; i += 1) {
                        const text = parent.innerText || parent.textContent || "";
                        if (text.length < 500) parts.push(text);
                        parent = parent.parentElement;
                    }
                    return parts.join(" ");
                };

                const scoreText = (text, weight) => {
                    const loweredText = String(text).toLowerCase();
                    let score = 0;
                    for (const keyword of loweredKeywords) {
                        if (keyword && loweredText.includes(keyword)) score += weight;
                    }
                    return score;
                };

                const scored = candidates.map((element, index) => {
                    if (!isVisible(element)) return { index, score: 0, top: Number.MAX_SAFE_INTEGER };
                    const rect = element.getBoundingClientRect();
                    const score =
                        scoreText(attrText(element), 20)
                        + scoreText(labelText(element), 12)
                        + scoreText(nearbyText(element), 4);
                    return { index, score, top: rect.top };
                }).filter((item) => item.score > 0);

                scored.sort((left, right) => {
                    if (right.score !== left.score) return right.score - left.score;
                    return left.top - right.top;
                });
                return scored.length ? scored[0].index : null;
            }""",
            {"candidateSelector": candidate_selector, "keywords": keywords},
        )
        return None if index is None else int(index)
    except Exception:
        return None

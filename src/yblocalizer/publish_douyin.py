from __future__ import annotations

from pathlib import Path
from typing import Callable
import re
import time


UPLOAD_URL = "https://creator.douyin.com/creator-micro/content/upload"
LogFn = Callable[[str], None]
_FIELD_SELECTOR = "input:not([type='file']):not([type='checkbox']):not([type='radio']), textarea, [contenteditable='true'], [contenteditable='plaintext-only']"


def assist_publish(
    video_path: Path,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    profile_dir: Path | None = None,
    browser: str = "chromium",
    wait_for_review: bool = True,
    headless: bool = False,
    log: LogFn | None = None,
) -> None:
    """Upload one video to Douyin's creator form and fill recognized fields.

    The routine deliberately stops before any final publish or submit action.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright is not installed. Run: pip install -r requirements.txt && playwright install chromium") from exc

    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    browser = _normalize_browser(browser)
    profile_dir = profile_dir or _default_profile_dir(browser)
    profile_dir.mkdir(parents=True, exist_ok=True)
    logger = log or _print_log

    with sync_playwright() as playwright:
        context = _launch_context(playwright, profile_dir, browser, headless)
        video_selected = False
        try:
            page = context.new_page()
            logger(f"正在打开抖音创作者中心投稿页（{browser}）：{UPLOAD_URL}")
            page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60_000)
            _wait_for_login(page, logger, timeout_seconds=180)
            if UPLOAD_URL.rstrip("/") not in (page.url or "").rstrip("/"):
                page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60_000)

            logger("如页面要求登录，请在这个抖音专用浏览器窗口完成登录；登录状态会保存在独立 Profile 中。")
            started = time.monotonic()
            _select_video_file(page, video_path, logger, timeout_seconds=300)
            video_selected = True
            logger(f"已选择成片：{video_path.name}（{_elapsed(started)}）。等待抖音接收并处理视频…")
            _wait_for_caption_field(page, logger, video_path, timeout_seconds=_upload_timeout(video_path))

            caption = _caption_with_tags(description, tags or [])
            title_field = _find_field(page, ("标题", "作品标题", "作品名称"), exclude=("描述", "简介"))
            caption_field = _find_field(page, ("作品描述", "描述", "简介", "文案", "添加作品描述", "添加描述"))
            if title_field is not None:
                _fill(title_field, title)
                logger("已填写标题。")
            else:
                logger("当前页面没有识别到独立标题栏；标题会并入作品描述（若描述栏可用）。")
                if caption_field is not None and title.strip():
                    caption = f"{title.strip()}\n\n{caption}" if caption else title.strip()

            if caption and caption_field is not None:
                _fill(caption_field, caption)
                logger("已填写作品描述和话题标签。请在抖音页面确认话题是否已被识别为可点击话题。")
            elif caption:
                logger("未识别到作品描述栏；请在抖音页面手动粘贴标题、正文和话题。")

            logger("Douyin upload assistance reached the pre-submit step.")
            logger("成片和可识别的文案已尽量填好；尚未正式投稿。请检查封面、可见范围、声明及话题，再手动发布。")
            if wait_for_review:
                logger("抖音浏览器会保持打开。检查/发布结束后请关闭该窗口。")
                _wait_until_browser_closed(context)
        except Exception:
            if video_selected and wait_for_review:
                logger("自动填表遇到问题；抖音浏览器会保留供你检查和手动继续。尚未正式投稿。")
                _wait_until_browser_closed(context)
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def _normalize_browser(browser: str) -> str:
    value = (browser or "chromium").strip().lower()
    return "msedge" if value in {"edge", "msedge", "microsoft edge", "microsoft-edge"} else "chromium"


def _default_profile_dir(browser: str) -> Path:
    return Path("data") / ("douyin-edge-profile" if _normalize_browser(browser) == "msedge" else "douyin-profile")


def _launch_context(playwright, profile_dir: Path, browser: str, headless: bool):
    options = {
        "user_data_dir": str(profile_dir),
        "headless": headless,
        "viewport": {"width": 1440, "height": 960},
    }
    if browser == "msedge":
        options["channel"] = "msedge"
    return playwright.chromium.launch_persistent_context(**options)


def _wait_for_login(page, logger: LogFn, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    login_seen = False
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        if not _login_required(page):
            return
        if not login_seen:
            logger("抖音页面需要登录，请在打开的浏览器窗口完成登录。")
            login_seen = True
        if time.monotonic() - last_heartbeat >= 10:
            logger("等待抖音登录完成…")
            last_heartbeat = time.monotonic()
        time.sleep(2)
    if login_seen:
        raise RuntimeError("等待抖音登录超时。请完成登录后重新启动投稿辅助。")


def _login_required(page) -> bool:
    try:
        url = (page.url or "").lower()
        text = page.locator("body").inner_text(timeout=700).replace(" ", "")
    except Exception:
        return False
    return any(marker in url for marker in ("login", "passport", "sso")) or any(
        marker in text for marker in ("扫码登录", "验证码登录", "手机号登录", "登录后继续", "登录抖音")
    )


def _select_video_file(page, video_path: Path, logger: LogFn, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        if _login_required(page):
            _wait_for_login(page, logger, timeout_seconds=180)
            if UPLOAD_URL.rstrip("/") not in (page.url or "").rstrip("/"):
                page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60_000)
        locator = _video_input(page)
        if locator is not None:
            locator.set_input_files(str(video_path))
            logger("已找到抖音视频上传控件。")
            return
        if _open_video_picker(page, video_path):
            return
        now = time.monotonic()
        if now - last_heartbeat >= 10:
            logger(f"等待抖音上传控件… 当前页面：{page.url}")
            last_heartbeat = now
        time.sleep(2)
    raise RuntimeError("抖音投稿页中没有找到明确的视频上传控件。请确认已登录且页面已加载，再重新启动投稿辅助。")


def _video_input(page):
    try:
        inputs = page.locator("input[type='file']")
        for index in range(inputs.count()):
            locator = inputs.nth(index)
            try:
                details = locator.evaluate(
                    """element => {
                      const accept = (element.getAttribute('accept') || '').toLowerCase();
                      const parent = element.closest('label, [role=button], button, [class*=upload]') || element.parentElement;
                      const nearby = (parent?.innerText || parent?.getAttribute('aria-label') || '').toLowerCase();
                      return { accept, nearby };
                    }"""
                )
            except Exception:
                continue
            accept = details.get("accept", "")
            nearby = details.get("nearby", "")
            if "video" in accept or any(ext in accept for ext in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")) or any(
                marker in nearby for marker in ("视频", "上传视频")
            ):
                return locator
    except Exception:
        pass
    return None


def _open_video_picker(page, video_path: Path) -> bool:
    selectors = [
        "button:has-text('上传视频')", "button:has-text('选择视频')", "button:has-text('上传作品')",
        "[role='button']:has-text('上传视频')", "[role='button']:has-text('选择视频')",
        "text=上传视频", "text=选择视频", "text=点击上传视频", "text=上传作品",
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.is_visible(timeout=150):
                locator.scroll_into_view_if_needed(timeout=1_000)
                with page.expect_file_chooser(timeout=1_500) as chooser_info:
                    locator.click(timeout=1_500)
                chooser_info.value.set_files(str(video_path))
                return True
        except Exception:
            continue
    return False


def _wait_for_caption_field(page, logger: LogFn, video_path: Path, timeout_seconds: int) -> None:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_heartbeat = 0.0
    while time.monotonic() < deadline:
        if _find_field(page, ("作品描述", "描述", "简介", "文案", "添加作品描述", "添加描述")) is not None or _find_field(
            page, ("标题", "作品标题", "作品名称"), exclude=("描述", "简介")
        ) is not None:
            logger(f"抖音作品编辑表单已出现（{_elapsed(started)}）。")
            return
        try:
            body = page.locator("body").inner_text(timeout=800)
        except Exception:
            body = ""
        if "上传失败" in body:
            raise RuntimeError("抖音页面提示视频上传失败。请检查浏览器中的具体提示后重试。")
        now = time.monotonic()
        if now - last_heartbeat >= 10:
            marker = _upload_marker(body, video_path.name)
            logger(f"等待抖音接收/处理视频并显示编辑表单（{_elapsed(started)}）{marker}")
            last_heartbeat = now
        time.sleep(2)
    raise RuntimeError("视频已交给抖音页面，但编辑表单迟迟未出现。请在浏览器检查上传状态；不会自动发布。")


def _find_field(page, keywords: tuple[str, ...], exclude: tuple[str, ...] = ()):
    try:
        fields = page.locator(_FIELD_SELECTOR)
        fallback = None
        fallback_score = 0
        for index in range(fields.count()):
            locator = fields.nth(index)
            if not locator.is_visible(timeout=100):
                continue
            metadata = locator.evaluate(
                """element => {
                  const direct = [element.getAttribute('placeholder'), element.getAttribute('aria-label'), element.getAttribute('name'), element.getAttribute('data-placeholder')].filter(Boolean).join(' ');
                  const labels = [Array.from(element.labels || []).map(label => label.innerText || '').join(' '), element.closest('label')?.innerText || ''].filter(Boolean).join(' ');
                  const parent = element.parentElement?.innerText || '';
                  return { direct, labels, parent };
                }"""
            )
            normalize = lambda value: re.sub(r"\s+", "", str(value)).lower()
            direct, labels, parent = (normalize(metadata.get(key, "")) for key in ("direct", "labels", "parent"))
            if any(normalize(item) in direct or normalize(item) in labels for item in exclude):
                continue
            if any(normalize(item) in direct for item in keywords):
                return locator
            if any(normalize(item) in labels for item in keywords) and fallback_score < 3:
                fallback, fallback_score = locator, 3
            elif len(parent) <= 80 and any(normalize(item) in parent for item in keywords) and fallback_score < 1:
                fallback, fallback_score = locator, 1
        return fallback
    except Exception:
        return None


def _caption_with_tags(description: str, tags: list[str]) -> str:
    caption = description.strip()
    hashtags = [f"#{tag.strip().lstrip('#')}" for tag in tags if tag.strip().lstrip("#")]
    if hashtags:
        joined = " ".join(hashtags)
        caption = f"{caption}\n\n{joined}" if caption else joined
    return caption


def _fill(locator, value: str) -> None:
    locator.scroll_into_view_if_needed(timeout=2_000)
    locator.click(timeout=2_000)
    try:
        locator.fill(value, timeout=3_000)
    except Exception:
        locator.press("Control+a")
        locator.type(value, delay=10)


def _upload_timeout(video_path: Path) -> int:
    try:
        size_mb = video_path.stat().st_size / (1024 * 1024)
    except OSError:
        return 1800
    return max(900, min(5400, int(300 + size_mb * 4)))


def _upload_marker(body: str, filename: str) -> str:
    for marker in ("上传完成", "上传中", "处理中", "转码中", "解析中", "上传失败"):
        if marker in body:
            return f"；页面状态：{marker}"
    match = re.search(r"\d{1,3}\s*%", body)
    if match:
        return f"；页面进度：{match.group(0)}"
    if filename and Path(filename).stem in body:
        return "；页面已显示文件名"
    return ""


def _wait_until_browser_closed(context) -> None:
    while context.pages:
        if all(page.is_closed() for page in context.pages):
            return
        time.sleep(1)


def _elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"


def _print_log(message: str) -> None:
    print(message, flush=True)

from __future__ import annotations

from pathlib import Path
from string import Formatter
import json
import os
import re


def _custom_templates_file() -> Path:
    # Resolve on demand instead of during module import.  The workbench sets
    # YBLOCALIZER_DATA_DIR during startup, and this keeps template storage out
    # of whichever directory happened to launch the EXE.
    data_dir = os.environ.get("YBLOCALIZER_DATA_DIR") or (
        Path(os.environ.get("APPDATA") or Path.home()) / "YouTubeBiliLocalizer"
    )
    path = Path(data_dir) / "custom_templates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


ALLOWED_TEMPLATE_FIELDS = {"source", "custom_text"}
PUBLISH_PLATFORMS = ("bilibili", "douyin")

# B 站默认文案：保留出处占位，便于授权转载说明。
DEFAULT_BILIBILI_TEMPLATES = {
    "授权本地化": (
        "已获授权进行中文本地化与字幕制作。\n"
        "原视频链接：{source}\n"
        "原作者/来源：请在发布前补充\n"
        "说明：本视频仅用于授权分享与学习交流。"
    ),
    "字幕翻译分享": (
        "本视频为授权素材的中文字幕版本，已完成转写、翻译和字幕压制。\n"
        "原视频链接：{source}\n"
        "如需了解原内容，请访问原视频页面。"
    ),
    "学习笔记": (
        "学习笔记与中文字幕整理。\n"
        "原视频链接：{source}\n"
        "翻译可能存在细节偏差，欢迎友好指正。"
    ),
    "技术教程翻译": (
        "【中文字幕】技术教程翻译与本地化。\n"
        "原视频链接：{source}\n"
        "本翻译仅供学习参考，版权归原作者所有。"
    ),
    "影视剪辑分享": (
        "精彩片段中文字幕版。\n"
        "原视频链接：{source}\n"
        "剪辑与字幕制作：AI 辅助完成，仅供爱好者交流。"
    ),
    "播客/访谈翻译": (
        "【中文字幕】播客/访谈内容翻译。\n"
        "原视频链接：{source}\n"
        "内容为嘉宾个人观点，翻译如有不准确之处欢迎指正。"
    ),
    "游戏实况翻译": (
        "【中文字幕】游戏实况/攻略翻译。\n"
        "原视频链接：{source}\n"
        "游戏画面与音频版权归原作者/发行商所有。"
    ),
    "自定义模板": (
        "{custom_text}\n"
        "原视频链接：{source}"
    ),
}

# 抖音默认文案：不含原视频链接占位（平台习惯与 B 站不同）。
DEFAULT_DOUYIN_TEMPLATES = {
    "中文字幕版": (
        "中文字幕版。\n"
        "翻译由本地工具辅助完成，细节如有偏差欢迎指正。"
    ),
    "学习笔记": (
        "学习笔记整理。\n"
        "内容仅供个人学习参考。"
    ),
    "技术教程": (
        "【中文字幕】技术内容整理。\n"
        "仅供学习交流。"
    ),
    "影视剪辑": (
        "精彩片段中文字幕版。\n"
        "剪辑与字幕由本地工具辅助完成。"
    ),
    "播客访谈": (
        "【中文字幕】播客/访谈整理。\n"
        "内容为嘉宾个人观点，翻译如有不准确之处欢迎指正。"
    ),
    "游戏实况": (
        "【中文字幕】游戏实况/攻略整理。\n"
        "游戏画面与音频版权归原作者/发行商所有。"
    ),
    "自定义模板": "{custom_text}",
}

# Backward-compatible alias used by older callers / Tk GUI.
DEFAULT_TEMPLATES = DEFAULT_BILIBILI_TEMPLATES
DESCRIPTION_TEMPLATES = DEFAULT_BILIBILI_TEMPLATES


def normalize_publish_platform(platform: str | None) -> str:
    value = str(platform or "bilibili").strip().lower()
    if value in {"dy", "tiktok", "抖音"}:
        return "douyin"
    if value in {"bili", "bilibili", "b站", "哔哩哔哩"}:
        return "bilibili"
    if value == "douyin":
        return "douyin"
    return "bilibili"


def default_templates_for(platform: str | None) -> dict[str, str]:
    if normalize_publish_platform(platform) == "douyin":
        return dict(DEFAULT_DOUYIN_TEMPLATES)
    return dict(DEFAULT_BILIBILI_TEMPLATES)


def get_all_templates(platform: str | None = "bilibili") -> dict[str, str]:
    """获取指定平台的模板（内置 + 该平台自定义）。"""
    platform = normalize_publish_platform(platform)
    templates = default_templates_for(platform)
    custom = _load_custom_templates().get(platform, {})
    templates.update(custom)
    return templates


def build_publish_description(
    platform: str | None,
    template_name: str,
    source: str,
    include_source_link: bool = True,
    custom_text: str = "",
    extra_lines: list[str] | None = None,
    template_body: str | None = None,
) -> str:
    """按平台构建投稿文案。抖音侧不会附带原视频链接。"""
    platform = normalize_publish_platform(platform)
    if platform == "douyin":
        include_source_link = False

    source_text = source.strip() if include_source_link and source.strip() else ""
    if include_source_link and not source_text:
        source_text = "请在发布前补充"

    if template_body is not None and str(template_body).strip():
        validate_template(template_body)
        template = template_body
    else:
        templates = get_all_templates(platform)
        template = templates.get(template_name) or default_templates_for(platform).get(
            next(iter(default_templates_for(platform)))
        )

    description = _format_template(
        template or "",
        source=source_text if include_source_link else "",
        custom_text=custom_text,
    ).strip()

    if include_source_link and source.strip() and source.strip() not in description:
        description = f"{description}\n原视频链接：{source.strip()}"
    elif not include_source_link:
        description = _strip_empty_source_lines(description, source.strip())

    if extra_lines:
        for line in extra_lines:
            if line.strip():
                description = f"{description}\n{line.strip()}"

    return description.strip()


def build_bilibili_description(
    template_name: str,
    source: str,
    include_source_link: bool = True,
    custom_text: str = "",
    extra_lines: list[str] | None = None,
    template_body: str | None = None,
) -> str:
    """构建 B 站发布文案（兼容旧调用）。"""
    return build_publish_description(
        "bilibili",
        template_name,
        source,
        include_source_link=include_source_link,
        custom_text=custom_text,
        extra_lines=extra_lines,
        template_body=template_body,
    )


def ensure_source_link(description: str, source: str, include_source_link: bool = True) -> str:
    description = description.strip()
    source = source.strip()
    if not include_source_link or not source:
        return description
    if source in description:
        return description
    if description:
        return f"{description}\n原视频链接：{source}"
    return build_bilibili_description("授权本地化", source, include_source_link=True)


def _strip_empty_source_lines(description: str, source: str = "") -> str:
    """Remove blank source-link lines when Douyin/no-link mode fills {source} with empty text."""
    cleaned: list[str] = []
    source = (source or "").strip()
    for line in description.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"原视频链接[:：]\s*(请在发布前补充)?", stripped):
            continue
        if source and stripped == source:
            continue
        if source and re.fullmatch(rf"原视频链接[:：]\s*{re.escape(source)}", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _load_custom_templates() -> dict[str, dict[str, str]]:
    """Load custom templates keyed by platform. Migrates legacy flat files."""
    template_file = _custom_templates_file()
    if not template_file.exists():
        return {"bilibili": {}, "douyin": {}}
    try:
        data = json.loads(template_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"bilibili": {}, "douyin": {}}
    if not isinstance(data, dict):
        return {"bilibili": {}, "douyin": {}}

    # New shape: {"bilibili": {...}, "douyin": {...}}
    if any(key in data for key in PUBLISH_PLATFORMS) and all(
        key in PUBLISH_PLATFORMS or not isinstance(value, str) for key, value in data.items()
    ):
        result = {"bilibili": {}, "douyin": {}}
        for platform in PUBLISH_PLATFORMS:
            bucket = data.get(platform) or {}
            if isinstance(bucket, dict):
                result[platform] = {
                    str(name): str(body)
                    for name, body in bucket.items()
                    if str(name).strip() and isinstance(body, str)
                }
        return result

    # Legacy flat shape: {"模板名": "正文"} → treat as Bilibili customs.
    legacy = {
        str(name): str(body)
        for name, body in data.items()
        if str(name).strip() and isinstance(body, str)
    }
    return {"bilibili": legacy, "douyin": {}}


def _write_custom_templates(data: dict[str, dict[str, str]]) -> None:
    template_file = _custom_templates_file()
    template_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bilibili": data.get("bilibili", {}),
        "douyin": data.get("douyin", {}),
    }
    template_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_custom_template(name: str, template: str, platform: str | None = "bilibili") -> None:
    """保存用户自定义模板到指定平台。"""
    platform = normalize_publish_platform(platform)
    validate_template(template)
    data = _load_custom_templates()
    data.setdefault(platform, {})[name] = template
    _write_custom_templates(data)


def delete_custom_template(name: str, platform: str | None = "bilibili") -> bool:
    """删除指定平台的自定义模板。"""
    platform = normalize_publish_platform(platform)
    data = _load_custom_templates()
    bucket = data.get(platform, {})
    if name not in bucket:
        return False
    del bucket[name]
    data[platform] = bucket
    _write_custom_templates(data)
    return True


def get_template_names(platform: str | None = "bilibili") -> list[str]:
    """获取指定平台可用模板名称列表。"""
    return list(get_all_templates(platform).keys())


def preview_template(
    template_name: str,
    source: str = "https://example.com/video",
    custom_text: str = "",
    platform: str | None = "bilibili",
) -> str:
    """预览模板生成的文案效果。"""
    platform = normalize_publish_platform(platform)
    return build_publish_description(
        platform,
        template_name,
        source,
        include_source_link=(platform != "douyin"),
        custom_text=custom_text,
    )


def validate_template(template: str) -> None:
    """Validate a user template before saving it."""
    unknown_fields = sorted(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name and field_name.split(".", 1)[0].split("[", 1)[0] not in ALLOWED_TEMPLATE_FIELDS
    )
    if unknown_fields:
        allowed = ", ".join(f"{{{name}}}" for name in sorted(ALLOWED_TEMPLATE_FIELDS))
        unknown = ", ".join(f"{{{name}}}" for name in unknown_fields)
        raise ValueError(f"模板变量不支持：{unknown}。可用变量：{allowed}")
    try:
        _format_template(template, source="https://example.com/video", custom_text="自定义内容")
    except Exception as exc:
        raise ValueError(f"模板格式错误：{exc}") from exc


def _format_template(template: str, source: str, custom_text: str) -> str:
    return template.format(source=source, custom_text=custom_text)

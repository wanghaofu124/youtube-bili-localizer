from __future__ import annotations

from pathlib import Path
from string import Formatter
import json
import os


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
DEFAULT_TEMPLATES = {
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

def get_all_templates() -> dict[str, str]:
    """获取所有模板，包括默认模板和用户自定义模板。"""
    templates = dict(DEFAULT_TEMPLATES)
    custom = _load_custom_templates()
    templates.update(custom)
    return templates


def build_bilibili_description(
    template_name: str,
    source: str,
    include_source_link: bool = True,
    custom_text: str = "",
    extra_lines: list[str] | None = None,
    template_body: str | None = None,
) -> str:
    """
    构建B站发布文案。
    
    Args:
        template_name: 模板名称
        source: 原视频链接
        include_source_link: 是否包含原视频链接
        custom_text: 自定义模板时的额外文本
        extra_lines: 额外追加的行（如标签、鸣谢等）
        template_body: 可选的模板正文。传入时直接使用该正文作为模板，
            不再按 template_name 查询已保存模板（用于「草稿」场景）。
    """
    source_text = source.strip() if include_source_link and source.strip() else "请在发布前补充"

    if template_body is not None and str(template_body).strip():
        validate_template(template_body)
        template = template_body
    else:
        templates = get_all_templates()
        template = templates.get(template_name)
        if template is None:
            template = DEFAULT_TEMPLATES["授权本地化"]

    description = _format_template(template, source=source_text, custom_text=custom_text).strip()
    
    # 确保原视频链接在文案中
    if include_source_link and source.strip() and source.strip() not in description:
        description = f"{description}\n原视频链接：{source.strip()}"
    
    # 追加额外行
    if extra_lines:
        for line in extra_lines:
            if line.strip():
                description = f"{description}\n{line.strip()}"
    
    return description


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


def _load_custom_templates() -> dict[str, str]:
    """从文件加载用户自定义模板。"""
    template_file = _custom_templates_file()
    if not template_file.exists():
        return {}
    try:
        data = json.loads(template_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_custom_template(name: str, template: str) -> None:
    """保存用户自定义模板到文件。"""
    validate_template(template)
    template_file = _custom_templates_file()
    template_file.parent.mkdir(parents=True, exist_ok=True)
    templates = _load_custom_templates()
    templates[name] = template
    template_file.write_text(
        json.dumps(templates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_custom_template(name: str) -> bool:
    """删除用户自定义模板。"""
    template_file = _custom_templates_file()
    templates = _load_custom_templates()
    if name not in templates:
        return False
    del templates[name]
    template_file.write_text(
        json.dumps(templates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def get_template_names() -> list[str]:
    """获取所有可用模板名称列表。"""
    return list(get_all_templates().keys())


def preview_template(template_name: str, source: str = "https://example.com/video", custom_text: str = "") -> str:
    """预览模板生成的文案效果。"""
    return build_bilibili_description(template_name, source, custom_text=custom_text)


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

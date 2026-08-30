from __future__ import annotations

import os
import math
from pathlib import Path
import shutil
from typing import Any, Callable

from .dependencies import dependency_capabilities
from .download import po_token_provider_status
from .ocr_subtitle import _normalize_ocr_language, available_ocr_languages
from .render import render_encoder_status
from .performance import normalize_resource_profile


COLOR_MAP = {
    "白色": "&H00FFFFFF", "黄色": "&H0000FFFF", "青色": "&H00FFFF00",
    "绿色": "&H0000FF00", "黑色": "&H00000000", "灰色": "&H00808080", "蓝色": "&H00FF0000",
}
COLOR_NAME_BY_VALUE = {value.upper(): name for name, value in COLOR_MAP.items()}

MAX_DOWNLOAD_SECONDS = 86_400
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 5_000
MAX_MODEL_LENGTH = 128
MAX_FONT_NAME_LENGTH = 100
MAX_TARGET_LANGUAGE_LENGTH = 32
MAX_TAGS = 20
MAX_TAG_LENGTH = 30


class InputValidationError(ValueError):
    """Structured validation error safe to return through the local API."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_input",
        field: str | None = None,
        limits: dict[str, int | float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.limits = limits or {}

    def payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {"error": str(self), "code": self.code}
        if self.field:
            result["field"] = self.field
        if self.limits:
            result["limits"] = self.limits
        return result


def _invalid_type(field: str, expected: str) -> InputValidationError:
    return InputValidationError(
        f"{field} 必须是{expected}。", code="invalid_type", field=field,
    )


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise _invalid_type(field, "布尔值 true 或 false")
    return value


def require_boolean(value: Any, field: str) -> bool:
    """Validate a top-level API boolean without Python truthy coercion."""
    return _strict_bool(value, field)


def require_text(value: Any, field: str, maximum: int, *, allow_empty: bool = True) -> str:
    text = _strict_text(value, field, maximum)
    if not allow_empty and not text:
        raise InputValidationError(
            f"{field} 不能为空。", code="required", field=field,
        )
    return text


def _strict_number(
    value: Any,
    field: str,
    minimum: float,
    maximum: float,
    *,
    integer: bool = False,
) -> int | float:
    if type(value) not in {int, float}:
        raise _invalid_type(field, "数字")
    number = float(value)
    if not math.isfinite(number):
        raise InputValidationError(
            f"{field} 必须是有限数字。", code="non_finite_number", field=field,
        )
    if integer and not number.is_integer():
        raise InputValidationError(
            f"{field} 必须是整数。", code="integer_required", field=field,
        )
    if number < minimum or number > maximum:
        raise InputValidationError(
            f"{field} 必须在 {minimum:g} 到 {maximum:g} 之间。",
            code="out_of_range", field=field,
            limits={"min": minimum, "max": maximum},
        )
    return int(number) if integer else number


def _strict_text(value: Any, field: str, maximum: int, *, allow_none: bool = False) -> str:
    if value is None and allow_none:
        return ""
    if not isinstance(value, str):
        raise _invalid_type(field, "文本")
    text = value.strip()
    if len(text) > maximum:
        raise InputValidationError(
            f"{field} 不能超过 {maximum} 个字符。", code="too_long", field=field,
            limits={"max_length": maximum},
        )
    return text


def _validate_strict_source(source: dict[str, Any]) -> None:
    bool_fields = (
        "require_reuse_allowed", "smart_translation", "smart_subtitle_layout",
        "publish_to_bilibili", "include_source_link", "close_after_fill", "prefer_platform_subtitles",
    )
    for field in bool_fields:
        if field in source:
            _strict_bool(source[field], field)

    number_fields = {
        "beam_size": (1, 10, True),
        "ocr_interval": (0.2, 60, False),
        "ocr_crop_ratio": (0.05, 1, False),
        "ocr_min_chars": (1, 100, True),
        "subtitle_margin_ratio": (0.01, 0.4, False),
        "render_crf": (14, 32, True),
        "font_size": (8, 96, True),
    }
    for field, (minimum, maximum, integer) in number_fields.items():
        if field in source:
            _strict_number(source[field], field, minimum, maximum, integer=integer)

    if "max_seconds" in source and source["max_seconds"] is not None:
        _strict_number(source["max_seconds"], "max_seconds", 1, MAX_DOWNLOAD_SECONDS, integer=True)

    text_fields = {
        "title": MAX_TITLE_LENGTH,
        "description": MAX_DESCRIPTION_LENGTH,
        "whisper_model_size": MAX_MODEL_LENGTH,
        "translate_model": MAX_MODEL_LENGTH,
        "font_name": MAX_FONT_NAME_LENGTH,
        "target_lang": MAX_TARGET_LANGUAGE_LENGTH,
        "source_language": MAX_TARGET_LANGUAGE_LENGTH,
    }
    for field, maximum in text_fields.items():
        if field in source and source[field] is not None:
            _strict_text(source[field], field, maximum)

    if "tags" in source:
        tags = source["tags"]
        if not isinstance(tags, list):
            raise _invalid_type("tags", "文本数组")
        if len(tags) > MAX_TAGS:
            raise InputValidationError(
                f"tags 最多允许 {MAX_TAGS} 个标签。", code="too_many_items", field="tags",
                limits={"max_items": MAX_TAGS},
            )
        for index, item in enumerate(tags):
            if not isinstance(item, str):
                raise _invalid_type(f"tags[{index}]", "文本")
            if len(item.strip()) > MAX_TAG_LENGTH:
                raise InputValidationError(
                    f"第 {index + 1} 个标签不能超过 {MAX_TAG_LENGTH} 个字符。",
                    code="too_long", field=f"tags[{index}]",
                    limits={"max_length": MAX_TAG_LENGTH},
                )


def _color(value: Any, fallback: str) -> str:
    text = _text(value)
    if text in COLOR_MAP:
        return COLOR_MAP[text]
    return text.upper() if text.upper() in COLOR_NAME_BY_VALUE else fallback


def _effect_from_values(outline: Any, shadow: Any) -> str:
    has_outline, has_shadow = bool(float(outline or 0)), bool(float(shadow or 0))
    if has_outline and has_shadow:
        return "描边+阴影"
    if has_shadow:
        return "阴影"
    if has_outline:
        return "描边"
    return "无"


def _text(value: Any) -> str:
    """Normalize optional UI strings without turning ``None`` into 'None'."""
    return "" if value is None else str(value).strip()


def default_options(output_dir: str = "outputs/workbench_demo") -> dict[str, Any]:
    """The sole public default set used by the workbench and its API clients."""
    return {
        "title": "", "require_reuse_allowed": False,
        "cookies_from_browser": "", "cookies_file": "", "max_seconds": 10,
        "youtube_po_token_mode": "auto", "youtube_proxy": "",
        "download_quality": "1080p", "render_encoder": "auto",
        "resource_profile": "balanced",
        "subtitle_source": "audio", "prefer_platform_subtitles": True,
        "whisper_model_size": "small", "source_language": "",
        "beam_size": 5, "ocr_interval": 0.5, "ocr_crop_ratio": 0.30, "ocr_min_chars": 3,
        "ocr_language": "eng",
        "subtitle_margin_ratio": 0.055, "render_crf": 20,
        "translator": "deepseek", "target_lang": "zh-Hans", "translate_model": "",
        "smart_translation": True, "smart_subtitle_layout": True,
        "font_name": "Microsoft YaHei", "font_size": 24,
        "subtitle_display_mode": "translated", "subtitle_color": "白色",
        "subtitle_outline_color": "黑色", "subtitle_effect": "描边",
        "output_dir": output_dir, "description": "", "tags": [],
        "publish_to_bilibili": False, "include_source_link": True,
        "bilibili_browser": "chromium", "close_after_fill": False,
    }


def normalize_options(value: Any, output_dir: str, *, strict: bool = True) -> dict[str, Any]:
    if strict and value is not None and not isinstance(value, dict):
        raise _invalid_type("options", "对象")
    source = value if isinstance(value, dict) else {}
    if strict:
        _validate_strict_source(source)
    raw = {**default_options(output_dir), **source}
    translator = str(raw["translator"]).lower()
    if translator not in {"deepseek", "openai", "none"}:
        raise InputValidationError("翻译器必须是 deepseek、openai 或 none。", code="invalid_choice", field="translator")
    subtitle_source = str(raw["subtitle_source"])
    if subtitle_source not in {"auto", "audio", "ocr", "merged"}:
        raise InputValidationError("字幕来源选项无效。", code="invalid_choice", field="subtitle_source")
    display_mode = str(raw["subtitle_display_mode"])
    if display_mode not in {"translated", "bilingual-source-first", "bilingual-translation-first"}:
        raise InputValidationError("字幕显示模式无效。", code="invalid_choice", field="subtitle_display_mode")
    effect = (
        str(source["subtitle_effect"])
        if "subtitle_effect" in source
        else _effect_from_values(source.get("subtitle_outline", 1), source.get("subtitle_shadow", 0))
    )
    if effect not in {"描边", "阴影", "描边+阴影", "无"}:
        raise InputValidationError("字幕效果选项无效。", code="invalid_choice", field="subtitle_effect")
    try:
        max_seconds = raw.get("max_seconds")
        if strict:
            max_seconds = None if max_seconds is None else int(max_seconds)
            font_size = int(raw["font_size"])
        else:
            max_seconds = None if max_seconds in {None, "", 0, "0"} else max(1, min(MAX_DOWNLOAD_SECONDS, int(max_seconds)))
            font_size = max(8, min(96, int(raw["font_size"])))
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "字号和 URL 时长必须是有效数字。", code="invalid_number",
        ) from exc
    cookies_file = _text(raw.get("cookies_file")) or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
    cookies_from_browser = _text(raw.get("cookies_from_browser")) or os.getenv("YBLOCALIZER_COOKIES_FROM_BROWSER", "").strip() or None
    youtube_po_token_mode = _text(raw.get("youtube_po_token_mode")).lower() or "auto"
    if youtube_po_token_mode not in {"auto", "off"}:
        raise InputValidationError("YouTube 浏览器验证模式必须是 auto 或 off。", code="invalid_choice", field="youtube_po_token_mode")
    youtube_proxy = _text(raw.get("youtube_proxy")) or os.getenv("YBLOCALIZER_YOUTUBE_PROXY", "").strip() or None
    download_quality = _text(raw.get("download_quality")).lower() or "1080p"
    if download_quality not in {"720p", "1080p", "original"}:
        raise InputValidationError("下载画质必须是 720p、1080p 或 original。", code="invalid_choice", field="download_quality")
    render_encoder = str(raw.get("render_encoder", "auto")).strip().lower()
    if render_encoder not in {"auto", "cpu", "nvidia"}:
        raise InputValidationError("渲染编码器必须是 auto、cpu 或 nvidia。", code="invalid_choice", field="render_encoder")
    try:
        resource_profile = normalize_resource_profile(raw.get("resource_profile"))
    except ValueError as exc:
        raise InputValidationError(str(exc), code="invalid_choice", field="resource_profile") from exc
    if cookies_file:
        cookies_from_browser = None
    return {
        "title": _text(raw["title"]) or None,
        "require_reuse_allowed": raw["require_reuse_allowed"] if strict else bool(raw["require_reuse_allowed"]),
        "cookies_from_browser": cookies_from_browser, "cookies_file": cookies_file,
        "youtube_po_token_mode": youtube_po_token_mode, "youtube_proxy": youtube_proxy,
        "download_quality": download_quality, "render_encoder": render_encoder,
        "resource_profile": resource_profile,
        "max_seconds": max_seconds, "subtitle_source": subtitle_source,
        "prefer_platform_subtitles": raw["prefer_platform_subtitles"] if strict else bool(raw["prefer_platform_subtitles"]),
        "whisper_model_size": str(raw["whisper_model_size"]),
        "source_language": _text(raw["source_language"]) or None,
        "beam_size": int(raw["beam_size"]) if strict else max(1, min(10, int(raw["beam_size"]))),
        "ocr_interval": float(raw["ocr_interval"]) if strict else max(0.2, min(60.0, float(raw["ocr_interval"]))),
        "ocr_crop_ratio": float(raw["ocr_crop_ratio"]) if strict else max(0.05, min(1.0, float(raw["ocr_crop_ratio"]))),
        "ocr_min_chars": int(raw["ocr_min_chars"]) if strict else max(1, min(100, int(raw["ocr_min_chars"]))),
        "ocr_language": _normalize_ocr_language(raw.get("ocr_language")),
        "subtitle_margin_ratio": float(raw["subtitle_margin_ratio"]) if strict else max(0.01, min(0.4, float(raw["subtitle_margin_ratio"]))),
        "render_crf": int(raw["render_crf"]) if strict else max(14, min(32, int(raw["render_crf"]))),
        "translator": translator, "target_lang": str(raw["target_lang"]).strip() or "zh-Hans",
        "translate_model": _text(raw["translate_model"]) or None,
        "smart_translation": raw["smart_translation"] if strict else bool(raw["smart_translation"]),
        "smart_subtitle_layout": raw["smart_subtitle_layout"] if strict else bool(raw["smart_subtitle_layout"]),
        "font_name": str(raw["font_name"]).strip() or "Microsoft YaHei", "font_size": font_size,
        "subtitle_display_mode": display_mode,
        "subtitle_color": _color(raw["subtitle_color"], "&H00FFFFFF"),
        "subtitle_outline_color": _color(raw["subtitle_outline_color"], "&H00000000"),
        "subtitle_outline": 0 if effect in {"阴影", "无"} else 1,
        "subtitle_shadow": 1 if effect in {"阴影", "描边+阴影"} else 0,
        "output_dir": str(raw["output_dir"]), "description": _text(raw["description"]),
        "tags": [str(item).strip() for item in raw["tags"] if str(item).strip()] if isinstance(raw["tags"], list) else [],
        "publish_to_bilibili": raw["publish_to_bilibili"] if strict else bool(raw["publish_to_bilibili"]),
        "include_source_link": raw["include_source_link"] if strict else bool(raw["include_source_link"]),
        "bilibili_browser": "msedge" if str(raw["bilibili_browser"]).lower() in {"edge", "msedge"} else "chromium",
        "close_after_fill": raw["close_after_fill"] if strict else bool(raw["close_after_fill"]),
    }


def options_for_public(options: dict[str, Any]) -> dict[str, Any]:
    """Return browser-safe options while retaining the UI representation."""
    public = dict(options)
    public["subtitle_color"] = COLOR_NAME_BY_VALUE.get(str(options.get("subtitle_color", "")).upper(), "白色")
    public["subtitle_outline_color"] = COLOR_NAME_BY_VALUE.get(str(options.get("subtitle_outline_color", "")).upper(), "黑色")
    public["subtitle_effect"] = _effect_from_values(options.get("subtitle_outline"), options.get("subtitle_shadow"))
    public["youtube_proxy_configured"] = bool(options.get("youtube_proxy"))
    public["cookies_file_configured"] = bool(options.get("cookies_file"))
    public["cookies_from_browser_configured"] = bool(options.get("cookies_from_browser"))
    public["youtube_proxy"] = ""
    public["cookies_file"] = ""
    return public


def options_for_storage(options: dict[str, Any]) -> dict[str, Any]:
    """Serialize canonical task choices without local credentials or paths."""
    stored = dict(options)
    stored["cookies_file_configured"] = bool(options.get("cookies_file"))
    stored["youtube_proxy_configured"] = bool(options.get("youtube_proxy"))
    stored["cookies_file"] = None
    stored["youtube_proxy"] = None
    return stored


def options_from_storage(value: Any, output_dir: str) -> dict[str, Any]:
    """Restore choices and resolve sensitive values from current local settings."""
    stored = dict(value) if isinstance(value, dict) else {}
    if stored.get("cookies_file_configured"):
        stored["cookies_file"] = os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
    if stored.get("youtube_proxy_configured"):
        stored["youtube_proxy"] = os.getenv("YBLOCALIZER_YOUTUBE_PROXY", "").strip() or None
    return normalize_options(stored, output_dir, strict=False)


def capabilities() -> dict[str, Any]:
    return dependency_capabilities()


def preflight(
    payload: dict[str, Any], output_dir: str,
    profile_in_use: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    options = normalize_options(payload.get("options"), output_dir)
    blocking: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    add_block = lambda code, message: blocking.append({"code": code, "message": message})
    add_warn = lambda code, message: warnings.append({"code": code, "message": message})
    source_url = str(payload.get("source_url", "")).strip()
    if not source_url and not payload.get("material_id"): add_block("source_missing", "还没有选择素材。")
    if not bool(payload.get("authorized", False)): add_block("authorization_required", "还没有确认处理授权。")
    if options["target_lang"].lower().startswith("zh") and options["translator"] == "none": add_block("translator_required", "中文字幕不能使用 none 翻译器。")
    if options["translator"] == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"): add_block("deepseek_key_missing", "DeepSeek API Key 尚未配置。")
    if options["translator"] == "openai" and not os.getenv("OPENAI_API_KEY"): add_block("openai_key_missing", "OpenAI API Key 尚未配置。")
    if options["cookies_file"] and not Path(options["cookies_file"]).expanduser().is_file(): add_block("cookies_missing", "Cookies 文件不存在，请重新选择。")
    if source_url and options["youtube_po_token_mode"] == "auto":
        po_status = po_token_provider_status()
        if not po_status["available"]:
            add_warn("youtube_po_provider_missing", "安装包缺少 YouTube 自动浏览器验证组件；将使用 yt-dlp 常规模式，受限视频可能失败。")
        elif not po_status["browser_path"]:
            add_warn("youtube_po_browser_unknown", "未预先检测到 Chrome/Edge；开始任务时验证组件会再次自动查找浏览器。")
    if source_url and options["download_quality"] == "original":
        source_height = int(payload.get("source_height") or 0)
        detail = f"（检测到最高 {source_height}p）" if source_height >= 2160 else ""
        add_warn("original_quality_load", f"原始画质可能显著增加下载、OCR 和字幕渲染负载{detail}；普通电脑推荐 1080p。")
    if options["render_encoder"] == "nvidia" and not render_encoder_status()["nvidia"]:
        add_block("nvenc_unavailable", "未检测到可用的 NVIDIA NVENC；请改用自动或 CPU 渲染。")
    device = str(payload.get("device", "cpu")).lower()
    compute_type = str(payload.get("compute_type", "int8")).lower()
    if device == "cpu" and compute_type in {"float16", "int8_float16"}: add_block("precision_incompatible", "CPU 不支持当前 float16 精度，请使用 int8 或 float32。")
    caps = capabilities()
    if not caps["ffmpeg"]["available"]: add_block("ffmpeg_missing", "未检测到 FFmpeg，无法提取音频或渲染视频。")
    if options["subtitle_source"] == "ocr" and not caps["tesseract"]["available"]: add_block("tesseract_required", "纯 OCR 模式需要 Tesseract，请安装后重试。")
    elif options["subtitle_source"] in {"merged", "auto"} and not caps["tesseract"]["available"]: add_warn("ocr_unavailable", "未安装 Tesseract；如需 OCR 将自动退回音频转写。")
    if options["subtitle_source"] in {"ocr", "merged"} and caps["tesseract"]["available"]:
        installed = available_ocr_languages()
        missing = [item for item in options["ocr_language"].split("+") if item not in installed]
        if missing:
            add_block("ocr_language_missing", f"Tesseract 缺少语言包：{', '.join(missing)}。请安装语言数据或更改 OCR 语言。")
    if options["publish_to_bilibili"] and profile_in_use and profile_in_use(options["bilibili_browser"]): add_block("bilibili_profile_busy", "B 站自动化浏览器正在使用中，请关闭后再开始全流程。")
    return {"ready": not blocking, "blocking": blocking, "warnings": warnings, "normalized_options": options}

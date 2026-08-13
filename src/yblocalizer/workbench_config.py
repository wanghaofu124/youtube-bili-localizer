from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Callable


COLOR_MAP = {
    "白色": "&H00FFFFFF", "黄色": "&H0000FFFF", "青色": "&H00FFFF00",
    "绿色": "&H0000FF00", "黑色": "&H00000000", "灰色": "&H00808080", "蓝色": "&H00FF0000",
}


def default_options(output_dir: str = "outputs/workbench_demo") -> dict[str, Any]:
    """The sole public default set used by the workbench and its API clients."""
    return {
        "title": "", "require_reuse_allowed": False,
        "cookies_from_browser": "", "cookies_file": "", "max_seconds": 10,
        "subtitle_source": "audio", "whisper_model_size": "small", "source_language": "",
        "beam_size": 5, "ocr_interval": 1.0, "ocr_crop_ratio": 0.30, "ocr_min_chars": 3,
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


def normalize_options(value: Any, output_dir: str) -> dict[str, Any]:
    raw = {**default_options(output_dir), **(value if isinstance(value, dict) else {})}
    translator = str(raw["translator"]).lower()
    if translator not in {"deepseek", "openai", "none"}:
        raise ValueError("translator must be deepseek, openai, or none.")
    subtitle_source = str(raw["subtitle_source"])
    if subtitle_source not in {"auto", "audio", "ocr", "merged"}:
        raise ValueError("Unknown subtitle source.")
    display_mode = str(raw["subtitle_display_mode"])
    if display_mode not in {"translated", "bilingual-source-first", "bilingual-translation-first"}:
        raise ValueError("Unknown subtitle display mode.")
    effect = str(raw["subtitle_effect"])
    try:
        max_seconds = raw.get("max_seconds")
        max_seconds = None if max_seconds in {None, "", 0, "0"} else max(1, int(max_seconds))
        font_size = max(8, min(96, int(raw["font_size"])))
    except (TypeError, ValueError) as exc:
        raise ValueError("Font size and URL duration must be valid numbers.") from exc
    cookies_file = str(raw.get("cookies_file", "")).strip() or os.getenv("YBLOCALIZER_COOKIES_FILE", "").strip() or None
    cookies_from_browser = str(raw.get("cookies_from_browser", "")).strip() or None
    if cookies_file:
        cookies_from_browser = None
    return {
        "title": str(raw["title"]).strip() or None,
        "require_reuse_allowed": bool(raw["require_reuse_allowed"]),
        "cookies_from_browser": cookies_from_browser, "cookies_file": cookies_file,
        "max_seconds": max_seconds, "subtitle_source": subtitle_source,
        "whisper_model_size": str(raw["whisper_model_size"]),
        "source_language": str(raw["source_language"]).strip() or None,
        "beam_size": max(1, min(10, int(raw["beam_size"]))),
        "ocr_interval": max(0.2, float(raw["ocr_interval"])),
        "ocr_crop_ratio": max(0.05, min(1.0, float(raw["ocr_crop_ratio"]))),
        "ocr_min_chars": max(1, int(raw["ocr_min_chars"])),
        "subtitle_margin_ratio": max(0.01, min(0.4, float(raw["subtitle_margin_ratio"]))),
        "render_crf": max(14, min(32, int(raw["render_crf"]))),
        "translator": translator, "target_lang": str(raw["target_lang"]).strip() or "zh-Hans",
        "translate_model": str(raw["translate_model"]).strip() or None,
        "smart_translation": bool(raw["smart_translation"]),
        "smart_subtitle_layout": bool(raw["smart_subtitle_layout"]),
        "font_name": str(raw["font_name"]).strip() or "Microsoft YaHei", "font_size": font_size,
        "subtitle_display_mode": display_mode,
        "subtitle_color": COLOR_MAP.get(str(raw["subtitle_color"]), "&H00FFFFFF"),
        "subtitle_outline_color": COLOR_MAP.get(str(raw["subtitle_outline_color"]), "&H00000000"),
        "subtitle_outline": 0 if effect in {"阴影", "无"} else 1,
        "subtitle_shadow": 1 if effect in {"阴影", "描边+阴影"} else 0,
        "output_dir": str(raw["output_dir"]), "description": str(raw["description"]).strip(),
        "tags": [str(item).strip() for item in raw["tags"] if str(item).strip()] if isinstance(raw["tags"], list) else [],
        "publish_to_bilibili": bool(raw["publish_to_bilibili"]),
        "include_source_link": bool(raw["include_source_link"]),
        "bilibili_browser": "msedge" if str(raw["bilibili_browser"]).lower() in {"edge", "msedge"} else "chromium",
        "close_after_fill": bool(raw["close_after_fill"]),
    }


def capabilities() -> dict[str, Any]:
    checks = {
        "ffmpeg": ("FFmpeg", True, "提取音频与硬字幕渲染"),
        "node": ("Node.js", False, "OCR 辅助服务"),
        "tesseract": ("Tesseract", False, "OCR 字幕识别引擎"),
    }
    return {
        key: {"label": label, "required": required, "purpose": purpose, "available": shutil.which(key) is not None}
        for key, (label, required, purpose) in checks.items()
    }


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
    device = str(payload.get("device", "cpu")).lower()
    compute_type = str(payload.get("compute_type", "int8")).lower()
    if device == "cpu" and compute_type in {"float16", "int8_float16"}: add_block("precision_incompatible", "CPU 不支持当前 float16 精度，请使用 int8 或 float32。")
    caps = capabilities()
    if not caps["ffmpeg"]["available"]: add_block("ffmpeg_missing", "未检测到 FFmpeg，无法提取音频或渲染视频。")
    if options["subtitle_source"] == "ocr" and not caps["tesseract"]["available"]: add_block("tesseract_required", "纯 OCR 模式需要 Tesseract，请安装后重试。")
    elif options["subtitle_source"] in {"merged", "auto"} and not caps["tesseract"]["available"]: add_warn("ocr_unavailable", "未安装 Tesseract；如需 OCR 将自动退回音频转写。")
    if options["publish_to_bilibili"] and profile_in_use and profile_in_use(options["bilibili_browser"]): add_block("bilibili_profile_busy", "B 站自动化浏览器正在使用中，请关闭后再开始全流程。")
    return {"ready": not blocking, "blocking": blocking, "warnings": warnings, "normalized_options": options}

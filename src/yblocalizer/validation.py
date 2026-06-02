from __future__ import annotations

from .models import Segment


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def validate_target_language(segments: list[Segment], target_lang: str, provider: str) -> None:
    normalized = target_lang.lower()
    if not normalized.startswith("zh"):
        return
    final_text = "\n".join(segment.final_text for segment in segments).strip()
    if not final_text:
        raise RuntimeError("Translation produced empty subtitles.")
    if provider.lower() == "none":
        raise RuntimeError(
            "translator=none is only for debugging. It cannot produce Chinese subtitles. "
            "Choose openai or deepseek, and configure the matching API key."
        )
    if not contains_cjk(final_text):
        raise RuntimeError(
            "Translation finished, but the output does not contain Chinese characters. "
            "The video was not rendered to avoid producing an English-subtitle result by mistake."
        )

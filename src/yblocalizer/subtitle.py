from __future__ import annotations

from pathlib import Path
import re

from .models import Segment


def format_srt_timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def segments_to_srt(
    segments: list[Segment],
    display_mode: str = "translated",
    smart_layout: bool = True,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(segment.start)
        end = segment.end
        if index < len(segments):
            next_start = segments[index].start
            if end > next_start - 0.05:
                # 规整时间重叠：两条字幕同时显示会互相叠加占满画面，
                # 将本条结束时间截到下一条开始前，保证任何时刻至多一条字幕。
                end = max(segment.start + 0.1, next_start - 0.05)
        end = format_srt_timestamp(end)
        text = format_segment_text(
            segment,
            display_mode=display_mode,
            smart_layout=smart_layout,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
        )
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def format_segment_text(
    segment: Segment,
    display_mode: str = "translated",
    smart_layout: bool = True,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> str:
    source = segment.text.strip()
    translated = segment.final_text.strip()
    if smart_layout:
        duration = max(0.1, segment.end - segment.start)
        translated = _layout_subtitle_text(
            translated,
            duration=duration,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            prefer_cjk=_has_cjk(translated),
        )
        source = _layout_subtitle_text(
            source,
            duration=duration,
            max_chars_per_line=max(32, max_chars_per_line * 2),
            max_lines=1 if display_mode != "source" else 2,
            prefer_cjk=False,
        )
    if display_mode == "source":
        return source
    if display_mode == "bilingual-source-first":
        return f"{source}\n{translated}" if translated and translated != source else source
    if display_mode == "bilingual-translation-first":
        return f"{translated}\n{source}" if translated and translated != source else translated
    return translated


def write_srt(
    path: Path,
    segments: list[Segment],
    display_mode: str = "translated",
    smart_layout: bool = True,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            segments_to_srt(
                segments,
                display_mode=display_mode,
                smart_layout=smart_layout,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
            ),
            encoding="utf-8-sig",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _layout_subtitle_text(
    text: str,
    duration: float,
    max_chars_per_line: int,
    max_lines: int,
    prefer_cjk: bool,
) -> str:
    text = _normalize_subtitle_text(text)
    if not text:
        return text
    dynamic_limit = _dynamic_line_limit(text, duration, max_chars_per_line, prefer_cjk=prefer_cjk)
    output: list[str] = []
    for paragraph in text.splitlines():
        output.extend(_wrap_line(paragraph, dynamic_limit, prefer_cjk=prefer_cjk))
    return _limit_lines(output, max(1, max_lines), prefer_cjk=prefer_cjk)


def _normalize_subtitle_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    text = re.sub(r"\s+([，。！？、；：,.!?;:])", r"\1", text)
    text = re.sub(r"([（《“])\s+", r"\1", text)
    text = re.sub(r"\s+([）》”])", r"\1", text)
    return text


def _dynamic_line_limit(text: str, duration: float, configured: int, prefer_cjk: bool) -> int:
    configured = max(8, configured)
    if not prefer_cjk:
        return max(28, configured)
    readable = 12 if duration < 1.4 else 16 if duration < 2.4 else configured
    if len(text) <= readable + 3:
        return max(readable, len(text))
    return max(10, min(configured, readable if duration < 2.4 else configured))


def _wrap_line(text: str, max_chars: int, prefer_cjk: bool) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if prefer_cjk or _has_cjk(text):
        return _wrap_cjk_line(text, max_chars)
    return _wrap_latin_line(text, max_chars)


def _wrap_cjk_line(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = _best_cjk_break(remaining, max_chars)
        lines.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        lines.append(remaining)
    return lines


def _best_cjk_break(text: str, max_chars: int) -> int:
    lower = max(6, int(max_chars * 0.55))
    upper = min(len(text), max_chars + 1)
    for punctuation in "，、；：,;: ":
        candidates = [
            idx + 1
            for idx in range(lower, min(len(text), max_chars + 4))
            if text[idx] == punctuation
        ]
        if candidates:
            return max(candidates)
    for punctuation in "。！？.!?":
        candidates = [idx + 1 for idx in range(lower, min(len(text), max_chars + 4)) if text[idx] == punctuation]
        if candidates:
            return max(candidates)
    return max_chars


def _wrap_latin_line(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _limit_lines(lines: list[str], max_lines: int, prefer_cjk: bool) -> str:
    cleaned = [line for line in (item.strip() for item in lines) if line]
    if len(cleaned) <= max_lines:
        return "\n".join(cleaned)
    kept = cleaned[: max_lines - 1]
    joiner = "" if prefer_cjk else " "
    kept.append(joiner.join(cleaned[max_lines - 1 :]))
    return "\n".join(kept)


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))

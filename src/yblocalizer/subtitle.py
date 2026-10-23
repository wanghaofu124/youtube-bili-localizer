from __future__ import annotations

from html import unescape
from pathlib import Path
import re
import unicodedata

from .models import Segment


_SRT_TIMING = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)


def read_srt_segments(path: Path, max_seconds: float | None = None) -> list[Segment]:
    """Read external SRT data into normalized pipeline segments."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    output: list[Segment] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        match = _SRT_TIMING.search(lines[timing_index])
        if not match:
            continue
        start = _parse_srt_timestamp(match.group("start"))
        end = _parse_srt_timestamp(match.group("end"))
        if max_seconds is not None:
            if start >= max_seconds:
                continue
            end = min(end, max_seconds)
        caption = " ".join(lines[timing_index + 1 :])
        caption = unescape(re.sub(r"<[^>]+>", "", caption))
        caption = re.sub(r"\s+", " ", caption).strip()
        if not caption or end <= start:
            continue
        if output and caption == output[-1].text and start <= output[-1].end + 0.15:
            output[-1].end = max(output[-1].end, end)
            continue
        output.append(Segment(round(start, 3), round(end, 3), caption))
    return output


def _parse_srt_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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
            if end > next_start - 0.05 and next_start > segment.start:
                # 规整时间重叠：两条字幕同时显示会互相叠加占满画面，
                # 紧邻字幕缩小间隔，避免 100ms 最小时长反过来制造重叠。
                gap = min(0.05, (next_start - segment.start) / 2)
                end = min(end, next_start - gap)
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
    return _format_segment_parts(segment, display_mode, smart_layout, max_chars_per_line, max_lines)[0]


def cue_source_line_counts(
    segments: list[Segment],
    display_mode: str,
    smart_layout: bool = True,
    max_chars_per_line: int = 18,
    max_lines: int = 2,
) -> dict[int, int]:
    """1-based cue index -> number of source-language lines in that bilingual cue."""
    return {
        index: _format_segment_parts(segment, display_mode, smart_layout, max_chars_per_line, max_lines)[1]
        for index, segment in enumerate(segments, start=1)
    }


def _format_segment_parts(
    segment: Segment,
    display_mode: str,
    smart_layout: bool,
    max_chars_per_line: int,
    max_lines: int,
) -> tuple[str, int]:
    """Return the cue text and how many of its lines are the bilingual source."""
    source = segment.text.strip()
    translated = segment.final_text.strip()
    bilingual = display_mode in {"bilingual-source-first", "bilingual-translation-first"}
    if smart_layout:
        duration = max(0.1, segment.end - segment.start)
        if bilingual:
            translated, source = _layout_bilingual_pair(
                translated,
                source,
                duration=duration,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
            )
        else:
            translated = _layout_subtitle_text(
                translated,
                duration=duration,
                max_chars_per_line=max_chars_per_line,
                max_lines=max(1, max_lines),
                prefer_cjk=_has_cjk(translated),
            )
            source = _layout_subtitle_text(
                source,
                duration=duration,
                max_chars_per_line=max(32, max_chars_per_line * 2),
                max_lines=max(1, max_lines) if display_mode == "source" else 1,
                prefer_cjk=_has_cjk(source),
            )
    if display_mode == "source":
        return source, 0
    two_parts = bool(source and translated and translated != source)
    source_lines = len(source.splitlines()) if two_parts else 0
    if display_mode == "bilingual-source-first":
        return (f"{source}\n{translated}" if two_parts else source or translated), source_lines
    if display_mode == "bilingual-translation-first":
        return (f"{translated}\n{source}" if two_parts else translated), source_lines
    return translated, 0


def translated_layout_issues(
    segments: list[Segment], *, smart_layout: bool = True,
    max_chars_per_line: int = 18,
    max_display_units_per_line: float = 36.5,
    display_mode: str = "translated",
) -> list[tuple[int, float, str]]:
    """Report cues that exceed the same row geometry used to write the SRT."""
    issues: list[tuple[int, float, str]] = []
    for index, segment in enumerate(segments, start=1):
        text = format_segment_text(
            segment,
            display_mode=display_mode,
            smart_layout=smart_layout,
            max_chars_per_line=max_chars_per_line,
        )
        lines = text.splitlines()
        width = lambda line: sum(
            1.0 if unicodedata.east_asian_width(char) in {"W", "F"} else 0.55
            for char in line if not char.isspace()
        )
        if len(lines) > 2 or any(width(line) > max_display_units_per_line for line in lines):
            issues.append((index, segment.start, text))
    return issues


def _layout_bilingual_pair(
    translated: str,
    source: str,
    *,
    duration: float,
    max_chars_per_line: int,
    max_lines: int,
) -> tuple[str, str]:
    """Prefer one Chinese line + one English line for bilingual hard-subs.

    Only open a second row for both languages when the Chinese side itself is
    too long for a single on-screen line. A long English sentence alone must
    not force Chinese onto two rows.
    """
    translated = _normalize_subtitle_text(translated)
    source = _normalize_subtitle_text(source)
    # Bilingual Chinese can sit a bit wider than monolingual hard-subs so a
    # normal sentence stays on one row instead of stacking under English.
    cjk_one_line = max(32, max_chars_per_line + 14)
    latin_one_line = max(64, max_chars_per_line * 3)
    line_budget = 2 if len(translated) > cjk_one_line else 1
    line_budget = min(max(1, max_lines), line_budget)

    if line_budget == 1:
        return (
            _layout_subtitle_text(
                translated,
                duration=duration,
                max_chars_per_line=max(cjk_one_line, len(translated)),
                max_lines=1,
                prefer_cjk=_has_cjk(translated),
            ),
            _layout_subtitle_text(
                source,
                duration=duration,
                max_chars_per_line=latin_one_line,
                max_lines=2,
                prefer_cjk=_has_cjk(source),
            ),
        )

    return (
        _layout_subtitle_text(
            translated,
            duration=duration,
            max_chars_per_line=max_chars_per_line,
            max_lines=2,
            prefer_cjk=_has_cjk(translated),
        ),
        _layout_subtitle_text(
            source,
            duration=duration,
            max_chars_per_line=max(32, max_chars_per_line * 2),
            max_lines=2,
            prefer_cjk=_has_cjk(source),
        ),
    )


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
    if prefer_cjk:
        if max_lines < 2 or len(text) <= max_chars_per_line:
            return text
        # Chinese has no whitespace between words. If there is no known-safe
        # phrase boundary, one intact row is better than a fabricated cut.
        return _balanced_cjk_pair(text, max_chars_per_line) or text
    dynamic_limit = _dynamic_line_limit(text, duration, max_chars_per_line, prefer_cjk=prefer_cjk)
    output: list[str] = []
    for paragraph in text.splitlines():
        output.extend(_wrap_line(paragraph, dynamic_limit, prefer_cjk=prefer_cjk))
    return _limit_lines(
        output,
        max(1, max_lines),
        prefer_cjk=prefer_cjk,
        source_text=text,
        preferred_width=dynamic_limit,
    )


def _balanced_cjk_pair(text: str, width: int) -> str | None:
    """Choose a readable two-row break only at an explicit phrase boundary."""
    if len(text) <= width:
        return text
    cut = _best_safe_cjk_break(text, width)
    if cut is None:
        return None
    return f"{text[:cut]}\n{text[cut:]}"


_CJK_BREAK_PUNCTUATION = "，。！？；：、,.;:!?"
_CJK_CLAUSE_STARTERS = (
    "但是", "不过", "然而", "所以", "因此", "然后", "接着", "于是",
    "同时", "另外", "而且", "并且", "或者", "如果", "因为", "虽然",
    "还有", "现在", "接下来",
)


def _safe_cjk_breaks(text: str) -> list[int]:
    """Return only boundaries that do not require guessing a Chinese word edge."""
    breaks = {
        index + 1 for index, char in enumerate(text)
        if char in _CJK_BREAK_PUNCTUATION or char.isspace()
    }
    for starter in _CJK_CLAUSE_STARTERS:
        breaks.update(match.start() for match in re.finditer(re.escape(starter), text))
    return sorted(cut for cut in breaks if 3 <= cut <= len(text) - 3)


def _best_safe_cjk_break(text: str, preferred_width: int) -> int | None:
    candidates = [
        cut for cut in _safe_cjk_breaks(text)
        if max(len(text[:cut].strip()), len(text[cut:].strip())) <= 36
    ]
    if not candidates:
        return None

    def score(cut: int) -> tuple[float, float]:
        left, right = text[:cut].strip(), text[cut:].strip()
        overflow = max(0, len(left) - preferred_width) + max(0, len(right) - preferred_width)
        # A grammatical cut outranks balanced geometry. Among safe cuts, keep
        # both lines reasonably close in length without splitting a word.
        return overflow, abs(len(left) - len(right))

    return min(candidates, key=score)


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
        return min(configured, max(readable, len(text)))
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
        split_at = _best_safe_cjk_break(remaining, max_chars)
        if split_at is None:
            break
        lines.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        lines.append(remaining)
    return lines


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


def _limit_lines(
    lines: list[str],
    max_lines: int,
    prefer_cjk: bool,
    *,
    source_text: str | None = None,
    preferred_width: int | None = None,
) -> str:
    cleaned = [line for line in (item.strip() for item in lines) if line]
    if len(cleaned) <= max_lines:
        return "\n".join(cleaned)
    # Density planning should normally split long captions into timed cues.
    # This is a last-mile visual guard for dense or short-span fallbacks: keep
    # every character, but rebalance the text so a single cue cannot silently
    # spill into a third or fourth simultaneous row.
    text = source_text if source_text is not None else ("".join(cleaned) if prefer_cjk else " ".join(cleaned))
    if prefer_cjk:
        if max_lines < 2:
            return text
        return _balanced_cjk_pair(text, preferred_width or 18) or text
    width = max(preferred_width or 8, (len(text) + max_lines - 1) // max_lines)
    for _ in range(max(1, len(text) - width + 1)):
        wrapped = _wrap_latin_line(text, width)
        if len(wrapped) <= max_lines:
            return "\n".join(wrapped)
        width += 1
    # Even an unbreakable Latin token must not reintroduce visual overflow.
    return "\n".join(
        text[index : index + width]
        for index in range(0, len(text), width)
    )


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))

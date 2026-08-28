from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable

from .dependencies import resolve_command
from .models import Segment, save_segments
from .performance import ffmpeg_thread_args, normalize_resource_profile
from .runtime import CancellationRequested
from .subtitle import write_srt
from .util import require_command, run


COMMON_SUBTITLE_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "back",
    "be",
    "because",
    "been",
    "better",
    "but",
    "by",
    "can",
    "come",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "for",
    "from",
    "get",
    "go",
    "going",
    "got",
    "great",
    "had",
    "has",
    "have",
    "he",
    "he's",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "it's",
    "just",
    "know",
    "let",
    "let's",
    "like",
    "me",
    "more",
    "my",
    "need",
    "needed",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "race",
    "racing",
    "raining",
    "see",
    "she",
    "so",
    "stop",
    "super",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "think",
    "this",
    "time",
    "to",
    "today",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "with",
    "world",
    "would",
    "yeah",
    "you",
    "your",
}

DOMAIN_SUBTITLE_WORDS = {
    "automotive",
    "bull",
    "burton",
    "champion",
    "championship",
    "content",
    "creator",
    "formula",
    "gt",
    "jeremiah",
    "max",
    "motorsports",
    "nissan",
    "panasonic",
    "racing",
    "red",
    "verstappen",
}


@dataclass(frozen=True, slots=True)
class OCRCandidate:
    text: str
    score: float
    region: str


def extract_ocr_subtitles(
    video_path: Path,
    work_dir: Path,
    segments_json: Path,
    srt_path: Path,
    interval_seconds: float = 0.5,
    crop_bottom_ratio: float = 0.30,
    min_chars: int = 3,
    cancel_check: Callable[[], bool] | None = None,
    ocr_language: str = "eng",
    log: Callable[[str], None] | None = None,
    resource_profile: str = "balanced",
    progress: Callable[[float], None] | None = None,
) -> list[Segment]:
    """Extract burned-in subtitles with one frame pass and adaptive regions.

    Each frame tries the previously successful region first.  A strong result
    stops immediately; weak or empty results expand to the other common
    caption regions.  This supports captions that move around the screen while
    avoiding the former four complete FFmpeg extraction passes.
    """
    require_command("ffmpeg")
    language = _check_ocr_runtime(ocr_language)

    video_path = video_path.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    interval_seconds = max(0.2, float(interval_seconds))
    crop_bottom_ratio = min(1.0, max(0.2, float(crop_bottom_ratio)))
    profile = normalize_resource_profile(resource_profile)
    logger = log or (lambda _message: None)
    logger(f"OCR 开始：语言 {language}，每 {interval_seconds:g} 秒取帧；低置信度时自动扩大画面区域。")
    with tempfile.TemporaryDirectory(prefix=".ocr-temp-", dir=work_dir) as temporary:
        frame_dir = Path(temporary)
        _extract_video_frames(
            video_path=video_path,
            frame_dir=frame_dir,
            interval_seconds=interval_seconds,
            cancel_check=cancel_check,
            resource_profile=profile,
        )
        frames = sorted(frame_dir.glob("frame-*.*"))
        if not frames:
            raise RuntimeError("OCR 没有取得任何视频帧，请检查视频文件和 FFmpeg。")
        stats: dict[str, int] = {"frames": len(frames), "ocr_calls": 0, "expanded_frames": 0}
        segments = _frames_to_segments(
            frames,
            interval_seconds=interval_seconds,
            min_chars=min_chars,
            crop_bottom_ratio=crop_bottom_ratio,
            ocr_language=language,
            cancel_check=cancel_check,
            stats=stats,
            progress=progress,
        )
        logger(
            f"OCR 完成：{stats['frames']} 帧、{stats['ocr_calls']} 次识别、"
            f"{stats['expanded_frames']} 帧扩大了搜索区域；临时帧已清理。"
        )
    if not segments:
        raise RuntimeError(
            "OCR 没有找到可信的画面字幕。请检查 OCR 语言和字幕清晰度，"
            "或改用音频转写；临时帧已经清理。"
        )
    save_segments(segments_json, segments)
    write_srt(srt_path, segments, display_mode="source")
    return segments


def _ocr_region_specs(crop_bottom_ratio: float) -> list[tuple[str, float, float]]:
    bottom = min(1.0, max(0.2, crop_bottom_ratio))
    return [
        ("bottom", 1.0 - bottom, bottom),
        ("top", 0.0, 0.32),
        ("middle", 0.22, 0.56),
        ("full", 0.0, 1.0),
    ]


def _extract_video_frames(
    video_path: Path,
    frame_dir: Path,
    interval_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
    resource_profile: str = "balanced",
) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frame_dir / "frame-%06d.jpg"
    fps_value = 1.0 / interval_seconds
    frame_filter = f"fps={fps_value:.4f},scale=w='min(1920,iw)':h=-2"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            frame_filter,
            "-vsync",
            "0",
            "-q:v",
            "2",
            *ffmpeg_thread_args(resource_profile),
            str(frame_pattern),
        ],
        cancel_check=cancel_check,
        resource_profile=resource_profile,
    )


def _best_region_segments(region_results: list[tuple[str, list[Segment]]]) -> list[Segment]:
    if not region_results:
        return []
    _name, segments = max(region_results, key=lambda item: _score_segments(item[1]))
    if _score_segments(segments) <= 0:
        return []
    return segments


def _score_segments(segments: list[Segment]) -> int:
    unique_texts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if ocr_text_quality(text) < 0.55:
            continue
        if any(_similar_text(text, existing) for existing in unique_texts):
            continue
        unique_texts.append(text)
    return sum(int(len(text) * ocr_text_quality(text)) for text in unique_texts) + len(unique_texts) * 20


def _check_ocr_runtime(ocr_language: str = "eng") -> str:
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "OCR subtitle mode requires pytesseract and Pillow. Install them with: pip install pytesseract Pillow"
        ) from exc
    tesseract_cmd = _find_tesseract_command()
    if not tesseract_cmd:
        raise RuntimeError(
            "OCR subtitle mode requires the Tesseract OCR executable. Install Tesseract and add tesseract.exe to PATH."
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    language = _normalize_ocr_language(ocr_language)
    available = available_ocr_languages(tesseract_cmd)
    missing = [item for item in language.split("+") if item not in available]
    if missing:
        raise RuntimeError(
            f"Tesseract 缺少 OCR 语言包：{', '.join(missing)}。"
            f"当前已安装：{', '.join(sorted(available)) or '无法读取'}。"
        )
    return language


def _find_tesseract_command() -> str | None:
    found = resolve_command("tesseract")
    if found:
        return str(found)
    for candidate in [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _normalize_ocr_language(value: str | None) -> str:
    language = str(value or "eng").strip().lower()
    if not re.fullmatch(r"[a-z0-9_+-]{2,80}", language):
        raise ValueError("OCR 语言代码无效；请使用 eng、chi_sim 或 eng+chi_sim 这类 Tesseract 代码。")
    return language


def available_ocr_languages(tesseract_cmd: str | None = None) -> frozenset[str]:
    command = tesseract_cmd or _find_tesseract_command()
    if not command:
        return frozenset()
    try:
        completed = subprocess.run(
            [command, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    lines = (completed.stdout + "\n" + completed.stderr).splitlines()
    return frozenset(
        line.strip().lower()
        for line in lines
        if re.fullmatch(r"[A-Za-z0-9_+-]+", line.strip())
        and not line.lower().startswith("list of available")
    )


def _frames_to_segments(
    frames: list[Path],
    interval_seconds: float,
    min_chars: int,
    crop_bottom_ratio: float = 0.30,
    ocr_language: str = "eng",
    cancel_check: Callable[[], bool] | None = None,
    stats: dict[str, int] | None = None,
    progress: Callable[[float], None] | None = None,
) -> list[Segment]:
    import pytesseract
    from PIL import Image

    counters = stats if stats is not None else {}
    counters.setdefault("frames", len(frames))
    counters.setdefault("ocr_calls", 0)
    counters.setdefault("expanded_frames", 0)
    segments: list[Segment] = []
    active_text = ""
    active_start = 0.0
    active_end = 0.0
    active_hits = 0
    preferred_region = "bottom"

    def finish_active() -> None:
        """Commit only text that was confirmed by neighbouring frames.

        A single Tesseract pass frequently interprets foliage, soil, UI chrome,
        or compression blocks as words.  Real burned-in captions normally
        survive across adjacent samples, while that background noise changes
        from frame to frame.  Requiring temporal confirmation is therefore a
        stronger signal than accepting a plausible-looking string in one
        isolated image.
        """
        nonlocal active_text, active_start, active_end, active_hits
        if active_text and active_hits >= 2:
            segments.append(Segment(active_start, active_end, active_text))
        active_text = ""
        active_hits = 0

    for index, frame in enumerate(frames):
        _check_cancel(cancel_check)
        start = index * interval_seconds
        end = start + interval_seconds
        with Image.open(frame) as image:
            candidate, attempted = _adaptive_frame_candidate(
                pytesseract,
                image,
                crop_bottom_ratio=crop_bottom_ratio,
                language=ocr_language,
                preferred_region=preferred_region,
            )
        counters["ocr_calls"] += attempted
        if attempted > 1:
            counters["expanded_frames"] += 1
        text = candidate.text if candidate else ""
        if candidate:
            preferred_region = candidate.region
        if progress:
            progress((index + 1) / max(1, len(frames)))
        if len(text) < min_chars:
            finish_active()
            continue
        if active_text and _similar_text(active_text, text):
            active_text = _prefer_longer(active_text, text)
            active_end = end
            active_hits += 1
            continue
        finish_active()
        active_text = text
        active_start = start
        active_end = end
        active_hits = 1

    finish_active()
    return _smooth_segments(segments)


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise CancellationRequested("任务已被用户中断")


def _adaptive_frame_candidate(
    pytesseract,
    image,
    *,
    crop_bottom_ratio: float,
    language: str,
    preferred_region: str = "bottom",
) -> tuple[OCRCandidate | None, int]:
    specs = {name: (y_ratio, height_ratio) for name, y_ratio, height_ratio in _ocr_region_specs(crop_bottom_ratio)}
    order = list(dict.fromkeys([preferred_region, "bottom", "full", "top", "middle"]))
    best: OCRCandidate | None = None
    attempted = 0
    for region in order:
        if region not in specs:
            continue
        attempted += 1
        candidate = _ocr_region_candidate(pytesseract, image, region, *specs[region], language=language)
        if candidate and (best is None or candidate.score > best.score):
            best = candidate
        if candidate and candidate.score >= 0.74:
            break
    return (best if best and best.score >= 0.48 else None), attempted


def _ocr_region_candidate(
    pytesseract,
    image,
    region: str,
    y_ratio: float,
    height_ratio: float,
    *,
    language: str,
) -> OCRCandidate | None:
    width, height = image.size
    top = max(0, min(height - 1, round(height * y_ratio)))
    bottom = max(top + 1, min(height, round(height * (y_ratio + height_ratio))))
    cropped = image.crop((0, top, width, bottom))
    prepared = _prepare_ocr_image(cropped)
    text, confidence = _ocr_frame_candidate(pytesseract, prepared, language=language, sparse=region == "full")
    quality = ocr_text_quality(text)
    if not text or quality <= 0:
        return None
    score = min(1.0, quality * 0.58 + confidence * 0.42)
    return OCRCandidate(text=text, score=score, region=region)


def _prepare_ocr_image(image):
    from PIL import ImageOps

    image = ImageOps.grayscale(image)
    width, height = image.size
    if width < 1600:
        scale = 1600 / max(1, width)
        image = image.resize((int(width * scale), int(height * scale)))
    image = ImageOps.autocontrast(image)
    return image


def _ocr_frame_text(pytesseract, image, language: str = "eng") -> str:
    return _ocr_frame_candidate(pytesseract, image, language=language)[0]


def _ocr_frame_candidate(
    pytesseract,
    image,
    *,
    language: str = "eng",
    sparse: bool = False,
) -> tuple[str, float]:
    psm = 11 if sparse else 6
    config = f"--psm {psm}"
    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=12,
        )
    except Exception:
        try:
            fallback = _clean_ocr_fallback_text(
                pytesseract.image_to_string(image, lang=language, config=config, timeout=12)
            )
        except Exception:
            return "", 0.0
        return fallback, 0.45 if fallback else 0.0

    grouped: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
    texts = data.get("text", [])
    for index, raw_word in enumerate(texts):
        word = _clean_ocr_word(raw_word)
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = 0.0
        if confidence < 25:
            continue
        key = (
            int(data.get("block_num", [0] * len(texts))[index]),
            int(data.get("par_num", [0] * len(texts))[index]),
            int(data.get("line_num", [0] * len(texts))[index]),
        )
        grouped.setdefault(key, []).append((word, confidence))

    candidates: list[tuple[str, float]] = []
    for entries in grouped.values():
        line = _clean_ocr_text(" ".join(word for word, _confidence in entries))
        if _looks_like_subtitle_line(line):
            candidates.append((line, sum(confidence for _word, confidence in entries) / len(entries)))
    if candidates:
        text = _dedupe_joined_lines([line for line, _confidence in candidates])
        confidence = sum(value for _line, value in candidates) / len(candidates) / 100.0
        return text, min(1.0, max(0.0, confidence))

    try:
        fallback = _clean_ocr_fallback_text(
            pytesseract.image_to_string(image, lang=language, config=config, timeout=12)
        )
    except Exception:
        return "", 0.0
    return (fallback, 0.42) if _looks_like_subtitle_line(fallback) else ("", 0.0)


def _clean_ocr_word(word: str) -> str:
    word = re.sub(r"[\[\]{}<>|_~`=^*#\\]", "", word or "").strip()
    word = word.strip(".,!?;:\"'()")
    if not any(character.isalpha() for character in word):
        return ""
    if len(word) > 28:
        return ""
    alpha_count = sum(1 for character in word if character.isalpha())
    if alpha_count < max(1, len(word) * 0.45):
        return ""
    return word


def _clean_ocr_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        cleaned = cleaned.strip("|_~`·•")
        cleaned = re.sub(r"(?<=\w)[/\\;:|](?=\w)", " ", cleaned)
        cleaned = re.sub(r"[^\w\s.,!?;:'\"()，。！？；：、…·-]", " ", cleaned, flags=re.UNICODE)
        cleaned = cleaned.replace("_", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if not any(character.isalpha() for character in cleaned):
            continue
        lines.append(cleaned)
    joined = " ".join(lines)
    joined = re.sub(r"\s+([,.!?;:])", r"\1", joined)
    return joined.strip()


def _clean_ocr_fallback_text(text: str) -> str:
    candidates = []
    for line in text.splitlines():
        cleaned = _clean_ocr_text(line)
        if _looks_like_subtitle_line(cleaned):
            candidates.append(cleaned)
    return _dedupe_joined_lines(candidates)


def _looks_like_subtitle_line(line: str) -> bool:
    return ocr_text_quality(line) >= 0.55


def ocr_text_quality(line: str) -> float:
    line = _clean_ocr_text(line)
    if len(line) < 4:
        return 0.0
    visible = [character for character in line if not character.isspace()]
    text_chars = [character for character in visible if character.isalpha() or character.isdigit()]
    if not visible or len(text_chars) / len(visible) < 0.55:
        return 0.0
    if re.search(r"([^\w\s])\1{3,}", line, flags=re.UNICODE):
        return 0.0

    cjk_count = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", line))
    if cjk_count:
        if cjk_count < 2 and len(text_chars) < 5:
            return 0.0
        density = min(1.0, len(text_chars) / max(6, len(visible)))
        return min(0.92, 0.60 + density * 0.25 + min(0.07, cjk_count / 100))

    words = re.findall(r"[^\W\d_][\w'-]*", line, flags=re.UNICODE)
    if not words:
        return 0.0
    if len(words) == 1 and len(words[0]) < 5:
        return 0.0
    plausible = sum(1 for word in words if _generic_word_quality(word) > 0)
    plausible_ratio = plausible / len(words)
    if plausible_ratio < 0.60:
        return 0.0
    if len(words) >= 4 and sum(1 for word in words if len(word) <= 2) / len(words) > 0.70:
        return 0.0
    length_score = min(0.16, len(text_chars) / 120)
    return min(0.94, 0.52 + plausible_ratio * 0.24 + length_score)


def _generic_word_quality(word: str) -> float:
    raw = word.strip("'-")
    value = raw.lower()
    if not value:
        return 0.0
    if len(value) <= 2:
        return 0.6 if value.isalpha() else 0.0
    if re.fullmatch(r"([a-z])\1{2,}", value):
        return 0.0
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", value):
        return 0.0
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return 0.0
    if all(ord(character) < 128 for character in letters):
        return 0.85 if re.search(r"[aeiouy]", value) or raw.isupper() else 0.25
    return 0.85


def _looks_like_transcript_sentence(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z']*", line)
    if len(words) < 4:
        return False
    lower_words = [word.lower() for word in words]
    known_hits = sum(1 for word in lower_words if _is_known_word(word))
    wordish_hits = sum(1 for word in lower_words if _looks_wordish(word))
    short_ratio = sum(1 for word in lower_words if len(word) <= 2) / len(lower_words)
    known_ratio = known_hits / len(lower_words)
    gibberish_ratio = (len(lower_words) - wordish_hits) / len(lower_words)
    return (
        known_hits >= max(2, int(len(lower_words) * 0.35))
        and known_ratio >= 0.30
        and gibberish_ratio <= 0.25
        and short_ratio <= 0.55
        and any(word in lower_words for word in {"i", "you", "we", "it", "that", "this", "the", "is", "are", "was", "to", "he's", "let's"})
    )


def _looks_like_visual_label(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z']*", line)
    lower_words = [word.lower() for word in words]
    return len(words) >= 2 and sum(1 for word in lower_words if word in DOMAIN_SUBTITLE_WORDS) >= 2


def _looks_like_lower_third_text(line: str) -> bool:
    normalized = line.lower()
    return bool(
        re.search(r"\bf[ -]?1\b", normalized)
        or re.search(r"\bformula one\b", normalized)
        or re.search(r"\bworld champion\b", normalized)
        or re.search(r"\bmax verstappen\b", normalized)
    )


def _is_known_logo_text(line: str) -> bool:
    words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z']*", line)}
    return {"panasonic", "automotive"}.issubset(words)


def _is_known_word(word: str) -> bool:
    return word in COMMON_SUBTITLE_WORDS or word in DOMAIN_SUBTITLE_WORDS


def _looks_wordish(word: str) -> bool:
    if _is_known_word(word):
        return True
    if len(word) <= 2:
        return word in COMMON_SUBTITLE_WORDS
    if len(word) <= 4:
        return False
    if len(word) > 10 and not re.search(r"(ing|tion|ment|ship|ally|ology|sport)s?$", word):
        return False
    if re.fullmatch(r"([a-z])\1{2,}", word):
        return False
    if not re.search(r"[aeiouy]", word):
        return word.upper() in {"F1", "GT"} or word in {"gt"}
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word):
        return False
    return True


def _legacy_looks_like_subtitle_line(line: str) -> bool:
    if len(line) < 4:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z']*", line)
    if len(words) < 2:
        return False
    alpha_chars = len(re.findall(r"[A-Za-z]", line))
    if alpha_chars < len(line) * 0.55:
        return False
    common = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "can",
        "do",
        "for",
        "from",
        "got",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "just",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "they",
        "this",
        "to",
        "up",
        "we",
        "with",
        "you",
    }
    common_hits = sum(1 for word in words if word.lower() in common)
    return common_hits >= 1 or len(words) >= 5


def _dedupe_joined_lines(lines: list[str]) -> str:
    output: list[str] = []
    for line in lines:
        if any(_similar_text(line, existing) for existing in output):
            continue
        output.append(line)
    return " ".join(output)


def _similar_text(left: str, right: str) -> bool:
    if left == right:
        return True
    return SequenceMatcher(None, left.lower(), right.lower()).ratio() >= 0.82


def _prefer_longer(left: str, right: str) -> str:
    return right if len(right) > len(left) else left


def _smooth_segments(segments: list[Segment]) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        if segment.end - segment.start < 0.4:
            continue
        if ocr_text_quality(segment.text) < 0.55:
            continue
        if output and _similar_text(output[-1].text, segment.text) and segment.start - output[-1].end <= 1.5:
            output[-1] = Segment(output[-1].start, segment.end, _prefer_longer(output[-1].text, segment.text))
        else:
            output.append(segment)
    return output

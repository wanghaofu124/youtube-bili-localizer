from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import shutil
from typing import Callable

from .models import Segment, save_segments
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


def extract_ocr_subtitles(
    video_path: Path,
    work_dir: Path,
    segments_json: Path,
    srt_path: Path,
    interval_seconds: float = 1.0,
    crop_bottom_ratio: float = 0.30,
    min_chars: int = 3,
    cancel_check: Callable[[], bool] | None = None,
) -> list[Segment]:
    """Extract burned-in English subtitles from video frames with OCR.

    The extractor scans several common caption regions because Shorts and meme
    videos often place text near the top instead of the bottom.
    """
    require_command("ffmpeg")
    _check_ocr_runtime()

    video_path = video_path.resolve()
    interval_seconds = max(0.5, float(interval_seconds))
    crop_bottom_ratio = min(1.0, max(0.2, float(crop_bottom_ratio)))
    region_results: list[tuple[str, list[Segment]]] = []
    for region_name, y_ratio, height_ratio in _ocr_region_specs(crop_bottom_ratio):
        frame_dir = work_dir / f"ocr_frames_{region_name}"
        _extract_region_frames(
            video_path=video_path,
            frame_dir=frame_dir,
            interval_seconds=interval_seconds,
            y_ratio=y_ratio,
            height_ratio=height_ratio,
            cancel_check=cancel_check,
        )
        frames = sorted(frame_dir.glob("frame-*.png"))
        if not frames:
            continue
        region_segments = _frames_to_segments(frames, interval_seconds=interval_seconds, min_chars=min_chars)
        if region_segments:
            region_results.append((region_name, region_segments))

    segments = _best_region_segments(region_results)
    if not segments:
        raise RuntimeError(
            "OCR did not find readable subtitle text. Try a clearer video, larger subtitles, or use audio transcription."
        )
    save_segments(segments_json, segments)
    write_srt(srt_path, segments, display_mode="source")
    return segments


def _ocr_region_specs(crop_bottom_ratio: float) -> list[tuple[str, float, float]]:
    bottom = min(1.0, max(0.2, crop_bottom_ratio))
    return [
        ("bottom", 1.0 - bottom, bottom),
        ("top", 0.0, 0.30),
        ("upper", 0.0, 0.45),
        ("middle", 0.25, 0.50),
    ]


def _extract_region_frames(
    video_path: Path,
    frame_dir: Path,
    interval_seconds: float,
    y_ratio: float,
    height_ratio: float,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frame_dir / "frame-%06d.png"
    fps_value = 1.0 / interval_seconds
    crop_filter = f"fps={fps_value:.4f},crop=iw:ih*{height_ratio:.3f}:0:ih*{y_ratio:.3f}"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            crop_filter,
            "-vsync",
            "0",
            str(frame_pattern),
        ],
        cancel_check=cancel_check,
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


def _check_ocr_runtime() -> None:
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


def _find_tesseract_command() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _frames_to_segments(frames: list[Path], interval_seconds: float, min_chars: int) -> list[Segment]:
    import pytesseract
    from PIL import Image

    segments: list[Segment] = []
    active_text = ""
    active_start = 0.0
    active_end = 0.0

    for index, frame in enumerate(frames):
        start = index * interval_seconds
        end = start + interval_seconds
        with Image.open(frame) as image:
            prepared = _prepare_ocr_image(image)
            text = _ocr_frame_text(pytesseract, prepared)
        if len(text) < min_chars:
            if active_text:
                segments.append(Segment(active_start, active_end, active_text))
                active_text = ""
            continue
        if active_text and _similar_text(active_text, text):
            active_text = _prefer_longer(active_text, text)
            active_end = end
            continue
        if active_text:
            segments.append(Segment(active_start, active_end, active_text))
        active_text = text
        active_start = start
        active_end = end

    if active_text:
        segments.append(Segment(active_start, active_end, active_text))
    return _smooth_segments(segments)


def _prepare_ocr_image(image):
    from PIL import ImageOps

    image = ImageOps.grayscale(image)
    width, height = image.size
    if width < 1600:
        scale = 1600 / max(1, width)
        image = image.resize((int(width * scale), int(height * scale)))
    image = ImageOps.autocontrast(image)
    return image


def _ocr_frame_text(pytesseract, image) -> str:
    try:
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return _clean_ocr_fallback_text(pytesseract.image_to_string(image, lang="eng", config="--psm 6"))

    grouped: dict[tuple[int, int, int], list[str]] = {}
    texts = data.get("text", [])
    for index, raw_word in enumerate(texts):
        word = _clean_ocr_word(raw_word)
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = 0.0
        if confidence < 30:
            continue
        key = (
            int(data.get("block_num", [0] * len(texts))[index]),
            int(data.get("par_num", [0] * len(texts))[index]),
            int(data.get("line_num", [0] * len(texts))[index]),
        )
        grouped.setdefault(key, []).append(word)

    candidates: list[str] = []
    for words in grouped.values():
        line = _clean_ocr_text(" ".join(words))
        if _looks_like_subtitle_line(line):
            candidates.append(line)
    if candidates:
        return _dedupe_joined_lines(candidates)

    fallback = _clean_ocr_fallback_text(pytesseract.image_to_string(image, lang="eng", config="--psm 6"))
    return fallback if _looks_like_subtitle_line(fallback) else ""


def _clean_ocr_word(word: str) -> str:
    word = re.sub(r"[\[\]{}<>|_~`=^*#\\]", "", word or "").strip()
    word = word.strip(".,!?;:\"'()")
    if not re.search(r"[A-Za-z]", word):
        return ""
    if len(word) > 28:
        return ""
    alpha_count = len(re.findall(r"[A-Za-z]", word))
    if alpha_count < max(1, len(word) * 0.45):
        return ""
    return word


def _clean_ocr_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        cleaned = cleaned.strip("|_~`·•")
        cleaned = re.sub(r"(?<=[A-Za-z])[/\\;:|](?=[A-Za-z])", " ", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9.,!?;:'\"() -]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if not re.search(r"[A-Za-z]", cleaned):
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
    if _is_known_logo_text(line):
        return 0.62
    if _looks_like_visual_label(line):
        return 0.72
    if _looks_like_lower_third_text(line):
        return 0.76
    if _looks_like_transcript_sentence(line):
        return 0.82

    words = re.findall(r"[A-Za-z][A-Za-z']*", line)
    if len(words) < 2:
        return 0.0
    lower_words = [word.lower() for word in words]
    alpha_chars = len(re.findall(r"[A-Za-z]", line))
    if alpha_chars < len(line) * 0.55:
        return 0.0

    short_ratio = sum(1 for word in lower_words if len(word) <= 2) / len(lower_words)
    if len(lower_words) >= 4 and short_ratio > 0.65:
        return 0.0

    known_hits = sum(1 for word in lower_words if _is_known_word(word))
    wordish_hits = sum(1 for word in lower_words if _looks_wordish(word))
    gibberish_hits = len(lower_words) - wordish_hits
    gibberish_ratio = gibberish_hits / len(lower_words)
    known_ratio = known_hits / len(lower_words)

    if len(lower_words) <= 2 and known_hits < len(lower_words):
        return 0.0
    if len(lower_words) <= 2 and not any(word in DOMAIN_SUBTITLE_WORDS for word in lower_words):
        return 0.0
    if known_hits < 2:
        return 0.0
    if gibberish_ratio > 0.30:
        return 0.0
    if known_hits == 0 and len(lower_words) < 5:
        return 0.0
    if len(lower_words) >= 5 and known_ratio < 0.18:
        return 0.0
    if short_ratio > 0.5 and known_ratio < 0.45:
        return 0.0

    score = 0.45 + min(0.35, known_ratio * 0.7) + min(0.2, (1.0 - gibberish_ratio) * 0.2)
    return min(1.0, score)


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

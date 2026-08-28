from __future__ import annotations

from difflib import SequenceMatcher
import re

from .models import Segment
from .ocr_subtitle import ocr_text_quality


def merge_audio_ocr_segments(audio_segments: list[Segment], ocr_segments: list[Segment]) -> list[Segment]:
    """Merge speech transcript and on-screen OCR text on a shared timeline."""
    ocr_segments = [item for item in ocr_segments if ocr_text_quality(item.text) >= 0.55]
    if not audio_segments:
        return sorted(ocr_segments, key=lambda item: (item.start, item.end))
    if not ocr_segments:
        return sorted(audio_segments, key=lambda item: (item.start, item.end))

    merged: list[Segment] = []
    consumed_ocr: set[int] = set()

    for audio in sorted(audio_segments, key=lambda item: (item.start, item.end)):
        overlaps = [
            (index, ocr)
            for index, ocr in enumerate(ocr_segments)
            if _overlap_seconds(audio, ocr) >= 0.2
            or _overlap_ratio(audio, ocr) >= 0.35
            or _center_inside(ocr, audio)
        ]
        useful_ocr: list[Segment] = []
        for index, ocr in overlaps:
            if index in consumed_ocr:
                continue
            if ocr_text_quality(ocr.text) < 0.75:
                consumed_ocr.add(index)
                continue
            if _similar_text(audio.text, ocr.text) and not _has_visual_label(ocr.text, audio.text):
                consumed_ocr.add(index)
                continue
            useful_ocr.append(ocr)
            consumed_ocr.add(index)

        if useful_ocr:
            start = min([audio.start, *[item.start for item in useful_ocr]])
            end = max([audio.end, *[item.end for item in useful_ocr]])
            ocr_text = " / ".join(_dedupe_texts([item.text for item in useful_ocr]))
            text = _format_merged_text(audio.text, ocr_text)
            merged.append(Segment(start=start, end=end, text=text))
        else:
            merged.append(audio)

    for index, ocr in enumerate(ocr_segments):
        if index not in consumed_ocr:
            if _should_keep_standalone_ocr(ocr):
                merged.append(ocr)

    return _smooth_merged_segments(sorted(merged, key=lambda item: (item.start, item.end)))


def _format_merged_text(audio_text: str, ocr_text: str) -> str:
    audio_text = audio_text.strip()
    ocr_text = ocr_text.strip()
    if audio_text and ocr_text:
        return f"Speech: {audio_text} | On-screen text: {ocr_text}"
    return audio_text or ocr_text


def _overlap_seconds(left: Segment, right: Segment) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _overlap_ratio(left: Segment, right: Segment) -> float:
    overlap = _overlap_seconds(left, right)
    shorter = max(0.001, min(left.end - left.start, right.end - right.start))
    return overlap / shorter


def _center_inside(inner: Segment, outer: Segment) -> bool:
    center = (inner.start + inner.end) / 2
    return outer.start <= center <= outer.end


def _similar_text(left: str, right: str) -> bool:
    left_norm = _normalize_for_compare(left)
    right_norm = _normalize_for_compare(right)
    if not left_norm or not right_norm:
        return False
    if left_norm in right_norm or right_norm in left_norm:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.78


def _has_visual_label(ocr_text: str, audio_text: str) -> bool:
    if re.search(r"\*[A-Za-z][^*]{0,24}\*", ocr_text):
        return True
    match = re.match(r"\s*([A-Za-z][A-Za-z ]{0,24}):", ocr_text)
    if not match:
        return False
    label = match.group(1).strip().lower()
    return label not in _normalize_for_compare(audio_text)


def _should_keep_standalone_ocr(ocr: Segment) -> bool:
    text = ocr.text.strip()
    quality = ocr_text_quality(text)
    cjk_count = len(re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    if cjk_count >= 4:
        return quality >= 0.70 and ocr.end - ocr.start >= 0.6
    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    if quality < 0.75 or len(words) < 3:
        return False
    if _is_brand_only_text(text):
        return False
    if ocr.end - ocr.start < 1.2 and len(words) < 5:
        return False
    return True


def _is_brand_only_text(text: str) -> bool:
    words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z']*", text)}
    brand_words = {"automotive", "bull", "dunlop", "lexus", "nissan", "panasonic", "red"}
    return bool(words) and words.issubset(brand_words | {"a", "the"})


def _normalize_for_compare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(speech|on[- ]screen text|screen text|audio)\b", " ", text)
    text = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", text)
    return text


def _dedupe_texts(texts: list[str]) -> list[str]:
    output: list[str] = []
    for text in (item.strip() for item in texts):
        if not text:
            continue
        if any(_similar_text(text, existing) for existing in output):
            continue
        output.append(text)
    return output


def _smooth_merged_segments(segments: list[Segment]) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        if output and _is_continuation(output[-1], segment):
            output[-1] = Segment(
                start=output[-1].start,
                end=max(output[-1].end, segment.end),
                text=f"{output[-1].text.rstrip()} {segment.text.lstrip()}",
            )
            continue
        if output and _similar_text(output[-1].text, segment.text) and segment.start - output[-1].end <= 0.8:
            output[-1] = Segment(
                start=output[-1].start,
                end=max(output[-1].end, segment.end),
                text=_prefer_richer_text(output[-1].text, segment.text),
            )
            continue
        output.append(segment)
    return output


def _is_continuation(previous: Segment, current: Segment) -> bool:
    if "On-screen text:" in previous.text or "On-screen text:" in current.text:
        return False
    if current.start - previous.end > 0.5:
        return False
    previous_text = previous.text.strip()
    current_text = current.text.strip()
    if not previous_text or not current_text:
        return False
    if re.search(r"[.!?。！？]$", previous_text):
        return False
    match = re.match(r"[A-Za-z']+", current_text)
    if not match:
        return False
    first = match.group(0).lower()
    return first in {"and", "as", "but", "for", "in", "of", "or", "that", "to", "with"} or first[:1].islower()


def _prefer_richer_text(left: str, right: str) -> str:
    return right if len(_normalize_for_compare(right)) > len(_normalize_for_compare(left)) else left

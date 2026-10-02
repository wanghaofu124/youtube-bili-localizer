from datetime import datetime
from pathlib import Path

from yblocalizer.util import (
    allocate_job_folder_name,
    extract_youtube_video_id,
    job_folder_label,
)


def test_extract_youtube_video_id() -> None:
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=ojCBDoEdG74") == "ojCBDoEdG74"
    assert extract_youtube_video_id("https://youtu.be/ojCBDoEdG74?t=12") == "ojCBDoEdG74"
    assert extract_youtube_video_id("https://www.youtube.com/shorts/abcDEF12345") == "abcDEF12345"
    assert extract_youtube_video_id("https://example.com/watch") is None


def test_job_folder_label_prefers_id_and_title() -> None:
    label = job_folder_label(
        title="I Tested TEMU's CHEAPEST Aquarium Products!",
        source_url="https://www.youtube.com/watch?v=ojCBDoEdG74",
    )
    assert label.startswith("ojCBDoEdG74_")
    assert "TEMU" in label or "Aquarium" in label or "I-Tested" in label


def test_allocate_job_folder_name_strips_media_extension(tmp_path: Path) -> None:
    now = datetime(2026, 9, 23, 18, 48, 12)
    name = allocate_job_folder_name(
        tmp_path,
        title="authorized-demo-10s.mp4",
        source_name="authorized-demo-10s.mp4",
        now=now,
    )
    assert name == "20260923-184812_authorized-demo-10s"
    assert ".mp4" not in name


def test_allocate_job_folder_name_unique(tmp_path: Path) -> None:
    now = datetime(2026, 9, 23, 18, 48, 12)
    first = allocate_job_folder_name(
        tmp_path,
        title="Hello World",
        source_url="https://youtu.be/abcdefghijk",
        now=now,
    )
    assert first == "20260923-184812_abcdefghijk_Hello-World"
    (tmp_path / first).mkdir()
    second = allocate_job_folder_name(
        tmp_path,
        title="Hello World",
        source_url="https://youtu.be/abcdefghijk",
        now=now,
    )
    assert second == "20260923-184812_abcdefghijk_Hello-World-2"

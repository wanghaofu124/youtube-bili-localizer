from pathlib import Path

import pytest

from yblocalizer.runtime import PipelineEvent, PipelineStage
from yblocalizer.workbench_api import DEMO_VIDEO, DemoJob, Material, WorkbenchHandler, WorkbenchJobs, _upsert_env, _resolve_output_root, _safe_options, build_server, public_path, stage_from_log
from yblocalizer.workbench_config import default_options, preflight


def test_workbench_maps_existing_pipeline_logs_to_stages() -> None:
    assert stage_from_log("1/5 Preparing source video...") == ("准备素材", 12)
    assert stage_from_log("3/5 Transcribing audio with faster-whisper...") == ("转写字幕", 48)
    assert stage_from_log("4/5 Translating subtitles with deepseek...") == ("翻译字幕", 72)
    assert stage_from_log("5/5 Rendering hard subtitles with ffmpeg...") == ("渲染成片", 90)


def test_workbench_job_never_moves_progress_backwards() -> None:
    material = Material("demo", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True)
    job = DemoJob(id="demo", material=material, device="cuda", compute_type="float16")
    job.apply_event(PipelineEvent(PipelineStage.RENDERING, 90, "rendering"))
    job.apply_event(PipelineEvent(PipelineStage.TRANSCRIBING, 48, "transcribing"))
    assert job.progress == 90
    assert job.stage == "语音转写"


def test_workbench_default_subtitle_source_is_audio() -> None:
    assert default_options()["subtitle_source"] == "audio"


def test_workbench_public_path_is_relative_to_project(tmp_path: Path) -> None:
    assert public_path(Path("demo") / "authorized-demo-10s.mp4") == "demo/authorized-demo-10s.mp4"
    assert public_path(tmp_path / "private.mp4") == "private.mp4"


def test_workbench_creates_authorized_url_job(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = WorkbenchJobs()
    monkeypatch.setattr("yblocalizer.workbench_api.threading.Thread.start", lambda self: None)
    job = jobs.create_url("https://www.youtube.com/watch?v=demo", "cuda", "float16", authorized=True, options={"subtitle_source": "ocr", "font_size": 30, "output_dir": "outputs/custom"})
    assert job.material.source_url == "https://www.youtube.com/watch?v=demo"
    assert job.material.path is None
    assert job.options["subtitle_source"] == "ocr"
    assert job.options["font_size"] == 30
    assert job.output_root == _resolve_output_root("outputs/custom")


def test_workbench_cancel_marks_running_job_as_cancelling(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = WorkbenchJobs()
    monkeypatch.setattr("yblocalizer.workbench_api.threading.Thread.start", lambda self: None)
    job = jobs.create_url("https://www.youtube.com/watch?v=demo", "cpu", "int8", authorized=True)
    job.status = "running"
    cancelled = jobs.cancel(job.id)
    assert cancelled is job
    assert cancelled.status == "cancelling"
    assert cancelled.cancellation.cancelled is True
    assert any("Cancellation requested" in line for line in cancelled.logs)


@pytest.mark.parametrize("url", ["example.com/video", "ftp://example.com/video", "https://"])
def test_workbench_rejects_invalid_url(url: str) -> None:
    with pytest.raises(ValueError, match="valid HTTP"):
        WorkbenchJobs().create_url(url, "cpu", "int8", authorized=True)


def test_workbench_options_normalize_style_and_duration() -> None:
    options = _safe_options({"subtitle_effect": "描边+阴影", "subtitle_color": "黄色", "max_seconds": "25", "font_size": "500", "publish_to_bilibili": True, "include_source_link": False, "bilibili_browser": "msedge", "tags": ["AI", "字幕"]})
    assert options["subtitle_color"] == "&H0000FFFF"
    assert options["subtitle_outline"] == 1
    assert options["subtitle_shadow"] == 1
    assert options["font_size"] == 96
    assert options["max_seconds"] == 25
    assert options["publish_to_bilibili"] is True
    assert options["include_source_link"] is False
    assert options["bilibili_browser"] == "msedge"
    assert options["tags"] == ["AI", "字幕"]


def test_workbench_options_reject_unknown_subtitle_source() -> None:
    with pytest.raises(ValueError, match="subtitle source"):
        _safe_options({"subtitle_source": "invented"})


def test_upsert_env_persists_only_the_selected_cookie_browser(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    _upsert_env(env_path, {"YBLOCALIZER_COOKIES_FROM_BROWSER": "edge"})
    assert env_path.read_text(encoding="utf-8") == "YBLOCALIZER_COOKIES_FROM_BROWSER=edge\n"


def test_native_publish_video_metadata_is_tokenized(tmp_path: Path) -> None:
    video = tmp_path / "finished.mp4"
    video.write_bytes(b"not-a-real-video")
    (tmp_path / "publish_metadata.json").write_text('{"title":"智能标题","tags":["AI","字幕"]}', encoding="utf-8")
    jobs = WorkbenchJobs()
    token = jobs.register_native_publish_file(video)
    detail = jobs.publish_file_metadata(token)
    assert detail["token"] == token
    assert detail["name"] == "finished.mp4"
    assert detail["title"] == "智能标题"
    assert detail["tags"] == ["AI", "字幕"]
    assert jobs.publish_file_from_token(token) == video.resolve()


def test_readiness_reports_cpu_float16_and_profile_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(WorkbenchHandler)
    monkeypatch.setattr(handler, "_publish_profile_process_ids", lambda browser: ["123"])
    result = handler._readiness({
        "source_url": "https://example.com/video", "authorized": True, "device": "cpu", "compute_type": "float16",
        "options": {"translator": "none", "target_lang": "en", "publish_to_bilibili": True},
    })
    assert result["ready"] is False
    assert any("float16" in issue for issue in result["issues"])
    assert any("B 站" in issue for issue in result["issues"])


def test_history_defaults_to_current_output_root_and_hides_private_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_root = tmp_path / "current"
    current_job = current_root / "job-a"
    old_job = tmp_path / "old" / "job-b"
    current_job.mkdir(parents=True)
    old_job.mkdir(parents=True)
    records = [
        {"id": "current", "title": "Current", "status": "completed", "stage": "Done", "progress": 100,
         "error": None, "output_dir": str(current_job), "rendered_video": str(current_job / "rendered.mp4"),
         "options": {"cookies_file": "secret.txt"}, "created_at": 1, "finished_at": 2},
        {"id": "old", "title": "Old", "status": "completed", "stage": "Done", "progress": 100,
         "error": None, "output_dir": str(old_job), "rendered_video": None,
         "options": {"cookies_file": "old-secret.txt"}, "created_at": 1, "finished_at": 2},
    ]
    handler = object.__new__(WorkbenchHandler)
    monkeypatch.setattr("yblocalizer.workbench_api.job_db.list_jobs", lambda limit=None: records)
    monkeypatch.setattr("yblocalizer.workbench_api._resolve_output_root", lambda value: Path(value))

    visible = handler._history_records("current", str(current_root))

    assert [item["id"] for item in visible] == ["current"]
    assert "options" not in visible[0]
    assert visible[0]["output_exists"] is True
    assert visible[0]["rendered_exists"] is False


def test_history_current_scope_is_used_for_server_side_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "output"
    selected = root / "job"
    selected.mkdir(parents=True)
    records = [{"id": "visible", "output_dir": str(selected)}]
    handler = object.__new__(WorkbenchHandler)
    monkeypatch.setattr("yblocalizer.workbench_api.job_db.list_jobs", lambda limit=None: records)
    monkeypatch.setattr("yblocalizer.workbench_api._resolve_output_root", lambda value: Path(value))
    assert [item["id"] for item in handler._history_records("current", str(root), include_private=True)] == ["visible"]


def test_diagnostics_reports_missing_tools_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(WorkbenchHandler)
    monkeypatch.setattr("yblocalizer.workbench_api.shutil.which", lambda command: "C:/tools/ffmpeg.exe" if command == "ffmpeg" else None)
    result = handler._diagnostics()
    assert result["ready"] is True
    assert result["checks"][0]["available"] is True
    assert all("C:/" not in item["message"] for item in result["checks"])


def test_preflight_treats_tesseract_as_conditional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.workbench_config.shutil.which", lambda command: "tool" if command == "ffmpeg" else None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "configured")
    audio = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "audio"}}, "outputs")
    merged = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "merged"}}, "outputs")
    ocr = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "ocr"}}, "outputs")
    assert audio["ready"] and not audio["warnings"]
    assert merged["ready"] and merged["warnings"][0]["code"] == "ocr_unavailable"
    assert not ocr["ready"] and ocr["blocking"][-1]["code"] == "tesseract_required"


def test_workbench_server_can_bind_an_ephemeral_loopback_port() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
    finally:
        server.server_close()

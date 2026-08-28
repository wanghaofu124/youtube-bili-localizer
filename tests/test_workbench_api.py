from pathlib import Path
from http.client import HTTPConnection
from io import BytesIO
import json
import socket
import threading

import pytest

from yblocalizer import db as job_db
from yblocalizer.runtime import PipelineEvent, PipelineStage
from yblocalizer.workbench_api import DEMO_VIDEO, DemoJob, Material, WorkbenchHandler, WorkbenchJobs, _ffconcat_path, _upsert_env, _resolve_output_root, _safe_options, build_server, public_path, stage_from_log
from yblocalizer.workbench_config import InputValidationError, default_options, normalize_options, options_for_public, options_for_storage, options_from_storage, preflight


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
    assert default_options()["download_quality"] == "1080p"
    assert default_options()["render_encoder"] == "auto"


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
    with pytest.raises(ValueError, match="HTTP"):
        WorkbenchJobs().create_url(url, "cpu", "int8", authorized=True)


def test_workbench_options_normalize_style_and_duration() -> None:
    options = _safe_options({"subtitle_effect": "描边+阴影", "subtitle_color": "黄色", "max_seconds": 25, "font_size": 30, "publish_to_bilibili": True, "include_source_link": False, "bilibili_browser": "msedge", "tags": ["AI", "字幕"]})
    assert options["subtitle_color"] == "&H0000FFFF"
    assert options["subtitle_outline"] == 1
    assert options["subtitle_shadow"] == 1
    assert options["font_size"] == 30
    assert options["max_seconds"] == 25
    assert options["publish_to_bilibili"] is True
    assert options["include_source_link"] is False
    assert options["bilibili_browser"] == "msedge"
    assert options["tags"] == ["AI", "字幕"]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("max_seconds", 0.7, "integer_required"),
        ("max_seconds", -1, "out_of_range"),
        ("max_seconds", 2**32, "out_of_range"),
        ("ocr_interval", float("nan"), "non_finite_number"),
        ("ocr_interval", float("inf"), "non_finite_number"),
        ("smart_translation", "false", "invalid_type"),
        ("publish_to_bilibili", 0, "invalid_type"),
    ],
)
def test_strict_options_reject_ambiguous_or_unbounded_values(field: str, value: object, code: str) -> None:
    with pytest.raises(InputValidationError) as captured:
        _safe_options({field: value})
    assert captured.value.field == field
    assert captured.value.code == code


def test_strict_options_reject_oversized_text_and_tags() -> None:
    with pytest.raises(InputValidationError, match="200"):
        _safe_options({"title": "x" * 201})
    with pytest.raises(InputValidationError, match="20"):
        _safe_options({"tags": [f"tag-{index}" for index in range(21)]})
    with pytest.raises(InputValidationError, match="30"):
        _safe_options({"tags": ["x" * 31]})


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/video", "http://localhost/video", "http://192.168.1.10/video", "file:///C:/video.mp4"],
)
def test_workbench_rejects_private_or_local_urls(url: str) -> None:
    with pytest.raises(InputValidationError):
        WorkbenchJobs().create_url(url, "cpu", "int8", authorized=True)


def test_ffconcat_path_escapes_apostrophes(tmp_path: Path) -> None:
    value = _ffconcat_path(tmp_path / "导演's clip.mp4")
    assert "导演" in value
    assert "'\\''" in value


def test_invalid_local_media_never_enters_material_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = WorkbenchJobs()
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="为空"):
        jobs.add_local_file(empty, authorized=True)

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    monkeypatch.setattr("yblocalizer.workbench_api._probe_media", lambda *_args: (None, None, None))
    with pytest.raises(ValueError, match="有效视频"):
        jobs.add_local_file(corrupt, authorized=True)
    assert [item["id"] for item in jobs.materials()] == ["demo"]


def test_failed_uploaded_media_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.workbench_api._probe_media", lambda *_args: (None, None, None))
    jobs = WorkbenchJobs()
    with pytest.raises(ValueError, match="有效视频"):
        jobs.add_upload("broken.mp4", BytesIO(b"broken"), authorized=True)
    from yblocalizer import workbench_api
    assert not list(workbench_api.UPLOAD_ROOT.glob("*broken.mp4"))


def test_style_normalization_is_idempotent() -> None:
    first = normalize_options({
        "subtitle_color": "黄色", "subtitle_outline_color": "蓝色",
        "subtitle_effect": "描边+阴影",
    }, "outputs/test")
    second = normalize_options(first, "outputs/test")
    assert second["subtitle_color"] == first["subtitle_color"] == "&H0000FFFF"
    assert second["subtitle_outline_color"] == first["subtitle_outline_color"] == "&H00FF0000"
    assert (second["subtitle_outline"], second["subtitle_shadow"]) == (1, 1)
    assert options_for_public(second)["subtitle_effect"] == "描边+阴影"


def test_storage_options_never_contain_local_credentials() -> None:
    options = _safe_options({
        "cookies_file": "C:/private/cookies.txt",
        "youtube_proxy": "http://name:password@example.test:8080",
    })
    stored = options_for_storage(options)
    assert stored["cookies_file"] is None
    assert stored["youtube_proxy"] is None
    assert stored["cookies_file_configured"] is True
    assert stored["youtube_proxy_configured"] is True


def test_restored_task_resolves_current_local_secrets_without_exposing_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YBLOCALIZER_COOKIES_FILE", "C:/current/cookies.txt")
    monkeypatch.setenv("YBLOCALIZER_YOUTUBE_PROXY", "http://user:secret@127.0.0.1:7890")
    restored = options_from_storage({
        "translator": "none", "target_lang": "en",
        "cookies_file_configured": True, "youtube_proxy_configured": True,
    }, "outputs/test")
    assert restored["cookies_file"] == "C:/current/cookies.txt"
    assert restored["youtube_proxy"] == "http://user:secret@127.0.0.1:7890"
    public = json.dumps(options_for_public(restored), ensure_ascii=False)
    assert "current/cookies" not in public
    assert "user:secret" not in public


def test_workbench_options_reject_unknown_subtitle_source() -> None:
    with pytest.raises(InputValidationError) as captured:
        _safe_options({"subtitle_source": "invented"})
    assert captured.value.field == "subtitle_source"
    assert captured.value.code == "invalid_choice"


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


def test_subtitle_saves_keep_the_previous_valid_edit(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    source = job.work_dir / "segments.translated.json"
    save_segments(source, [Segment(0, 1, "hello", "你好")])
    job.artifacts.translated_segments = str(source)
    job.stages["translate"]["status"] = "completed"

    jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "第一次", "deleted": False}])
    first = Path(job.artifacts.translated_segments or "")
    jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "第二次", "deleted": False}])
    second = Path(job.artifacts.translated_segments or "")

    assert first.name == "segments.translated.edited.r1.json"
    assert second.name == "segments.translated.edited.r2.json"
    assert first.is_file() and second.is_file()


def test_failed_subtitle_save_removes_inactive_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    source = job.work_dir / "segments.translated.json"
    save_segments(source, [Segment(0, 1, "hello", "你好")])
    job.artifacts.translated_segments = str(source)
    job.stages["translate"]["status"] = "completed"
    monkeypatch.setattr("yblocalizer.workbench_api.write_srt", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "新字幕", "deleted": False}])

    assert job.artifacts.translated_segments == str(source)
    assert not (job.work_dir / "segments.translated.edited.r1.json").exists()
    assert not (job.work_dir / "zh.edited.r1.srt").exists()


def test_shutdown_marks_unresponsive_stage_as_interrupted(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    release = threading.Event()
    worker = threading.Thread(target=lambda: release.wait(2), daemon=True)
    worker.start()
    job.status, job.current_stage = "running", "extract"
    job.auto_run = True
    job.stages["extract"]["status"] = "running"
    jobs._running_id, jobs._worker = job.id, worker

    jobs.shutdown(timeout=0)

    assert job.cancellation.cancelled
    assert job.status == "interrupted"
    assert job.stages["extract"]["status"] == "interrupted"
    assert job.next_stage == "extract"
    assert job.auto_run is False
    release.set()
    worker.join(2)


def test_modified_artifacts_commit_as_one_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    old_rendered = job.work_dir / "rendered.mp4"
    old_rendered.write_bytes(b"old-render")
    new_raw = job.work_dir / "muted-test.mp4"
    new_raw.write_bytes(b"new-raw")
    subtitle = job.work_dir / "zh.edited.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    segments = job.work_dir / "segments.translated.edited.json"
    segments.write_text('[{"start":0,"end":1,"text":"hello","translated_text":"你好"}]', encoding="utf-8")
    job.rendered_video = old_rendered
    job.artifacts.rendered_video = str(old_rendered)
    job.artifacts.translated_srt = str(subtitle)
    job.artifacts.translated_segments = str(segments)
    for row in job.stages.values():
        row["status"] = "completed"

    monkeypatch.setattr("yblocalizer.workbench_api.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workbench_api._probe_media", lambda *_args: (5.0, 1280, 720))
    def fake_burn(_video, _subtitle, target, **_kwargs):
        target.write_bytes(b"new-render")
        return target
    monkeypatch.setattr("yblocalizer.workbench_api.burn_subtitles", fake_burn)

    jobs._finish_modify(job, new_raw, "muted", subtitle, segments, "mute")

    assert job.artifacts.original_video == str(source.resolve())
    assert job.artifacts.raw_video == str(new_raw)
    assert job.artifacts.rendered_video == str(old_rendered)
    assert old_rendered.read_bytes() == b"new-render"
    assert job.artifacts.revision == 1
    assert job.artifacts.last_edit == "mute"
    manifest = json.loads((job.work_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["raw_video"] == str(new_raw)
    assert manifest["artifacts"]["revision"] == 1


def test_failed_modify_keeps_previous_rendered_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    previous = job.work_dir / "rendered.mp4"
    previous.write_bytes(b"previous")
    job.rendered_video = previous
    job.artifacts.rendered_video = str(previous)
    new_raw = job.work_dir / "trimmed-test.mp4"
    new_raw.write_bytes(b"candidate")
    subtitle = job.work_dir / "zh.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    monkeypatch.setattr("yblocalizer.workbench_api.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workbench_api.burn_subtitles", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")))

    with pytest.raises(RuntimeError, match="render failed"):
        jobs._finish_modify(job, new_raw, "trim", subtitle, operation="trim")

    assert previous.read_bytes() == b"previous"
    assert job.rendered_video == previous
    assert job.artifacts.raw_video == str(source.resolve())


def test_cancelled_modify_keeps_previous_rendered_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    previous = job.work_dir / "rendered.mp4"
    previous.write_bytes(b"previous")
    job.rendered_video = previous
    job.artifacts.rendered_video = str(previous)
    new_raw = job.work_dir / "muted-test.mp4"
    new_raw.write_bytes(b"candidate")
    subtitle = job.work_dir / "zh.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    monkeypatch.setattr("yblocalizer.workbench_api.validate_media", lambda *_args: True)
    def cancel_after_render(_video, _subtitle, target, **_kwargs):
        target.write_bytes(b"candidate-render")
        job.cancellation.cancel()
        return target
    monkeypatch.setattr("yblocalizer.workbench_api.burn_subtitles", cancel_after_render)

    from yblocalizer.runtime import CancellationRequested
    with pytest.raises(CancellationRequested):
        jobs._finish_modify(job, new_raw, "mute", subtitle, operation="mute")

    assert previous.read_bytes() == b"previous"
    assert job.artifacts.raw_video == str(source.resolve())


def test_multiple_mute_ranges_use_a_valid_filter_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    subtitle = job.work_dir / "zh.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    segments = job.work_dir / "segments.translated.json"
    segments.write_text('[{"start":0,"end":1,"text":"hello","translated_text":"你好"}]', encoding="utf-8")
    job.artifacts.translated_srt = str(subtitle)
    job.artifacts.translated_segments = str(segments)
    commands: list[list[str]] = []
    def fake_run(command, **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"muted")
    captured: dict[str, object] = {}
    def fake_finish(_job, _raw, _message, subtitle=None, segments_file=None, operation="modify"):
        captured.update({"subtitle": subtitle, "segments": segments_file, "operation": operation})
    monkeypatch.setattr("yblocalizer.workbench_api.run_command", fake_run)
    monkeypatch.setattr(jobs, "_finish_modify", fake_finish)

    jobs._run_mute(job, [(0.0, 1.0), (2.0, 3.0)])

    audio_filter = commands[0][commands[0].index("-af") + 1]
    assert ",volume=" in audio_filter
    assert "+volume=" not in audio_filter
    assert captured == {"subtitle": subtitle, "segments": segments, "operation": "mute"}


def test_test_history_scanner_only_accepts_pytest_temp_outputs(tmp_path: Path) -> None:
    pytest_output = tmp_path / "out" / "job-test"
    normal_output = Path("D:/Videos/real-job")
    for job_id, output in (("test-row", pytest_output), ("real-row", normal_output)):
        job_db.record_job(
            job_id=job_id, material_id=None, source_url=None, title=job_id,
            status="draft", stage="waiting", progress=0, error=None,
            output_dir=str(output), rendered_video=None, device="cpu", compute_type="int8",
            options={}, created_at=1, started_at=None, finished_at=None,
        )
    handler = object.__new__(WorkbenchHandler)
    candidates = handler._test_history_candidates()
    assert [row["id"] for row in candidates] == ["test-row"]
    assert candidates[0]["reason"] == "输出目录位于 pytest 独立临时目录中"


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
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda command, tools_root=None: Path("tool") if command == "ffmpeg" else None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "configured")
    audio = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "audio"}}, "outputs")
    merged = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "merged"}}, "outputs")
    ocr = preflight({"material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8", "options": {"subtitle_source": "ocr"}}, "outputs")
    assert audio["ready"] and not audio["warnings"]
    assert merged["ready"] and merged["warnings"][0]["code"] == "ocr_unavailable"
    assert not ocr["ready"] and ocr["blocking"][-1]["code"] == "tesseract_required"


def test_preflight_warns_before_original_4k_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yblocalizer.dependencies.resolve_command",
        lambda command, tools_root=None: Path("tool") if command == "ffmpeg" else None,
    )
    result = preflight(
        {
            "source_url": "https://www.youtube.com/watch?v=demo",
            "authorized": True,
            "device": "cpu",
            "compute_type": "int8",
            "source_height": 2160,
            "options": {
                "translator": "none",
                "target_lang": "en",
                "youtube_po_token_mode": "off",
                "download_quality": "original",
            },
        },
        "outputs",
    )
    assert result["ready"]
    assert any(item["code"] == "original_quality_load" for item in result["warnings"])


def test_workbench_server_can_bind_an_ephemeral_loopback_port() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
    finally:
        server.server_close()


def test_mutating_api_requires_csrf_origin_and_json_content_type() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    connection = HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", "/api/bootstrap")
        bootstrap_response = connection.getresponse()
        token = json.loads(bootstrap_response.read().decode("utf-8"))["csrf_token"]

        connection.request(
            "POST", "/api/preflight", body='{"authorized":true}',
            headers={"Content-Type": "text/plain", "Origin": "https://untrusted.example", "X-YBL-CSRF": token},
        )
        rejected_origin = connection.getresponse()
        assert rejected_origin.status == 403
        rejected_origin.read()

        connection.request(
            "POST", "/api/preflight", body='{"authorized":true}',
            headers={"Content-Type": "text/plain", "X-YBL-CSRF": token},
        )
        rejected_type = connection.getresponse()
        assert rejected_type.status == 415
        rejected_type.read()

        payload = json.dumps({
            "material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8",
            "options": {"translator": "none", "target_lang": "en"},
        })
        connection.request(
            "POST", "/api/preflight", body=payload,
            headers={"Content-Type": "application/json", "X-YBL-CSRF": token},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert "ready" in json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_json_api_rejects_non_object_non_finite_invalid_utf8_and_oversized_bodies() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    bootstrap = HTTPConnection(host, port, timeout=10)
    bootstrap.request("GET", "/api/bootstrap")
    token = json.loads(bootstrap.getresponse().read().decode("utf-8"))["csrf_token"]
    bootstrap.close()

    def post(body: bytes) -> tuple[int, dict[str, object]]:
        connection = HTTPConnection(host, port, timeout=10)
        connection.request(
            "POST", "/api/preflight", body=body,
            headers={"Content-Type": "application/json", "X-YBL-CSRF": token, "Connection": "close"},
        )
        response = connection.getresponse()
        result = response.status, json.loads(response.read().decode("utf-8"))
        connection.close()
        return result

    try:
        assert post(b"[]") == (400, {"error": "JSON 请求体必须是对象。", "code": "object_required"})
        assert post(b'{"value":NaN}')[1]["code"] == "non_finite_number"
        assert post(b"\xff\xfe")[1]["code"] == "invalid_utf8"
        with socket.create_connection((host, port), timeout=5) as client:
            request = (
                "POST /api/preflight HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Content-Type: application/json\r\n"
                f"X-YBL-CSRF: {token}\r\n"
                f"Content-Length: {1024 * 1024 + 1}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            client.sendall(request)
            chunks = []
            while chunk := client.recv(4096):
                chunks.append(chunk)
            oversized_response = b"".join(chunks)
        assert b" 413 " in oversized_response
        assert b"body_too_large" in oversized_response
    finally:
        server.shutdown()
        server.server_close()


def test_workbench_server_caps_active_connections() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    slots = server._connection_slots  # type: ignore[attr-defined]
    acquired = 0
    try:
        for _ in range(32):
            assert slots.acquire(blocking=False)
            acquired += 1
        with socket.create_connection(server.server_address, timeout=5) as client:
            client.sendall(b"GET /api/bootstrap HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            response = client.recv(4096)
        assert b"503 Service Unavailable" in response
        assert b"too_many_connections" in response
    finally:
        for _ in range(acquired):
            slots.release()
        server.shutdown()
        server.server_close()


def test_each_server_uses_a_distinct_csrf_token() -> None:
    first = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    second = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    try:
        assert first.RequestHandlerClass.csrf_token != second.RequestHandlerClass.csrf_token
    finally:
        first.server_close()
        second.server_close()


def test_staged_job_api_creates_draft_prepares_then_runs_one_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda size, root=None: tmp_path / size)
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    def fake_start(self, job, stages, *, auto_run):
        job.status, job.current_stage, job.auto_run = "running", stages[0], auto_run
        return job
    monkeypatch.setattr(WorkbenchJobs, "_start_stage_sequence", fake_start)
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    host, port = server.server_address[:2]
    connection = HTTPConnection(host, port, timeout=10)
    try:
        connection.request("GET", "/api/bootstrap")
        token = json.loads(connection.getresponse().read().decode("utf-8"))["csrf_token"]
        headers = {"Content-Type": "application/json", "X-YBL-CSRF": token}
        payload = json.dumps({
            "material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8",
            "options": {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")},
        })
        connection.request("POST", "/api/jobs", body=payload, headers=headers)
        created_response = connection.getresponse()
        created = json.loads(created_response.read().decode("utf-8"))
        assert created_response.status == 201
        assert created["status"] == "draft"
        assert created["stages"]["acquire"]["status"] == "completed"

        connection.request("POST", f"/api/jobs/{created['id']}/prepare", body="{}", headers=headers)
        prepared = json.loads(connection.getresponse().read().decode("utf-8"))
        assert prepared["status"] == "ready"
        assert not [row for row in prepared["checks"] if row["status"] == "blocking"]

        connection.request("POST", f"/api/jobs/{created['id']}/stages/extract/run", body="{}", headers=headers)
        running = json.loads(connection.getresponse().read().decode("utf-8"))
        assert running["status"] == "running"
        assert running["current_stage"] == "extract"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()

from pathlib import Path
from http.client import HTTPConnection
from http import HTTPStatus
from io import BytesIO
import hashlib
import json
import socket
import threading
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from yblocalizer import db as job_db
from yblocalizer.runtime import PipelineEvent, PipelineStage
from yblocalizer.transcribe import WhisperCudaOutOfMemory
from yblocalizer.workbench_api import DEMO_VIDEO, DemoJob, Material, WorkbenchHandler, WorkbenchJobs, _ffconcat_path, _upsert_env, _resolve_output_root, _recovery_for_error, _safe_options, _probe_translation_service, build_server, public_path, stage_from_log
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


def test_cuda_failure_exposes_user_confirmed_recovery_in_snapshot() -> None:
    stage, error, expected_id = "extract", WhisperCudaOutOfMemory("CUDA out of memory"), "retry_extract_with_lower_resource"
    job = DemoJob(
        id="recovery-test", material=Material("demo", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True),
        device="cuda", compute_type="float16",
    )
    job.status = "failed"
    job.stages[stage].update({"status": "failed", "error_code": getattr(error, "code", "")})
    job.stages[stage]["recovery"] = _recovery_for_error(error, stage, job)

    recovery = job.snapshot()["recovery"]

    assert recovery["id"] == expected_id
    assert recovery["requires_confirmation"] is True
    assert recovery["stage"] == stage


def test_dependency_refresh_query_clears_cached_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(WorkbenchHandler)
    handler.path = "/api/dependencies?refresh=1"
    handler.dependencies = SimpleNamespace(snapshot=lambda _ocr_language="eng": {"dependencies": []})
    result: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: result.update(payload=payload, status=status)  # type: ignore[method-assign]
    cleared: list[bool] = []
    monkeypatch.setattr("yblocalizer.workbench_api.clear_dependency_cache", lambda: cleared.append(True))

    handler.do_GET()

    assert cleared == [True]
    assert result["payload"] == {"dependencies": []}


def test_material_media_route_accepts_normal_material_id(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"video")
    handler = object.__new__(WorkbenchHandler)
    handler.path = "/api/materials/abc-123/media"
    handler.jobs = SimpleNamespace(material_path=lambda material_id: media if material_id == "abc-123" else None)
    served: list[Path] = []
    handler._serve_media = lambda path: served.append(path)  # type: ignore[method-assign]

    handler.do_GET()

    assert served == [media]


def test_options_route_decodes_unicode_job_ids_before_matching() -> None:
    job_id = "20260928-KO-UPECA-VEĆU-RIBU"
    handler = object.__new__(WorkbenchHandler)
    handler.path = f"/api/jobs/{quote(job_id, safe='')}/options"
    handler._authorize_mutation = lambda _path: True  # type: ignore[method-assign]
    handler._payload = lambda: {"options": {}}  # type: ignore[method-assign]
    updated_job = SimpleNamespace(snapshot=lambda: {"id": job_id})
    received: list[str] = []

    def update_options(received_id: str, _payload: dict[str, object]) -> tuple[object, list[str]]:
        received.append(received_id)
        return updated_job, []

    handler.jobs = SimpleNamespace(update_options=update_options)
    result: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: result.update(payload=payload, status=status)  # type: ignore[method-assign]

    handler.do_PATCH()

    assert received == [job_id]
    assert result["status"] == HTTPStatus.OK
    assert result["payload"] == {"id": job_id, "stale_stages": []}


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
    pipeline_options = jobs._pipeline_options(job)
    assert pipeline_options.ocr_fallback_to_audio is False
    assert pipeline_options.prefer_platform_subtitles is False


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


def test_prepare_cannot_overwrite_running_or_revive_cancelled_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import threading

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {"output_dir": str(tmp_path)})
    entered, release = threading.Event(), threading.Event()

    def slow_preflight(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return {"blocking": [], "warnings": []}

    monkeypatch.setattr("yblocalizer.workbench_api.preflight", slow_preflight)
    monkeypatch.setattr(jobs, "_prepare_required_stages", lambda _job: set())
    worker = threading.Thread(target=jobs.prepare, args=(job.id,))
    worker.start()
    assert entered.wait(5)
    try:
        assert job.status == "preparing"
        with pytest.raises(RuntimeError, match="准备检查"):
            jobs._start_stage_sequence(job, ["acquire"], auto_run=False)
        assert jobs.cancel(job.id).status == "cancelling"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert job.status == "cancelled"
    assert job.checks == []


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


def test_native_publish_metadata_uses_persisted_attempt_pointer_after_restart(tmp_path: Path) -> None:
    attempt = tmp_path / "job" / ".attempt-translate-restarted"
    attempt.mkdir(parents=True)
    video = attempt / "rendered.v2.mp4"
    video.write_bytes(b"video")
    metadata = attempt / "publish_metadata.json"
    metadata.write_text(json.dumps({"title": "恢复标题", "tags": ["恢复"]}), encoding="utf-8")
    job_db.record_job(
        "restart-job", None, None, "restart-job", "completed", "完成", 100, None,
        str(attempt.parent), str(video), "cpu", "int8", {}, 1, 1, 2,
        artifacts={"rendered_video": str(video), "publish_metadata": str(metadata)},
    )
    jobs = WorkbenchJobs()
    token = jobs.register_native_publish_file(video)
    detail = jobs.publish_file_metadata(token)
    assert detail["title"] == "恢复标题"
    assert detail["tags"] == ["恢复"]


def test_cues_reads_extract_attempt_pointer(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    material = Material("cue-material", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True)
    job = DemoJob("cue-job", material, "cpu", "int8", work_dir=tmp_path)
    attempt = tmp_path / ".attempt-extract-test"
    attempt.mkdir()
    source = attempt / "segments.source.json"
    save_segments(source, [Segment(0, 1, "hello")])
    job.artifacts.source_segments = str(source)
    job.stages["extract"]["status"] = "completed"
    jobs._jobs[job.id] = job
    assert jobs.cues(job.id)["cues"][0]["source"] == "hello"


def test_cues_empty_draft_is_ready_response_with_no_edit_owner(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = DemoJob("empty-cues", Material("empty", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8", work_dir=tmp_path)
    jobs._jobs[job.id] = job

    assert jobs.cues(job.id) == {"cues": [], "ready": False, "translation_ready": False}


def test_cues_known_legacy_job_without_work_dir_is_empty_response() -> None:
    jobs = WorkbenchJobs()
    job = DemoJob("legacy-empty-cues", Material("legacy-empty", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8")
    jobs._jobs[job.id] = job

    assert jobs.cues(job.id) == {"cues": [], "ready": False, "translation_ready": False}


@pytest.mark.parametrize("artifact_name", ["source_segments", "translated_segments"])
def test_cues_missing_committed_subtitle_pointer_is_controlled_error(
    tmp_path: Path, artifact_name: str,
) -> None:
    jobs = WorkbenchJobs()
    job = DemoJob("missing-pointer", Material("missing", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8", work_dir=tmp_path)
    setattr(job.artifacts, artifact_name, str(tmp_path / f"missing-{artifact_name}.json"))
    jobs._jobs[job.id] = job

    with pytest.raises(RuntimeError, match="committed subtitle file is missing"):
        jobs.cues(job.id)


def test_cues_translated_only_legacy_job_without_work_dir_reads_pointer(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = DemoJob("translated-only", Material("translated", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8")
    translated = tmp_path / "segments.translated.json"
    save_segments(translated, [Segment(0, 1, "hello", "你好")])
    job.artifacts.translated_segments = str(translated)
    job.artifacts.source_segments = str(tmp_path / "missing-source.json")
    jobs._jobs[job.id] = job

    result = jobs.cues(job.id)
    assert result["ready"] is True
    assert result["translation_ready"] is True
    assert result["cues"][0]["translated"] == "你好"


def test_cues_adds_review_issues_and_defaults_legacy_marker_to_false(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = DemoJob("review-cues", Material("review", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8")
    translated = tmp_path / "segments.translated.json"
    translated.write_text(json.dumps([
        {"start": 0.0, "end": 0.5, "text": "hello", "translated_text": "你好", "needs_review": True},
        {"start": 0.4, "end": 1.2, "text": "there", "translated_text": "世界"},
    ], ensure_ascii=False), encoding="utf-8")
    job.artifacts.translated_segments = str(translated)
    jobs._jobs[job.id] = job

    cues = jobs.cues(job.id)["cues"]

    assert cues[0]["needs_review"] is True
    assert {"short-duration", "overlap"}.issubset(cues[0]["issues"])
    assert cues[1]["needs_review"] is False


def test_cues_rejects_corrupt_subtitle_file_as_controlled_error(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = DemoJob("bad-cues", Material("bad", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8", work_dir=tmp_path)
    source = tmp_path / "segments.source.json"
    source.write_text("not json", encoding="utf-8")
    job.artifacts.source_segments = str(source)
    jobs._jobs[job.id] = job

    with pytest.raises(RuntimeError, match="Subtitle data is invalid"):
        jobs.cues(job.id)


def test_cues_rejects_corrupt_auxiliary_subtitle_file_as_controlled_error(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = DemoJob("bad-aux-cues", Material("bad-aux", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True), "cpu", "int8", work_dir=tmp_path)
    source = tmp_path / "segments.source.json"
    save_segments(source, [Segment(0, 1, "hello")])
    (tmp_path / "segments.audio.json").write_text("not json", encoding="utf-8")
    job.artifacts.source_segments = str(source)
    jobs._jobs[job.id] = job

    with pytest.raises(RuntimeError, match="Subtitle data is invalid"):
        jobs.cues(job.id)


def test_publish_fields_reads_translate_attempt_metadata(tmp_path: Path) -> None:
    output = tmp_path / "job"
    attempt = output / ".attempt-translate-test"
    attempt.mkdir(parents=True)
    video = attempt / "rendered.mp4"
    video.write_bytes(b"video")
    metadata = attempt / "publish_metadata.json"
    metadata.write_text(json.dumps({"title": "Attempt title", "tags": ["attempt"]}), encoding="utf-8")
    jobs = WorkbenchJobs()
    material = Material("publish-material", DEMO_VIDEO, "demo.mp4", 10, 1280, 720, True)
    job = DemoJob("publish-job", material, "cpu", "int8", work_dir=output)
    job.artifacts.rendered_video = str(video)
    job.artifacts.publish_metadata = str(metadata)
    handler = object.__new__(WorkbenchHandler)
    handler.jobs = jobs
    title, _description, tags = handler._compose_publish_fields({}, video, job)
    assert title == "Attempt title"
    assert tags == ["attempt"]


def test_publish_assist_falls_back_to_metadata_and_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out" / "job-pub"
    output.mkdir(parents=True)
    video = output / "rendered.mp4"
    video.write_bytes(b"video")
    (output / "publish_metadata.json").write_text(
        json.dumps({"title": "海狮与金枪鱼", "tags": ["中文字幕", "科普"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    jobs = WorkbenchJobs()
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "zh-Hans", "output_dir": str(tmp_path / "out"),
        "title": "Sea Lion vs Tuna",
    })
    job.work_dir = output
    job.rendered_video = video
    job.artifacts.rendered_video = str(video)
    job.artifacts.publish_metadata = str(output / "publish_metadata.json")
    job.material.source_url = "https://www.youtube.com/watch?v=demo"
    job.result = {"rendered_video": "job-pub/rendered.mp4", "output_dir": "job-pub"}
    jobs._jobs[job.id] = job

    captured: dict[str, object] = {}

    def fake_assist(**kwargs):
        captured.update(kwargs)
        session = jobs.publish_session()
        session.status = "finished"

    monkeypatch.setattr("yblocalizer.workbench_api.assist_publish", fake_assist)
    monkeypatch.setattr("yblocalizer.workbench_api.threading.Thread.start", lambda self: self.run())

    handler = object.__new__(WorkbenchHandler)
    handler.jobs = jobs
    responses: list[tuple[int, dict]] = []

    def fake_json(payload, status=HTTPStatus.OK):
        responses.append((int(status), payload))

    handler._json = fake_json  # type: ignore[method-assign]
    handler._publish_profile_process_ids = lambda browser: []  # type: ignore[method-assign]
    handler._publish_profile_dir = lambda browser: tmp_path / "profile"  # type: ignore[method-assign]
    handler._start_publish_assist({
        "confirmed": True,
        "job_id": job.id,
        "browser": "chromium",
        "close_after_fill": False,
        "title": "",
        "description": "",
        "tags": [],
        "include_source_link": True,
        "template": "授权本地化",
    })
    assert responses[-1][0] == int(HTTPStatus.ACCEPTED)
    assert captured["title"] == "海狮与金枪鱼"
    assert "中文字幕" in captured["tags"]
    assert "https://www.youtube.com/watch?v=demo" in str(captured["description"])


def test_publish_copy_endpoint_reads_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "job-copy"
    output.mkdir()
    video = output / "rendered.mp4"
    video.write_bytes(b"video")
    (output / "publish_metadata.json").write_text(
        json.dumps({"title": "本地化标题", "tags": ["标签A"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    jobs = WorkbenchJobs()
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "zh-Hans", "output_dir": str(tmp_path),
    })
    job.work_dir = output
    job.rendered_video = video
    job.artifacts.rendered_video = str(video)
    job.artifacts.publish_metadata = str(output / "publish_metadata.json")
    jobs._jobs[job.id] = job

    handler = object.__new__(WorkbenchHandler)
    handler.jobs = jobs
    responses: list[dict] = []
    handler._json = lambda payload, status=HTTPStatus.OK: responses.append(payload)  # type: ignore[method-assign]
    handler._publish_copy({"job_id": job.id, "generate": False})
    assert responses[-1]["title"] == "本地化标题"
    assert responses[-1]["tags"] == ["标签A"]
    assert responses[-1]["has_metadata"] is True


def _subtitle_save_base(job, path: Path) -> dict[str, object]:
    from yblocalizer.models import load_segments

    return {
        "base_revision": int(job.artifacts.revision),
        "base_fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
        "base_ids": list(range(len(load_segments(path)))),
    }


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

    jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "第一次", "deleted": False}], **_subtitle_save_base(job, source))
    first = Path(job.artifacts.translated_segments or "")
    jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "第二次", "deleted": False}], **_subtitle_save_base(job, first))
    second = Path(job.artifacts.translated_segments or "")

    assert first.name.startswith("segments.translated.edited.r1.")
    assert second.name.startswith("segments.translated.edited.r2.")
    assert first.suffix == ".json" and second.suffix == ".json"
    assert first.is_file() and second.is_file()


def test_subtitle_save_persists_source_and_manual_review_without_changing_raw_source(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, load_segments, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    original = job.work_dir / "segments.translated.json"
    raw_source = job.work_dir / "segments.source.json"
    save_segments(original, [Segment(0, 1, "source before", "译文之前")])
    save_segments(raw_source, [Segment(0, 1, "source before")])
    original_bytes = original.read_bytes()
    raw_source_bytes = raw_source.read_bytes()
    job.artifacts.translated_segments = str(original)
    job.artifacts.source_segments = str(raw_source)
    job.stages["translate"]["status"] = "completed"

    jobs.save_cues(job.id, [{
        "start": 0, "end": 1, "source": "修正原文", "translated": "修正译文",
        "deleted": False, "needs_review": True,
    }], **_subtitle_save_base(job, original))
    saved = load_segments(Path(job.artifacts.translated_segments or ""))[0]

    assert saved.text == "修正原文"
    assert saved.translated_text == "修正译文"
    assert saved.needs_review is True
    assert original.read_bytes() == original_bytes
    assert raw_source.read_bytes() == raw_source_bytes


def test_single_cue_retranslation_uses_task_model_and_draft_context_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "ollama", "translate_model": "qwen3.5:9b", "source_language": "en",
        "target_lang": "zh-Hans", "output_dir": str(tmp_path / "out"),
    })
    translated = job.work_dir / "segments.translated.json"
    save_segments(translated, [Segment(0, 1, "before", "之前"), Segment(1, 2, "old source", "旧译文"), Segment(2, 3, "after", "之后")])
    original_bytes = translated.read_bytes()
    job.artifacts.translated_segments = str(translated)
    job.stages["translate"]["status"] = "completed"
    captured: dict[str, object] = {}

    def fake_translate(source_text: str, **kwargs: object) -> str:
        captured["source"] = source_text
        captured["reserved_job"] = jobs._subtitle_model_action_id
        captured.update(kwargs)
        return "候选译文"

    monkeypatch.setattr("yblocalizer.workbench_api.translate_single_cue", fake_translate)
    result = jobs.retranslate_cue(job.id, 1, "draft source", [
        {"cue_id": 0, "source": "draft before", "translated": "草稿之前"},
        {"cue_id": 2, "source": "draft after", "translated": "草稿之后"},
    ])

    assert result["translation"] == "候选译文"
    assert result["provider"] == "Ollama 本地模型"
    assert captured["source"] == "draft source"
    assert captured["reserved_job"] == job.id
    assert captured["provider"] == "ollama"
    assert captured["model"] == "qwen3.5:9b"
    context = json.loads(str(captured["context"]))
    assert [entry["source"] for entry in context] == ["draft before", "draft after"]
    assert translated.read_bytes() == original_bytes
    assert jobs._subtitle_model_action_id is None


def test_single_cue_model_failure_keeps_subtitle_file_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "ollama", "translate_model": "qwen3.5:9b", "output_dir": str(tmp_path / "out"),
    })
    translated = job.work_dir / "segments.translated.json"
    save_segments(translated, [Segment(0, 1, "source", "原译文")])
    original_bytes = translated.read_bytes()
    job.artifacts.translated_segments = str(translated)
    job.stages["translate"]["status"] = "completed"
    def fail_translation(*_args: object, **_kwargs: object) -> str:
        assert jobs._subtitle_model_action_id == job.id
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("yblocalizer.workbench_api.translate_single_cue", fail_translation)

    with pytest.raises(RuntimeError, match="model unavailable"):
        jobs.retranslate_cue(job.id, 0, "corrected source")

    assert translated.read_bytes() == original_bytes
    assert jobs._subtitle_model_action_id is None


def test_compaction_preview_uses_unsaved_draft_and_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "ollama", "translate_model": "qwen3.5:9b", "output_dir": str(tmp_path / "out"),
    })
    translated = job.work_dir / "segments.translated.json"
    save_segments(translated, [Segment(0, 2, "source", "saved translation")])
    original_bytes = translated.read_bytes()
    job.artifacts.translated_segments = str(translated)
    job.stages["translate"]["status"] = "completed"
    captured: dict[str, object] = {}

    def fake_compact(segments: list[Segment], indexes: list[int], **kwargs: object) -> tuple[list[Segment], list[int]]:
        captured["source"] = segments[0].text
        captured["translation"] = segments[0].translated_text
        captured["reserved_job"] = jobs._subtitle_model_action_id
        captured.update(kwargs)
        return ([Segment(segments[0].start, segments[0].end, segments[0].text, "draft compact")], [])

    monkeypatch.setattr("yblocalizer.workbench_api.compact_overflowing_subtitles", fake_compact)
    result = jobs.compact_cues(job.id, [0], [{
        "cue_id": 0, "source": "draft source", "translated": "draft translation",
    }])

    assert captured["source"] == "draft source"
    assert captured["translation"] == "draft translation"
    assert captured["provider"] == "ollama"
    assert captured["reserved_job"] == job.id
    assert result["request_count"] == 1
    assert result["candidates"][0]["original"] == "draft translation"
    assert result["candidates"][0]["candidate"] == "draft compact"
    assert translated.read_bytes() == original_bytes
    assert jobs._subtitle_model_action_id is None


def test_subtitle_model_action_blocks_save_without_marking_pipeline_interrupted(tmp_path: Path) -> None:
    from yblocalizer.models import Segment, save_segments

    jobs = WorkbenchJobs()
    job = jobs.create("demo", "cpu", "int8", {
        "translator": "ollama", "translate_model": "qwen3.5:9b", "output_dir": str(tmp_path / "out"),
    })
    translated = job.work_dir / "segments.translated.json"
    save_segments(translated, [Segment(0, 1, "source", "译文")])
    job.artifacts.translated_segments = str(translated)
    job.stages["translate"]["status"] = "completed"
    job.status = "completed"
    jobs._subtitle_model_action_id = job.id

    with pytest.raises(RuntimeError, match="正在运行"):
        jobs.save_cues(job.id, [{"start": 0, "end": 1, "translated": "新译文"}])
    jobs.shutdown(timeout=0)

    assert job.status == "completed"
    assert translated.is_file()


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
        jobs.save_cues(
            job.id, [{"start": 0, "end": 1, "translated": "新字幕", "deleted": False}],
            **_subtitle_save_base(job, source),
        )

    assert job.artifacts.translated_segments == str(source)
    assert not list(job.work_dir.glob("segments.translated.edited.r1.*.json"))
    assert not list(job.work_dir.glob("zh.edited.r1.*.srt"))


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


def _record_history_row(job_id: str, output: Path, status: str = "completed") -> None:
    job_db.record_job(
        job_id, None, None, job_id, status, "完成", 100, None,
        str(output), str(output / "rendered.mp4"), "cpu", "int8", {"output_dir": str(output.parent)}, 1, 1, 2,
    )


def test_history_delete_preview_and_confirmed_output_delete_are_fingerprinted(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-1"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    (output / "job_manifest.json").write_text(json.dumps({"job_id": "task-1"}), encoding="utf-8")
    _record_history_row("task-1", output)
    jobs = WorkbenchJobs()

    plan = jobs.history_delete_plan(["task-1"], "record_and_outputs")
    assert plan["task_count"] == 1 and plan["file_count"] == 2
    result = jobs.delete_history(["task-1"], "record_and_outputs", plan["fingerprint"])
    assert result["deleted"] == 1
    assert not output.exists()
    assert job_db.list_jobs() == []
    assert jobs.get("task-1") is None


def test_history_batch_delete_reports_partial_failure_per_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "outputs" / "task-first"
    second = tmp_path / "outputs" / "task-second"
    for output in (first, second):
        output.mkdir(parents=True)
        (output / "rendered.mp4").write_bytes(b"video")
        _record_history_row(output.name, output)
    jobs = WorkbenchJobs()
    plan = jobs.history_delete_plan([first.name, second.name], "record_and_outputs")
    from yblocalizer.storage import delete_paths as real_delete_paths

    def delete_one_then_fail(paths: list[Path], root: Path) -> int:
        if paths[0] == second:
            raise OSError("file in use")
        return real_delete_paths(paths, root)

    monkeypatch.setattr("yblocalizer.workbench_api.delete_paths", delete_one_then_fail)
    result = jobs.delete_history([first.name, second.name], "record_and_outputs", plan["fingerprint"])
    assert result["deleted_ids"] == [first.name]
    assert [item["status"] for item in result["results"]] == ["deleted", "partial"]
    assert not first.exists() and second.exists()
    assert [row["id"] for row in job_db.list_jobs(None)] == [second.name]


def test_restorable_hides_jobs_whose_local_outputs_were_wiped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yblocalizer.workbench_api._validate_imported_media",
        lambda *_args: (5.0, 1280, 720),
    )
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    assert any(row["id"] == job.id for row in jobs.restorable())
    import shutil
    shutil.rmtree(job.work_dir)
    job.raw_video = None
    job.artifacts.raw_video = None
    job.artifacts.original_video = None
    assert all(row["id"] != job.id for row in jobs.restorable())


def test_history_delete_rejects_active_batch_without_changes(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-active"
    output.mkdir(parents=True)
    _record_history_row("task-active", output, "running")
    jobs = WorkbenchJobs()
    with pytest.raises(RuntimeError, match="运行中的任务"):
        jobs.history_delete_plan(["task-active"], "record_only")
    assert [row["id"] for row in job_db.list_jobs()] == ["task-active"]


def test_history_delete_rejects_stale_output_preview(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-stale"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    (output / "job_manifest.json").write_text(json.dumps({"job_id": "task-stale"}), encoding="utf-8")
    _record_history_row("task-stale", output)
    jobs = WorkbenchJobs()
    plan = jobs.history_delete_plan(["task-stale"], "record_and_outputs")
    (output / "rendered.mp4").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="过期"):
        jobs.delete_history(["task-stale"], "record_and_outputs", plan["fingerprint"])
    assert output.exists()
    assert job_db.list_jobs()[0]["id"] == "task-stale"


def test_history_delete_missing_output_is_record_only_effectively(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-missing"
    _record_history_row("task-missing", output)
    jobs = WorkbenchJobs()
    plan = jobs.history_delete_plan(["task-missing"], "record_and_outputs")
    assert plan["file_count"] == plan["bytes"] == 0
    result = jobs.delete_history(["task-missing"], "record_and_outputs", plan["fingerprint"])
    assert result["deleted"] == 1
    assert job_db.list_jobs() == []


def test_history_delete_requires_valid_ids_and_output_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-id"
    output.mkdir(parents=True)
    _record_history_row("task-id", output)
    jobs = WorkbenchJobs()
    with pytest.raises(ValueError, match="duplicates"):
        jobs.history_delete_plan(["task-id", "task-id"], "record_only")
    for invalid_id in ("../task-id", "task/child", r"task\child", ".", "..", "task-id."):
        with pytest.raises(ValueError, match="invalid"):
            jobs.history_delete_plan([invalid_id], "record_only")
    with pytest.raises(ValueError, match="fingerprint"):
        jobs.delete_history(["task-id"], "record_and_outputs", "")


def test_history_delete_accepts_legacy_mp4_id_without_deleting_outputs(tmp_path: Path) -> None:
    job_id = "20260923-185105_authorized-demo-10s.mp4"
    output = tmp_path / "outputs" / job_id
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    _record_history_row(job_id, output)
    jobs = WorkbenchJobs()

    record_plan = jobs.history_delete_plan([job_id], "record_only")
    assert record_plan["job_ids"] == [job_id]
    assert record_plan["file_count"] == 0
    output_plan = jobs.history_delete_plan([job_id], "record_and_outputs")
    assert output_plan["file_count"] == 1
    assert output.exists()


def test_internal_history_clear_allows_over_200_but_keeps_active_guard() -> None:
    ids = [f"bulk-{index:03d}" for index in range(201)]
    for job_id in ids:
        job_db.record_job(job_id, None, None, job_id, "completed", "完成", 100, None, None, None, "cpu", "int8", {}, 1, 1, 2)
    jobs = WorkbenchJobs()
    plan = jobs.history_delete_plan(ids, "record_only", allow_many=True)
    assert plan["task_count"] == 201
    result = jobs.delete_history(ids, "record_only", allow_many=True)
    assert result["deleted"] == 201
    assert job_db.list_jobs() == []

    active_ids = [f"active-{index:03d}" for index in range(201)]
    for index, job_id in enumerate(active_ids):
        job_db.record_job(job_id, None, None, job_id, "running" if index == 200 else "completed", "处理中", 50, None, None, None, "cpu", "int8", {}, 1, 1, None)
    with pytest.raises(RuntimeError, match="运行中的任务"):
        jobs.delete_history(active_ids, "record_only", allow_many=True)
    assert len(job_db.list_jobs(None)) == 201


def test_history_delete_requires_configured_output_root(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-root"
    output.mkdir(parents=True)
    job_db.record_job(
        "task-root", None, None, "task-root", "completed", "完成", 100, None,
        str(output), None, "cpu", "int8", {}, 1, 1, 2,
    )
    plan = WorkbenchJobs().history_delete_plan(["task-root"], "record_and_outputs")
    assert plan["record_only_required_count"] == 1
    assert plan["tasks"][0]["disposition"] == "record_only_required"


def test_history_delete_allows_derived_raw_video_but_protects_external_source(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-material"
    output.mkdir(parents=True)
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    derived = output / "raw.mp4"
    derived.write_bytes(b"derived")
    job_db.record_job(
        "task-material", "material-1", None, "task-material", "completed", "完成", 100, None,
        str(output), None, "cpu", "int8", {"output_dir": str(output.parent)}, 1, 1, 2,
        artifacts={"original_video": str(source), "raw_video": str(derived), "source": str(source)},
    )
    jobs = WorkbenchJobs()
    plan = jobs.history_delete_plan(["task-material"], "record_and_outputs")
    assert plan["bytes"] == len(b"derived")

    protected = tmp_path / "outputs" / "task-protected"
    protected.mkdir()
    inside = protected / "imported.mp4"
    inside.write_bytes(b"source")
    job_db.record_job(
        "task-protected", "material-2", None, "task-protected", "completed", "完成", 100, None,
        str(protected), None, "cpu", "int8", {"output_dir": str(protected.parent)}, 1, 1, 2,
        artifacts={"original_video": str(inside), "raw_video": str(inside), "source": str(inside)},
    )
    protected_plan = jobs.history_delete_plan(["task-protected"], "record_and_outputs")
    assert protected_plan["tasks"][0]["disposition"] == "record_only_required"
    assert "受保护" in protected_plan["tasks"][0]["reason"]


def test_output_delete_rejects_unscanned_path(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    handler = object.__new__(WorkbenchHandler)
    captured: dict[str, object] = {}
    handler._outputs_snapshot = lambda _root: {"tasks": []}  # type: ignore[method-assign]
    handler._json = lambda payload, status=HTTPStatus.OK: captured.update(payload=payload, status=status)  # type: ignore[method-assign]
    handler._delete_output_paths({"confirmed": True, "output_dir": str(root), "paths": ["."]})
    assert captured["status"] == HTTPStatus.BAD_REQUEST
    assert "扫描" in str(captured["payload"])


def test_output_paths_active_checks_persisted_task(tmp_path: Path) -> None:
    output = tmp_path / "outputs" / "task-live"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    _record_history_row("task-live", output, "running")
    jobs = WorkbenchJobs()
    assert jobs.output_paths_active([output, output / "rendered.mp4"])


def test_output_delete_rejects_active_task_with_conflict(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = root / "task-live-api"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    _record_history_row("task-live-api", output, "running")
    handler = object.__new__(WorkbenchHandler)
    handler.jobs = WorkbenchJobs()
    captured: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: captured.update(payload=payload, status=status)  # type: ignore[method-assign]
    handler._delete_output_paths({"confirmed": True, "output_dir": str(root), "paths": ["task-live-api"]})
    assert captured["status"] == HTTPStatus.CONFLICT
    assert output.exists()


def test_output_delete_preview_rejects_stale_file_and_syncs_registered_task(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = root / "task-preview"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    (output / "job_manifest.json").write_text(json.dumps({"job_id": output.name}), encoding="utf-8")
    _record_history_row(output.name, output)
    handler = object.__new__(WorkbenchHandler)
    handler.jobs = WorkbenchJobs()
    captured: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: captured.update(payload=payload, status=status)  # type: ignore[method-assign]
    payload = {"output_dir": str(root), "paths": [output.name]}
    plan = handler._output_delete_plan(payload)
    assert plan["file_count"] == 2 and plan["bytes"] >= 5
    assert plan["targets"][0]["record_will_remove"] is True
    (output / "rendered.mp4").write_bytes(b"changed")
    handler._delete_output_paths({**payload, "confirmed": True, "fingerprint": plan["fingerprint"]})
    assert captured["status"] == HTTPStatus.BAD_REQUEST
    assert output.exists() and job_db.list_jobs(None)
    fresh = handler._output_delete_plan(payload)
    handler._delete_output_paths({**payload, "confirmed": True, "fingerprint": fresh["fingerprint"]})
    assert captured["status"] == HTTPStatus.OK
    assert captured["payload"]["results"][0]["record_removed"] is True  # type: ignore[index]
    assert not output.exists() and not job_db.list_jobs(None)


def test_output_delete_preview_rejects_registered_record_only_fallback(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    output = root / "task-untrusted"
    output.mkdir(parents=True)
    (output / "rendered.mp4").write_bytes(b"video")
    job_db.record_job(
        output.name, None, None, output.name, "completed", "完成", 100, None,
        str(output), None, "cpu", "int8", {}, 1, 1, 2,
    )
    handler = object.__new__(WorkbenchHandler)
    handler.jobs = WorkbenchJobs()
    with pytest.raises(RuntimeError, match="仅删除记录或混合删除"):
        handler._output_delete_plan({"output_dir": str(root), "paths": [output.name]})
    assert output.exists()


def test_output_file_batch_delete_reports_each_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "outputs"
    task = root / "orphan"
    task.mkdir(parents=True)
    first = task / "rendered.mp4"
    second = task / "source.srt"
    first.write_bytes(b"video")
    second.write_bytes(b"subtitle")
    handler = object.__new__(WorkbenchHandler)
    handler.jobs = WorkbenchJobs()
    captured: dict[str, object] = {}
    handler._json = lambda payload, status=HTTPStatus.OK: captured.update(payload=payload, status=status)  # type: ignore[method-assign]
    payload = {"output_dir": str(root), "paths": ["orphan/rendered.mp4", "orphan/source.srt"]}
    plan = handler._output_delete_plan(payload)
    from yblocalizer.storage import delete_paths as real_delete_paths

    def delete_one_then_fail(paths: list[Path], allowed_root: Path) -> int:
        if paths[0] == second:
            raise OSError("file in use")
        return real_delete_paths(paths, allowed_root)

    monkeypatch.setattr("yblocalizer.workbench_api.delete_paths", delete_one_then_fail)
    handler._delete_output_paths({**payload, "confirmed": True, "fingerprint": plan["fingerprint"]})
    results = captured["payload"]["results"]  # type: ignore[index]
    assert [item["status"] for item in results] == ["deleted", "partial"]
    assert not first.exists() and second.exists()


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


def test_ollama_preflight_blocks_until_selected_model_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "material_id": "demo", "authorized": True, "device": "cpu", "compute_type": "int8",
        "options": {"translator": "ollama", "target_lang": "zh-Hans", "translate_model": "qwen3.5:4b"},
    }
    monkeypatch.setattr("yblocalizer.workbench_config.ollama_local_status", lambda: {
        "runtime_state": "ready", "runtime_available": True, "runtime_message": "ready",
        "models": {"ollama-qwen35-4b": False, "ollama-qwen35-9b": False},
    })
    result = preflight(payload, "outputs", required_stages={"translate"})
    assert any(item["code"] == "ollama-qwen35-4b" for item in result["blocking"])
    gemma_payload = {
        **payload,
        "options": {"translator": "ollama", "target_lang": "zh-Hans", "translate_model": "translategemma:12b"},
    }
    gemma_result = preflight(gemma_payload, "outputs", required_stages={"translate"})
    assert any(item["code"] == "ollama-translategemma-12b" for item in gemma_result["blocking"])
    monkeypatch.setattr("yblocalizer.workbench_api.ollama_local_status", lambda: {
        "runtime_state": "ready", "runtime_available": True, "runtime_message": "ready",
        "models": {"ollama-qwen35-4b": True, "ollama-qwen35-9b": False},
    })
    ready, message = _probe_translation_service("ollama", "qwen3.5:4b")
    assert ready and "qwen3.5:4b" in message
    monkeypatch.setattr("yblocalizer.workbench_api.ollama_local_status", lambda: {
        "runtime_state": "ready", "runtime_available": True, "runtime_message": "ready",
        "models": {"ollama-qwen35-4b": False, "ollama-qwen35-9b": False, "ollama-translategemma-12b": True},
    })
    ready, message = _probe_translation_service("ollama", "translategemma:12b")
    assert ready and "translategemma:12b" in message


def test_deepseek_preflight_accepts_legacy_flash_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}).encode()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("yblocalizer.workbench_api.urlopen", lambda *_args, **_kwargs: FakeResponse())
    ready, message = _probe_translation_service("deepseek", "deepseek-v4-flash")
    assert ready and "验证通过" in message


def test_ollama_options_reject_unregistered_model() -> None:
    with pytest.raises(InputValidationError, match="已登记"):
        normalize_options({"translator": "ollama", "translate_model": "somewhere/unknown:latest"}, "outputs")


def test_translategemma_is_an_allowed_local_translation_model() -> None:
    options = normalize_options({"translator": "ollama", "translate_model": "translategemma:12b"}, "outputs")
    assert options["translate_model"] == "translategemma:12b"


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


def test_job_api_rejects_ambiguous_video_source() -> None:
    server = build_server(Path("frontend/dist"), "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection(*server.server_address[:2], timeout=10)
    try:
        connection.request("GET", "/api/bootstrap")
        token = json.loads(connection.getresponse().read().decode("utf-8"))["csrf_token"]
        connection.request("POST", "/api/jobs", body=json.dumps({
            "source_url": "https://www.youtube.com/watch?v=demo",
            "material_id": "demo", "authorized": True,
        }), headers={"Content-Type": "application/json", "X-YBL-CSRF": token})
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        assert response.status == 400
        assert payload["code"] == "ambiguous_source"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_staged_job_api_creates_draft_prepares_then_runs_one_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda size, root=None: tmp_path / size)
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
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

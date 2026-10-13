from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yblocalizer import db as job_db
from yblocalizer.dependencies import require_whisper_model
from yblocalizer.models import Segment, load_segments, save_segments
from yblocalizer.subtitle import write_srt
from yblocalizer.subtitle_speech_pack import TimedWord, save_words, words_path_for_segments
from yblocalizer.workflow import (
    WorkflowArtifacts,
    atomic_write_manifest,
    invalidate_downstream,
    invalidation_stage,
    new_stage_states,
    run_acquire,
    run_extract,
    run_render,
    run_translate,
    run_stage,
    STAGE_RUNNERS,
    validate_media,
    validate_segments,
    validate_srt,
)
from yblocalizer.runtime import PipelineContext
from yblocalizer.workbench_api import WorkbenchJobs, _restored_created_at


def test_asr_evidence_artifact_round_trips_without_raw_path_loss() -> None:
    artifacts = WorkflowArtifacts(asr_evidence="job/asr-evidence.json", asr_timeline_shift_seconds=-0.75)
    restored = WorkflowArtifacts.from_dict(artifacts.to_dict())
    assert restored.asr_evidence == "job/asr-evidence.json"
    assert restored.asr_timeline_shift_seconds == -0.75


@pytest.fixture(autouse=True)
def valid_imported_test_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow unit tests use byte stubs; media validation has dedicated tests."""
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))


def test_translategemma_workflow_uses_translation_only_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "segments.source.json"
    save_segments(source_path, [Segment(0, 1.2, "Hello there.")])
    artifacts = WorkflowArtifacts(source_segments=str(source_path), title="Sample")
    observed: dict[str, object] = {}

    def forbidden_general_model(*_args, **_kwargs):
        raise AssertionError("TranslateGemma must not receive unrelated general-model instructions")

    def fake_translate_file(input_json: Path, output_json: Path, output_srt: Path, **kwargs):
        observed.update(kwargs)
        translated = [
            Segment(item.start, item.end, item.text, "你好。")
            for item in load_segments(input_json)
        ]
        save_segments(output_json, translated)
        write_srt(output_srt, translated, display_mode="translated")
        return translated

    monkeypatch.setattr("yblocalizer.workflow.resegment_source_segments", forbidden_general_model)
    monkeypatch.setattr("yblocalizer.workflow.correct_source_segments", forbidden_general_model)
    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate_file)
    options = SimpleNamespace(
        translator="ollama", translate_model="translategemma:12b", source_language="en",
        subtitle_source="ocr", smart_translation=True, target_lang="zh-Hans", batch_size=40,
        subtitle_display_mode="translated", smart_subtitle_layout=True,
        resource_profile="balanced", title="Sample",
    )
    messages: list[str] = []

    result = run_translate(options, tmp_path, artifacts, PipelineContext(), messages.append)

    assert observed["source_lang"] == "en"
    assert Path(result.translated_segments).name == "segments.translated.json"
    assert Path(result.translated_segments).parent.name.startswith(".attempt-translate-")
    assert Path(result.translated_srt).name == "zh.srt"
    assert result.publish_metadata and Path(result.publish_metadata).is_file()
    assert any("跳过通用模型语义切句与源文校对" in message for message in messages)


@pytest.mark.parametrize(
    ("configured", "mode", "probability", "expected"),
    [("en", "audio", 0.99, "en"), ("", "audio", 0.80, "es"), ("", "audio", 0.79, None), ("", "merged", 0.99, None)],
)
def test_translate_resolves_asr_language_only_for_confident_pure_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: str, mode: str, probability: float, expected: str | None,
) -> None:
    source_path = tmp_path / "segments.source.json"
    save_segments(source_path, [Segment(0, 1, "hola")])
    evidence_path = tmp_path / "asr-evidence.json"
    evidence_path.write_text(json.dumps({"detected_language": "es", "language_probability": probability}), encoding="utf-8")
    artifacts = WorkflowArtifacts(source_segments=str(source_path), asr_evidence=str(evidence_path), subtitle_extraction_mode=mode)
    captured: dict[str, object] = {}

    def fake_translate(input_json: Path, output_json: Path, output_srt: Path, **kwargs):
        captured.update(kwargs)
        translated = [Segment(item.start, item.end, item.text, "你好") for item in load_segments(input_json)]
        save_segments(output_json, translated); write_srt(output_srt, translated, display_mode="translated")
        return translated

    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate)
    options = SimpleNamespace(translator="none", translate_model="", source_language=configured, subtitle_source="ocr", smart_translation=False, target_lang="zh-Hans", batch_size=1, subtitle_display_mode="translated", smart_subtitle_layout=True, resource_profile="balanced", title="Sample")
    run_translate(options, tmp_path, artifacts, PipelineContext(), lambda _message: None)
    assert captured["source_lang"] == expected


def test_translate_maps_raw_asr_suspect_interval_through_ocr_shift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "segments.source.json"
    save_segments(source_path, [Segment(0, 5, "first"), Segment(5, 10, "second")])
    evidence_path = tmp_path / "asr-evidence.json"
    evidence_path.write_text(json.dumps({"suspect_intervals": [{"start": 7, "end": 8}]}), encoding="utf-8")
    artifacts = WorkflowArtifacts(source_segments=str(source_path), asr_evidence=str(evidence_path), asr_timeline_shift_seconds=-2.0, subtitle_extraction_mode="audio")

    def fake_translate(input_json: Path, output_json: Path, output_srt: Path, **_kwargs):
        translated = load_segments(input_json)
        for item in translated: item.translated_text = "你好"
        save_segments(output_json, translated); write_srt(output_srt, translated, display_mode="translated")
        return translated

    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate)
    options = SimpleNamespace(translator="none", translate_model="", source_language="en", subtitle_source="audio", smart_translation=False, target_lang="zh-Hans", batch_size=1, subtitle_display_mode="translated", smart_subtitle_layout=True, resource_profile="balanced", title="Sample")
    result = run_translate(options, tmp_path, artifacts, PipelineContext(), lambda _message: None)
    translated = load_segments(Path(result.translated_segments))
    assert [item.needs_review for item in translated] == [False, True]


def test_audio_translation_uses_speech_cues_without_crossing_long_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "segments.source.json"
    save_segments(source_path, [Segment(442.38, 507.11, "We've got ourselves a beautiful yellow fish.")])
    save_words(words_path_for_segments(source_path), [
        TimedWord(442.38, 442.56, "We've"),
        TimedWord(504.95, 505.18, " got"),
        TimedWord(505.19, 507.0, " ourselves a beautiful yellow fish."),
    ])
    observed: dict[str, object] = {}

    def fake_translate(input_json: Path, output_json: Path, output_srt: Path, **kwargs):
        source_items = load_segments(input_json)
        observed["source"] = source_items
        observed.update(kwargs)
        translated = [Segment(item.start, item.end, item.text, "中文") for item in source_items]
        save_segments(output_json, translated)
        write_srt(output_srt, translated, display_mode="translated")
        return translated

    monkeypatch.setattr("yblocalizer.workflow.translate_segments_file", fake_translate)
    options = SimpleNamespace(
        translator="none", translate_model="", source_language="en",
        subtitle_source="audio", smart_translation=True, target_lang="zh-Hans", batch_size=40,
        subtitle_display_mode="translated", smart_subtitle_layout=True,
        resource_profile="balanced", title="Sample",
    )
    artifacts = WorkflowArtifacts(
        source_segments=str(source_path), subtitle_extraction_mode="audio", title="Sample",
    )

    run_translate(options, tmp_path, artifacts, PipelineContext(), lambda _message: None)

    assert observed["strict_alignment"] is True
    assert observed["source_timing_slots"] is None
    cues = observed["source"]
    assert isinstance(cues, list) and len(cues) == 2
    assert cues[0].end < 443 and cues[1].start == 504.95


@pytest.mark.parametrize("repair_fails", [False, True])
@pytest.mark.parametrize("font_plan_fails", [False, True])
def test_render_keeps_going_with_dense_translation_and_preserves_old_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair_fails: bool, font_plan_fails: bool,
) -> None:
    translated = tmp_path / "segments.translated.json"
    save_segments(translated, [Segment(
        7.0, 10.0, "source",
        "一段明显超出两行画面容量的字幕，一段明显超出两行画面容量的字幕，一段明显超出两行画面容量的字幕。",
    )])
    old_video = tmp_path / "old-render.mp4"
    old_video.write_bytes(b"old")
    raw_video = tmp_path / "raw.mp4"
    raw_video.write_bytes(b"raw")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workflow.subtitle_line_capacity", lambda *_args, **_kwargs: 18)
    if font_plan_fails:
        def failed_font_plan(*_args, **_kwargs):
            raise RuntimeError("无法计算逐条字号")
        monkeypatch.setattr("yblocalizer.workflow._plan_cue_font_sizes", failed_font_plan)
    def fake_compact(items, *_args, **_kwargs):
        if repair_fails:
            raise RuntimeError("翻译服务暂时不可用")
        return items, [0]
    monkeypatch.setattr("yblocalizer.workflow.compact_overflowing_subtitles", fake_compact)
    def fake_burn(_video, subtitle, output, **_kwargs):
        assert "一段明显超出两行画面容量" in subtitle.read_text(encoding="utf-8-sig")
        output.write_bytes(b"new")
        return output
    monkeypatch.setattr("yblocalizer.workflow.burn_subtitles", fake_burn)
    artifacts = WorkflowArtifacts(
        raw_video=str(raw_video), translated_segments=str(translated),
        rendered_video=str(old_video),
    )
    options = SimpleNamespace(
        subtitle_display_mode="translated", smart_subtitle_layout=True,
        font_name="Arial", font_size=36, subtitle_color="&H00FFFFFF",
        subtitle_outline_color="&H00000000", subtitle_outline=1, subtitle_shadow=0,
        render_crf=20, render_encoder="cpu", resource_profile="balanced",
        subtitle_margin_ratio=0.055, translator="none", translate_model="",
        target_lang="zh-Hans",
    )
    result = run_render(options, tmp_path, artifacts, PipelineContext(), lambda _message: None)
    assert Path(result.rendered_video).read_bytes() == b"new"
    assert result.translated_segments == str(translated)
    assert any("继续渲染" in warning for warning in result.content_warnings)
    assert old_video.read_bytes() == b"old"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"download_quality"}, "acquire"),
        ({"whisper_model_size"}, "extract"),
        ({"translator"}, "translate"),
        ({"font_size"}, "render"),
        ({"tags"}, "publish"),
    ],
)
def test_configuration_changes_invalidate_the_expected_stage(changed: set[str], expected: str) -> None:
    assert invalidation_stage(changed) == expected


def test_downstream_invalidation_preserves_upstream_checkpoint() -> None:
    states = new_stage_states(True)
    for row in states.values():
        row["status"] = "completed"
        row["progress"] = 100
    changed = invalidate_downstream(states, "translate")
    assert states["acquire"]["status"] == "completed"
    assert states["extract"]["status"] == "completed"
    assert changed == ["translate", "render", "publish"]
    assert all(states[name]["status"] == "stale" for name in changed)


def test_downstream_invalidation_clears_outdated_recovery_action() -> None:
    states = new_stage_states()
    states["extract"].update({
        "status": "failed", "error": "old failure", "error_code": "cuda_oom",
        "recovery": {"id": "retry_extract_with_lower_resource"},
    })

    invalidate_downstream(states, "extract")

    assert states["extract"]["status"] == "stale"
    assert states["extract"]["error_code"] is None
    assert states["extract"]["recovery"] is None


def test_workflow_stage_rejects_invalid_artifacts_before_returning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(STAGE_RUNNERS, "translate", lambda *_args: WorkflowArtifacts())

    with pytest.raises(RuntimeError, match="产物校验失败"):
        run_stage("translate", object(), tmp_path, WorkflowArtifacts(), None, lambda _message: None)  # type: ignore[arg-type]


def test_manifest_write_replaces_complete_json(tmp_path: Path) -> None:
    target = tmp_path / "job_manifest.json"
    atomic_write_manifest(target, {"workflow_version": 1, "next_stage": "extract"})
    atomic_write_manifest(target, {"workflow_version": 1, "next_stage": "translate"})
    assert json.loads(target.read_text(encoding="utf-8"))["next_stage"] == "translate"
    assert not target.with_suffix(".json.tmp").exists()


def test_subtitle_checkpoint_validation_rejects_malformed_files(tmp_path: Path) -> None:
    segments = tmp_path / "segments.json"
    subtitle = tmp_path / "subtitle.srt"
    segments.write_text('[{"start":0,"end":1,"text":"hello","translated_text":null}]', encoding="utf-8")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    assert validate_segments(segments)
    assert validate_srt(subtitle)
    segments.write_text("{}", encoding="utf-8")
    subtitle.write_text("not an srt", encoding="utf-8")
    assert not validate_segments(segments)
    assert not validate_srt(subtitle)


def test_media_validation_uses_the_managed_ffprobe_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "input.mp4"
    media.write_bytes(b"media")
    managed = tmp_path / "tools" / "ffprobe"
    seen: list[str] = []

    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name: managed if name == "ffprobe" else None)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen.append(command[0])
        return SimpleNamespace(returncode=0, stdout='{"streams":[{"index":0}]}')

    monkeypatch.setattr("yblocalizer.workflow.subprocess.run", fake_run)
    assert validate_media(media)
    assert seen == [str(managed)]


def test_transcription_refuses_to_download_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="不会自动下载模型"):
        require_whisper_model("small", tmp_path)


def test_platform_subtitles_skip_audio_extraction_and_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    platform = tmp_path / "demo.en.srt"
    platform.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n", encoding="utf-8"
    )
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr(
        "yblocalizer.workflow._extract_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("audio extraction must be skipped")),
    )
    monkeypatch.setattr(
        "yblocalizer.workflow.transcribe_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Whisper must be skipped")),
    )
    options = SimpleNamespace(
        subtitle_source="auto", prefer_platform_subtitles=True, max_seconds=None,
        whisper_model_size="small", source_language="en", device="cuda", compute_type="float16",
        beam_size=5, resource_profile="balanced", ocr_fallback_to_audio=False,
    )

    artifacts = run_extract(
        options, tmp_path,
        WorkflowArtifacts(raw_video=str(raw), platform_subtitles=str(platform)),
        PipelineContext(), lambda _line: None,
    )

    assert artifacts.subtitle_extraction_mode == "platform"
    assert artifacts.audio is None
    assert json.loads(Path(artifacts.source_segments).read_text(encoding="utf-8"))[0]["text"] == "Hello world"


def test_ocr_only_does_not_call_whisper_when_ocr_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workflow.extract_ocr_subtitles", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("OCR failed")))
    monkeypatch.setattr("yblocalizer.workflow.transcribe_audio", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Whisper must not run")))
    options = SimpleNamespace(
        subtitle_source="ocr", prefer_platform_subtitles=False, max_seconds=None,
        ocr_interval=1.0, ocr_crop_ratio=0.5, ocr_min_chars=3, ocr_language="eng",
        resource_profile="balanced", ocr_fallback_to_audio=False,
    )
    with pytest.raises(RuntimeError, match="OCR failed"):
        run_extract(options, tmp_path, WorkflowArtifacts(raw_video=str(raw)), PipelineContext(), lambda _line: None)


def test_job_creation_only_creates_a_draft_and_local_source_is_ready(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    assert job.status == "draft"
    assert job.current_stage is None
    assert job.stages["acquire"]["status"] == "completed"
    assert job.next_stage == "extract"
    assert job.artifacts.raw_video == str(source.resolve())


def test_local_acquire_retry_keeps_registered_original_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    artifacts = WorkflowArtifacts(source=str(source), source_kind="file", original_video=str(source), raw_video=str(source))
    options = SimpleNamespace(source=str(source), source_kind="file", title="demo", i_have_rights=True)

    result = run_acquire(options, tmp_path / "work", artifacts, PipelineContext(), lambda _line: None)

    assert result.raw_video == result.original_video == str(source)
    assert not (tmp_path / "work" / "raw.mp4").exists()


def test_interrupted_publish_is_not_silently_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "publish_to_bilibili": True,
        "output_dir": str(tmp_path / "out"),
    })
    for stage in ("acquire", "extract", "translate", "render"):
        job.stages[stage]["status"] = "completed"
    job.stages["publish"]["status"] = "interrupted"
    job.status, job.current_stage, job.next_stage = "interrupted", "publish", "publish"
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)

    jobs._verify_checkpoints(job)

    assert job.next_stage == "publish"
    assert job.checkpoint_validation == "verified"
    assert job.snapshot()["can_resume"] is False
    with pytest.raises(RuntimeError, match="核对草稿"):
        jobs.resume(job.id)


def test_rendered_job_waits_for_enabled_publish_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "publish_to_bilibili": True,
        "output_dir": str(tmp_path / "out"),
    })
    for stage in ("acquire", "extract", "translate", "render"):
        job.stages[stage]["status"] = "completed"
    job.status = "completed"
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)

    jobs._verify_checkpoints(job)

    assert job.next_stage == "publish"
    assert job.status == "ready"
    assert job.checkpoint_validation == "verified"


def test_artifact_public_summary_never_exposes_absolute_paths(tmp_path: Path) -> None:
    video = tmp_path / "private.mp4"
    video.write_bytes(b"video")
    summary = WorkflowArtifacts(raw_video=str(video)).public_summary()
    assert summary["raw_video"]["name"] == "private.mp4"
    assert str(tmp_path) not in json.dumps(summary)


def test_style_change_only_invalidates_render_and_publish(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    for row in job.stages.values():
        row["status"], row["progress"] = "completed", 100
    job.checks = [{"id": "ffmpeg", "status": "passed"}]
    job.status = "ready"
    updated, stale = jobs.update_options(job.id, {"options": {**job.options, "font_size": 32}})
    assert updated.stages["acquire"]["status"] == "completed"
    assert updated.stages["extract"]["status"] == "completed"
    assert updated.stages["translate"]["status"] == "completed"
    assert stale == ["render", "publish"]
    assert updated.next_stage == "render"
    assert updated.checks == [{"id": "ffmpeg", "status": "passed"}]
    assert updated.status == "ready"
    assert "重新检查准备" not in (updated.stage or "")


def test_cc_gate_toggle_keeps_prepare_checks_and_clears_acquire_error(tmp_path: Path) -> None:
    jobs = WorkbenchJobs()
    job = jobs.create_url(
        "https://www.youtube.com/watch?v=test", "cpu", "int8", True,
        {
            "translator": "none", "target_lang": "en", "require_reuse_allowed": True,
            "output_dir": str(tmp_path / "out"),
        },
    )
    job.checks = [{"id": "source", "status": "passed"}]
    job.status = "failed"
    job.stages["acquire"].update({
        "status": "failed", "progress": 5,
        "error": "该视频未被 YouTube 标记为“可转载”",
    })
    updated, stale = jobs.update_options(job.id, {"options": {"require_reuse_allowed": False}})
    assert updated.options["require_reuse_allowed"] is False
    assert updated.checks == [{"id": "source", "status": "passed"}]
    assert updated.status == "ready"
    assert updated.stages["acquire"]["status"] == "stale"
    assert updated.stages["acquire"]["error"] is None
    assert stale == ["acquire"]
    assert jobs._pipeline_options(updated).require_reuse_allowed is False


def test_partial_option_patch_preserves_unmentioned_values(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "title": "保留标题",
        "description": "保留简介", "font_size": 24, "output_dir": str(tmp_path / "out"),
    })
    updated, _ = jobs.update_options(job.id, {"options": {"font_size": 31}})
    assert updated.options["font_size"] == 31
    assert updated.options["title"] == "保留标题"
    assert updated.options["description"] == "保留简介"
    assert updated.options["translator"] == "none"


@pytest.mark.parametrize("field", ["include_source_link", "close_after_fill"])
def test_publish_setting_only_invalidates_publish(tmp_path: Path, field: str) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    for row in job.stages.values():
        row["status"], row["progress"] = "completed", 100
    updated, stale = jobs.update_options(job.id, {"options": {field: not job.options[field]}})
    assert stale == ["publish"]
    assert updated.next_stage == "publish"


def test_run_all_uses_the_remaining_real_stage_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.checks = [{"status": "passed"}]
    # The fixture uses a non-media placeholder; keep this order test focused on
    # stage selection while run_all now validates checkpoints before starting.
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)
    captured: list[str] = []
    def fake_start(current, stages, *, auto_run):
        captured.extend(stages)
        assert auto_run is True
        return current
    monkeypatch.setattr(jobs, "_start_stage_sequence", fake_start)
    jobs.run_all(job.id)
    assert captured == ["extract", "translate", "render"]


def test_run_all_mixed_stale_and_completed_stages_uses_contiguous_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.checks = [{"status": "passed"}]

    def fake_verify(current: DemoJob) -> None:
        current.stages["extract"]["status"] = "stale"
        current.stages["translate"]["status"] = "completed"
        current.stages["render"]["status"] = "completed"
        current.checkpoint_validation = "invalid"

    captured: list[str] = []
    monkeypatch.setattr(jobs, "_verify_checkpoints", fake_verify)
    monkeypatch.setattr(jobs, "_start_stage_sequence", lambda current, stages, *, auto_run: (captured.extend(stages) or current))

    jobs.run_all(job.id)

    assert captured == ["extract", "translate", "render"]


def test_run_all_rechecks_completed_render_before_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.checks = [{"status": "passed"}]
    for stage in ("acquire", "extract", "translate", "render"):
        job.stages[stage]["status"] = "completed"
    monkeypatch.setattr(
        "yblocalizer.workbench_api.validate_stage",
        lambda stage, _artifacts: stage != "render",
    )
    captured: list[str] = []
    monkeypatch.setattr(jobs, "_start_stage_sequence", lambda current, stages, *, auto_run: (captured.extend(stages) or current))

    jobs.run_all(job.id)

    assert captured == ["render"]


def test_restored_passed_checks_are_cleared_until_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        job_db,
        "list_jobs",
        lambda _limit=None: [{
            "id": "restored-checks",
            "material_id": "missing-material",
            "status": "draft",
            "stage": "等待准备检查",
            "checks": [{"id": "ffmpeg", "status": "passed"}],
            "options": {},
            "artifacts": {},
            "stage_states": {},
            "device": "cpu",
            "compute_type": "int8",
        }],
    )
    jobs = WorkbenchJobs(restore=True)
    job = jobs.get("restored-checks")
    assert job is not None
    assert job.checks == []
    with pytest.raises(RuntimeError, match="开始前准备"):
        jobs.run_all(job.id)


def test_url_access_problem_is_blocked_before_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.dependencies.resolve_command", lambda name, tools_root=None: Path(f"C:/tools/{name}.exe"))
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda size, root=None: tmp_path / size)
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.workbench_api.get_video_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Sign in to confirm you are not a bot")))
    jobs = WorkbenchJobs()
    job = jobs.create_url(
        "https://www.youtube.com/watch?v=test", "cpu", "int8", True,
        {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")},
    )
    prepared = jobs.prepare(job.id)
    access = next(row for row in prepared.checks if row["id"] == "source-access")
    assert access["status"] == "blocking"
    assert prepared.status == "draft"
    assert prepared.stages["acquire"]["status"] == "pending"


def test_prepare_first_url_checks_all_workflow_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = WorkbenchJobs()
    job = jobs.create_url(
        "https://www.youtube.com/watch?v=full", "cpu", "int8", True,
        {"translator": "none", "target_lang": "en", "youtube_po_token_mode": "off", "output_dir": str(tmp_path / "out")},
    )
    calls: list[str] = []
    monkeypatch.setattr("yblocalizer.workbench_api.get_video_metadata", lambda *_args, **_kwargs: (calls.append("url") or SimpleNamespace(title="full", duration=12, max_width=1280, max_height=720)))
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda *_args, **_kwargs: calls.append("whisper"))
    monkeypatch.setattr("yblocalizer.workbench_api._probe_translation_service", lambda _provider, _model=None: (calls.append("translator") or (True, "ok")))
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: (calls.append(name) or Path(name)))
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.workbench_config.capabilities", lambda: {"ffmpeg": {"available": True}, "tesseract": {"available": False}})

    prepared = jobs.prepare(job.id)
    check_ids = {row["id"] for row in prepared.checks}
    assert {"source-access", "ffmpeg", "ffprobe", "whisper-small", "translator", "encoder", "output"} <= check_ids
    assert {"url", "whisper", "translator", "ffmpeg", "ffprobe"} <= set(calls)


def test_prepare_ocr_only_skips_whisper_and_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cuda", "float16", {"subtitle_source": "ocr", "ocr_language": "eng", "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    monkeypatch.setattr(jobs, "_verify_checkpoints", lambda _job: None)
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Whisper must not be checked")))
    monkeypatch.setattr("yblocalizer.workbench_api.cuda_runtime_status", lambda: (_ for _ in ()).throw(AssertionError("CUDA must not be checked")))
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(name))
    monkeypatch.setattr("yblocalizer.workbench_api.available_ocr_languages", lambda *_args: {"eng"})
    monkeypatch.setattr("yblocalizer.workbench_config.available_ocr_languages", lambda *_args: {"eng"})
    monkeypatch.setattr("yblocalizer.workbench_config.capabilities", lambda: {"ffmpeg": {"available": True}, "tesseract": {"available": True}})
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    prepared = jobs.prepare(job.id)
    ids = {row["id"] for row in prepared.checks}
    assert "tesseract" in ids and "whisper-small" not in ids and "cuda" not in ids


def test_prepare_translate_only_skips_url_whisper_and_ocr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.stages["acquire"]["status"] = job.stages["extract"]["status"] = "completed"
    monkeypatch.setattr(jobs, "_verify_checkpoints", lambda _job: None)
    monkeypatch.setattr("yblocalizer.workbench_api.get_video_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate-only must not access URL")))
    monkeypatch.setattr("yblocalizer.workbench_api.require_whisper_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate-only must not check Whisper")))
    monkeypatch.setattr("yblocalizer.workbench_api.available_ocr_languages", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate-only must not check OCR")))
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(name))
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.workbench_api._probe_translation_service", lambda _provider, _model=None: (True, "ok"))
    monkeypatch.setattr("yblocalizer.workbench_config.capabilities", lambda: {"ffmpeg": {"available": True}, "tesseract": {"available": False}})

    prepared = jobs.prepare(job.id)
    assert "translator" in {row["id"] for row in prepared.checks}
    assert "whisper-small" not in {row["id"] for row in prepared.checks}


def test_prepare_render_only_is_offline_and_skips_model_and_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "deepseek", "output_dir": str(tmp_path / "out")})
    for name in ("acquire", "extract", "translate"):
        job.stages[name]["status"] = "completed"
    monkeypatch.setattr(jobs, "_verify_checkpoints", lambda _job: None)
    for path, label in (("yblocalizer.workbench_api.get_video_metadata", "URL"), ("yblocalizer.workbench_api.require_whisper_model", "Whisper"), ("yblocalizer.workbench_api._probe_translation_service", "API")):
        monkeypatch.setattr(path, lambda *_args, _label=label, **_kwargs: (_ for _ in ()).throw(AssertionError(f"render-only must not check {_label}")))
    monkeypatch.setattr("yblocalizer.workbench_api.resolve_command", lambda name: Path(name))
    monkeypatch.setattr("yblocalizer.workbench_api.render_encoder_status", lambda: {"cpu": True, "nvidia": False})
    monkeypatch.setattr("yblocalizer.workbench_config.capabilities", lambda: {"ffmpeg": {"available": True}, "tesseract": {"available": False}})

    prepared = jobs.prepare(job.id)
    assert {row["id"] for row in prepared.checks} >= {"ffmpeg", "ffprobe", "encoder", "output"}
    assert "translator" not in {row["id"] for row in prepared.checks}


def test_restore_reads_summaries_without_probing_every_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    created = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    monkeypatch.setattr(WorkbenchJobs, "_verify_checkpoints", lambda *_args: (_ for _ in ()).throw(AssertionError("startup probed media")))
    restored = WorkbenchJobs(restore=True)
    rows = restored.restorable()
    assert any(row["id"] == created.id and row["checkpoint_validation"] == "pending" for row in rows)


def test_undated_legacy_jobs_sort_behind_real_recent_jobs() -> None:
    assert _restored_created_at({"created_at": None, "started_at": None, "updated_at": None}) == 0
    assert _restored_created_at({"created_at": None, "started_at": 123.5, "updated_at": 456}) == 123.5


def test_ambiguous_legacy_edits_preserve_existing_render_and_block_rerender(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {
        "translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out"),
    })
    job.workflow_version = 1
    rendered = job.work_dir / "rendered.mp4"
    rendered.write_bytes(b"last-good")
    job.artifacts.rendered_video = str(rendered)
    job.rendered_video = rendered
    (job.work_dir / "trimmed-a.mp4").write_bytes(b"first-edit")
    (job.work_dir / "muted-b.mp4").write_bytes(b"second-edit")
    for row in job.stages.values():
        row["status"] = "completed"
    monkeypatch.setattr("yblocalizer.workbench_api.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)

    jobs._verify_checkpoints(job)

    assert job.workflow_version == 2
    assert job.artifacts.edit_state == "legacy-ambiguous"
    assert job.artifacts.rendered_video == str(rendered)
    assert rendered.read_bytes() == b"last-good"
    with pytest.raises(RuntimeError, match="不能重新渲染"):
        jobs.rerender(job.id)


def test_standard_v1_task_migrates_without_changing_video_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.workflow_version = 1
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)
    jobs._verify_checkpoints(job)
    assert job.workflow_version == 2
    assert job.artifacts.original_video == str(source.resolve())
    assert job.artifacts.raw_video == str(source.resolve())
    assert job.artifacts.edit_state == "clean"


def test_single_coherent_v1_edit_becomes_current_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(material.id, "cpu", "int8", {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")})
    job.workflow_version = 1
    trimmed = job.work_dir / "trimmed-only.mp4"
    trimmed.write_bytes(b"trimmed")
    segments = job.work_dir / "segments.trimmed.json"
    segments.write_text('[{"start":0,"end":1,"text":"hello","translated_text":"你好"}]', encoding="utf-8")
    subtitle = job.work_dir / "zh.trimmed.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")
    monkeypatch.setattr("yblocalizer.workbench_api.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workbench_api.validate_stage", lambda *_args: True)
    jobs._verify_checkpoints(job)
    assert job.workflow_version == 2
    assert job.artifacts.original_video == str(source.resolve())
    assert job.artifacts.raw_video == str(trimmed)
    assert job.artifacts.translated_segments == str(segments)
    assert job.artifacts.translated_srt == str(subtitle)
    assert (job.artifacts.revision, job.artifacts.last_edit, job.artifacts.edit_state) == (1, "trimmed", "migrated")


def test_acquire_discards_stale_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    work = tmp_path / "work"
    work.mkdir()
    stale = work / "audio.wav"
    stale.write_bytes(b"old-audio")
    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    artifacts = WorkflowArtifacts(
        source=str(source),
        source_kind="file",
        original_video=str(source),
        raw_video=str(source),
        audio=str(stale),
    )
    options = SimpleNamespace(source=str(source), source_kind="file", title="demo", i_have_rights=True)

    result = run_acquire(options, work, artifacts, PipelineContext(), lambda _line: None)

    assert result.audio is None
    # Acquire only clears the new snapshot pointer; versioned prior media is
    # retained so a failed/cancelled retry can still be resumed safely.
    assert stale.exists()


def test_extract_always_refreshes_cached_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    stale = tmp_path / "audio.wav"
    stale.write_bytes(b"stale")
    refreshed: list[Path] = []

    def fake_extract(video: Path, output: Path, ctx: object) -> Path:
        del video, ctx
        refreshed.append(output)
        output.write_bytes(b"fresh")
        return output

    monkeypatch.setattr("yblocalizer.workflow.validate_media", lambda *_args: True)
    monkeypatch.setattr("yblocalizer.workflow._extract_audio", fake_extract)
    monkeypatch.setattr(
        "yblocalizer.workflow.transcribe_audio",
        lambda audio, **_kwargs: (_ for _ in ()).throw(AssertionError(f"stop-after-audio:{audio}")),
    )
    options = SimpleNamespace(
        subtitle_source="audio",
        prefer_platform_subtitles=False,
        max_seconds=None,
        whisper_model_size="small",
        source_language="en",
        device="cpu",
        compute_type="int8",
        beam_size=1,
        resource_profile="balanced",
        ocr_fallback_to_audio=False,
    )
    artifacts = WorkflowArtifacts(raw_video=str(raw), audio=str(stale))

    with pytest.raises(AssertionError, match="stop-after-audio"):
        run_extract(options, tmp_path, artifacts, PipelineContext(), lambda _line: None)

    assert len(refreshed) == 1
    assert refreshed[0].name == "audio.wav"
    assert refreshed[0].parent.name.startswith(".attempt-extract-")
    assert (tmp_path / "audio.wav").read_bytes() == b"stale"


def test_rerunning_extract_marks_translate_and_render_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    jobs = WorkbenchJobs()
    material = jobs.add_local_file(source, authorized=True)
    job = jobs.create(
        material.id,
        "cpu",
        "int8",
        {"translator": "none", "target_lang": "en", "output_dir": str(tmp_path / "out")},
    )
    for name in ("acquire", "extract", "translate", "render"):
        job.stages[name]["status"] = "completed"
        job.stages[name]["progress"] = 100
    job.next_stage = None
    job.checks = [{"id": "ready", "status": "ok", "message": "ready"}]

    class _IdleThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            return None

    monkeypatch.setattr("yblocalizer.workbench_api.threading.Thread", _IdleThread)
    monkeypatch.setattr(WorkbenchJobs, "_checks_ready", lambda self, current: True)
    monkeypatch.setattr(WorkbenchJobs, "_prerequisites_complete", lambda self, current, stage: True)

    updated = jobs.run_stage(job.id, "extract")

    assert updated.stages["acquire"]["status"] == "completed"
    assert updated.stages["extract"]["status"] == "stale"
    assert updated.stages["translate"]["status"] == "stale"
    assert updated.stages["render"]["status"] == "stale"
    assert updated.next_stage == "extract"
    jobs._running_id = None

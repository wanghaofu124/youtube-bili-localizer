from __future__ import annotations

import json
from pathlib import Path

import pytest

from yblocalizer.dependencies import require_whisper_model
from yblocalizer.workflow import (
    WorkflowArtifacts,
    atomic_write_manifest,
    invalidate_downstream,
    invalidation_stage,
    new_stage_states,
    validate_segments,
    validate_srt,
)
from yblocalizer.workbench_api import WorkbenchJobs


@pytest.fixture(autouse=True)
def valid_imported_test_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow unit tests use byte stubs; media validation has dedicated tests."""
    monkeypatch.setattr("yblocalizer.workbench_api._validate_imported_media", lambda *_args: (5.0, 1280, 720))


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


def test_transcription_refuses_to_download_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="不会自动下载模型"):
        require_whisper_model("small", tmp_path)


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
    updated, stale = jobs.update_options(job.id, {"options": {**job.options, "font_size": 32}})
    assert updated.stages["acquire"]["status"] == "completed"
    assert updated.stages["extract"]["status"] == "completed"
    assert updated.stages["translate"]["status"] == "completed"
    assert stale == ["render", "publish"]
    assert updated.next_stage == "render"


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
    captured: list[str] = []
    def fake_start(current, stages, *, auto_run):
        captured.extend(stages)
        assert auto_run is True
        return current
    monkeypatch.setattr(jobs, "_start_stage_sequence", fake_start)
    jobs.run_all(job.id)
    assert captured == ["extract", "translate", "render"]


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

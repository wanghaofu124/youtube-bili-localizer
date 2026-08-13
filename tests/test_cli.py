from argparse import Namespace

from yblocalizer import cli


def test_process_command_maps_cli_arguments_to_pipeline(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda options: captured.setdefault("options", options))
    args = cli.build_parser().parse_args(
        ["process", "--file", "demo/authorized-demo-10s.mp4", "--i-have-rights", "--translator", "deepseek", "--subtitle-source", "merged"]
    )
    args.func(args)
    assert captured["options"].source_kind == "file"
    assert captured["options"].subtitle_source == "merged"
    assert captured["options"].i_have_rights is True


def test_process_command_defaults_to_audio_subtitles() -> None:
    args = cli.build_parser().parse_args(
        ["process", "--file", "demo/authorized-demo-10s.mp4", "--i-have-rights"]
    )
    assert args.subtitle_source == "audio"


def test_publish_command_keeps_manual_review_wait(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "assist_publish", lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    args = cli.build_parser().parse_args(
        [
            "publish", "--video", "demo/artifacts/rendered.mp4", "--title", "演示", "--source-url", "https://example.com/source", "--tags", "字幕,演示",
        ]
    )
    args.func(args)
    assert captured["kwargs"]["wait_for_review"] is True
    assert captured["kwargs"]["tags"] == ["字幕", "演示"]
    assert "https://example.com/source" in captured["kwargs"]["description"]

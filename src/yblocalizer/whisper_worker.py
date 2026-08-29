from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["--native-smoke-test"]:
        import ctranslate2  # noqa: F401
        import onnxruntime  # noqa: F401
        return 0
    if len(args) != 1:
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    events_path = Path(request["events_path"])

    def log(message: str) -> None:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"message": message}, ensure_ascii=False) + "\n")

    os.environ["YBLOCALIZER_WHISPER_WORKER"] = "1"
    try:
        from .transcribe import _transcribe_audio_in_process
        _transcribe_audio_in_process(
            Path(request["audio_path"]),
            Path(request["segments_json"]),
            Path(request["srt_path"]),
            model_size=request["model_size"],
            language=request.get("language"),
            device=request["device"],
            compute_type=request["compute_type"],
            initial_prompt=request.get("initial_prompt"),
            beam_size=request["beam_size"],
            resource_profile=request["resource_profile"],
            log=log,
        )
    except Exception as exc:
        result_path.write_text(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        return 1
    result_path.write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

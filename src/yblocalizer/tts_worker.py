"""VoxCPM TTS worker, run with the separate VoxCPM venv Python.

Usage: python tts_worker.py request.json

This file is executed by path, so it must not import yblocalizer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        return 2
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    events_path = Path(request["events_path"])

    def log(message: str) -> None:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"message": message}, ensure_ascii=False) + "\n")

    try:
        import soundfile as sf
        import torch
        from voxcpm import VoxCPM

        started = time.perf_counter()
        model = VoxCPM.from_pretrained(
            request["model_dir"], load_denoiser=False, local_files_only=True, optimize=False,
        )
        sample_rate = model.tts_model.sample_rate
        log(f"VoxCPM 模型已加载，用时 {time.perf_counter() - started:.1f} 秒。")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        items = []
        for item in request["items"]:
            begin = time.perf_counter()
            wav = model.generate(text=item["text"], cfg_value=2.0, inference_timesteps=10)
            sf.write(item["out"], wav, sample_rate)
            elapsed = time.perf_counter() - begin
            items.append({"index": item["index"], "out": item["out"], "duration": len(wav) / sample_rate, "seconds": elapsed})
            log(f"已合成第 {item['index'] + 1} 句，{len(wav) / sample_rate:.2f} 秒音频，用时 {elapsed:.1f} 秒。")
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        result = {"status": "completed", "sample_rate": sample_rate, "items": items, "peak_vram_bytes": peak}
    except Exception as exc:
        result_path.write_text(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        return 1
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

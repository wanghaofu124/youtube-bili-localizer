"""Optional VoxCPM dubbing: synthesize, align, build a dub track and mix."""
from __future__ import annotations

import json
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable

from . import dependencies
from .models import Segment
from .runtime import CancellationRequested

DUB_MIX_MODES = ("duck", "dub_only")
MAX_GAP_BORROW = 0.5
MIN_TARGET = 0.05
MAX_TEMPO = 1.35
ORIGINAL_GAIN = 0.25
DUB_GAIN = 1.0
WORKER_PATH = Path(__file__).with_name("tts_worker.py")

LogFn = Callable[[str], None]


def target_durations(segments: list[Segment], total: float) -> list[float]:
    """Line length plus the gap before the next line, borrowing at most 0.5s."""
    targets = []
    for index, segment in enumerate(segments):
        next_start = segments[index + 1].start if index + 1 < len(segments) else total
        gap = max(0.0, min(MAX_GAP_BORROW, next_start - segment.end))
        targets.append(max(segment.end - segment.start + gap, MIN_TARGET))
    return targets


def tempo_for(duration: float, target: float) -> float:
    """atempo factor: 1.0 when it fits, else speed up, capped at MAX_TEMPO."""
    return 1.0 if duration <= target else min(duration / target, MAX_TEMPO)


def align_command(ffmpeg: str, source: Path, output: Path, target: float, tempo: float, sample_rate: int) -> list[str]:
    command = [ffmpeg, "-y", "-v", "error", "-i", str(source)]
    if tempo > 1.0:
        command += ["-af", f"atempo={tempo:.4f}"]
    return command + ["-t", f"{target:.3f}", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output)]


def mix_command(ffmpeg: str, video: Path, track: Path, output: Path, mode: str) -> list[str]:
    command = [ffmpeg, "-y", "-v", "error", "-i", str(video), "-i", str(track)]
    if mode == "dub_only":
        command += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        # Original at 0.25, further ducked by a sidechain wherever the dub speaks.
        graph = (
            f"[1:a]volume={DUB_GAIN},asplit=2[dub][key];[0:a]volume={ORIGINAL_GAIN}[orig];"
            "[orig][key]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=300[duck];"
            "[duck][dub]amix=inputs=2:duration=first:normalize=0[a]"
        )
        command += ["-filter_complex", graph, "-map", "0:v:0", "-map", "[a]"]
    return command + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]


def build_track(clips: list[tuple[float, Path]], total: float, sample_rate: int, output: Path) -> None:
    """Place mono 16-bit clips at their start times on a silent track."""
    frames = bytearray(int(total * sample_rate) * 2)
    for start, path in clips:
        with wave.open(str(path), "rb") as clip:
            data = clip.readframes(clip.getnframes())
        offset = int(start * sample_rate) * 2
        data = data[: max(0, len(frames) - offset)]
        frames[offset:offset + len(data)] = data
    with wave.open(str(output), "wb") as track:
        track.setnchannels(1)
        track.setsampwidth(2)
        track.setframerate(sample_rate)
        track.writeframes(bytes(frames))


def probe_duration(path: Path) -> float:
    ffprobe = dependencies.resolve_command("ffprobe")
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True, creationflags=dependencies.NO_WINDOW,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def run_tts_worker(python: str, model_dir: str, items: list[dict[str, Any]], control_dir: Path,
                   log: LogFn, cancel_check: Callable[[], bool]) -> dict[str, Any]:
    """Run tts_worker.py in the VoxCPM venv, like the Whisper worker."""
    control_dir.mkdir(parents=True, exist_ok=True)
    request_path = control_dir / "request.json"
    result_path = control_dir / "result.json"
    events_path = control_dir / "events.jsonl"
    result_path.unlink(missing_ok=True)
    events_path.write_text("", encoding="utf-8")
    request_path.write_text(json.dumps({
        "model_dir": model_dir, "items": items,
        "result_path": str(result_path), "events_path": str(events_path),
    }, ensure_ascii=False), encoding="utf-8")
    offset = 0
    with (control_dir / "worker.log").open("w", encoding="utf-8", errors="replace") as diagnostic:
        process = subprocess.Popen([python, str(WORKER_PATH), str(request_path)], stdout=diagnostic, stderr=subprocess.STDOUT,
                                   creationflags=dependencies.NO_WINDOW)
        try:
            while process.poll() is None:
                if cancel_check():
                    raise CancellationRequested("任务已被用户中断")
                offset = _forward_events(events_path, offset, log)
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    _forward_events(events_path, offset, log)
    if not result_path.is_file():
        raise RuntimeError(f"VoxCPM 配音子进程异常退出（{process.returncode}），详见 {control_dir / 'worker.log'}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(f"VoxCPM 配音失败：{result.get('error')}")
    return result


def _forward_events(path: Path, offset: int, log: LogFn) -> int:
    if not path.is_file():
        return offset
    with path.open("rb") as handle:
        handle.seek(offset)
        for line in handle:
            if line.endswith(b"\n"):
                try:
                    log(json.loads(line.decode("utf-8"))["message"])
                except (ValueError, KeyError):
                    pass
                offset += len(line)
    return offset


def llm_rewrite(options: Any) -> Callable[[str, float], str | None]:
    """Ask the translation LLM to shorten one line; None when no LLM is configured."""
    provider = str(getattr(options, "translator", "none")).lower()

    def rewrite(text: str, ratio: float) -> str | None:
        if provider not in {"deepseek", "openai", "ollama"}:
            return None
        from .translate import _chat_client, _completion_create, ai_provider_request_info
        env = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "ollama": "YBLOCALIZER_OLLAMA"}[provider]
        base_url = {
            "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "openai": os.getenv("OPENAI_BASE_URL", "").strip() or None, "ollama": f"{dependencies.OLLAMA_BASE_URL}/v1",
        }[provider]
        model = ai_provider_request_info(provider, getattr(options, "translate_model", None))[1]
        limit = max(2, int(len(text) / ratio * 0.9))
        response = _completion_create(
            _chat_client(env, base_url), checkpoint_path=None, kind="dub_rewrite", provider=env, model=model,
            messages=[
                {"role": "system", "content": "你是配音稿编辑。把给定的一句配音台词压缩改写得更短，保留原意，口语自然。只输出改写后的句子。"},
                {"role": "user", "content": f"改写到不超过 {limit} 个字：\n{text}"},
            ],
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip() or None

    return rewrite


def dub_video(
    video: Path, segments: list[Segment], work_dir: Path, *, python: str, model_dir: str, mix_mode: str,
    log: LogFn, cancel_check: Callable[[], bool], rewrite: Callable[[str, float], str | None],
    tts: Callable[..., dict[str, Any]] = run_tts_worker,
) -> tuple[Path | None, Path]:
    ffmpeg = dependencies.resolve_command("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法生成配音，请先在依赖页安装 FFmpeg。")
    ffmpeg = str(ffmpeg)
    dub_dir = work_dir / "dub"
    dub_dir.mkdir(parents=True, exist_ok=True)
    total = probe_duration(video)
    targets = target_durations(segments, total)
    texts = {index: segment.final_text.strip() for index, segment in enumerate(segments) if segment.final_text.strip()}
    items = [{"index": index, "text": text, "out": str(dub_dir / f"line-{index:04d}.wav")} for index, text in texts.items()]
    if not items:
        log("没有需要配音的句子，跳过配音。")
        report_path = dub_dir / "dub_report.json"
        report_path.write_text(json.dumps({"mix_mode": mix_mode, "lines": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        return None, report_path
    log(f"开始合成配音，共 {len(items)} 句。")
    result = tts(python, model_dir, items, dub_dir / "round1", log, cancel_check)
    sample_rate = int(result["sample_rate"])
    peak = int(result.get("peak_vram_bytes") or 0)
    lines = {row["index"]: row for row in result["items"]}
    rewritten: dict[int, str] = {}
    retry = []
    for index, row in lines.items():
        ratio = row["duration"] / targets[index]
        if ratio > MAX_TEMPO:
            try:
                shorter = rewrite(texts[index], ratio)
            except Exception as error:
                log(f"第 {index + 1} 句压缩改写失败，沿用原文：{error}")
                shorter = None
            if shorter and shorter != texts[index]:
                rewritten[index] = shorter
                retry.append({"index": index, "text": shorter, "out": str(dub_dir / f"line-{index:04d}-rewrite.wav")})
    if retry:
        log(f"{len(retry)} 句超过 {MAX_TEMPO} 倍时长，已压缩改写后重合成。")
        second = tts(python, model_dir, retry, dub_dir / "round2", log, cancel_check)
        peak = max(peak, int(second.get("peak_vram_bytes") or 0))
        lines.update({row["index"]: row for row in second["items"]})
    clips: list[tuple[float, Path]] = []
    report = []
    for index in sorted(lines):
        row, target = lines[index], targets[index]
        tempo = tempo_for(row["duration"], target)
        truncated = row["duration"] / target > MAX_TEMPO
        if truncated:
            log(f"第 {index + 1} 句配音仍超长（{row['duration']:.2f}s / 目标 {target:.2f}s），已加速到 {MAX_TEMPO} 倍并截断。")
        aligned = dub_dir / f"aligned-{index:04d}.wav"
        subprocess.run(align_command(ffmpeg, Path(row["out"]), aligned, target, tempo, sample_rate), check=True, capture_output=True,
                       creationflags=dependencies.NO_WINDOW)
        clips.append((segments[index].start, aligned))
        report.append({
            "index": index, "text": rewritten.get(index, texts[index]), "rewritten": index in rewritten,
            "duration": round(row["duration"], 3), "target": round(target, 3), "tempo": round(tempo, 4),
            "truncated": truncated, "seconds": round(row.get("seconds", 0.0), 2),
        })
    track = dub_dir / "dub_track.wav"
    build_track(clips, total, sample_rate, track)
    candidate = dub_dir / "mix.next.mp4"
    mixed = subprocess.run(mix_command(ffmpeg, video, track, candidate, mix_mode), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", creationflags=dependencies.NO_WINDOW)
    if mixed.returncode != 0:
        raise RuntimeError(f"配音混音失败：{mixed.stderr.strip()}")
    output = video.with_name(f"{video.stem}_dub.mp4")
    candidate.replace(output)
    report_path = dub_dir / "dub_report.json"
    report_path.write_text(json.dumps({
        "mix_mode": mix_mode, "sample_rate": sample_rate, "peak_vram_bytes": peak, "lines": report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"配音版已生成：{output.name}")
    return output, report_path

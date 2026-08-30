from __future__ import annotations

import os
import re
import json
import ctypes
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from .cancellation import check_cancelled as legacy_check_cancelled
from .dependencies import require_whisper_model, resolve_command
from .models import Segment, save_segments
from .performance import cpu_thread_limit, normalize_resource_profile
from .subtitle import write_srt

LogFn = Callable[[str], None]


def transcribe_audio(
    audio_path: Path,
    segments_json: Path,
    srt_path: Path,
    model_size: str = "tiny",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
    beam_size: int = 5,
    log: LogFn | None = None,
    cancel_check: Callable[[], None] | None = None,
    resource_profile: str = "balanced",
) -> list[Segment]:
    if (getattr(sys, "frozen", False) or os.environ.get("YBLOCALIZER_FORCE_WHISPER_WORKER") == "1") and os.environ.get("YBLOCALIZER_WHISPER_WORKER") != "1":
        return _transcribe_audio_isolated(
            audio_path, segments_json, srt_path, model_size, language, device,
            compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile,
        )
    return _transcribe_audio_in_process(
        audio_path, segments_json, srt_path, model_size, language, device,
        compute_type, initial_prompt, beam_size, log, cancel_check, resource_profile,
    )


def _transcribe_audio_in_process(
    audio_path: Path,
    segments_json: Path,
    srt_path: Path,
    model_size: str = "tiny",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    initial_prompt: str | None = None,
    beam_size: int = 5,
    log: LogFn | None = None,
    cancel_check: Callable[[], None] | None = None,
    resource_profile: str = "balanced",
) -> list[Segment]:
    check = cancel_check or legacy_check_cancelled
    check()
    if device.strip().lower() in {"cuda", "auto"}:
        _prepend_nvidia_cuda_dll_paths()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc

    logger = log or (lambda _message: None)
    profile = normalize_resource_profile(resource_profile)
    thread_limit = cpu_thread_limit(profile)
    logger(f"资源模式：{profile}；Whisper CPU 线程上限 {thread_limit}。")
    last_error: Exception | None = None
    for resolved_device, resolved_compute_type, note in _transcribe_attempts(device, compute_type):
        if note:
            logger(note)
        try:
            model = WhisperModel(
                str(require_whisper_model(model_size)),
                device=resolved_device,
                compute_type=resolved_compute_type,
                cpu_threads=thread_limit,
                num_workers=1,
            )
            # word_timestamps=True：让 whisper 用词级对齐重算段边界。
            # 实测不开启时（仅 VAD 粗块），有 BGM 的视频会把数秒静默并入同一条段
            # （如 0-11s 一条字幕），开启后能精确切到真实语音边界（如 0.0-1.0 / 5.9-7.5）。
            whisper_segments, _info = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=_clean_initial_prompt(initial_prompt),
                vad_filter=True,
                beam_size=max(1, min(10, beam_size)),
                condition_on_previous_text=True,
                word_timestamps=True,
            )
            raw_segments = []
            for item in whisper_segments:
                check()
                text = item.text.strip()
                if text:
                    raw_segments.append(Segment(start=float(item.start), end=float(item.end), text=text))
            segments = _split_long_segments(raw_segments)
            segments = _align_segment_starts(segments, audio_path)
            break
        except Exception as exc:
            last_error = exc
            if resolved_device == "cuda":
                logger(
                    "CUDA transcription failed; trying a safer fallback. "
                    f"device={resolved_device}, compute_type={resolved_compute_type}, error={exc}"
                )
                continue
            raise RuntimeError(
                "faster-whisper transcription failed on CPU fallback. "
                f"device={resolved_device}, compute_type={resolved_compute_type}, error={exc}"
            ) from exc
    else:
        raise RuntimeError(f"faster-whisper transcription failed: {last_error}") from last_error
    save_segments(segments_json, segments)
    write_srt(srt_path, segments, display_mode="source")
    return segments


def _transcribe_audio_isolated(
    audio_path: Path,
    segments_json: Path,
    srt_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    initial_prompt: str | None,
    beam_size: int,
    log: LogFn | None,
    cancel_check: Callable[[], None] | None,
    resource_profile: str,
) -> list[Segment]:
    logger = log or (lambda _message: None)
    check = cancel_check or legacy_check_cancelled
    segments_json.parent.mkdir(parents=True, exist_ok=True)
    control_dir = Path(tempfile.mkdtemp(prefix=".whisper-worker-", dir=segments_json.parent))
    try:
        attempts = [(device, compute_type)]
        if device.strip().lower() in {"cuda", "auto"}:
            attempts.append(("cpu", "int8"))
        last_error: RuntimeError | None = None
        for index, (attempt_device, attempt_compute) in enumerate(attempts):
            if index:
                logger("Whisper 子进程异常退出，主程序仍在运行；正在改用 CPU / int8 重试。")
            try:
                items, temporary_json, temporary_srt = _run_whisper_worker(
                    control_dir, audio_path, model_size, language, attempt_device,
                    attempt_compute, initial_prompt, beam_size, logger, check, resource_profile,
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            temporary_json.replace(segments_json)
            temporary_srt.replace(srt_path)
            return items
        raise last_error or RuntimeError("Whisper 子进程未能启动。")
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)


def _run_whisper_worker(
    control_dir: Path,
    audio_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str,
    initial_prompt: str | None,
    beam_size: int,
    logger: LogFn,
    check: Callable[[], None],
    resource_profile: str,
) -> tuple[list[Segment], Path, Path]:
    request_path = control_dir / f"request-{device}.json"
    result_path = control_dir / f"result-{device}.json"
    events_path = control_dir / f"events-{device}.jsonl"
    diagnostic_path = control_dir / f"diagnostic-{device}.log"
    temporary_json = control_dir / f"segments-{device}.json"
    temporary_srt = control_dir / f"subtitles-{device}.srt"
    request_path.write_text(json.dumps({
        "audio_path": str(audio_path),
        "segments_json": str(temporary_json),
        "srt_path": str(temporary_srt),
        "result_path": str(result_path),
        "events_path": str(events_path),
        "model_size": model_size,
        "language": language,
        "device": device,
        "compute_type": compute_type,
        "initial_prompt": initial_prompt,
        "beam_size": beam_size,
        "resource_profile": resource_profile,
    }, ensure_ascii=False), encoding="utf-8")
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--whisper-worker", str(request_path)]
    else:
        command = [sys.executable, "-m", "yblocalizer.whisper_worker", str(request_path)]
    event_offset = 0
    with diagnostic_path.open("w", encoding="utf-8", errors="replace") as diagnostic:
        process = subprocess.Popen(command, stdout=diagnostic, stderr=subprocess.STDOUT, text=True)
        try:
            while process.poll() is None:
                check()
                event_offset = _forward_worker_events(events_path, event_offset, logger)
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        event_offset = _forward_worker_events(events_path, event_offset, logger)
    if process.returncode != 0:
        code = process.returncode & 0xFFFFFFFF
        detail = ""
        if result_path.is_file():
            try:
                detail = str(json.loads(result_path.read_text(encoding="utf-8")).get("error") or "")
            except (OSError, ValueError):
                pass
        if detail:
            raise RuntimeError(f"Whisper 转写失败：{detail}；主程序和已下载素材均已保留。")
        raise RuntimeError(f"Whisper 转写子进程异常退出（0x{code:08X}），主程序和已下载素材均已保留。")
    if not result_path.is_file() or not temporary_json.is_file() or not temporary_srt.is_file():
        raise RuntimeError("Whisper 转写子进程未生成完整结果，已保留原有检查点。")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(f"Whisper 转写失败：{result.get('error') or '未知错误'}")
    from .models import load_segments
    return load_segments(temporary_json), temporary_json, temporary_srt


def _forward_worker_events(path: Path, offset: int, logger: LogFn) -> int:
    if not path.is_file():
        return offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                message = json.loads(line).get("message")
            except ValueError:
                message = None
            if message:
                logger(str(message))
        return handle.tell()


def _clean_initial_prompt(initial_prompt: str | None) -> str | None:
    if not initial_prompt:
        return None
    cleaned = " ".join(initial_prompt.split())
    return cleaned[:1000] or None


def _split_long_segments(segments: list[Segment], max_duration: float = 4.8) -> list[Segment]:
    output: list[Segment] = []
    for segment in segments:
        output.extend(_split_long_segment(segment, max_duration=max_duration))
    return output


def _align_segment_starts(segments: list[Segment], audio_path: Path) -> list[Segment]:
    """把每条字幕的开始时间对齐到音频里的真实语音起点。

    faster-whisper 的段时间戳（含词级对齐）实测系统性比真实语音早
    0.1~0.5 秒（尤其有 BGM 时），导致字幕在声音出现前就显示。
    这里用短窗 RMS 能量在每条段 start 前后找第一个语音帧；
    只后移不提前：若检测到更晚的语音起点则对齐过去，否则至少后移 0.25s，
    保证字幕不早于声音出现。
    """
    try:
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        return segments
    try:
        sample_rate, data = wavfile.read(str(audio_path))
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data
        mono = mono.astype(np.float32) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)
    except Exception:
        return segments
    window = max(80, sample_rate // 40)  # 25ms
    rms = np.sqrt(np.convolve(mono * mono, np.ones(window) / window, mode="same"))
    # 阈值：取 40 百分位的 3 倍，且不低于 -30dBFS
    threshold = max(0.0316, float(np.percentile(rms, 40)) * 3.0)

    def _first_voice_frame(start: float, end: float) -> float | None:
        begin = max(0, int((start - 0.6) * sample_rate))
        stop = int((end + 0.2) * sample_rate)
        segment = rms[begin:stop]
        hits = np.where(segment > threshold)[0]
        if len(hits) == 0:
            return None
        # 需要至少连续 3 帧（约 75ms）才算语音起点，避免瞬时噪声
        groups = np.split(hits, np.where(np.diff(hits) > sample_rate // 25)[0] + 1)
        for group in groups:
            if len(group) >= 3:
                return (begin + group[0]) / sample_rate
        return None

    aligned: list[Segment] = []
    for segment in segments:
        start = segment.start
        end = segment.end
        voice_at = _first_voice_frame(start, end)
        if voice_at is not None and voice_at > start + 0.1:
            start = voice_at
        else:
            # 能量检测不可靠（BGM 强时）也至少后移 0.25s，消除"字幕早于声音"
            start = start + 0.25
        shift = start - segment.start
        aligned.append(Segment(start=start, end=end + shift, text=segment.text))
    # 校正后可能出现重叠，规整一下
    for index, segment in enumerate(aligned):
        if index < len(aligned) - 1:
            next_start = aligned[index + 1].start
            if segment.end > next_start - 0.05:
                segment.end = max(segment.start + 0.1, next_start - 0.05)
    return aligned


def _split_long_segment(segment: Segment, max_duration: float) -> list[Segment]:
    duration = segment.end - segment.start
    if duration <= max_duration:
        return [segment]
    parts = _sentence_parts(segment.text)
    if len(parts) <= 1:
        return [segment]
    weights = [max(1, len(re.sub(r"\s+", "", part))) for part in parts]
    total = sum(weights)
    output: list[Segment] = []
    cursor = segment.start
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            end = segment.end
        else:
            end = cursor + duration * weights[index] / total
        if end - cursor >= 0.25:
            output.append(Segment(start=cursor, end=end, text=part))
        cursor = end
    return output or [segment]


def _sentence_parts(text: str) -> list[str]:
    parts = [part.strip() for part in re.findall(r"[^.!?]+[.!?]*", text) if part.strip()]
    if len(parts) <= 1:
        return [text.strip()]
    return parts


def _transcribe_attempts(device: str, compute_type: str) -> list[tuple[str, str, str]]:
    requested_device = (device or "cpu").strip().lower()
    requested_compute = (compute_type or "int8").strip().lower()
    if requested_compute == "default":
        requested_compute = "float16" if requested_device == "cuda" else "int8"

    if requested_device == "cuda":
        attempts = [(requested_device, requested_compute, "Using CUDA for faster-whisper transcription.")]
        if requested_compute not in {"float16", "int8_float16"}:
            attempts.append(("cuda", "float16", "Retrying CUDA with compute_type=float16."))
        attempts.append(("cpu", "int8", "Falling back to CPU transcription with compute_type=int8."))
        return _dedupe_attempts(attempts)

    if requested_device == "auto":
        return [
            ("cuda", "float16", "Auto device: trying CUDA first."),
            ("cpu", "int8", "Auto device: falling back to CPU if CUDA is unavailable."),
        ]

    return [(requested_device, requested_compute, "")]


def _dedupe_attempts(attempts: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for device, compute_type, note in attempts:
        key = (device, compute_type)
        if key in seen:
            continue
        seen.add(key)
        output.append((device, compute_type, note))
    return output


def _nvidia_cuda_dll_directories() -> list[Path]:
    roots: list[Path] = []
    explicit_directories: list[Path] = []
    try:
        import nvidia
    except ImportError:
        pass
    else:
        roots.extend(Path(item) for item in getattr(nvidia, "__path__", ()))
        module_file = getattr(nvidia, "__file__", None)
        if module_file:
            roots.append(Path(module_file).parent)

    configured = os.environ.get("YBLOCALIZER_CUDA_DLL_DIR", "").strip()
    if configured:
        explicit_directories.append(Path(configured))

    # The onedir application intentionally does not bundle ~1.7 GB of optional
    # NVIDIA libraries. Reuse an existing Python CUDA runtime when present so a
    # developer or power user does not have to copy DLLs into the application.
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for access in (winreg.KEY_READ, winreg.KEY_READ | winreg.KEY_WOW64_32KEY, winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
                    try:
                        with winreg.OpenKey(hive, r"Software\Python\PythonCore", 0, access) as python_core:
                            index = 0
                            while True:
                                try:
                                    version = winreg.EnumKey(python_core, index)
                                except OSError:
                                    break
                                index += 1
                                try:
                                    with winreg.OpenKey(python_core, version + r"\InstallPath", 0, access) as install_key:
                                        install_root = Path(str(winreg.QueryValue(install_key, None)))
                                except OSError:
                                    continue
                                roots.append(install_root / "Lib" / "site-packages" / "nvidia")
                    except OSError:
                        continue

        except ImportError:
            pass

        toolkit = Path(os.environ.get("ProgramFiles", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit.is_dir():
            roots.extend(path for path in toolkit.glob("v*/bin") if path.is_dir())

    candidates: list[Path] = list(explicit_directories)
    for root in roots:
        if root.name.lower() == "bin":
            candidates.append(root)
        else:
            candidates.extend([
                root / "cuda_runtime" / "bin",
                root / "cublas" / "bin",
                root / "cudnn" / "bin",
            ])
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def _prepend_nvidia_cuda_dll_paths() -> None:
    candidates = _nvidia_cuda_dll_directories()
    existing = os.environ.get("PATH", "")
    parts = [str(path) for path in candidates]
    if not parts:
        return
    current_lower = {item.lower() for item in existing.split(os.pathsep) if item}
    missing = [item for item in parts if item.lower() not in current_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + [existing])


def cuda_runtime_status() -> dict[str, object]:
    """Verify the CUDA libraries that faster-whisper actually loads.

    ``nvidia-smi`` only proves that a display driver sees the GPU.  It does not
    prove that cuBLAS/cuDNN are installed, which previously let preparation pass
    and then fail several minutes later in transcription.
    """
    if resolve_command("nvidia-smi") is None:
        return {
            "available": False,
            "missing": ["NVIDIA driver"],
            "message": "未检测到 NVIDIA 显卡驱动（nvidia-smi）。",
        }

    _prepend_nvidia_cuda_dll_paths()
    if os.name == "nt":
        required = (
            "cublas64_12.dll", "cublasLt64_12.dll",
            "cudnn64_9.dll", "cudnn_ops64_9.dll",
        )
        loader = ctypes.WinDLL
    else:
        required = ("libcublas.so.12", "libcudnn.so.9")
        loader = ctypes.CDLL

    search_dirs = _nvidia_cuda_dll_directories()
    missing: list[str] = []
    for name in required:
        resolved = shutil.which(name)
        if not resolved:
            resolved = next((str(path / name) for path in search_dirs if (path / name).is_file()), None)
        try:
            loader(resolved or name)
        except (OSError, TypeError):
            missing.append(name)

    if missing:
        return {
            "available": False,
            "missing": missing,
            "message": "显卡驱动可用，但缺少或无法加载 CUDA 转写运行库：" + "、".join(missing) + "。",
        }
    return {
        "available": True,
        "missing": [],
        "message": "NVIDIA 驱动、cuBLAS 和 cuDNN 均可加载。",
    }

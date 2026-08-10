from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

from .cancellation import CancellationRequested


def require_command(name: str) -> None:
    if shutil.which(name) is not None:
        return
    discovered = _discover_windows_command(name)
    if discovered:
        os.environ["PATH"] = f"{discovered.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        return
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def _discover_windows_command(name: str) -> Path | None:
    if not sys.platform.startswith("win"):
        return None
    executable = f"{name}.exe" if not name.lower().endswith(".exe") else name
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if not winget_packages.exists():
        return None
    matches = sorted(winget_packages.rglob(executable), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def run(command: list[str], cwd: Path | None = None, cancel_check: Callable[[], bool] | None = None) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command, cwd=cwd, text=True, errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    output: list[str] = []
    cancelled = False

    def _watch_cancel() -> None:
        nonlocal cancelled
        while process.poll() is None:
            if cancel_check and cancel_check():
                cancelled = True
                _terminate_process_tree(process)
                return
            time.sleep(0.1)

    if cancel_check:
        threading.Thread(target=_watch_cancel, daemon=True).start()

    while True:
        try:
            line = process.stdout.readline() if process.stdout else ""
        except (OSError, ValueError):
            line = ""
        if not line and process.poll() is not None:
            break
        if line:
            output.append(line.rstrip())
    if cancelled:
        raise CancellationRequested("任务已被用户中断")
    if process.returncode != 0:
        tail = "\n".join(output[-40:])
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {printable}\n{tail}")


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate only the known child command after the user presses cancel."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def slugify(value: str, fallback: str = "video") -> str:
    value = re.sub(r"[^\w\-.]+", "-", value, flags=re.UNICODE).strip("-._")
    return value[:80] or fallback


def timestamp_id(prefix: str = "job") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def project_root() -> Path:
    return Path.cwd()


def ensure_rights_confirmed(confirmed: bool) -> None:
    if confirmed:
        return
    raise SystemExit(
        "Refusing to continue without --i-have-rights. Use this tool only for videos you own, "
        "videos you have explicit permission to process, or content whose license allows this use."
    )


def warn_optional_dependency(package: str, install_hint: str) -> None:
    print(f"Missing optional dependency: {package}", file=sys.stderr)
    print(f"Install hint: {install_hint}", file=sys.stderr)

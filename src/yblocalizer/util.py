from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


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


def run(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    completed = subprocess.run(command, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {printable}")


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

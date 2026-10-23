import json
from pathlib import Path

from yblocalizer import __version__


def test_versions_are_unified() -> None:
    root = Path(__file__).resolve().parents[1]
    package = json.loads((root / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert __version__ == package["version"] == "0.3.0"
    assert "v0.3.0" in (root / "PORTABLE-README.txt").read_text(encoding="utf-8")

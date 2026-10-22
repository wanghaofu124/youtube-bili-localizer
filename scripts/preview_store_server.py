"""Isolated loopback API preview server for Store screenshots."""
from __future__ import annotations
import argparse, json, os, signal
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()
data_dir = Path(args.data_dir).resolve()
data_dir.mkdir(parents=True, exist_ok=True)
(data_dir / ".env").write_text("", encoding="utf-8")
os.environ["APPDATA"] = str(data_dir)
os.environ["YBLOCALIZER_DATA_DIR"] = str(data_dir)
for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "YBLOCALIZER_OLLAMA", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"):
    os.environ.pop(key, None)

from yblocalizer.workbench_api import ASSET_ROOT, build_server

server = build_server(ASSET_ROOT / "frontend" / "dist", "127.0.0.1", 0)
info = {"host": "127.0.0.1", "port": server.server_address[1], "pid": os.getpid(), "data_dir": str(data_dir)}
(data_dir / "preview.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: server.shutdown())
try:
    server.serve_forever(poll_interval=0.2)
finally:
    server.server_close()

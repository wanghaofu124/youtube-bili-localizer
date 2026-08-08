# Contributing

## Local checks

```powershell
pip install -e ".[dev]"
python -m compileall -q src
python -m pytest -q
```

Tests mock every external integration. Do not add test cases that download media, load a Whisper model, call an LLM, start Playwright, or require credentials.

## Updating the public demo or benchmark

Only use `demo/authorized-demo-10s.mp4` as public input. Run the real benchmark locally:

```powershell
python scripts/benchmark_demo.py --device cpu --compute-type int8 --export-demo
python scripts/benchmark_demo.py --device cuda --compute-type float16
```

Review the reports in `benchmarks/runs/` for errors and sensitive paths. Copy approved reports to `benchmarks/cpu.json` and `benchmarks/cuda.json`; then update the README with only values present in those files. Do not commit `outputs/`, browser data, credentials, or unlicensed media.

# Benchmarks

Run the real core pipeline against the authorized 10-second demo on the local
machine. The command loads `.env` only to call the configured translation
provider; it does not open a browser or publish anything.

```powershell
python scripts/benchmark_demo.py --device cpu --compute-type int8
python scripts/benchmark_demo.py --device cuda --compute-type float16
```

Each run writes a timestamped JSON report to `benchmarks/runs/`. Inspect the
report, then copy a reviewed and redacted result to this directory as
`cpu.json` or `cuda.json` before citing any number in the README or resume. Reports record
the exact hardware, model settings, input duration, stage timings, output
summary, and failures.

Committed results are summaries of successful local runs. The ignored raw
reports remain available locally for audit, while the public summaries exclude
absolute output paths and verbose tool build configuration.

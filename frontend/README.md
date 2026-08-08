# Workbench demo

This warm-color React workbench is connected to the existing Python pipeline.
It only starts the repository's authorised 10-second demo source; no browser,
account, cookie, or arbitrary URL is used by this demo.

## Run the real demo

```powershell
cd D:\codex\youtube-bili-localizer
powershell -ExecutionPolicy Bypass -File scripts\run_workbench_demo.ps1
```

Open `http://127.0.0.1:8765`, then select **开始处理**. The UI starts a local
Python job, polls real pipeline logs, and reports the generated SRT and hard
subtitle video under `outputs/workbench_demo/`.

The current `.env` must contain the configured `DEEPSEEK_API_KEY`; `ffmpeg`,
faster-whisper, and its selected CPU/CUDA runtime must be available. The
default device is CUDA/float16, and the advanced panel can switch to CPU/int8.

## Frontend-only development

```powershell
cd frontend
npm install
npm run dev
```

Keep `scripts\run_workbench_demo.ps1` running in another terminal; Vite proxies
`/api` to the local Python bridge. `npm run build` creates the static bundle
served by the bridge for the portfolio demo.

# YouTube 视频本地化工具（YouTube Bili Localizer）

面向**已获授权视频**的 Python 本地化工具：将本地文件或授权 URL 转为中文字幕硬字幕视频，并在人工确认前辅助填写 B 站投稿信息。

> 仅处理你拥有版权、已取得明确授权或许可证允许下载、改编与转载的素材。本项目不绕过登录、验证码、风控或版权限制；B 站流程不会自动点击“立即投稿”。

## 演示与能力

仓库内提供一段已授权的 10 秒输入样例：[`demo/authorized-demo-10s.mp4`](demo/authorized-demo-10s.mp4)。运行一次本地 demo 后，可在 `demo/artifacts/` 查看生成的源字幕、中文字幕和硬字幕成片。

![授权样例的中文字幕硬字幕输出](demo/artifacts/preview.png)

```mermaid
flowchart LR
    A[授权 URL / 本地视频] --> B[yt-dlp 导入或本地复制]
    B --> C[faster-whisper 音频转写]
    B --> D[OCR 画面文字]
    C --> E[时间轴合并与字幕校对]
    D --> E
    E --> F[LLM 上下文翻译]
    F --> G[SRT / 双语排版]
    G --> H[FFmpeg 硬字幕渲染]
    H --> I[Playwright 投稿信息辅助]
```

- URL 或本地文件输入；`yt-dlp` 支持读取用户已登录浏览器的 cookies（不会绕过平台验证）。
- `faster-whisper` 支持 CPU、CUDA 和可配置精度；OCR 可识别画面内英文字幕与信息标签。
- 音频/OCR 可分别使用，或按时间轴合并；转写、翻译和输出均保留可检查的 JSON/SRT 中间产物。
- OpenAI / DeepSeek 可进行源字幕校对、上下文分批翻译和投稿元数据生成。
- 支持中文、原文、双语字幕，动态换行、字体、描边、阴影和位置配置。
- 图形工作台（WebView 界面）：素材、处理、字幕精修、视频修改（删除/保留/静音/导出/重排）、发布辅助、任务历史；另保留 CLI。
- 字幕时间轴精修与视频片段级编辑均自动保持字幕对齐，处理结果（转写/OCR/翻译对照）实时可见。

## 5 分钟跑通

### 0. 前置要求（Windows）

1. **Python 3.10+**：从 <https://www.python.org/downloads/> 安装，勾选 **Add python.exe to PATH**；
2. **ffmpeg**：管理员 PowerShell 执行 `winget install Gyan.FFmpeg`，装完后重开终端（刷新 PATH）；
3. **Node.js**（可选但强烈建议）：YouTube 视频流需要 JS 挑战求解；
4. **Tesseract OCR**（可选，仅「画面字幕 OCR」需要）：`winget install UB-Mannheim.TesseractOCR`；
5. **NVIDIA GPU**（可选）：有 CUDA 时可用 `--device cuda` 加速转写。

### 1. 源码安装

```powershell
git clone https://github.com/wanghaofu124/youtube-bili-localizer.git
cd youtube-bili-localizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
playwright install chromium
```

复制 `.env.example` 为 `.env`，填写一个翻译服务的 API Key：

```powershell
copy .env.example .env
```

### 2. 处理授权样例

CPU 通用配置：

```powershell
yblocalizer process --file demo\authorized-demo-10s.mp4 --i-have-rights --translator deepseek --subtitle-source audio --device cpu --compute-type int8 --output-dir outputs\demo
```

有 NVIDIA CUDA 环境时可改用：`--device cuda --compute-type float16`。成功后目录结构如下：

```text
outputs/demo/job-*/
├── source.srt                 # 英文源字幕
├── segments.source.json       # 带时间轴的源分段
├── zh.srt                     # 中文或双语字幕
├── segments.translated.json   # 翻译后的分段
├── publish_metadata.json      # 建议标题和标签
└── rendered.mp4               # 最终硬字幕视频
```

完整的 GUI、URL 输入、CUDA、OCR、字幕样式和投稿辅助说明请看[使用教程](docs/使用教程.md)。

## Windows EXE 打包（可选）

源码运行即可正常使用全部功能；如需免安装 EXE，用 PyInstaller 自行打包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
```

产物位于 `dist\`（`YouTubeBiliLocalizer.exe` 为图形界面、`yblocalizer.exe` 为命令行）。目标机器仍需安装 `ffmpeg`；OCR 模式另需 Tesseract；YouTube 下载建议安装 Node.js。Whisper 模型首次转写时自动下载。

> **数据位置**：运行时的配置（`.env`）、任务输出、字幕、浏览器 Profile 与任务历史（SQLite）都保存在用户数据目录 `C:\Users\<用户名>\AppData\Roaming\YouTubeBiliLocalizer\`，与 EXE 安装目录分离——升级或重建 EXE 不会删除你的数据。

## 工作台前端

暖色 React 工作台在 [`frontend/`](frontend/) 实现（Vite + React，由本地 HTTP 桥 `yblocalizer.workbench_api` 提供 API）。包含视频预览、真实下载进度、字幕时间轴编辑、逐条字幕时间/文字/删除编辑、视频片段级修改（删除/保留/静音/导出/重排）、处理结果实时对照、B 站投稿辅助状态、任务历史与输出文件管理。前端构建：`cd frontend && npm run build`。

## 开发验证与性能基准

```powershell
# CI 同款：不下载媒体、不加载模型、不调用 API、不启动浏览器
python -m pytest -q
python -m compileall -q src

# 真实本机基准：会加载模型并调用 .env 配置的翻译服务
python scripts/benchmark_demo.py --device cpu --compute-type int8 --export-demo
python scripts/benchmark_demo.py --device cuda --compute-type float16
```

基准会对生产流水线的导入、音频提取、转写、字幕审校、翻译、投稿元数据生成及渲染逐阶段计时，并输出版本化 JSON 到 `benchmarks/runs/`。

### 已记录的本机基准

同一 10 秒、1280×720 授权样例，以 `small` Whisper、DeepSeek 翻译与 FFmpeg 渲染完整运行。端到端数据包含网络翻译和元数据生成延迟，不能泛化为其他视频或网络环境的性能承诺。

| 配置 | 转写 | 翻译 | 端到端 |
| --- | ---: | ---: | ---: |
| CPU / int8 | 36.085 秒 | 28.361 秒 | 74.492 秒 |
| RTX 5070 CUDA / float16 | 11.426 秒 | 21.656 秒 | 44.814 秒 |

完整配置、阶段耗时与产物摘要见 [CPU 记录](benchmarks/cpu.json) 和 [CUDA 记录](benchmarks/cuda.json)。

GitHub Actions 在 Ubuntu 的 Python 3.10 与 3.12 上运行编译检查和 mock 测试。测试覆盖字幕时间轴与排版、OCR/音频合并、翻译行数修复、中文结果验证、文案模板、输出目录安全清理、CLI 参数映射和完整流水线的 mock 成功/失败路径。

## 架构与关键取舍

v0.2 将 React 工作台确定为唯一新增功能的 UI，旧 Tk GUI 进入兼容维护。默认值、能力探测与预检统一由后端提供；任务使用独立取消令牌和结构化阶段事件，跨启动历史由 SQLite 管理。详细边界见 [v0.2 架构说明](docs/architecture-v0.2.md)。

| 设计 | 原因 |
| --- | --- |
| 重型依赖惰性导入 | CLI 帮助、测试和非模型功能不需要下载或加载 Whisper / Playwright。 |
| 音频与 OCR 双路径 | 语音转写覆盖对白，OCR 补足画面标签与已烧录字幕；合并时按时间重叠与文本相似度去重。 |
| 上下文分批翻译 | 控制单次 API 请求大小，同时在批次前后提供上下文，避免逐句直译。 |
| 动态字幕排版 | 根据语种、可用时长和字符数调整换行，限制屏幕行数以提升可读性。 |
| 人工确认投稿 | Playwright 仅协助上传和填写；用户核对分类、版权声明和内容后手动提交。 |

## 公开资产与安全

- 仅 `demo/authorized-demo-10s.mp4` 与从它生成的 `demo/artifacts/` 可作为公开演示媒体。
- `outputs/`、浏览器诊断截图、cookies、Profile、API 响应、日志及无明确授权素材一律忽略，禁止提交。
- `.env` 已忽略；请勿提交 API Key。有关本地开发与基准更新方式见 [benchmarks/README.md](benchmarks/README.md)。

# YouTube Bili Localizer

一个用于授权视频本地化的桌面工具：获取 YouTube 或本地视频素材，自动转写字幕、翻译成中文、压制硬字幕，并辅助打开 B 站创作中心填写投稿信息。

> 只处理你拥有版权、已获得作者明确授权，或许可证允许下载、改编和转载的视频。本项目不会绕过 YouTube、B 站或任何平台的登录、验证码、风控和版权限制。

## 详细教程

第一次使用请先看这里：

[详细使用教程](docs/使用教程.md)

教程包含安装、API Key 配置、GUI 全流程、CUDA、字幕模式、B 站发布辅助、命令行示例和常见问题。

## 功能

- 使用 `yt-dlp` 获取授权视频素材，支持读取浏览器 cookies。
- 使用 `faster-whisper` 从音频转写字幕，支持 CPU、CUDA 或自动降级。
- 支持音频字幕、画面 OCR 字幕，以及“音频 + OCR 合并翻译”。
- 支持 `none` 调试翻译器，以及 `openai` / `deepseek` 生成中文字幕。
- 翻译前可用模型校对源字幕，修正明显 ASR/OCR 错误、专有名词和上下文问题。
- 支持中文单语、原文 + 中文、中英双语字幕显示。
- 可设置字体、字号、字幕颜色、描边、阴影和位置。
- 使用 `ffmpeg` 将 `.srt` 字幕压制进视频。
- 使用 Playwright 辅助打开 B 站创作中心、上传视频、填写标题、简介、标签和封面，并在提交前停住等待人工确认。
- 提供 Tkinter 图形界面，支持输出目录占用查看和选择性清理。

## 快速安装

```powershell
cd D:\codex\youtube-bili-localizer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
playwright install chromium
```

还需要安装 `ffmpeg`，并确保 `ffmpeg` 在系统 `PATH` 中。

## API Key

复制示例配置：

```powershell
copy .env.example .env
```

按需填写：

```env
OPENAI_API_KEY=
OPENAI_TRANSLATE_MODEL=gpt-4.1-mini

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_TRANSLATE_MODEL=deepseek-v4-flash
```

`.env` 已加入 `.gitignore`，不要把真实密钥提交到 GitHub。

## 启动 GUI

无终端窗口启动：

```powershell
.\launch_gui.vbs
```

调试模式启动：

```powershell
.\launch_gui_debug.bat
```

也可以直接运行：

```powershell
python -m yblocalizer.gui
```

## 命令行示例

处理授权 YouTube 链接：

```powershell
python -m yblocalizer process --url "https://www.youtube.com/watch?v=VIDEO_ID" --i-have-rights --require-reuse-allowed --max-seconds 60 --translator deepseek
```

处理本地视频：

```powershell
python -m yblocalizer process --file "D:\path\input.mp4" --i-have-rights --translator deepseek
```

辅助发布到 B 站：

```powershell
python -m yblocalizer publish --video "outputs\job-xxx\rendered.mp4" --title "标题" --description "已获授权转载/本地化。" --source-url "https://www.youtube.com/watch?v=VIDEO_ID"
```

更多说明见 [详细使用教程](docs/使用教程.md)。

## 免责声明

请只在合法授权范围内使用本工具。使用者需要自行确认素材版权、转载授权、平台规则和最终发布内容。本项目不提供规避平台限制、批量搬运未授权内容或绕过安全机制的能力。

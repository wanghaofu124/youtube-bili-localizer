# v0.2 架构边界

工作台采用模块化单体。依赖方向固定为：桌面壳/HTTP/CLI/Tk 适配器 → 应用服务与配置 → 流水线领域逻辑 → yt-dlp、Whisper、FFmpeg、Playwright 等基础设施。

## 决策

- React 工作台是唯一新增功能的 UI；Tk GUI 只维护启动与核心兼容性。
- `workbench_config.py` 是工作台默认值、归一化、能力探测和预检规则的唯一来源。前端通过 `/api/bootstrap` 取得默认值，通过 `/api/preflight` 取得结构化阻塞项与警告。
- 默认字幕来源为音频。Tesseract 对音频模式无要求，对合并/自动模式只给警告，对纯 OCR 模式才阻塞。
- `PipelineContext` 为每个任务持有独立取消令牌和结构化事件回调。维护期 Tk 入口的旧全局取消函数只是兼容适配器，不参与工作台任务。
- SQLite 是跨启动任务历史的事实来源；输出目录扫描只描述当前磁盘占用。启动时遗留的运行态记录会迁移为 `interrupted`。
- 演示入口只读取仓库内预生成的授权成片与字幕，不加载模型、不请求 API；真实素材在创建任务前必须通过预检。

## 稳定边界

- B 站自动化停在人工提交前。
- API 不回传 Key、Cookies 内容或任意未登记路径。
- 用户 `.env`、SQLite、输出目录和浏览器 Profile 位于 `%APPDATA%\YouTubeBiliLocalizer`，升级不会覆盖。
- v0.2 不承诺批处理、缓存、下载重试或自动投稿。

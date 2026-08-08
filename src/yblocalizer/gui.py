from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from .pipeline import PipelineOptions, run_pipeline, request_cancellation, reset_cancellation, is_cancellation_requested, CancellationError
from .download import get_video_metadata
from .publish_bili import UPLOAD_URL, assist_publish, open_upload_page
from .publish_text import (
    DEFAULT_TEMPLATES as DESCRIPTION_TEMPLATES,
    build_bilibili_description,
    ensure_source_link,
    get_all_templates,
    get_template_names,
    save_custom_template,
    delete_custom_template,
    preview_template,
)
from .storage import delete_paths, format_bytes, scan_outputs


STAGE_PROGRESS = {
    "1/5": (12, "准备素材"),
    "2/5": (28, "提取音频"),
    "3/5": (48, "生成字幕"),
    "4/5": (70, "翻译字幕"),
    "5/5": (88, "压制视频"),
}

PALETTE = {
    "page": "#eef3f2",
    "panel": "#fbfcfb",
    "panel_alt": "#f4f8f6",
    "line": "#d8e1df",
    "text": "#1f2d32",
    "muted": "#5d6b72",
    "header": "#d9f0ea",
    "header_accent": "#ffddd2",
    "accent": "#2a9d8f",
    "accent_dark": "#1f6f67",
    "blue": "#457b9d",
    "warm": "#e76f51",
    "log_bg": "#f7faf8",
    "log_text": "#27363b",
}


class LocalizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        if load_dotenv:
            load_dotenv()
        self.title("YouTube Bili Localizer")
        self.geometry("1080x800")
        self.minsize(980, 720)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.bilibili_page_worker: threading.Thread | None = None
        self.result_dir: Path | None = None
        self._bili_pre_submit_notice_shown = False
        self._run_started_at: float | None = None
        self._elapsed_timer_id: str | None = None
        self._build_vars()
        self._configure_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.after(100, self._drain_logs)

    def _build_vars(self) -> None:
        self.source_kind = tk.StringVar(value="url")
        self.url = tk.StringVar(value="")
        self.file_path = tk.StringVar(value="")
        self.title_text = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=str(Path("outputs").resolve()))
        self.i_have_rights = tk.BooleanVar(value=False)
        self.require_reuse_allowed = tk.BooleanVar(value=True)
        self.full_video = tk.BooleanVar(value=False)
        self.duration_value = tk.StringVar(value="10")
        self.duration_unit = tk.StringVar(value="秒")
        self.total_duration_text = tk.StringVar(value="总时长：未读取")
        self.cookies_from_browser = tk.StringVar(value="")
        self.translator = tk.StringVar(value="deepseek")
        self.target_lang = tk.StringVar(value="zh-Hans")
        self.translate_model = tk.StringVar(value="")
        self.api_key = tk.StringVar(value=os.getenv("DEEPSEEK_API_KEY", ""))
        self.subtitle_source = tk.StringVar(value="音频 + 画面文字合并")
        self.whisper_model = tk.StringVar(value="small")
        self.source_language = tk.StringVar(value="自动检测")
        self.device = tk.StringVar(value="cpu")
        self.compute_type = tk.StringVar(value="int8")
        self.font_size = tk.StringVar(value="24")
        self.subtitle_mode = tk.StringVar(value="中文单语")
        self.font_name = tk.StringVar(value="Microsoft YaHei")
        self.subtitle_color = tk.StringVar(value="白色")
        self.subtitle_outline_color = tk.StringVar(value="黑色")
        self.subtitle_effect = tk.StringVar(value="描边")
        self.smart_translation_layout = tk.BooleanVar(value=True)
        self.publish_to_bilibili = tk.BooleanVar(value=False)
        self.close_bilibili_browser_after_fill = tk.BooleanVar(value=False)
        self.bilibili_browser = tk.StringVar(value="Chromium")
        self.direct_publish_video = tk.StringVar(value="")
        self.description_template = tk.StringVar(value="授权本地化")
        self.include_source_link = tk.BooleanVar(value=True)
        self.tags = tk.StringVar(value="")
        self.output_summary = tk.StringVar(value="尚未扫描输出目录")
        self.run_state = tk.StringVar(value="就绪")
        self.current_stage = tk.StringVar(value="等待开始")
        self.elapsed_text = tk.StringVar(value="用时 00:00")
        self.progress_value = tk.IntVar(value=0)
        self.guide_status = tk.StringVar(
            value="按顺序完成：选择素材 -> 设置字幕翻译 -> 可选发布文案 -> 开始全流程。"
        )

    def _configure_styles(self) -> None:
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.configure(background=PALETTE["page"])
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=PALETTE["page"], foreground=PALETTE["text"])
        style.configure("TFrame", background=PALETTE["page"])
        style.configure("Panel.TFrame", background=PALETTE["panel"])
        style.configure("TLabelframe", background=PALETTE["panel"], bordercolor=PALETTE["line"], relief="solid")
        style.configure("TLabelframe.Label", background=PALETTE["page"], foreground=PALETTE["text"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", background=PALETTE["page"], foreground=PALETTE["text"])
        style.configure("TButton", padding=(12, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#e8f1ef")])
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor=PALETTE["line"], lightcolor=PALETTE["line"], darkcolor=PALETTE["line"], padding=4)
        style.configure("TCombobox", fieldbackground="#ffffff", bordercolor=PALETTE["line"], arrowcolor=PALETTE["muted"], padding=4)
        style.configure("TCheckbutton", background=PALETTE["panel"], foreground=PALETTE["text"])
        style.configure("TRadiobutton", background=PALETTE["panel"], foreground=PALETTE["text"])
        style.configure("Header.TFrame", background=PALETTE["header"])
        style.configure("HeaderTitle.TLabel", background=PALETTE["header"], foreground="#173b3f", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("HeaderText.TLabel", background=PALETTE["header"], foreground="#40616a")
        style.configure("Chip.TLabel", background="#ffffff", foreground=PALETTE["accent_dark"], padding=(10, 4), font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("WarmChip.TLabel", background=PALETTE["header_accent"], foreground="#7c2d12", padding=(10, 4), font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Status.TFrame", background="#fff9f1")
        style.configure("Status.TLabel", background="#fff9f1", foreground=PALETTE["text"])
        style.configure("Horizontal.TProgressbar", troughcolor="#f4e7db", background=PALETTE["accent"], bordercolor="#f4e7db", lightcolor=PALETTE["accent"], darkcolor=PALETTE["accent"])
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"), background=PALETTE["accent"], foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", PALETTE["accent_dark"]), ("disabled", "#a7b7b4")], foreground=[("disabled", "#f8fafc")])
        style.configure("Danger.TButton", foreground="#8a2d1b")

    def _build_ui(self) -> None:
        shell = ttk.Frame(self)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(shell, highlightthickness=0, background=PALETTE["page"])
        main_scroll = ttk.Scrollbar(shell, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=main_scroll.set)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        main_scroll.grid(row=0, column=1, sticky="ns")

        root = ttk.Frame(self.main_canvas, padding=18)
        self._main_window_id = self.main_canvas.create_window((0, 0), window=root, anchor="nw")
        root.bind("<Configure>", self._update_main_scroll_region)
        self.main_canvas.bind("<Configure>", self._resize_scroll_content)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(6, weight=1)

        header = ttk.Frame(root, padding=(22, 16), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="YouTube Bili Localizer", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="授权视频本地化工作台。把下载、字幕、翻译、压制和 B 站发布辅助收进一条清晰的流水线。",
            style="HeaderText.TLabel",
            wraplength=940,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        chips = ttk.Frame(header, style="Header.TFrame")
        chips.grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Label(chips, text="授权素材", style="Chip.TLabel").grid(row=0, column=0, padx=(0, 8))
        ttk.Label(chips, text="音频 + OCR", style="Chip.TLabel").grid(row=0, column=1, padx=(0, 8))
        ttk.Label(chips, text="中文排版", style="Chip.TLabel").grid(row=0, column=2, padx=(0, 8))
        ttk.Label(chips, text="提交前人工确认", style="WarmChip.TLabel").grid(row=0, column=3)

        guide = ttk.LabelFrame(root, text="0. 快捷操作", padding=12)
        guide.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        guide.columnconfigure(5, weight=1)
        ttk.Button(guide, text="YouTube 任务", command=self._guide_youtube).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(guide, text="本地视频任务", command=self._guide_local_file).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(guide, text="推荐设置", command=self._apply_easy_defaults).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(guide, text="检查配置", command=self._check_basic_readiness).grid(row=0, column=3, padx=(0, 8))
        more_button = ttk.Menubutton(guide, text="更多工具")
        more_menu = tk.Menu(more_button, tearoff=False)
        more_menu.add_command(label="启用 B 站发布辅助", command=self._guide_bilibili)
        more_menu.add_command(label="登录/检查 B 站页面", command=self._open_bilibili_upload_page)
        more_menu.add_separator()
        more_menu.add_command(label="刷新全部状态", command=self._refresh_all)
        more_menu.add_command(label="输出文件管理", command=self._open_storage_manager)
        more_menu.add_command(label="关闭后台 B 站浏览器", command=self._close_bilibili_background_browser)
        more_button["menu"] = more_menu
        more_button.grid(row=0, column=4, padx=(0, 8))
        ttk.Label(guide, textvariable=self.guide_status, wraplength=920).grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0))

        status = ttk.Frame(guide, padding=(10, 8), style="Status.TFrame")
        status.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.run_state, style="Status.TLabel", width=12).grid(row=0, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(status, variable=self.progress_value, maximum=100, mode="determinate")
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(status, textvariable=self.current_stage, style="Status.TLabel", width=18).grid(row=0, column=2, sticky="e")
        ttk.Label(status, textvariable=self.elapsed_text, style="Status.TLabel", width=12).grid(row=0, column=3, sticky="e", padx=(8, 0))

        source = ttk.LabelFrame(root, text="1. 素材来源", padding=12)
        source.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        source.columnconfigure(1, weight=1)
        ttk.Radiobutton(source, text="YouTube URL", variable=self.source_kind, value="url").grid(row=0, column=0, sticky="w")
        self.url_entry = ttk.Entry(source, textvariable=self.url)
        self.url_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self.url_entry.bind("<FocusOut>", self._on_url_focus_out)
        self.url_entry.bind("<Return>", self._on_url_return)
        ttk.Button(source, text="读取时长", command=lambda: self._start_metadata_load(silent=False)).grid(row=0, column=2, sticky="e")
        ttk.Radiobutton(source, text="本地视频", variable=self.source_kind, value="file").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source, textvariable=self.file_path).grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(source, text="选择文件", command=self._choose_file).grid(row=1, column=2, pady=(8, 0))
        ttk.Label(source, text="标题").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source, textvariable=self.title_text).grid(row=2, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Checkbutton(source, text="我确认拥有处理/转载该视频的授权或许可证", variable=self.i_have_rights).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(source, text="YouTube URL 必须标记为 Creative Commons reuse allowed", variable=self.require_reuse_allowed).grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Label(source, text="URL 读取长度").grid(row=5, column=0, sticky="w", pady=(8, 0))
        duration_controls = ttk.Frame(source)
        duration_controls.grid(row=5, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Checkbutton(duration_controls, text="完整视频", variable=self.full_video, command=self._sync_duration_state).grid(row=0, column=0, sticky="w")
        self.duration_entry = ttk.Entry(duration_controls, textvariable=self.duration_value, width=10)
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=(12, 4))
        self.duration_unit_box = ttk.Combobox(duration_controls, textvariable=self.duration_unit, values=["秒", "分钟", "小时"], state="readonly", width=8)
        self.duration_unit_box.grid(row=0, column=2, sticky="w")
        ttk.Label(source, textvariable=self.total_duration_text).grid(row=5, column=2, sticky="w", pady=(8, 0))
        ttk.Label(source, text="YouTube Cookies").grid(row=6, column=0, sticky="w", pady=(8, 0))
        cookies_box = ttk.Combobox(
            source,
            textvariable=self.cookies_from_browser,
            values=["", "chrome", "edge", "firefox", "brave", "chromium"],
            state="readonly",
            width=14,
        )
        cookies_box.grid(row=6, column=1, sticky="w", padx=8, pady=(8, 0))
        cookies_box.bind("<<ComboboxSelected>>", lambda _event: self._start_metadata_load(silent=True))
        self._sync_duration_state()

        settings = ttk.LabelFrame(root, text="2. 字幕与翻译", padding=12)
        settings.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for column in range(7):
            settings.columnconfigure(column, weight=1)
        ttk.Label(settings, text="翻译器").grid(row=0, column=0, sticky="w")
        translator_box = ttk.Combobox(settings, textvariable=self.translator, values=["deepseek", "openai", "none"], state="readonly", width=14)
        translator_box.grid(row=0, column=1, sticky="w")
        translator_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_api_key())
        ttk.Label(settings, text="API Key").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.api_key, show="*", width=34).grid(row=0, column=3, columnspan=3, sticky="ew")
        ttk.Button(settings, text="保存 Key 到 .env", command=self._save_api_key).grid(row=0, column=6, sticky="e", padx=(8, 0))
        ttk.Label(settings, text="模型").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.translate_model, width=20).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="目标语言").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.target_lang, width=12).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字幕来源").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Combobox(
            settings,
            textvariable=self.subtitle_source,
            values=["自动检测", "音频 + 画面文字合并", "音频转写", "画面英文字幕 OCR"],
            state="readonly",
            width=18,
        ).grid(row=1, column=5, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字幕模式").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            settings,
            textvariable=self.subtitle_mode,
            values=["中文单语", "原文在上 + 中文在下", "中文在上 + 原文在下"],
            state="readonly",
            width=20,
        ).grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="Whisper").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.whisper_model, values=["tiny", "base", "small", "medium", "large-v3"], state="readonly", width=14).grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="源语言").grid(row=2, column=4, sticky="w", pady=(8, 0))
        ttk.Combobox(
            settings,
            textvariable=self.source_language,
            values=["自动检测", "en", "zh", "ja", "ko", "fr", "de", "es", "ru"],
            state="readonly",
            width=12,
        ).grid(row=2, column=5, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="设备").grid(row=3, column=0, sticky="w", pady=(8, 0))
        device_box = ttk.Combobox(settings, textvariable=self.device, values=["cpu", "auto", "cuda"], state="readonly", width=14)
        device_box.grid(row=3, column=1, sticky="w", pady=(8, 0))
        device_box.bind("<<ComboboxSelected>>", lambda _event: self._sync_compute_type_for_device())
        ttk.Label(settings, text="计算类型").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.compute_type, values=["int8", "int8_float16", "default", "float16", "float32"], state="readonly", width=14).grid(row=3, column=3, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字体").grid(row=3, column=4, sticky="w", pady=(8, 0))
        ttk.Combobox(
            settings,
            textvariable=self.font_name,
            values=["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Noto Sans CJK SC"],
            width=18,
        ).grid(row=3, column=5, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字号").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.font_size, width=8).grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字幕颜色").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.subtitle_color, values=["白色", "黄色", "青色", "绿色", "黑色"], state="readonly", width=10).grid(row=4, column=3, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="边框颜色").grid(row=4, column=4, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.subtitle_outline_color, values=["黑色", "白色", "灰色", "蓝色"], state="readonly", width=10).grid(row=4, column=5, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="样式").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.subtitle_effect, values=["描边", "阴影", "描边+阴影", "无"], state="readonly", width=12).grid(row=5, column=1, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            settings,
            text="智能上下文翻译与排版",
            variable=self.smart_translation_layout,
        ).grid(row=5, column=2, columnspan=4, sticky="w", pady=(8, 0))

        publish = ttk.LabelFrame(root, text="3. B 站发布辅助", padding=12)
        publish.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        publish.columnconfigure(1, weight=1)
        ttk.Checkbutton(publish, text="渲染完成后打开 B 站创作中心并辅助上传填表", variable=self.publish_to_bilibili).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(publish, text="B 站浏览器").grid(row=0, column=3, sticky="e", padx=(8, 4))
        ttk.Combobox(
            publish,
            textvariable=self.bilibili_browser,
            values=["Chromium", "Microsoft Edge"],
            state="readonly",
            width=16,
        ).grid(row=0, column=4, sticky="e")
        
        # 模板选择行
        ttk.Label(publish, text="简介模板").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.template_box = ttk.Combobox(
            publish,
            textvariable=self.description_template,
            values=get_template_names(),
            state="readonly",
            width=20,
        )
        self.template_box.grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        self.template_box.bind("<<ComboboxSelected>>", lambda _event: self._generate_description())
        ttk.Checkbutton(publish, text="简介自动附原视频链接", variable=self.include_source_link).grid(row=1, column=2, sticky="w", pady=(8, 0))
        
        # 模板管理按钮
        template_mgr = ttk.Frame(publish)
        template_mgr.grid(row=1, column=3, columnspan=2, sticky="e", pady=(8, 0))
        template_tools = ttk.Menubutton(template_mgr, text="模板工具")
        template_menu = tk.Menu(template_tools, tearoff=False)
        template_menu.add_command(label="预览当前文案", command=self._preview_description)
        template_menu.add_command(label="新建模板", command=self._open_template_editor_new)
        template_menu.add_command(label="编辑当前模板", command=self._open_template_editor_edit)
        template_menu.add_command(label="删除当前模板", command=self._delete_template)
        template_tools["menu"] = template_menu
        template_tools.grid(row=0, column=0)
        
        # 文案生成按钮行
        publish_buttons = ttk.Frame(publish)
        publish_buttons.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(4, 0))
        ttk.Button(publish_buttons, text="更新文案", command=self._generate_description).grid(row=0, column=0, padx=(0, 8))
        self.open_upload_button = ttk.Button(publish_buttons, text="登录检查", command=self._open_bilibili_upload_page)
        self.open_upload_button.grid(row=0, column=1)
        ttk.Label(publish_buttons, text="  追加备注:", font=("", 9)).grid(row=0, column=2, padx=(12, 4))
        self.extra_lines_entry = ttk.Entry(publish_buttons, width=30)
        self.extra_lines_entry.grid(row=0, column=3, sticky="ew", padx=(0, 8))
        ttk.Button(publish_buttons, text="追加", command=self._append_extra_lines).grid(row=0, column=4)
        
        # 简介文本框
        ttk.Label(publish, text="简介").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        self.description_text = tk.Text(
            publish,
            height=5,
            wrap="word",
            relief="flat",
            padx=8,
            pady=8,
            background="#ffffff",
            foreground=PALETTE["text"],
            insertbackground=PALETTE["text"],
            selectbackground="#cce7df",
        )
        self.description_text.grid(row=3, column=1, columnspan=4, sticky="ew", padx=8, pady=(8, 0))
        
        # 标签行
        ttk.Label(publish, text="标签，逗号分隔").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(publish, textvariable=self.tags).grid(row=4, column=1, columnspan=4, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(publish, text="成品视频").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(publish, textvariable=self.direct_publish_video).grid(row=5, column=1, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(publish, text="选择成品视频", command=self._choose_direct_publish_video).grid(row=5, column=3, sticky="e", pady=(8, 0))
        self.direct_publish_button = ttk.Button(publish, text="直接上传到 B 站", command=self._start_direct_publish)
        self.direct_publish_button.grid(row=5, column=4, sticky="e", padx=(8, 0), pady=(8, 0))
        ttk.Checkbutton(
            publish,
            text="填完投稿信息后自动关闭浏览器（上传未完成会中断）",
            variable=self.close_bilibili_browser_after_fill,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(
            publish,
            text="关闭后台 B 站浏览器",
            command=self._close_bilibili_background_browser,
            style="Danger.TButton",
        ).grid(row=6, column=3, columnspan=2, sticky="e", pady=(8, 0))
        self._generate_description()

        output = ttk.LabelFrame(root, text="4. 输出与文件管理", padding=12)
        output.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        output.columnconfigure(1, weight=1)
        ttk.Label(output, text="保存位置").grid(row=0, column=0, sticky="w")
        ttk.Entry(output, textvariable=self.output_dir).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(output, text="选择保存位置", command=self._choose_output_dir).grid(row=0, column=2)
        ttk.Button(output, text="打开目录", command=self._open_result_dir).grid(row=0, column=3, padx=(8, 0))
        output_actions = ttk.Frame(output)
        output_actions.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        output_actions.columnconfigure(0, weight=1)
        ttk.Label(output_actions, textvariable=self.output_summary).grid(row=0, column=0, sticky="w")
        ttk.Button(output_actions, text="文件管理", command=self._open_storage_manager).grid(row=0, column=1, padx=(8, 0))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=12)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(12, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap="word",
            relief="flat",
            padx=10,
            pady=10,
            background=PALETTE["log_bg"],
            foreground=PALETTE["log_text"],
            insertbackground=PALETTE["text"],
            selectbackground="#d7ece6",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("stage", foreground=PALETTE["blue"])
        self.log_text.tag_configure("success", foreground=PALETTE["accent_dark"])
        self.log_text.tag_configure("warning", foreground=PALETTE["warm"])
        self.log_text.tag_configure("error", foreground="#b42318")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(root)
        actions.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, text="快捷键：Ctrl+R 开始，Esc 中断，F5 刷新，Ctrl+L 定位 URL").grid(row=0, column=0, sticky="w")
        self.run_button = ttk.Button(actions, text="开始全流程", command=self._start, style="Accent.TButton")
        self.run_button.grid(row=0, column=1, padx=(0, 8))
        self.cancel_button = ttk.Button(actions, text="中断任务", command=self._cancel, state=tk.DISABLED, style="Danger.TButton")
        self.cancel_button.grid(row=0, column=2, padx=(0, 8))
        self._refresh_output_summary()
        self._bind_main_scroll(root)

    def _update_main_scroll_region(self, _event: object) -> None:
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _resize_scroll_content(self, event: tk.Event) -> None:
        self.main_canvas.itemconfigure(self._main_window_id, width=event.width)

    def _bind_main_scroll(self, widget: tk.Widget) -> None:
        skip_types = (tk.Text, ttk.Treeview, ttk.Combobox)
        if not isinstance(widget, skip_types):
            widget.bind("<MouseWheel>", self._on_main_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_main_scroll(child)

    def _on_main_mousewheel(self, event: tk.Event) -> str:
        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        if delta:
            self.main_canvas.yview_scroll(delta, "units")
        return "break"

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-r>", lambda _event: self._start())
        self.bind_all("<Control-R>", lambda _event: self._start())
        self.bind_all("<Escape>", lambda _event: self._cancel())
        self.bind_all("<F5>", lambda _event: self._refresh_all())
        self.bind_all("<Control-l>", lambda _event: self._focus_url())
        self.bind_all("<Control-L>", lambda _event: self._focus_url())
        self.bind_all("<Control-o>", lambda _event: self._choose_file())
        self.bind_all("<Control-O>", lambda _event: self._choose_file())

    def _focus_url(self) -> None:
        self.source_kind.set("url")
        self.url_entry.focus_set()
        self.url_entry.selection_range(0, tk.END)
        self.main_canvas.yview_moveto(0.0)

    def _guide_youtube(self) -> None:
        self.source_kind.set("url")
        self.guide_status.set("粘贴 YouTube 链接，点击“读取时长”，确认授权；如遇登录/机器人检查，在 Cookies 里选择已登录的浏览器。")
        self.main_canvas.yview_moveto(0.0)

    def _guide_local_file(self) -> None:
        self.source_kind.set("file")
        self.guide_status.set("点击“选择文件”导入本地视频，确认授权后设置字幕与翻译；本地视频不会使用 URL 读取长度。")
        self.main_canvas.yview_moveto(0.0)

    def _guide_bilibili(self) -> None:
        self.publish_to_bilibili.set(True)
        self._generate_description()
        self.guide_status.set("已勾选 B 站发布辅助。请选择简介模板、检查原视频链接和标签；最终提交前仍会停在 B 站页面让你人工确认。")
        self.main_canvas.yview_moveto(0.45)

    def _apply_easy_defaults(self) -> None:
        self.translator.set("deepseek")
        self.target_lang.set("zh-Hans")
        self.subtitle_source.set("音频 + 画面文字合并")
        self.subtitle_mode.set("中文单语")
        self.source_language.set("自动检测")
        self.whisper_model.set("small")
        self.device.set("cpu")
        self.compute_type.set("int8")
        self.font_name.set("Microsoft YaHei")
        self.font_size.set("24")
        self.subtitle_color.set("白色")
        self.subtitle_outline_color.set("黑色")
        self.subtitle_effect.set("描边")
        self.smart_translation_layout.set(True)
        self.include_source_link.set(True)
        self._refresh_api_key()
        self._generate_description()
        self.guide_status.set("已应用推荐设置：DeepSeek 翻译、音频+画面文字合并、中文字幕、CPU 稳定模式。")

    def _check_basic_readiness(self) -> None:
        issues: list[str] = []
        source_kind = self.source_kind.get()
        source = self.url.get().strip() if source_kind == "url" else self.file_path.get().strip()
        if not source:
            issues.append("还没有选择素材")
        if not self.i_have_rights.get():
            issues.append("还没有确认授权")
        if source_kind == "url" and not self.full_video.get():
            try:
                self._parse_duration_seconds(self.duration_value.get(), self.duration_unit.get(), "URL 读取长度")
            except Exception:
                issues.append("URL 读取长度需要填写有效数字")
        translator = self.translator.get()
        target_lang = self.target_lang.get().strip() or "zh-Hans"
        if target_lang.lower().startswith("zh") and translator == "none":
            issues.append("中文字幕不能使用 none 翻译器")
        if translator == "deepseek" and not (self.api_key.get().strip() or os.getenv("DEEPSEEK_API_KEY")):
            issues.append("DeepSeek API Key 未填写")
        if translator == "openai" and not (self.api_key.get().strip() or os.getenv("OPENAI_API_KEY")):
            issues.append("OpenAI API Key 未填写")
        if issues:
            self.guide_status.set("还需要处理：" + "；".join(issues) + "。")
        else:
            self.guide_status.set("基础配置看起来可以运行。点击底部“开始全流程”，运行日志会显示每一步进度。")

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.webm *.mov"), ("All files", "*.*")])
        if path:
            self.file_path.set(path)
            self.source_kind.set("file")

    def _choose_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
            self._refresh_output_summary()

    def _on_url_focus_out(self, _event: object) -> None:
        self._start_metadata_load(silent=True)
        self._refresh_generated_description_if_needed()

    def _on_url_return(self, _event: object) -> None:
        self._start_metadata_load(silent=False)
        self._refresh_generated_description_if_needed()

    def _start_metadata_load(self, silent: bool) -> None:
        if self.source_kind.get() != "url":
            return
        url = self.url.get().strip()
        if not url:
            if not silent:
                messagebox.showerror("缺少 URL", "请先填写 YouTube URL。")
            return
        self.total_duration_text.set("总时长：读取中...")
        cookies = self.cookies_from_browser.get().strip() or None
        threading.Thread(target=self._metadata_worker, args=(url, cookies, silent), daemon=True).start()

    def _metadata_worker(self, url: str, cookies: str | None, silent: bool) -> None:
        try:
            metadata = get_video_metadata(url, cookies_from_browser=cookies)
            duration_text = self._format_duration(metadata.duration)
            license_text = metadata.license or "未知许可"
            text = f"总时长：{duration_text} | {license_text}"
            self.after(0, lambda: self.total_duration_text.set(text))
            if metadata.title:
                self.after(0, lambda title=metadata.title: self._set_title_if_empty(title))
        except Exception as exc:
            message = str(exc)
            self.after(0, lambda: self.total_duration_text.set("总时长：读取失败"))
            if not silent:
                self.after(0, lambda: messagebox.showerror("读取时长失败", message))

    def _sync_duration_state(self) -> None:
        state = tk.DISABLED if self.full_video.get() else tk.NORMAL
        self.duration_entry.configure(state=state)
        self.duration_unit_box.configure(state="disabled" if self.full_video.get() else "readonly")

    def _sync_compute_type_for_device(self) -> None:
        device = self.device.get().strip().lower()
        compute_type = self.compute_type.get().strip().lower()
        if device == "cpu" and compute_type in {"float16", "int8_float16"}:
            self.compute_type.set("int8")
            self.guide_status.set("CPU 转写已切换为 int8，稳定性更好。")
        elif device in {"cuda", "auto"} and compute_type == "int8":
            self.compute_type.set("float16")
            self.guide_status.set("CUDA/自动设备已切换为 float16；如果 CUDA 不可用，程序会自动降级 CPU。")

    def _set_title_if_empty(self, title: str) -> None:
        if not self.title_text.get().strip():
            self.title_text.set(title)

    def _generate_description(self) -> None:
        try:
            description = build_bilibili_description(
                self.description_template.get(),
                self._description_source(),
                include_source_link=self.include_source_link.get(),
            )
        except Exception as exc:
            self.guide_status.set(f"简介模板有问题：{exc}")
            messagebox.showerror("文案生成失败", str(exc), parent=self)
            return
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert("1.0", description)

    def _refresh_generated_description_if_needed(self) -> None:
        text = self._description_text()
        if not text.strip() or "请在发布前补充" in text:
            self._generate_description()

    def _preview_description(self) -> None:
        """预览当前模板生成的文案效果。"""
        template_name = self.description_template.get()
        source = self._description_source() or "https://example.com/video"
        try:
            preview = preview_template(template_name, source)
        except Exception as exc:
            messagebox.showerror("文案预览失败", str(exc), parent=self)
            return
        messagebox.showinfo(
            "文案预览",
            f"模板：{template_name}\n\n{preview}\n\n---\n当前文案已自动生成在简介框中。",
            parent=self,
        )

    def _append_extra_lines(self) -> None:
        """将额外行追加到文案中。"""
        extra_text = self.extra_lines_entry.get().strip()
        if not extra_text:
            messagebox.showinfo("提示", "请先在「额外鸣谢/备注」输入框中填写内容。", parent=self)
            return
        current = self._description_text()
        extra_lines = [line.strip() for line in re.split(r"[\r\n,，]+", extra_text) if line.strip()]
        for line in extra_lines:
            if line and line not in current:
                current = f"{current}\n{line}"
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert("1.0", current)
        self.extra_lines_entry.delete(0, tk.END)
        self.guide_status.set(f"已追加 {len(extra_lines)} 条额外内容到文案。")

    def _open_template_editor_new(self) -> None:
        """打开新建模板对话框。"""
        TemplateEditor(self, mode="new")

    def _open_template_editor_edit(self) -> None:
        """打开编辑模板对话框。"""
        template_name = self.description_template.get()
        if template_name in DESCRIPTION_TEMPLATES:
            messagebox.showinfo("提示", f"「{template_name}」是内置模板，不能编辑。\n请选择自定义模板或新建一个。", parent=self)
            return
        TemplateEditor(self, mode="edit", template_name=template_name)

    def _delete_template(self) -> None:
        """删除选中的自定义模板。"""
        template_name = self.description_template.get()
        if template_name in DESCRIPTION_TEMPLATES:
            messagebox.showinfo("提示", f"「{template_name}」是内置模板，不能删除。", parent=self)
            return
        if not template_name:
            messagebox.showinfo("提示", "请先选择一个自定义模板。", parent=self)
            return
        if messagebox.askyesno("确认删除", f"确定要删除自定义模板「{template_name}」吗？", parent=self):
            if delete_custom_template(template_name):
                self._refresh_template_list()
                self.description_template.set(get_template_names()[0] if get_template_names() else "授权本地化")
                self._generate_description()
                self.guide_status.set(f"已删除模板「{template_name}」。")
            else:
                messagebox.showerror("删除失败", f"无法删除模板「{template_name}」。", parent=self)

    def _refresh_template_list(self) -> None:
        """刷新模板下拉列表。"""
        names = get_template_names()
        self.template_box["values"] = names

    def _description_text(self) -> str:
        return self.description_text.get("1.0", tk.END).strip()

    def _description_source(self) -> str:
        if self.source_kind.get() == "url":
            return self.url.get().strip()
        return ""

    def _refresh_api_key(self) -> None:
        provider = self.translator.get()
        if provider == "deepseek":
            self.api_key.set(os.getenv("DEEPSEEK_API_KEY", self.api_key.get()))
        elif provider == "openai":
            self.api_key.set(os.getenv("OPENAI_API_KEY", self.api_key.get()))

    def _save_api_key(self) -> None:
        provider = self.translator.get()
        key = self.api_key.get().strip()
        if provider not in {"deepseek", "openai"}:
            messagebox.showinfo("无需保存", "none 翻译器不需要 API Key。")
            return
        if not key:
            messagebox.showerror("缺少 Key", "API Key 不能为空。")
            return
        env_key = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
        os.environ[env_key] = key
        updates = {env_key: key}
        if provider == "deepseek":
            updates.setdefault("DEEPSEEK_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
            updates.setdefault("DEEPSEEK_TRANSLATE_MODEL", os.getenv("DEEPSEEK_TRANSLATE_MODEL", "deepseek-v4-flash"))
        else:
            updates.setdefault("OPENAI_TRANSLATE_MODEL", os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini"))
        _upsert_env(Path(".env"), updates)
        messagebox.showinfo("已保存", f"{env_key} 已保存到 .env，后续命令行和 GUI 都能读取。")

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "当前任务还在运行。")
            return
        try:
            options = self._collect_options()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        if options.publish_to_bilibili and self._bilibili_page_worker_alive():
            messagebox.showinfo("B 站检查页已打开", "当前打开的是登录/检查窗口，不会自动上传视频。请先关闭它，再运行自动发布辅助。")
            return
        if options.publish_to_bilibili and not self._ensure_bilibili_profile_available_for_publish():
            return
        # 重置中断标志
        reset_cancellation()
        self._bili_pre_submit_notice_shown = False
        self.log_text.delete("1.0", tk.END)
        self.result_dir = None
        self._start_task_ui("全流程运行中")
        self.run_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.worker = threading.Thread(target=self._run_worker, args=(options,), daemon=True)
        self.worker.start()

    def _cancel(self) -> None:
        """中断当前运行的任务。"""
        if not self.worker or not self.worker.is_alive():
            return
        if messagebox.askyesno("确认中断", "确定要中断当前任务吗？\n已完成的步骤不会丢失。"):
            request_cancellation()
            self._log("\n正在请求中断任务...（等待当前步骤完成后停止）")
            self.cancel_button.configure(state=tk.DISABLED)

    def _collect_options(self) -> PipelineOptions:
        source_kind = self.source_kind.get()
        source = self.url.get().strip() if source_kind == "url" else self.file_path.get().strip()
        if not source:
            raise RuntimeError("请填写 YouTube URL 或选择本地视频。")
        if not self.i_have_rights.get():
            raise RuntimeError("请先确认你拥有处理/转载该视频的授权或许可证。")
        translator = self.translator.get()
        target_lang = self.target_lang.get().strip() or "zh-Hans"
        if target_lang.lower().startswith("zh") and translator == "none":
            raise RuntimeError("目标是中文字幕时不能使用 none 翻译器。请选择 deepseek 或 openai。")
        api_key = self.api_key.get().strip()
        if translator == "deepseek":
            if not api_key and not os.getenv("DEEPSEEK_API_KEY"):
                raise RuntimeError("请填写 DEEPSEEK_API_KEY，或在 .env 中配置。")
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
        elif translator == "openai":
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("请填写 OPENAI_API_KEY，或在 .env 中配置。")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        max_seconds = None
        if source_kind == "url" and not self.full_video.get():
            max_seconds = self._parse_duration_seconds(
                self.duration_value.get(),
                self.duration_unit.get(),
                "URL 读取长度",
            )
        font_size = self._parse_required_int(self.font_size.get(), "字幕字号")
        description = self._description_text()
        if self.publish_to_bilibili.get():
            source_for_description = source if source_kind == "url" else ""
            if not description:
                description = build_bilibili_description(
                    self.description_template.get(),
                    source_for_description,
                    include_source_link=self.include_source_link.get(),
                )
            else:
                description = ensure_source_link(
                    description,
                    source_for_description,
                    include_source_link=self.include_source_link.get(),
                )
        return PipelineOptions(
            source=source,
            source_kind=source_kind,
            output_dir=Path(self.output_dir.get().strip() or "outputs"),
            title=self.title_text.get().strip() or None,
            description=description,
            tags=[item.strip() for item in self.tags.get().split(",") if item.strip()],
            i_have_rights=True,
            require_reuse_allowed=self.require_reuse_allowed.get() and source_kind == "url",
            cookies_from_browser=self.cookies_from_browser.get().strip() or None,
            max_seconds=max_seconds,
            subtitle_source=self._subtitle_source_value(),
            whisper_model_size=self.whisper_model.get(),
            source_language=self._source_language_value(),
            device=self.device.get(),
            compute_type=self.compute_type.get(),
            translator=translator,
            target_lang=target_lang,
            translate_model=self.translate_model.get().strip() or None,
            smart_translation=self.smart_translation_layout.get(),
            smart_subtitle_layout=self.smart_translation_layout.get(),
            font_size=font_size,
            font_name=self.font_name.get().strip() or "Microsoft YaHei",
            subtitle_display_mode=self._subtitle_display_mode(),
            subtitle_color=self._ass_color(self.subtitle_color.get()),
            subtitle_outline_color=self._ass_color(self.subtitle_outline_color.get()),
            subtitle_outline=self._subtitle_outline(),
            subtitle_shadow=self._subtitle_shadow(),
            publish_to_bilibili=self.publish_to_bilibili.get(),
            include_source_link_in_description=self.include_source_link.get(),
            bilibili_browser=self._bilibili_browser_value(),
            bilibili_profile_dir=self._bilibili_profile_dir(),
            bilibili_wait_for_review=not self.close_bilibili_browser_after_fill.get(),
        )

    def _run_worker(self, options: PipelineOptions) -> None:
        try:
            result = run_pipeline(options, log=self._log)
            self.result_dir = result.work_dir
            self._log("")
            self._log("任务完成")
            self._log(f"Rendered video: {result.rendered_video}")
            if options.publish_to_bilibili:
                self._log("")
                self._log("B 站发布辅助已到提交前：视频和信息已尽量自动填好，但还没有发布。")
                self._log("请在打开的 B 站页面检查分区、创作声明、封面和平台提示，确认无误后手动点击“立即投稿”。")
                self.log_queue.put("__SHOW_BILI_PRE_SUBMIT_NOTICE__")
        except CancellationError:
            self._log("")
            self._log("任务已被用户中断。")
        except Exception as exc:
            self._log("")
            self._log("任务失败")
            self._log(str(exc))
        finally:
            self.log_queue.put("__ENABLE_RUN_BUTTON__")

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _drain_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if message == "__ENABLE_RUN_BUTTON__":
                self.run_button.configure(state=tk.NORMAL)
                self.cancel_button.configure(state=tk.DISABLED)
                self._stop_elapsed_timer()
                continue
            if message == "__SHOW_BILI_PRE_SUBMIT_NOTICE__":
                self._show_bili_pre_submit_notice()
                continue
            self._apply_log_state(message)
            self.log_text.insert(tk.END, message + "\n", self._log_tag(message))
            self.log_text.see(tk.END)
        self.after(100, self._drain_logs)

    def _start_task_ui(self, state: str) -> None:
        self._run_started_at = time.monotonic()
        self.progress_value.set(2)
        self.run_state.set(state)
        self.current_stage.set("启动中")
        self.elapsed_text.set("用时 00:00")
        self._tick_elapsed_timer()

    def _tick_elapsed_timer(self) -> None:
        if self._run_started_at is None:
            return
        elapsed = max(0, int(time.monotonic() - self._run_started_at))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            text = f"用时 {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            text = f"用时 {minutes:02d}:{seconds:02d}"
        self.elapsed_text.set(text)
        self._elapsed_timer_id = self.after(1000, self._tick_elapsed_timer)

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_timer_id:
            try:
                self.after_cancel(self._elapsed_timer_id)
            except tk.TclError:
                pass
        self._elapsed_timer_id = None
        self._run_started_at = None

    def _apply_log_state(self, message: str) -> None:
        for marker, (progress, stage) in STAGE_PROGRESS.items():
            if message.startswith(marker):
                self.progress_value.set(max(self.progress_value.get(), progress))
                self.current_stage.set(stage)
                self.run_state.set("运行中")
                return
        if "Opening Bilibili Creator Center" in message:
            self.progress_value.set(max(self.progress_value.get(), 94))
            self.current_stage.set("B 站填表")
            self.run_state.set("等待确认")
        elif "Bilibili upload assistance reached the pre-submit step." in message:
            self.progress_value.set(max(self.progress_value.get(), 98))
            self.current_stage.set("等待人工投稿")
            self.run_state.set("等待确认")
            self._show_bili_pre_submit_notice()
        elif message == "直接上传成品视频到 B 站":
            self.progress_value.set(20)
            self.current_stage.set("B 站上传")
            self.run_state.set("运行中")
        elif message in {"任务完成", "成品视频发布辅助已结束。"}:
            self.progress_value.set(100)
            self.current_stage.set("完成")
            self.run_state.set("已完成")
            self._refresh_output_summary()
        elif message in {"任务失败", "成品视频发布辅助失败"}:
            self.current_stage.set("查看日志")
            self.run_state.set("失败")
        elif message == "任务已被用户中断。":
            self.current_stage.set("已中断")
            self.run_state.set("已中断")
        elif "B 站发布辅助已到提交前" in message:
            self.progress_value.set(max(self.progress_value.get(), 98))
            self.current_stage.set("等待人工投稿")
            self.run_state.set("等待确认")

    @staticmethod
    def _log_tag(message: str) -> str:
        if not message:
            return ""
        if message.startswith(tuple(STAGE_PROGRESS.keys())):
            return "stage"
        if message in {"任务完成", "成品视频发布辅助已结束。"} or "已到提交前" in message:
            return "success"
        if message in {"任务失败", "成品视频发布辅助失败"} or "failed" in message.lower() or "失败" in message:
            return "error"
        if "等待" in message or "warning" in message.lower() or "请" in message:
            return "warning"
        return ""

    def _show_bili_pre_submit_notice(self) -> None:
        self.guide_status.set("B 站页面已到提交前：请检查页面并手动投稿；关闭 B 站浏览器后程序会结束等待。")
        if self._bili_pre_submit_notice_shown:
            return
        self._bili_pre_submit_notice_shown = True
        messagebox.showinfo(
            "B 站发布辅助已到提交前",
            "程序已把 B 站投稿页准备到提交前。\n\n"
            "现在还没有真正发布。\n"
            "请检查分区、创作声明、封面、简介、标签和平台提示；确认无误后，再手动点击“立即投稿”。\n\n"
            "如果页面已经处理完，你需要关闭这个 B 站浏览器窗口，GUI 才会结束等待。",
            parent=self,
        )

    def _open_result_dir(self) -> None:
        path = self.result_dir or Path(self.output_dir.get().strip() or "outputs")
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path.resolve())

    def _open_storage_manager(self) -> None:
        StorageManager(self, Path(self.output_dir.get().strip() or "outputs"))

    def _choose_direct_publish_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择已经压制好的成品视频",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.webm"), ("All files", "*.*")],
        )
        if not path:
            return
        video_path = Path(path)
        self.direct_publish_video.set(str(video_path))
        self._load_publish_metadata_for_video(video_path)
        if not self.title_text.get().strip():
            self.title_text.set(video_path.stem)
        self.guide_status.set("已选择成品视频；点击“直接上传到 B 站”即可跳过下载、转写、翻译和渲染。")

    def _load_publish_metadata_for_video(self, video_path: Path) -> None:
        metadata_path = video_path.parent / "publish_metadata.json"
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"读取智能标题/标签失败：{exc}")
            return
        title = str(metadata.get("title") or "").strip()
        tags = [str(tag).strip() for tag in metadata.get("tags") or [] if str(tag).strip()]
        if title and not self.title_text.get().strip():
            self.title_text.set(title)
        if tags and not self.tags.get().strip():
            self.tags.set(", ".join(tags))
        self._log(f"已从 {metadata_path} 读取智能标题/标签。")

    def _start_direct_publish(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "当前任务还在运行。")
            return
        if self._bilibili_page_worker_alive():
            messagebox.showinfo("B 站检查页已打开", "当前打开的是登录/检查窗口，不会自动上传视频。请先关闭它，再点击“直接上传到 B 站”。")
            return
        if not self._ensure_bilibili_profile_available_for_publish():
            return
        try:
            video_path = Path(self.direct_publish_video.get().strip())
            if not video_path.exists() or not video_path.is_file():
                raise RuntimeError("请先选择一个已经生成好的成品视频，例如 rendered.mp4。")
            if not self.i_have_rights.get():
                raise RuntimeError("请先确认你拥有处理/转载该视频的授权或许可证。")
            title = self.title_text.get().strip() or video_path.stem
            description = self._direct_publish_description()
            tags = [item.strip() for item in self.tags.get().split(",") if item.strip()]
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        self.run_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.DISABLED)
        self.result_dir = video_path.parent
        self._bili_pre_submit_notice_shown = False
        self.log_text.delete("1.0", tk.END)
        self._start_task_ui("B 站上传辅助")
        self.progress_value.set(20)
        self.current_stage.set("准备成品视频")
        self._log("直接上传成品视频到 B 站")
        self._log(f"Video: {video_path}")
        cover_path = self._default_cover_for_video(video_path)
        if cover_path:
            self._log(f"Cover: {cover_path}")
        self.worker = threading.Thread(
            target=self._direct_publish_worker,
            args=(video_path, title, description, tags, cover_path),
            daemon=True,
        )
        self.worker.start()

    def _direct_publish_description(self) -> str:
        description = self._description_text()
        source = self.url.get().strip() if self.url.get().strip().startswith(("http://", "https://")) else ""
        if not description:
            return build_bilibili_description(
                self.description_template.get(),
                source,
                include_source_link=self.include_source_link.get(),
            )
        return ensure_source_link(description, source, include_source_link=self.include_source_link.get())

    def _direct_publish_worker(self, video_path: Path, title: str, description: str, tags: list[str], cover_path: Path | None) -> None:
        try:
            assist_publish(
                video_path,
                title=title,
                description=description,
                tags=tags,
                cover_path=cover_path,
                profile_dir=self._bilibili_profile_dir(),
                browser=self._bilibili_browser_value(),
                screenshot_path=video_path.parent / "bilibili-upload-page.png",
                wait_for_review=not self.close_bilibili_browser_after_fill.get(),
                log=self._log,
            )
            self._log("")
            self._log("成品视频发布辅助已结束。")
        except Exception as exc:
            self._log("")
            self._log("成品视频发布辅助失败")
            self._log(str(exc))
        finally:
            self.log_queue.put("__ENABLE_RUN_BUTTON__")

    def _default_cover_for_video(self, video_path: Path) -> Path | None:
        for name in ["cover.jpg", "cover.png", "thumbnail.jpg", "thumbnail.png"]:
            candidate = video_path.parent / name
            if candidate.exists():
                return candidate
        return None

    def _open_bilibili_upload_page(self) -> None:
        if self._bilibili_page_worker_alive():
            messagebox.showinfo("B 站检查页已打开", "当前 B 站登录/检查窗口还在打开状态。它不会自动上传视频；检查完成后请先关闭该窗口。")
            return
        self.guide_status.set(f"正在打开 {self.bilibili_browser.get()} 的 B 站登录/检查页；这个窗口只用于登录确认，不会自动上传视频。")
        self._log("打开 B 站登录/检查页：这个窗口只用于登录确认，不会自动选择视频或上传。")
        self._log("确认能进入投稿页后，请关闭这个窗口，再点击“直接上传到 B 站”或运行全流程发布。")
        self.open_upload_button.configure(state=tk.DISABLED)
        self.bilibili_page_worker = threading.Thread(target=self._open_bilibili_upload_worker, daemon=True)
        self.bilibili_page_worker.start()

    def _open_bilibili_upload_worker(self) -> None:
        try:
            open_upload_page(
                profile_dir=self._bilibili_profile_dir(),
                browser=self._bilibili_browser_value(),
                log=self._log,
            )
        except Exception as exc:
            self._log(f"Bilibili upload page failed: {exc}")
            self.after(0, lambda: self.guide_status.set("打开投稿页失败，请查看运行日志。"))
        else:
            self.after(0, lambda: self.guide_status.set("B 站登录/检查窗口已关闭。现在可以点击“直接上传到 B 站”或运行全流程发布。"))
        finally:
            self.after(0, self._mark_bilibili_page_worker_finished)

    def _bilibili_page_worker_alive(self) -> bool:
        return self.bilibili_page_worker is not None and self.bilibili_page_worker.is_alive()

    def _mark_bilibili_page_worker_finished(self) -> None:
        self.bilibili_page_worker = None
        try:
            self.open_upload_button.configure(state=tk.NORMAL)
        except Exception:
            pass

    def _ensure_bilibili_profile_available_for_publish(self) -> bool:
        process_ids = self._bilibili_profile_process_ids()
        if not process_ids:
            return True
        browser_name = self.bilibili_browser.get()
        profile_dir = self._bilibili_profile_dir()
        pid_text = ", ".join(process_ids[:8])
        message = (
            f"{browser_name} 的 B 站自动化浏览器还在打开，占用了配置目录：\n{profile_dir}\n\n"
            f"请先关闭这个 B 站浏览器窗口，再重新点击上传。\n"
            f"检测到的进程 PID：{pid_text}"
        )
        self._log("B 站浏览器配置仍被占用，已停止启动新的上传任务。")
        self._log(f"Profile: {profile_dir}")
        self._log(f"PID: {pid_text}")
        messagebox.showinfo("请先关闭 B 站浏览器", message)
        return False

    def _close_bilibili_background_browser(self) -> None:
        process_ids = [pid for pid in self._bilibili_background_process_ids() if pid.isdigit()]
        browser_name = self.bilibili_browser.get()
        profile_dir = self._bilibili_profile_dir()
        if not process_ids:
            self.guide_status.set("没有发现 B 站后台浏览器或旧上传进程。")
            self._log("没有发现 B 站后台浏览器或旧上传进程。")
            return
        message = (
            f"将关闭 {browser_name} 的 B 站自动化浏览器，以及可能残留的旧上传进程。\n\n"
            f"当前配置目录：{profile_dir}\n"
            f"PID：{', '.join(process_ids[:12])}\n\n"
            "如果视频还在上传，关闭后上传会中断。确认关闭吗？"
        )
        if not messagebox.askyesno("关闭后台 B 站浏览器", message):
            return
        command = "Stop-Process -Id " + ",".join(process_ids) + " -Force"
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self.guide_status.set("关闭后台浏览器失败，请查看日志。")
            self._log(f"关闭 B 站后台浏览器失败：{exc}")
            return
        if completed.returncode == 0:
            self.bilibili_page_worker = None
            self._mark_bilibili_page_worker_finished()
            self.guide_status.set(f"已关闭 {len(process_ids)} 个 B 站后台浏览器进程。")
            self._log(f"已关闭 B 站后台浏览器 PID：{', '.join(process_ids)}")
        else:
            self.guide_status.set("关闭后台浏览器失败，请查看日志。")
            self._log(f"关闭 B 站后台浏览器失败：{completed.stderr.strip() or completed.stdout.strip()}")

    def _bilibili_profile_process_ids(self) -> list[str]:
        profile_dir = str((Path.cwd() / self._bilibili_profile_dir()).resolve())
        return self._process_ids_for_patterns([profile_dir])

    def _bilibili_background_process_ids(self) -> list[str]:
        profile_dirs = [
            str((Path.cwd() / Path("data/bilibili-profile")).resolve()),
            str((Path.cwd() / Path("data/bilibili-edge-profile")).resolve()),
        ]
        legacy_runners = [
            "edge_direct_publish_runner.py",
            "live_refill.py",
            "open_upload_page",
        ]
        return self._process_ids_for_patterns(profile_dirs + legacy_runners)

    def _process_ids_for_patterns(self, patterns: list[str]) -> list[str]:
        conditions = []
        for pattern in patterns:
            escaped = pattern.replace("'", "''")
            conditions.append(f"$_.CommandLine -like '*{escaped}*'")
        if not conditions:
            return []
        where_clause = " -or ".join(conditions)
        command = (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.ProcessId -ne $PID -and $_.Name -notin @('powershell.exe','pwsh.exe') -and $_.CommandLine -and ({where_clause}) }} | "
            "Select-Object -ExpandProperty ProcessId"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return []
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    def _bilibili_browser_value(self) -> str:
        if self.bilibili_browser.get() == "Microsoft Edge":
            return "msedge"
        return "chromium"

    def _bilibili_profile_dir(self) -> Path:
        if self._bilibili_browser_value() == "msedge":
            return Path("data/bilibili-edge-profile")
        return Path("data/bilibili-profile")

    def _refresh_all(self) -> None:
        """一键刷新所有状态：输出占用、模板列表、文案、时长信息、API Key。"""
        self.guide_status.set("正在刷新所有状态...")
        
        # 1. 刷新输出占用
        self._refresh_output_summary()
        
        # 2. 刷新模板列表
        self._refresh_template_list()
        
        # 3. 刷新文案（如果当前文案为空或包含占位符）
        self._refresh_generated_description_if_needed()
        
        # 4. 刷新 API Key（从环境变量读取最新值）
        self._refresh_api_key()
        
        # 5. 如果当前是 URL 模式，重新读取时长
        if self.source_kind.get() == "url" and self.url.get().strip():
            self._start_metadata_load(silent=True)
        
        self.guide_status.set("状态已刷新。")

    def _refresh_output_summary(self) -> None:
        output_root = Path(self.output_dir.get().strip() or "outputs")
        tasks = scan_outputs(output_root)
        total = sum(task.size for task in tasks)
        rendered = sum(file.size for task in tasks for file in task.files if file.kind == "成品视频")
        self.output_summary.set(
            f"当前输出：{len(tasks)} 个任务 | 总占用 {format_bytes(total)} | 成品视频 {format_bytes(rendered)}"
        )

    @staticmethod
    def _parse_duration_seconds(value: str, unit: str, label: str) -> int:
        value = value.strip()
        if not value:
            raise RuntimeError(f"{label} 不能为空；如需全片请选择完整视频。")
        parsed = float(value)
        multipliers = {"秒": 1, "分钟": 60, "小时": 3600}
        if unit not in multipliers:
            raise RuntimeError(f"{label} 单位无效。")
        seconds = round(parsed * multipliers[unit])
        if seconds <= 0:
            raise RuntimeError(f"{label} 必须大于 0。")
        return seconds

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "未知"
        total = max(0, round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}小时{minutes}分钟{secs}秒"
        if minutes:
            return f"{minutes}分钟{secs}秒"
        return f"{secs}秒"

    def _source_language_value(self) -> str | None:
        value = self.source_language.get().strip()
        return None if value in {"", "自动检测"} else value

    def _subtitle_source_value(self) -> str:
        if self.subtitle_source.get() == "自动检测":
            return "auto"
        if self.subtitle_source.get() == "音频 + 画面文字合并":
            return "merged"
        if self.subtitle_source.get() == "画面英文字幕 OCR":
            return "ocr"
        return "audio"

    def _subtitle_display_mode(self) -> str:
        mapping = {
            "中文单语": "translated",
            "原文在上 + 中文在下": "bilingual-source-first",
            "中文在上 + 原文在下": "bilingual-translation-first",
        }
        return mapping.get(self.subtitle_mode.get(), "translated")

    def _subtitle_outline(self) -> int:
        return 0 if self.subtitle_effect.get() in {"阴影", "无"} else 1

    def _subtitle_shadow(self) -> int:
        return 1 if self.subtitle_effect.get() in {"阴影", "描边+阴影"} else 0

    @staticmethod
    def _ass_color(name: str) -> str:
        colors = {
            "白色": "&H00FFFFFF",
            "黄色": "&H0000FFFF",
            "青色": "&H00FFFF00",
            "绿色": "&H0000FF00",
            "黑色": "&H00000000",
            "灰色": "&H00808080",
            "蓝色": "&H00FF0000",
        }
        return colors.get(name, "&H00FFFFFF")

    @staticmethod
    def _parse_required_int(value: str, label: str) -> int:
        parsed = int(value.strip())
        if parsed <= 0:
            raise RuntimeError(f"{label} 必须大于 0。")
        return parsed


def main() -> None:
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    app = LocalizerApp()
    app.mainloop()


def _upsert_env(path: Path, updates: dict[str, str]) -> None:
    existing: list[str] = []
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


class StorageManager(tk.Toplevel):
    def __init__(self, parent: tk.Tk, output_root: Path) -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.output_root = output_root.resolve()
        self.title("输出文件管理")
        self.geometry("1120x720")
        self.minsize(980, 620)
        self.tasks = []
        self.task_by_id: dict[str, object] = {}
        self.file_by_id: dict[str, Path] = {}
        self.file_usage_by_id: dict[str, object] = {}
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(16, 14, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        title = ttk.Label(header, text="输出文件管理", font=("Microsoft YaHei UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="查看每次下载、翻译、压制产生的文件占用；按住 Ctrl 可以多选。").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="刷新", command=self.refresh).grid(row=0, column=1, rowspan=2, padx=(8, 0), sticky="e")
        ttk.Button(header, text="打开输出目录", command=self._open_output_root).grid(row=0, column=2, rowspan=2, padx=(8, 0), sticky="e")

        dashboard = ttk.Frame(self, padding=(16, 0, 16, 10))
        dashboard.grid(row=1, column=0, sticky="ew")
        for column in range(6):
            dashboard.columnconfigure(column, weight=1)
        self.summary_cards: dict[str, tk.StringVar] = {}
        for column, label in enumerate(["总占用", "下载/原视频", "成品视频", "字幕", "中间文件", "预览/其他"]):
            var = tk.StringVar(value=f"{label}\n扫描中...")
            self.summary_cards[label] = var
            card = ttk.Label(dashboard, textvariable=var, relief="groove", anchor="center", padding=10)
            card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))

        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))

        task_frame = ttk.Frame(panes)
        task_frame.columnconfigure(0, weight=1)
        task_frame.rowconfigure(1, weight=1)
        ttk.Label(task_frame, text="1. 选择任务", font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.task_tree = ttk.Treeview(task_frame, columns=("name", "size", "files", "modified"), show="headings", selectmode="extended")
        self.task_tree.heading("name", text="任务")
        self.task_tree.heading("size", text="大小")
        self.task_tree.heading("files", text="文件数")
        self.task_tree.heading("modified", text="最近修改")
        self.task_tree.column("name", width=300)
        self.task_tree.column("size", width=110, anchor="e")
        self.task_tree.column("files", width=80, anchor="center")
        self.task_tree.column("modified", width=130, anchor="center")
        self.task_tree.grid(row=1, column=0, sticky="nsew")
        task_scroll = ttk.Scrollbar(task_frame, orient="vertical", command=self.task_tree.yview)
        task_scroll.grid(row=1, column=1, sticky="ns")
        self.task_tree.configure(yscrollcommand=task_scroll.set)
        self.task_tree.bind("<<TreeviewSelect>>", lambda _event: self._show_selected_files())
        self.task_tree.bind("<Double-1>", lambda _event: self._open_selected_task())
        panes.add(task_frame, weight=1)

        file_frame = ttk.Frame(panes)
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(2, weight=1)
        ttk.Label(file_frame, text="2. 查看/选择文件", font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.selection_text = tk.StringVar(value="未选择任务")
        ttk.Label(file_frame, textvariable=self.selection_text).grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.file_tree = ttk.Treeview(file_frame, columns=("kind", "size", "name"), show="headings", selectmode="extended")
        self.file_tree.heading("kind", text="类型")
        self.file_tree.heading("size", text="大小")
        self.file_tree.heading("name", text="文件")
        self.file_tree.column("kind", width=100)
        self.file_tree.column("size", width=100, anchor="e")
        self.file_tree.column("name", width=360)
        self.file_tree.grid(row=2, column=0, sticky="nsew")
        file_scroll = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_tree.yview)
        file_scroll.grid(row=2, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=file_scroll.set)
        self.file_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_file_selection_text())
        self.file_tree.bind("<Double-1>", lambda _event: self._open_selected_file())
        panes.add(file_frame, weight=1)

        actions = ttk.Frame(self, padding=(16, 0, 16, 14))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="打开任务目录", command=self._open_selected_task).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(actions, text="打开选中文件", command=self._open_selected_file).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(actions, text="删除整个任务", command=self._delete_selected_tasks).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(actions, text="删除选中文件", command=self._delete_selected_files).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(actions, text="关闭", command=self.destroy).grid(row=0, column=5, padx=(8, 0))

    def refresh(self) -> None:
        self.tasks = scan_outputs(self.output_root)
        self.task_by_id.clear()
        self.file_by_id.clear()
        self.file_usage_by_id.clear()
        self.task_tree.delete(*self.task_tree.get_children())
        self.file_tree.delete(*self.file_tree.get_children())
        total = sum(task.size for task in self.tasks)
        totals = self._category_totals()
        self.summary_cards["总占用"].set(f"总占用\n{format_bytes(total)}\n{len(self.tasks)} 个任务")
        self.summary_cards["下载/原视频"].set(f"下载/原视频\n{format_bytes(totals.get('下载/原视频', 0))}")
        self.summary_cards["成品视频"].set(f"成品视频\n{format_bytes(totals.get('成品视频', 0))}")
        self.summary_cards["字幕"].set(f"字幕\n{format_bytes(totals.get('字幕', 0))}")
        self.summary_cards["中间文件"].set(f"中间文件\n{format_bytes(totals.get('中间文件', 0))}")
        preview_other = totals.get("预览图", 0) + totals.get("其他", 0)
        self.summary_cards["预览/其他"].set(f"预览/其他\n{format_bytes(preview_other)}")
        self.selection_text.set("未选择任务")
        for index, task in enumerate(self.tasks):
            item_id = f"task-{index}"
            self.task_by_id[item_id] = task
            self.task_tree.insert(
                "",
                "end",
                iid=item_id,
                values=(task.name, format_bytes(task.size), len(task.files), self._format_modified(task.modified_at)),
            )

    def _show_selected_files(self) -> None:
        self.file_tree.delete(*self.file_tree.get_children())
        self.file_by_id.clear()
        self.file_usage_by_id.clear()
        selected = self.task_tree.selection()
        counter = 0
        selected_size = 0
        for task_id in selected:
            task = self.task_by_id.get(task_id)
            if not task:
                continue
            selected_size += task.size
            for file_usage in task.files:
                file_id = f"file-{counter}"
                counter += 1
                self.file_by_id[file_id] = file_usage.path
                self.file_usage_by_id[file_id] = file_usage
                self.file_tree.insert(
                    "",
                    "end",
                    iid=file_id,
                    values=(file_usage.kind, format_bytes(file_usage.size), file_usage.path.name),
                )
        if selected:
            self.selection_text.set(f"已选 {len(selected)} 个任务，共 {format_bytes(selected_size)}；右侧可选择单个文件删除。")
        else:
            self.selection_text.set("未选择任务")

    def _delete_selected_tasks(self) -> None:
        paths = []
        total = 0
        for task_id in self.task_tree.selection():
            task = self.task_by_id.get(task_id)
            if task:
                paths.append(task.path)
                total += task.size
        if not paths:
            messagebox.showinfo("没有选择", "请先选择要删除的任务目录。", parent=self)
            return
        if not messagebox.askyesno("确认删除", f"将删除 {len(paths)} 个任务，释放约 {format_bytes(total)}。确定吗？", parent=self):
            return
        deleted = delete_paths(paths, self.output_root)
        messagebox.showinfo("已删除", f"已释放 {format_bytes(deleted)}。", parent=self)
        self.refresh()
        self._refresh_parent_summary()

    def _delete_selected_files(self) -> None:
        paths = [self.file_by_id[file_id] for file_id in self.file_tree.selection() if file_id in self.file_by_id]
        if not paths:
            messagebox.showinfo("没有选择", "请先选择要删除的文件。", parent=self)
            return
        total = sum(path.stat().st_size for path in paths if path.exists())
        if not messagebox.askyesno("确认删除", f"将删除 {len(paths)} 个文件，释放约 {format_bytes(total)}。确定吗？", parent=self):
            return
        deleted = delete_paths(paths, self.output_root)
        messagebox.showinfo("已删除", f"已释放 {format_bytes(deleted)}。", parent=self)
        self.refresh()
        self._refresh_parent_summary()

    def _open_output_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        os.startfile(self.output_root)

    def _open_selected_task(self) -> None:
        selected = self.task_tree.selection()
        if not selected:
            messagebox.showinfo("没有选择", "请先选择一个任务。", parent=self)
            return
        task = self.task_by_id.get(selected[0])
        if task and task.path.exists():
            os.startfile(task.path)

    def _open_selected_file(self) -> None:
        selected = self.file_tree.selection()
        if not selected:
            messagebox.showinfo("没有选择", "请先选择一个文件。", parent=self)
            return
        path = self.file_by_id.get(selected[0])
        if path and path.exists():
            os.startfile(path)

    def _update_file_selection_text(self) -> None:
        selected = self.file_tree.selection()
        if not selected:
            self._show_selected_files_summary_only()
            return
        total = sum(self.file_usage_by_id[file_id].size for file_id in selected if file_id in self.file_usage_by_id)
        self.selection_text.set(f"已选 {len(selected)} 个文件，删除后可释放约 {format_bytes(total)}。")

    def _show_selected_files_summary_only(self) -> None:
        selected_tasks = self.task_tree.selection()
        if not selected_tasks:
            self.selection_text.set("未选择任务")
            return
        total = sum(self.task_by_id[task_id].size for task_id in selected_tasks if task_id in self.task_by_id)
        self.selection_text.set(f"已选 {len(selected_tasks)} 个任务，共 {format_bytes(total)}；右侧可选择单个文件删除。")

    def _category_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for task in self.tasks:
            for file_usage in task.files:
                totals[file_usage.kind] = totals.get(file_usage.kind, 0) + file_usage.size
        return totals

    def _refresh_parent_summary(self) -> None:
        if hasattr(self.parent_app, "_refresh_output_summary"):
            self.parent_app._refresh_output_summary()

    @staticmethod
    def _format_modified(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")


class TemplateEditor(tk.Toplevel):
    """自定义模板编辑器弹窗。"""
    
    def __init__(self, parent: LocalizerApp, mode: str = "new", template_name: str = "") -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.mode = mode
        self.template_name = template_name
        
        if mode == "new":
            self.title("新建自定义模板")
        else:
            self.title(f"编辑模板：{template_name}")
        
        self.geometry("600x450")
        self.minsize(500, 380)
        self.transient(parent)
        self.grab_set()
        
        self._build_ui()
        
        if mode == "edit" and template_name:
            self._load_template(template_name)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        
        # 模板名称
        name_frame = ttk.Frame(self, padding=(16, 14, 16, 8))
        name_frame.grid(row=0, column=0, sticky="ew")
        name_frame.columnconfigure(1, weight=1)
        ttk.Label(name_frame, text="模板名称", font=("", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=40)
        name_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        
        # 说明
        info_frame = ttk.Frame(self, padding=(16, 0, 16, 8))
        info_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            info_frame,
            text="编写文案模板，使用 {source} 表示原视频链接，{custom_text} 表示自定义文本。",
            foreground="#666",
            font=("", 9),
        ).grid(row=0, column=0, sticky="w")
        
        # 模板内容编辑区
        editor_frame = ttk.Frame(self, padding=(16, 0, 16, 8))
        editor_frame.grid(row=2, column=0, sticky="nsew")
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(1, weight=1)
        ttk.Label(editor_frame, text="模板内容", font=("", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.template_text = tk.Text(editor_frame, height=10, wrap="word", font=("Consolas", 10))
        self.template_text.grid(row=1, column=0, sticky="nsew")
        
        # 快速插入变量按钮
        var_frame = ttk.Frame(self, padding=(16, 0, 16, 8))
        var_frame.grid(row=3, column=0, sticky="ew")
        ttk.Label(var_frame, text="插入变量：").grid(row=0, column=0, sticky="w")
        ttk.Button(var_frame, text="{source}", command=lambda: self._insert_var("{source}")).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(var_frame, text="{custom_text}", command=lambda: self._insert_var("{custom_text}")).grid(row=0, column=2, padx=(4, 0))
        
        # 预览区域
        preview_frame = ttk.LabelFrame(self, text="预览效果", padding=8)
        preview_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))
        preview_frame.columnconfigure(0, weight=1)
        self.preview_var = tk.StringVar(value="填写模板内容后自动预览")
        ttk.Label(preview_frame, textvariable=self.preview_var, wraplength=540, foreground="#555").grid(row=0, column=0, sticky="w")
        
        # 按钮
        btn_frame = ttk.Frame(self, padding=(16, 0, 16, 14))
        btn_frame.grid(row=5, column=0, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        ttk.Button(btn_frame, text="保存模板", command=self._save).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btn_frame, text="取消", command=self.destroy).grid(row=0, column=2)
        
        # 绑定输入事件以自动预览
        self.template_text.bind("<KeyRelease>", lambda _: self._update_preview())
        self.name_var.trace_add("write", lambda *_: self._update_preview())

    def _insert_var(self, var: str) -> None:
        """在光标位置插入变量。"""
        self.template_text.insert(tk.INSERT, var)
        self._update_preview()

    def _load_template(self, name: str) -> None:
        """加载已有模板到编辑器。"""
        templates = get_all_templates()
        if name in templates:
            self.name_var.set(name)
            self.template_text.delete("1.0", tk.END)
            self.template_text.insert("1.0", templates[name])
            self._update_preview()

    def _update_preview(self) -> None:
        """更新预览效果。"""
        content = self.template_text.get("1.0", tk.END).strip()
        name = self.name_var.get().strip()
        if not content:
            self.preview_var.set("请填写模板内容")
            return
        try:
            preview = content.format(source="https://example.com/video", custom_text="自定义内容")
            if name:
                self.preview_var.set(f"模板「{name}」预览：\n{preview}")
            else:
                self.preview_var.set(f"预览：\n{preview}")
        except KeyError as e:
            self.preview_var.set(f"模板中有未定义的变量：{e}，请使用 {{source}} 或 {{custom_text}}")
        except Exception as e:
            self.preview_var.set(f"模板格式错误：{e}")

    def _save(self) -> None:
        """保存模板。"""
        name = self.name_var.get().strip()
        content = self.template_text.get("1.0", tk.END).strip()
        
        if not name:
            messagebox.showerror("缺少名称", "请填写模板名称。", parent=self)
            return
        if not content:
            messagebox.showerror("缺少内容", "请填写模板内容。", parent=self)
            return
        
        try:
            content.format(source="test", custom_text="test")
        except KeyError as e:
            messagebox.showerror(
                "模板变量错误",
                f"模板中使用了未定义的变量 {e}。\n支持的变量：{{source}}、{{custom_text}}",
                parent=self,
            )
            return
        except Exception as e:
            messagebox.showerror("模板格式错误", f"模板内容有误：{e}\n请检查大括号是否正确配对。", parent=self)
            return
        
        # 保存
        try:
            save_custom_template(name, content)
            self.parent_app._refresh_template_list()
            self.parent_app.description_template.set(name)
            self.parent_app._generate_description()
            self.parent_app.guide_status.set(f"已{'新建' if self.mode == 'new' else '更新'}模板「{name}」。")
            self.destroy()
        except Exception as e:
            messagebox.showerror("保存失败", f"保存模板时出错：{e}", parent=self)


if __name__ == "__main__":
    main()

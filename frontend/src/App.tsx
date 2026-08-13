import { useEffect, useMemo, useRef, useState } from "react";

type View = "material" | "process" | "subtitle" | "publish" | "files";
type Material = {
  id: string;
  name: string;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  authorized: boolean;
  is_demo: boolean;
};
type Cue = {
  id: number;
  start: number;
  end: number;
  source: string;
  translated: string;
  deleted?: boolean;
  kind?: string;
};
type Job = {
  id: string;
  status: string;
  stage: string;
  progress: number;
  logs: string[];
  error: string | null;
  result: {
    output_dir: string;
    source_srt: string;
    translated_srt: string;
    rendered_video: string;
  } | null;
  material: Material;
  device: string;
  compute_type: string;
  options: Options;
  started_at: number | null;
  finished_at: number | null;
  elapsed_seconds: number;
};
type Options = {
  title: string;
  require_reuse_allowed: boolean;
  cookies_from_browser: string;
  cookies_file: string;
  max_seconds: number | null;
  subtitle_source: string;
  whisper_model_size: string;
  source_language: string;
  beam_size: number;
  ocr_interval: number;
  ocr_crop_ratio: number;
  ocr_min_chars: number;
  subtitle_margin_ratio: number;
  render_crf: number;
  translator: string;
  target_lang: string;
  translate_model: string;
  smart_translation: boolean;
  smart_subtitle_layout: boolean;
  font_name: string;
  font_size: number;
  subtitle_display_mode: string;
  subtitle_color: string;
  subtitle_outline_color: string;
  subtitle_effect: string;
  output_dir: string;
};
type Template = { name: string; body: string };
type OutputFile = {
  id: string;
  name: string;
  kind: string;
  size: number;
  size_label: string;
};
type OutputTask = {
  id: string;
  name: string;
  size: number;
  size_label: string;
  modified_at: number;
  files: OutputFile[];
};
type Outputs = {
  root: string;
  task_count: number;
  total_size: string;
  category_totals: Record<string, { bytes: number; label: string }>;
  tasks: OutputTask[];
};
type HistoryJob = {
  id: string;
  title: string;
  status: string;
  stage: string;
  progress: number;
  error: string | null;
  output_dir: string | null;
  rendered_video: string | null;
  output_exists: boolean;
  rendered_exists: boolean;
  created_at: number | null;
  finished_at: number | null;
};
type Diagnostic = {
  id: string;
  label: string;
  purpose: string;
  available: boolean;
  message: string;
};

const defaults: Options = {
  title: "",
  require_reuse_allowed: false,
  cookies_from_browser: "",
  cookies_file: "",
  max_seconds: 10,
  subtitle_source: "merged",
  whisper_model_size: "small",
  source_language: "",
  beam_size: 5,
  ocr_interval: 1,
  ocr_crop_ratio: 0.3,
  ocr_min_chars: 3,
  subtitle_margin_ratio: 0.055,
  render_crf: 20,
  translator: "deepseek",
  target_lang: "zh-Hans",
  translate_model: "",
  smart_translation: true,
  smart_subtitle_layout: true,
  font_name: "Microsoft YaHei",
  font_size: 24,
  subtitle_display_mode: "translated",
  subtitle_color: "白色",
  subtitle_outline_color: "黑色",
  subtitle_effect: "描边",
  output_dir: "outputs/workbench_demo",
};
const nav: Array<[View, string, string]> = [
  ["material", "素材", "链接、本地视频与授权"],
  ["process", "处理", "识别、翻译与推理"],
  ["subtitle", "字幕", "预览、编辑与渲染"],
  ["publish", "发布", "B 站投稿辅助"],
  ["files", "文件", "输出与空间管理"],
];
const timecode = (seconds: number) =>
  `00:${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

const parseTime = (value: string): number | null => {
  const parts = value.trim().split(":").map((item) => Number(item));
  if (!parts.length || parts.some((item) => Number.isNaN(item))) return null;
  let seconds = 0;
  if (parts.length === 3) seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
  else if (parts.length === 2) seconds = parts[0] * 60 + parts[1];
  else if (parts.length === 1) seconds = parts[0];
  return seconds >= 0 && Number.isFinite(seconds) ? seconds : null;
};
const formatTime = (seconds: number) => {
  const value = Math.max(0, seconds);
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
};

const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const area = document.createElement("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(area);
      return ok;
    } catch {
      return false;
    }
  }
};

function CopyButton({ text, label = "复制" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <button
      className="copy-button"
      type="button"
      title="点击复制到剪贴板；也可以直接用鼠标选中文本后右键复制"
      disabled={!text}
      onClick={() => {
        void copyToClipboard(text).then((ok) => {
          if (ok) {
            setCopied(true);
            setFailed(false);
            window.setTimeout(() => setCopied(false), 2000);
          } else {
            setFailed(true);
            window.setTimeout(() => setFailed(false), 3000);
          }
        });
      }}
    >
      {copied ? "已复制" : failed ? "失败，用鼠标选中后右键复制" : label}
    </button>
  );
}

const formatElapsed = (seconds: number) => {
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return hours
    ? `用时 ${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `用时 ${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
};

const secondsFromDuration = (value: string, unit: "秒" | "分钟" | "小时") => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return Math.round(parsed * (unit === "小时" ? 3600 : unit === "分钟" ? 60 : 1));
};
const formatBytes = (bytes: number) => {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return index ? `${value.toFixed(1)} ${units[index]}` : `${Math.round(value)} B`;
};

type NativeBridge = {
  choose_publish_video: () => Promise<{ token: string } | null>;
  choose_material_video: () => Promise<{ material_id: string; name: string } | null>;
  choose_output_dir: () => Promise<{ path: string } | null>;
  choose_cookies_file: () => Promise<{ path: string } | null>;
};
declare global {
  interface Window { pywebview?: { api?: NativeBridge } }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const form = options?.body instanceof FormData;
  const response = await fetch(path, {
    ...options,
    headers: form
      ? options?.headers
      : { "Content-Type": "application/json", ...(options?.headers ?? {}) },
  });
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "本地服务请求失败");
  return payload;
}

function App() {
  const fileInput = useRef<HTMLInputElement>(null);
  const publishFileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [view, setView] = useState<View>("material");
  const [materials, setMaterials] = useState<Material[]>([]);
  const [materialId, setMaterialId] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [authorized, setAuthorized] = useState(false);
  const [fullVideo, setFullVideo] = useState(false);
  const [durationValue, setDurationValue] = useState("10");
  const [durationUnit, setDurationUnit] = useState<"秒" | "分钟" | "小时">("秒");
  const [device, setDevice] = useState("cuda");
  const [computeType, setComputeType] = useState("float16");
  const [options, setOptions] = useState<Options>(defaults);
  const [job, setJob] = useState<Job | null>(null);
  const [cues, setCues] = useState<Cue[]>([]);
  const [history, setHistory] = useState<Cue[][]>([]);
  const [future, setFuture] = useState<Cue[][]>([]);
  const [activeCue, setActiveCue] = useState<number | null>(null);
  const [preview, setPreview] = useState<"source" | "rendered">("source");
  const [online, setOnline] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [message, setMessage] = useState("正在连接本地服务…");
  const [metadata, setMetadata] = useState<{
    duration: number | null;
    license: string | null;
    view_count: number | null;
  } | null>(null);
  const [metadataNotice, setMetadataNotice] = useState("");
  const [metadataState, setMetadataState] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateName, setTemplateName] = useState("授权本地化");
  const [templateBody, setTemplateBody] = useState("");
  const [templateEditorOpen, setTemplateEditorOpen] = useState(false);
  const [templateDraftName, setTemplateDraftName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [extraLine, setExtraLine] = useState("");
  const [publishBrowser, setPublishBrowser] = useState("chromium");
  const [closeAfterFill, setCloseAfterFill] = useState(false);
  const [publishInPipeline, setPublishInPipeline] = useState(false);
  const [includeSourceLink, setIncludeSourceLink] = useState(true);
  const [nativePublishToken, setNativePublishToken] = useState("");
  const [nativePublishName, setNativePublishName] = useState("");
  const [profileMessage, setProfileMessage] = useState("");
  const [showFullLogs, setShowFullLogs] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [readiness, setReadiness] = useState<string[]>([]);
  const [outputs, setOutputs] = useState<Outputs | null>(null);
  const [selectedOutputPaths, setSelectedOutputPaths] = useState<string[]>([]);
  const [selectedOutputTaskIds, setSelectedOutputTaskIds] = useState<string[]>([]);
  const [selectedPublishVideo, setSelectedPublishVideo] = useState("");
  const [mode, setMode] = useState<"basic" | "advanced">(
    () => (localStorage.getItem("ybl-mode") as "basic" | "advanced") || "basic",
  );
  const [apiKeyConfigured, setApiKeyConfigured] = useState<boolean | null>(null);
  const [cookiesFileValid, setCookiesFileValid] = useState<boolean | null>(null);
  const [publishSession, setPublishSession] = useState<{
    status: string;
    message: string;
    logs: string[];
    error: string | null;
    active: boolean;
  }>({ status: "idle", message: "", logs: [], error: null, active: false });
  const [viewedTaskId, setViewedTaskId] = useState<string | null>(null);
  const [jobHistory, setJobHistory] = useState<HistoryJob[]>([]);
  const [historyScope, setHistoryScope] = useState<"current" | "all">("current");
  const [historyTotal, setHistoryTotal] = useState(0);
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [translationReady, setTranslationReady] = useState(true);
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null);
  const [cutStart, setCutStart] = useState("");
  const [cutEnd, setCutEnd] = useState("");
  const [cutRanges, setCutRanges] = useState<Array<[number, number]>>([]);
  const [modOp, setModOp] = useState<"cut" | "keep" | "mute">("cut");
  const [exportStart, setExportStart] = useState("");
  const [exportEnd, setExportEnd] = useState("");
  const [reorderPoints, setReorderPoints] = useState("");
  const [reorderOrder, setReorderOrder] = useState("");
  const [offsetValue, setOffsetValue] = useState("");

  const selected = materials.find((item) => item.id === materialId);
  const running = ["queued", "running", "cancelling"].includes(
    job?.status ?? "",
  );
  const dirty = history.length > 0;
  const publishVideoReady = Boolean(
    nativePublishToken || selectedPublishVideo || job?.result?.rendered_video,
  );
  const savedTemplateBody =
    templates.find((item) => item.name === templateName)?.body ?? "";
  const templateDraftDirty = templateBody !== savedTemplateBody;
  const duration =
    job?.material.duration_seconds ??
    selected?.duration_seconds ??
    metadata?.duration ??
    10;
  const previewSrc = job
    ? `/api/jobs/${job.id}/media/${preview === "rendered" && job.result ? "rendered" : "source"}`
    : selected
      ? `/api/materials/${selected.id}/media`
      : "";
  const cuePositions = useMemo(
    () =>
      cues.map((cue) => ({
        cue,
        left: `${(cue.start / duration) * 100}%`,
        width: `${Math.max(3, ((cue.end - cue.start) / duration) * 100)}%`,
      })),
    [cues, duration],
  );
  const selectedOutputBytes = useMemo(() => {
    if (!outputs) return 0;
    const taskIds = new Set(selectedOutputPaths.filter((path) => outputs.tasks.some((task) => task.id === path)));
    return outputs.tasks.reduce((total, task) => total + (taskIds.has(task.id) ? task.size : task.files.filter((file) => selectedOutputPaths.includes(file.id)).reduce((sum, file) => sum + file.size, 0)), 0);
  }, [outputs, selectedOutputPaths]);

  const updateOptions = <K extends keyof Options>(key: K, value: Options[K]) =>
    setOptions((current) => ({ ...current, [key]: value }));
  const loadMaterials = async () =>
    setMaterials(
      (await api<{ materials: Material[] }>("/api/materials")).materials,
    );
  const loadTemplates = async () => {
    const templates = (await api<{ templates: Template[] }>("/api/templates"))
      .templates;
    setTemplates(templates);
    setTemplateBody(
      (current) =>
        current || templates.find((item) => item.name === templateName)?.body || "",
    );
  };
  const loadSettings = async () => {
    const settings = await api<{
      cookies_from_browser?: string;
      cookies_file?: string;
      cookies_file_valid?: boolean | null;
      deepseek_key_configured?: boolean;
      openai_key_configured?: boolean;
      default_output_dir?: string;
    }>("/api/settings");
    const browser = settings.cookies_from_browser ?? "";
    if (["", "chrome", "edge", "firefox", "brave", "chromium"].includes(browser)) {
      updateOptions("cookies_from_browser", browser);
    }
    if (settings.cookies_file) {
      updateOptions("cookies_file", settings.cookies_file);
    }
    if (settings.default_output_dir) {
      setOptions((current) => current.output_dir === defaults.output_dir
        ? { ...current, output_dir: settings.default_output_dir! }
        : current);
    }
    setCookiesFileValid(settings.cookies_file_valid ?? null);
    setApiKeyConfigured(Boolean(settings.deepseek_key_configured));
  };
  const loadJobHistory = async () => {
    try {
      const query = new URLSearchParams({
        limit: "50",
        scope: historyScope,
        output_dir: options.output_dir,
      });
      const response = await api<{ jobs: HistoryJob[]; total: number }>(`/api/history/jobs?${query}`);
      setJobHistory(response.jobs);
      setHistoryTotal(response.total);
    } catch {
      /* history is best-effort */
    }
  };
  const loadDiagnostics = async () => {
    try {
      const response = await api<{ checks: Diagnostic[] }>("/api/diagnostics");
      setDiagnostics(response.checks);
    } catch {
      /* A local service error is already reported by the connection state. */
    }
  };
  const clearHistory = async () => {
    if (!historyTotal) return;
    const scopeLabel = historyScope === "current" ? "当前输出目录" : "全部输出目录";
    if (!window.confirm(`清除“${scopeLabel}”中的 ${historyTotal} 条任务记录吗？\n\n只删除历史列表记录，不会删除任何视频、字幕或输出文件。`)) return;
    try {
      const response = await api<{ message: string }>("/api/history/clear", {
        method: "POST",
        body: JSON.stringify({ confirmed: true, scope: historyScope, output_dir: options.output_dir }),
      });
      setMessage(response.message);
      await loadJobHistory();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "清除任务历史失败");
    }
  };
  const openHistoryPath = async (path: string) => {
    try {
      await api("/api/history/open", { method: "POST", body: JSON.stringify({ path }) });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法打开该任务输出");
    }
  };
  const loadOutputs = async () => {
    const snapshot = await api<Outputs>("/api/outputs/scan", {
      method: "POST",
      body: JSON.stringify({ output_dir: options.output_dir }),
    });
    setOutputs(snapshot);
  };
  const loadCues = async (id: string) => {
    try {
      const data = await api<{ cues: Cue[]; translation_ready?: boolean }>(
        `/api/jobs/${id}/cues`,
      );
      setCues(data.cues);
      setTranslationReady(data.translation_ready ?? true);
      setHistory([]);
      setFuture([]);
    } catch {
      /* Cues are not ready during download/transcription. */
    }
  };
  const taskOptions = () => {
    const maxSeconds = sourceUrl.trim() && !fullVideo
      ? secondsFromDuration(durationValue, durationUnit)
      : null;
    return {
      ...options,
      max_seconds: maxSeconds,
      description,
      tags: tags.split(",").map((value) => value.trim()).filter(Boolean),
      publish_to_bilibili: publishInPipeline,
      include_source_link: includeSourceLink,
      bilibili_browser: publishBrowser,
      close_after_fill: closeAfterFill,
    };
  };
  const loadPublishStatus = async () => {
    try {
      const status = await api<{ message: string }>(`/api/publish/status?browser=${publishBrowser}`);
      setProfileMessage(status.message);
    } catch (error) {
      setProfileMessage(error instanceof Error ? error.message : "无法读取 B 站浏览器状态");
    }
  };
  const loadPublishSession = async () => {
    try {
      const session = await api<{
        status: string;
        message: string;
        logs: string[];
        error: string | null;
        active: boolean;
      }>("/api/publish/assist/status");
      setPublishSession(session);
    } catch {
      /* The service is only reachable when the workbench EXE is running. */
    }
  };
  const cookieHintMatches = (text: string | null | undefined) =>
    !!text && /Cookies|机器人|登录验证|拦截/i.test(text);
  const checkReadiness = async () => {
    try {
      const response = await api<{ ready: boolean; issues: string[]; message: string }>("/api/readiness", {
        method: "POST",
        body: JSON.stringify({ source_url: sourceUrl, material_id: selected?.id, authorized, device, compute_type: computeType, options: taskOptions() }),
      });
      setReadiness(response.issues);
      setMessage(response.message);
      return response.ready;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "检查配置失败");
      return false;
    }
  };
  useEffect(() => {
    const onContextMenu = (event: MouseEvent) => {
      const selection = window.getSelection()?.toString().trim();
      if (selection) {
        event.preventDefault();
        setCtxMenu({ x: event.clientX, y: event.clientY });
      }
    };
    const hideMenu = () => setCtxMenu(null);
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("mousedown", hideMenu);
    window.addEventListener("blur", hideMenu);
    return () => {
      window.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mousedown", hideMenu);
      window.removeEventListener("blur", hideMenu);
    };
  }, []);
  const copySelection = () => {
    const text = window.getSelection()?.toString() ?? "";
    void copyToClipboard(text);
    setCtxMenu(null);
  };
  const selectAllText = () => {
    window.getSelection()?.selectAllChildren(document.body);
    setCtxMenu(null);
  };
  const toggleMode = () => {
    const next = mode === "basic" ? "advanced" : "basic";
    setMode(next);
    localStorage.setItem("ybl-mode", next);
    setMessage(next === "advanced" ? "已切换到高级模式，显示全部参数。" : "已切换到基础模式，仅显示常用参数。");
  };
  const applyRecommended = () => {
    setDevice("cpu");
    setComputeType("int8");
    setOptions((current) => ({ ...current, subtitle_source: "merged", whisper_model_size: "small", source_language: "", translator: "deepseek", target_lang: "zh-Hans", smart_translation: true, smart_subtitle_layout: true, font_name: "Microsoft YaHei", font_size: 24, subtitle_display_mode: "translated", subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "描边" }));
    setIncludeSourceLink(true);
    setMessage("已应用推荐设置：CPU 稳定模式、音频+OCR 合并与中文硬字幕。");
  };

  useEffect(() => {
    Promise.all([
      api<{ ok: boolean }>("/api/health"),
      loadMaterials(),
      loadTemplates(),
      loadSettings(),
      loadDiagnostics(),
    ])
      .then(() => {
        setOnline(true);
        setMessage("本地服务已就绪。选择素材后即可开始处理。");
      })
      .catch(() => setMessage("未连接本地服务，请启动工作台 EXE。"));
  }, []);
  useEffect(() => {
    if (running) setInspectorOpen(true);
    if (job?.status === "completed") {
      setInspectorOpen(false);
      setPreview("rendered");
      loadCues(job.id);
      loadOutputs().catch(() => undefined);
    }
  }, [running, job?.status]);
  useEffect(() => {
    if (!job || !running) return;
    const timer = window.setInterval(async () => {
      try {
        const latest = await api<Job>(`/api/jobs/${job.id}`);
        setJob(latest);
        if (latest.progress >= 48) loadCues(latest.id);
        if (latest.status === "failed")
          setMessage(`处理失败：${latest.error ?? "请查看检查器日志"}`);
      } catch (error) {
        setMessage(
          `读取任务失败：${error instanceof Error ? error.message : "服务不可用"}`,
        );
      }
    }, 750);
    return () => window.clearInterval(timer);
  }, [job?.id, running]);
  useEffect(() => {
    if (view === "files") {
      loadOutputs().catch((error) => setMessage(String(error)));
      void loadJobHistory();
    }
    if (view === "publish") void loadPublishSession();
  }, [view, historyScope, options.output_dir]);
  useEffect(() => {
    if (!publishSession.active) return;
    const timer = window.setInterval(loadPublishSession, 1500);
    return () => window.clearInterval(timer);
  }, [publishSession.active]);
  useEffect(() => {
    if (!job?.started_at || !running) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [job?.started_at, running]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.matches("input, textarea, select");
      if (event.ctrlKey && event.key.toLowerCase() === "o") {
        event.preventDefault(); void chooseNativeMaterial(); return;
      }
      if (event.ctrlKey && event.key.toLowerCase() === "l") {
        event.preventDefault(); setView("material"); document.getElementById("source-url")?.focus(); return;
      }
      if (event.ctrlKey && event.key.toLowerCase() === "r" && !typing) {
        event.preventDefault(); void start(); return;
      }
      if (event.key === "Escape" && running) {
        event.preventDefault(); void cancel();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [running, sourceUrl, selected?.id, authorized, device, computeType, options, description, tags, publishInPipeline, includeSourceLink, fullVideo, durationValue, durationUnit]);

  const readMetadata = async () => {
    if (!sourceUrl.trim()) {
      setMetadataState("error");
      setMetadataNotice("请先粘贴完整的视频链接，再读取标题与时长。");
      return;
    }
    try {
      setMetadataState("loading");
      setMetadataNotice("正在读取标题、时长与许可证信息…");
      setMessage("正在读取链接标题与时长…");
      const value = await api<{
        title: string | null;
        duration: number | null;
        license: string | null;
        view_count: number | null;
      }>("/api/metadata", {
        method: "POST",
        body: JSON.stringify({
          url: sourceUrl,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
        }),
      });
      setMetadata(value);
      if (value.title) updateOptions("title", value.title);
      setMetadataState("success");
      setMetadataNotice("已读取视频信息，标题已自动填入下方字段。");
      setMessage("已读取视频信息。");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "未知错误";
      setMetadataState("error");
      setMetadataNotice(`读取失败：${detail}`);
      setMessage(`读取失败：${detail}`);
    }
  };
  const importVideo = async (file: File) => {
    if (!authorized) {
      setMessage("请先勾选上方「我确认拥有处理、转载或发布该视频的授权/许可证」复选框，再导入本地视频。");
      document.getElementById("rights-check")?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const body = new FormData();
    body.append("video", file);
    body.append("authorized", "true");
    try {
      const material = await api<Material>("/api/materials", {
        method: "POST",
        body,
      });
      await loadMaterials();
      setMaterialId(material.id);
      setSourceUrl("");
      updateOptions("title", material.name.replace(/\.[^.]+$/, ""));
      setMessage(
        `已导入 ${material.name}（开发模式：此版本会复制到工作区；桌面版直接使用原路径）。`,
      );
    } catch (error) {
      setMessage(
        `导入失败：${error instanceof Error ? error.message : "未知错误"}`,
      );
    }
  };
  const importPublishVideo = async (file: File) => {
    const body = new FormData();
    body.append("video", file);
    body.append("authorized", "true");
    try {
      const detail = await api<{ token: string; name: string; title: string; tags: string[] }>("/api/publish/upload", { method: "POST", body });
      setNativePublishToken(detail.token); setNativePublishName(detail.name); setSelectedPublishVideo("__native__");
      updateOptions("title", detail.title); if (detail.tags.length) setTags(detail.tags.join(", "));
      setMessage(`开发模式已导入成品视频：${detail.name}。`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "导入成品视频失败"); }
  };
  const buildDebugText = () => {
    const cookies = options.cookies_file
      ? `cookies.txt（${options.cookies_file}）`
      : options.cookies_from_browser
        ? `浏览器（${options.cookies_from_browser}）`
        : "未配置";
    const lines = [
      `时间：${new Date().toLocaleString()}`,
      `素材：${job?.material?.name ?? selected?.name ?? "未选择"}`,
      `状态：${job?.status ?? "-"}（${job?.stage ?? "-"} ${job?.progress ?? 0}%）`,
      `翻译器：${options.translator}${apiKeyConfigured === false ? "（DeepSeek Key 未配置）" : ""}`,
      `Cookies：${cookies}`,
      `错误：${job?.error ?? "无"}`,
      "日志：",
      ...(job?.logs ?? []),
    ];
    return lines.join("\n");
  };
  const confirmDiscardSubtitles = (action: string) =>
    !dirty ||
    window.confirm(
      `当前字幕还有未保存的修改（已编辑 ${history.length} 次）。\n${action}，未保存的内容将被清空。\n\n确定继续吗？`,
    );
  const start = async () => {
    const url = sourceUrl.trim();
    if ((!selected && !url) || running) return;
    if (dirty && !confirmDiscardSubtitles("开始新任务会清空当前未保存的字幕修改")) return;
    if (!authorized) return setMessage("开始前请确认拥有处理或转载授权。");
    if (url && !fullVideo && !secondsFromDuration(durationValue, durationUnit)) {
      return setMessage("请填写有效的 URL 读取长度，或选择完整视频。");
    }
    if (!(await checkReadiness())) return;
    try {
      const created = await api<Job>("/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          material_id: selected?.id,
          source_url: url || undefined,
          authorized,
          device,
          compute_type: computeType,
          options: taskOptions(),
        }),
      });
      setJob(created);
      setCues([]);
      setPreview("source");
      setMessage(
        url ? "正在下载视频并开始处理。" : `已开始处理：${selected?.name}`,
      );
    } catch (error) {
      setMessage(
        `无法启动：${error instanceof Error ? error.message : "未知错误"}`,
      );
    }
  };
  const cancel = async () => {
    if (!job) return;
    try {
      setJob(
        await api<Job>(`/api/jobs/${job.id}/cancel`, {
          method: "POST",
          body: "{}",
        }),
      );
      setMessage("已请求中断，当前步骤结束后停止。");
    } catch (error) {
      setMessage(String(error));
    }
  };
  const save = async () => {
    if (!job || !dirty) return;
    for (const cue of cues) {
      if (!cue.deleted && cue.start >= cue.end) {
        setMessage(`第 ${cue.id + 1} 条字幕时间无效：开始时间必须小于结束时间。`);
        return;
      }
    }
    try {
      await api<Job>(`/api/jobs/${job.id}/cues`, {
        method: "PUT",
        body: JSON.stringify({
          cues: cues.map((cue) => ({
            start: cue.start,
            end: cue.end,
            translated: cue.translated,
            deleted: Boolean(cue.deleted),
          })),
        }),
      });
      setHistory([]);
      setFuture([]);
      setMessage("字幕已保存（含时间调整与删除）。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存字幕失败");
    }
  };
  const rerender = async () => {
    if (!job || dirty) return;
    try {
      setJob(
        await api<Job>(`/api/jobs/${job.id}/rerender`, {
          method: "POST",
          body: "{}",
        }),
      );
      setMessage("正在按当前字幕样式重新渲染。 ");
    } catch (error) {
      setMessage(String(error));
    }
  };
  const addCutRange = () => {
    if (!job) return;
    const duration = job.material.duration_seconds ?? 0;
    const start = parseTime(cutStart);
    const end = parseTime(cutEnd);
    if (start === null || end === null) {
      setMessage("请输入有效的开始/结束时间，如 0:05.0 或 1:30。");
      return;
    }
    if (start < 0 || end > duration || start >= end) {
      setMessage(`时间范围无效：应在 00:00.0 – ${formatTime(duration)} 之间且开始早于结束。`);
      return;
    }
    setCutRanges((ranges) => [...ranges, [start, end]]);
    setCutStart("");
    setCutEnd("");
  };
  const removeCutRange = (index: number) =>
    setCutRanges((ranges) => ranges.filter((_, i) => i !== index));
  const applyTrim = async () => {
    if (!job || running || !cutRanges.length) return;
    if (dirty) {
      setMessage("修改前请先「保存」字幕修改，否则未保存的改动会丢失。");
      return;
    }
    try {
      if (modOp === "cut") {
        setJob(
          await api<Job>(`/api/jobs/${job.id}/trim`, {
            method: "POST",
            body: JSON.stringify({ cut_ranges: cutRanges }),
          }),
        );
        setMessage("正在删除片段并重新生成成片…");
      } else {
        setJob(
          await api<Job>(`/api/jobs/${job.id}/modify`, {
            method: "POST",
            body: JSON.stringify({ op: modOp, ranges: cutRanges }),
          }),
        );
        setMessage(modOp === "keep" ? "正在截取保留片段并重新生成…" : "正在静音指定片段…");
      }
      setCutRanges([]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "修改失败");
    }
  };
  const exportSegment = async () => {
    if (!job || running) return;
    const start = parseTime(exportStart);
    const end = parseTime(exportEnd);
    if (start === null || end === null || start >= end) {
      setMessage("请填写有效的导出时间段（开始早于结束）。");
      return;
    }
    try {
      const result = await api<{ exported: string; name: string }>(
        `/api/jobs/${job.id}/modify`,
        { method: "POST", body: JSON.stringify({ op: "export", ranges: [[start, end]] }) },
      );
      setMessage(`已导出片段：${result.name}（${result.exported}）`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "导出失败");
    }
  };
  const applyReorder = async () => {
    if (!job || running) return;
    if (dirty) {
      setMessage("重排前请先「保存」字幕修改。");
      return;
    }
    const points = reorderPoints
      .split(/[,，]/)
      .map((value) => parseTime(value))
      .filter((value): value is number => value !== null);
    const order = reorderOrder
      .split(/[,，]/)
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isInteger(value) && value >= 0);
    if (!points.length || !order.length) {
      setMessage("请填写分割点（如 0:10,0:20）和新顺序（如 3,1,2）。");
      return;
    }
    try {
      setJob(
        await api<Job>(`/api/jobs/${job.id}/modify`, {
          method: "POST",
          body: JSON.stringify({ op: "reorder", points, order }),
        }),
      );
      setMessage("正在重排片段并重新生成…");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "重排失败");
    }
  };
  const updateCue = (id: number, translated: string) => {
    setHistory((value) => [...value, cues]);
    setFuture([]);
    setCues((value) =>
      value.map((cue) => (cue.id === id ? { ...cue, translated } : cue)),
    );
  };
  const updateCueTime = (id: number, field: "start" | "end", value: string) => {
    const parsed = parseTime(value);
    if (parsed === null) {
      setMessage(`时间格式无效（如 0:05.0 或 1:30）：${value}`);
      return;
    }
    setHistory((value) => [...value, cues]);
    setFuture([]);
    setCues((current) =>
      current.map((cue) => (cue.id === id ? { ...cue, [field]: parsed } : cue)),
    );
  };
  const toggleCueDeleted = (id: number) => {
    setHistory((value) => [...value, cues]);
    setFuture([]);
    setCues((current) =>
      current.map((cue) => (cue.id === id ? { ...cue, deleted: !cue.deleted } : cue)),
    );
  };
  const alignSubtitles = async () => {
    if (!job || running) return;
    if (dirty) {
      setMessage("对齐前请先「保存」字幕修改。");
      return;
    }
    try {
      setJob(
        await api<Job>(`/api/jobs/${job.id}/align`, { method: "POST", body: "{}" }),
      );
      setMessage("正在用画面字幕时间自动对齐音频字幕…完成后自动刷新。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "自动对齐失败");
    }
  };
  const applyOffset = () => {    const offset = Number(offsetValue);
    if (!job || !cues.length) return;
    if (!Number.isFinite(offset) || offset === 0) {
      setMessage("请输入有效的偏移秒数（如 0.5 或 -0.5，可为小数）。");
      return;
    }
    const duration = job.material.duration_seconds;
    setHistory((value) => [...value, cues]);
    setFuture([]);
    setCues((current) =>
      current.map((cue) => ({
        ...cue,
        start: Math.max(0, Math.round((cue.start + offset) * 100) / 100),
        end: duration
          ? Math.min(duration, Math.max(0.1, Math.round((cue.end + offset) * 100) / 100))
          : Math.max(0.1, Math.round((cue.end + offset) * 100) / 100),
      })),
    );
    setOffsetValue("");
    setMessage(`已将所有字幕整体偏移 ${offset} 秒（保持时长不变）。请检查后点「保存」。`);
  };
  const undo = () => {
    const previous = history.at(-1);
    if (!previous) return;
    setFuture((value) => [cues, ...value]);
    setCues(previous);
    setHistory((value) => value.slice(0, -1));
  };
  const redo = () => {
    const next = future[0];
    if (!next) return;
    setHistory((value) => [...value, cues]);
    setCues(next);
    setFuture((value) => value.slice(1));
  };
  const seek = (cue: Cue) => {
    setActiveCue(cue.id);
    if (video.current) {
      video.current.currentTime = cue.start;
      video.current.play();
    }
  };
  const saveSettings = async () => {
    try {
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({
          translator: options.translator,
          api_key: apiKey,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
        }),
      });
      setApiKey("");
      setSettingsOpen(false);
      setMessage("本地设置已保存，密钥未回显。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存设置失败");
    }
  };
  const chooseCookiesFile = async () => {
    const bridge = window.pywebview?.api;
    if (!bridge) {
      setMessage("此版本不支持原生文件选择，请直接输入 cookies.txt 的完整路径。");
      return;
    }
    try {
      const selection = await bridge.choose_cookies_file();
      if (!selection) return;
      updateOptions("cookies_file", selection.path);
      updateOptions("cookies_from_browser", "");
      setMessage(`已选择 Cookies 文件：${selection.path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "选择 Cookies 文件失败");
    }
  };
  const generateDescription = async () => {
    try {
      const response = await api<{ description: string }>(
        "/api/publish/description",
        {
          method: "POST",
          body: JSON.stringify({
            template: templateName,
            source_url: sourceUrl,
            include_source_link: includeSourceLink,
            custom_text: templateBody,
            template_body: templateBody,
            extra_lines: extraLine ? [extraLine] : [],
          }),
        },
      );
      setDescription(response.description);
      setMessage("已按当前模板内容生成简介（草稿内容优先于已保存模板）。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "生成简介失败");
    }
  };
  const saveTemplate = async () => {
    const name = templateDraftName.trim();
    if (!name) return setMessage("请填写模板名称。");
    try {
      await api("/api/templates", {
        method: "POST",
        body: JSON.stringify({ name, body: templateBody || description }),
      });
      await loadTemplates();
      setTemplateName(name);
      setTemplateEditorOpen(false);
      setMessage("模板已保存到本机。 ");
    } catch (error) {
      setMessage(String(error));
    }
  };
  const openTemplateEditor = (name = "") => {
    const current = templates.find((item) => item.name === (name || templateName));
    setTemplateDraftName(name || (templateName === "授权本地化" ? "" : templateName));
    if (name || !templateDraftDirty) {
      setTemplateBody(current?.body ?? templateBody);
    }
    setTemplateEditorOpen(true);
  };
  const insertTemplateVariable = (variable: string) => setTemplateBody((value) => `${value}${value && !value.endsWith(" ") ? " " : ""}${variable}`);
  const deleteTemplate = async () => {
    if (!window.confirm(`删除自定义模板“${templateName}”吗？`)) return;
    try {
      await api(`/api/templates/${encodeURIComponent(templateName)}`, {
        method: "DELETE",
      });
      await loadTemplates();
      setTemplateName("授权本地化");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除失败");
    }
  };
  const publishCheck = async () => {
    try {
      const response = await api<{ message: string }>("/api/publish/check", {
        method: "POST",
        body: JSON.stringify({ browser: publishBrowser }),
      });
      setMessage(response.message);
      await loadPublishStatus();
      void loadPublishSession();
    } catch (error) {
      setMessage(String(error));
    }
  };
  const publishAssist = async () => {
    if (!publishVideoReady) {
      setMessage("请先选择当前任务成片、输出目录中的成片，或任意本地成片，再打开投稿辅助。");
      return;
    }
    if (
      !window.confirm(
        "将打开 B 站投稿页并辅助填写。不会自动点击投稿；请在页面中人工检查并提交。继续吗？",
      )
    )
      return;
    try {
      const response = await api<{ message: string }>("/api/publish/assist", {
        method: "POST",
        body: JSON.stringify({
          job_id: !nativePublishToken && !selectedPublishVideo && job?.result ? job.id : "",
          video_id: selectedPublishVideo,
          native_video_token: nativePublishToken,
          output_dir: options.output_dir,
          confirmed: true,
          browser: publishBrowser,
          title: options.title,
          description,
          tags: tags
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean),
          close_after_fill: closeAfterFill,
        }),
      });
      setMessage(response.message);
      void loadPublishSession();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "发布辅助失败");
    }
  };
  const closePublishBrowser = async () => {
    if (!window.confirm("关闭 B 站自动化浏览器会中断上传。确认关闭吗？"))
      return;
    try {
      const response = await api<{ closed: number }>(
        "/api/publish/close-browser",
        { method: "POST", body: JSON.stringify({ confirmed: true }) },
      );
      setMessage(`已关闭 ${response.closed} 个后台浏览器进程。`);
    } catch (error) {
      setMessage(String(error));
    }
  };
  const deleteOutputs = async () => {
    if (
      !selectedOutputPaths.length ||
      !window.confirm(
        `将删除以下 ${selectedOutputPaths.length} 个项目，预计释放约 ${formatBytes(selectedOutputBytes)}：\n${selectedOutputPaths.join("\n")}\n\n此操作不可恢复，确定继续吗？`,
      )
    )
      return;
    try {
      const response = await api<{ deleted_size: string }>(
        "/api/outputs/delete",
        {
          method: "POST",
          body: JSON.stringify({
            output_dir: options.output_dir,
            paths: selectedOutputPaths,
            confirmed: true,
          }),
        },
      );
      setMessage(`已删除 ${response.deleted_size}。`);
      setSelectedOutputPaths([]);
      await loadOutputs();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除失败");
    }
  };
  const openOutput = async () => {
    try {
      await api("/api/outputs/open", {
        method: "POST",
        body: JSON.stringify({ output_dir: options.output_dir }),
      });
    } catch (error) {
      setMessage(String(error));
    }
  };
  const openOutputPath = async (path: string) => {
    try {
      await api("/api/outputs/open-path", { method: "POST", body: JSON.stringify({ output_dir: options.output_dir, path }) });
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法打开所选项目"); }
  };
  const chooseOutputDir = async () => {
    const bridge = window.pywebview?.api;
    if (!bridge) {
      setMessage("此版本不支持原生文件夹选择，请直接输入输出目录路径。");
      return;
    }
    try {
      const selection = await bridge.choose_output_dir();
      if (!selection) return;
      updateOptions("output_dir", selection.path);
      setMessage(`输出目录已设为：${selection.path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "选择输出目录失败");
    }
  };
  const [rightsHint, setRightsHint] = useState(false);
  const chooseNativeMaterial = async () => {
    if (!authorized) {
      setRightsHint(true);
      setMessage("请先勾选下方「我确认拥有处理、转载或发布该视频的授权/许可证」复选框，再导入本地视频。");
      document.getElementById("rights-check")?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    const bridge = window.pywebview?.api;
    if (!bridge || typeof bridge.choose_material_video !== "function") {
      fileInput.current?.click();
      return;
    }
    try {
      const selection = await bridge.choose_material_video();
      if (!selection) return;
      await loadMaterials();
      setMaterialId(selection.material_id);
      setSourceUrl("");
      setJob(null);
      setCues([]);
      updateOptions("title", selection.name.replace(/\.[^.]+$/, ""));
      setMessage(`已导入本地视频（直接使用原路径，不会复制到工作区）：${selection.name}。`);
    } catch (error) {
      // 原生对话框异常时退回浏览器文件选择器，保证一定有反应
      fileInput.current?.click();
    }
  };
  const chooseNativePublishVideo = async () => {
    const bridge = window.pywebview?.api;
    if (!bridge) {
      publishFileInput.current?.click();
      return;
    }
    try {
      const selection = await bridge.choose_publish_video();
      if (!selection) return;
      const detail = await api<{ token: string; name: string; title: string; tags: string[] }>("/api/publish/native-video", { method: "POST", body: JSON.stringify(selection) });
      setNativePublishToken(detail.token); setNativePublishName(detail.name); setSelectedPublishVideo("__native__");
      updateOptions("title", detail.title); if (detail.tags.length) setTags(detail.tags.join(", "));
      setMessage(`已选择成品视频：${detail.name}。`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "选择成品视频失败"); }
  };

  const materialView = (
    <section className="workspace-card material-view">
      <header>
        <p className="eyebrow">素材来源</p>
        <h1>从链接或本地视频开始</h1>
        <p>仅处理你已获授权的内容。URL 会先由 yt-dlp 下载到本机。</p>
      </header>
      {apiKeyConfigured === false && options.translator === "deepseek" && (
        <div className="welcome-banner">
          <div>
            <b>首次使用提示</b>
            <span>
              当前翻译器是 DeepSeek，但 API Key 还未配置。请先到「设置」填入 Key；也可以在「处理」页把翻译器改为「不翻译」；或者先用「演示视频」体验完整流程（不需要 Key）。
            </span>
          </div>
          <button className="primary small" onClick={() => setSettingsOpen(true)}>
            去设置
          </button>
        </div>
      )}
      {diagnostics.some((check) => !check.available) && (
        <section className="diagnostics-banner" aria-label="首次使用依赖检查">
          <div>
            <b>开始前还需安装 {diagnostics.filter((check) => !check.available).map((check) => check.label).join("、")}</b>
            <span>这些本机工具用于音频、OCR 与字幕渲染；安装后重启工作台即可重新检测。</span>
          </div>
          <button className="text-button" onClick={() => setShortcutHelpOpen(true)}>查看说明</button>
        </section>
      )}
      <div className="form-grid two">
        <label>
          视频链接
          <input
            id="source-url"
            value={sourceUrl}
            placeholder="粘贴 YouTube / B 站视频链接"
            onChange={(event) => {
              setSourceUrl(event.target.value);
              setMetadata(null);
              setMetadataNotice("");
              setMetadataState("idle");
            }}
            onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void readMetadata(); } }}
            disabled={running}
          />
        </label>
        <div className="inline-actions">
          <button
            onClick={readMetadata}
            disabled={running || metadataState === "loading"}
          >
            {metadataState === "loading" ? "正在读取…" : "读取标题与时长"}
          </button>
          <button
            className="primary"
            onClick={start}
            disabled={running || !sourceUrl.trim()}
          >
            下载并处理
          </button>
        </div>
      </div>
      {metadataState !== "idle" && (
        <div className="metadata-notice-wrap">
          <p className={`metadata-notice ${metadataState}`} role="status">
            {metadataNotice}
          </p>
          {metadataState === "error" && cookieHintMatches(metadataNotice) && (
            <button className="cookie-hint-button" onClick={() => setSettingsOpen(true)}>
              去设置 → YouTube Cookies 来源
            </button>
          )}
        </div>
      )}
      {metadata && (
        <div className="metadata">
          <span>
            {metadata.duration
              ? `${metadata.duration.toFixed(1)} 秒`
              : "时长未知"}
          </span>
          <span>{metadata.license || "许可证待确认"}</span>
          <span>{metadata.view_count?.toLocaleString() || "—"} 播放</span>
        </div>
      )}
      <div className="divider">
        <span>或导入本地素材</span>
      </div>
      <div className="form-grid two">
        <label>
          已导入视频
          <select
            value={materialId}
            onChange={(event) => {
              if (dirty && !confirmDiscardSubtitles("切换素材会清空当前未保存的字幕修改")) return;
              setMaterialId(event.target.value);
              setSourceUrl("");
              setJob(null);
              setCues([]);
            }}
          >
            <option value="">未选择视频</option>
            {materials.map((item) => (
              <option key={item.id} value={item.id}>
                {item.is_demo ? "演示 · " : "本地 · "}
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <div className="inline-actions">
          <button onClick={() => void chooseNativeMaterial()} disabled={running}>
            选择本地视频
          </button>
          {!authorized && (
            <span className="rights-hint">导入前需先勾选下方授权确认</span>
          )}
          <button
            className="primary"
            onClick={start}
            disabled={running || !selected}
          >
            处理本地视频
          </button>
        </div>
      </div>
      <p className="material-hint">
        桌面版导入本地视频时直接使用原路径，不会复制到工作区，也不占用双倍磁盘。
      </p>
      <div className="form-grid three">
        <label>
          标题
          <input
            value={options.title}
            onChange={(event) => updateOptions("title", event.target.value)}
          />
        </label>
        <label>
          URL 读取长度
          <div className="duration-control">
            <label className="check"><input type="checkbox" checked={fullVideo} disabled={!sourceUrl.trim()} onChange={(event) => setFullVideo(event.target.checked)} />完整视频</label>
            <input type="number" min="1" value={durationValue} disabled={!sourceUrl.trim() || fullVideo} onChange={(event) => setDurationValue(event.target.value)} />
            <select value={durationUnit} disabled={!sourceUrl.trim() || fullVideo} onChange={(event) => setDurationUnit(event.target.value as "秒" | "分钟" | "小时")}><option>秒</option><option>分钟</option><option>小时</option></select>
          </div>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={options.require_reuse_allowed}
            onChange={(event) =>
              updateOptions("require_reuse_allowed", event.target.checked)
            }
            disabled={!sourceUrl.trim()}
          />
          仅接受 CC reuse allowed
          <small className="check-hint">
            （大多数视频无此标记，勾选后可能无法下载）
          </small>
        </label>
      </div>
      <label className={`rights-row ${rightsHint && !authorized ? "rights-hint-active" : ""}`} id="rights-check">
        <input
          type="checkbox"
          checked={authorized}
          onChange={(event) => {
            setAuthorized(event.target.checked);
            if (event.target.checked) setRightsHint(false);
          }}
        />
        我确认拥有处理、转载或发布该视频的授权/许可证
      </label>
      <div className="card-footer task-tools">
        <button onClick={applyRecommended} disabled={running}>推荐设置</button>
        <button onClick={() => void checkReadiness()} disabled={running}>检查配置</button>
        {readiness.length > 0 && <span className="readiness-error">{readiness.join("；")}</span>}
      </div>
    </section>
  );
  const processView = (
    <section className="workspace-card">
      <header>
        <p className="eyebrow">处理配置</p>
        <h1>字幕、翻译与推理</h1>
        <p>这些参数会与任务一起保存，并显示在检查器中。</p>
      </header>
      {options.translator === "deepseek" && apiKeyConfigured === false && (
        <p className="metadata-notice error">
          DeepSeek API Key 未配置。请到「设置」填入 Key，或把翻译器改为「不翻译」。
        </p>
      )}
      {mode === "basic" ? (
        <div className="settings-grid">
          <label>
            字幕来源
            <select
              value={options.subtitle_source}
              onChange={(event) =>
                updateOptions("subtitle_source", event.target.value)
              }
            >
              <option value="auto">自动检测</option>
              <option value="merged">音频 + OCR 合并</option>
              <option value="audio">仅音频转写</option>
              <option value="ocr">仅画面字幕 OCR</option>
            </select>
          </label>
          <label>
            翻译器
            <select
              value={options.translator}
              onChange={(event) =>
                updateOptions("translator", event.target.value)
              }
            >
              <option value="deepseek">DeepSeek</option>
              <option value="openai">OpenAI</option>
              <option value="none">不翻译</option>
            </select>
          </label>
          <label>
            字幕排版
            <select
              value={options.subtitle_display_mode}
              onChange={(event) =>
                updateOptions("subtitle_display_mode", event.target.value)
              }
            >
              <option value="translated">中文单语</option>
              <option value="bilingual-source-first">原文在上 + 中文在下</option>
              <option value="bilingual-translation-first">中文在上 + 原文在下</option>
            </select>
          </label>
          <p className="mode-note">
            其余参数（模型、精度、OCR、字幕样式等）已隐藏，当前使用推荐默认值。
            <button
              className="mode-expand"
              onClick={() => {
                setMode("advanced");
                localStorage.setItem("ybl-mode", "advanced");
              }}
            >
              展开全部选项
            </button>
          </p>
        </div>
      ) : (
      <div className="settings-grid">
        <label>
          字幕来源
          <select
            value={options.subtitle_source}
            onChange={(event) =>
              updateOptions("subtitle_source", event.target.value)
            }
          >
            <option value="auto">自动检测</option>
            <option value="merged">音频 + OCR 合并</option>
            <option value="audio">仅音频转写</option>
            <option value="ocr">仅画面字幕 OCR</option>
          </select>
        </label>
        <label>
          Whisper 模型
          <select
            value={options.whisper_model_size}
            onChange={(event) =>
              updateOptions("whisper_model_size", event.target.value)
            }
          >
            {["tiny", "base", "small", "medium", "large-v3"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          源语言
          <select
            value={options.source_language}
            onChange={(event) =>
              updateOptions("source_language", event.target.value)
            }
          >
            <option value="">自动检测</option>
            {["en", "zh", "ja", "ko", "fr", "de", "es", "ru"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          翻译器
          <select
            value={options.translator}
            onChange={(event) =>
              updateOptions("translator", event.target.value)
            }
          >
            <option value="deepseek">DeepSeek</option>
            <option value="openai">OpenAI</option>
            <option value="none">不翻译</option>
          </select>
        </label>
        <label>
          翻译模型
          <input
            value={options.translate_model}
            placeholder="使用默认模型"
            onChange={(event) =>
              updateOptions("translate_model", event.target.value)
            }
          />
        </label>
        <label>
          目标语言
          <input
            value={options.target_lang}
            onChange={(event) =>
              updateOptions("target_lang", event.target.value)
            }
          />
        </label>
        <label>
          推理设备
          <select
            value={device}
            onChange={(event) => {
              const next = event.target.value;
              setDevice(next);
              if (next === "cpu" && ["float16", "int8_float16"].includes(computeType)) setComputeType("int8");
              if (["cuda", "auto"].includes(next) && computeType === "int8") setComputeType("float16");
            }}
          >
            <option value="cuda">CUDA</option>
            <option value="cpu">CPU</option>
            <option value="auto">Auto</option>
          </select>
        </label>
        <label>
          计算精度
          <select
            value={computeType}
            onChange={(event) => setComputeType(event.target.value)}
          >
            {["int8", "int8_float16", "default", "float16", "float32"].map(
              (value) => (
                <option key={value}>{value}</option>
              ),
            )}
          </select>
        </label>
        <label>
          转写束宽（beam_size）
          <select
            value={options.beam_size}
            onChange={(event) => updateOptions("beam_size", Number(event.target.value))}
          >
            <option value={1}>1（快）</option>
            <option value={5}>5（推荐）</option>
            <option value={10}>10（更准更慢）</option>
          </select>
        </label>
        <label>
          OCR 帧间隔（秒）
          <select
            value={options.ocr_interval}
            onChange={(event) => updateOptions("ocr_interval", Number(event.target.value))}
          >
            <option value={0.5}>0.5（更密）</option>
            <option value={1}>1（推荐）</option>
            <option value={2}>2（更快）</option>
          </select>
        </label>
        <label>
          OCR 识别区域
          <select
            value={options.ocr_crop_ratio}
            onChange={(event) => updateOptions("ocr_crop_ratio", Number(event.target.value))}
          >
            <option value={0.2}>仅底部 20%</option>
            <option value={0.3}>底部 30%（推荐）</option>
            <option value={0.5}>下半屏</option>
            <option value={1}>全画面</option>
          </select>
        </label>
        <label>
          OCR 最小字符数
          <input
            type="number"
            min="1"
            max="20"
            value={options.ocr_min_chars}
            onChange={(event) => updateOptions("ocr_min_chars", Number(event.target.value || 3))}
          />
        </label>
        <label>
          字幕底部位置
          <select
            value={options.subtitle_margin_ratio}
            onChange={(event) => updateOptions("subtitle_margin_ratio", Number(event.target.value))}
          >
            <option value={0.03}>3%（贴近底部）</option>
            <option value={0.055}>5.5%（推荐）</option>
            <option value={0.1}>10%（偏上）</option>
            <option value={0.15}>15%（更高）</option>
          </select>
        </label>
        <label>
          渲染质量（CRF）
          <select
            value={options.render_crf}
            onChange={(event) => updateOptions("render_crf", Number(event.target.value))}
          >
            <option value={18}>18（高质量，文件大）</option>
            <option value={20}>20（推荐）</option>
            <option value={23}>23（平衡）</option>
            <option value={28}>28（小文件）</option>
          </select>
        </label>
        <label>
          输出目录
          <div className="output-dir-control">
            <input
              value={options.output_dir}
              onChange={(event) =>
                updateOptions("output_dir", event.target.value)
              }
            />
            <button onClick={() => void chooseOutputDir()} disabled={running}>
              选择文件夹
            </button>
          </div>
        </label>
      </div>
      )}
      <div className="toggle-row">
        <label className="check">
          <input
            type="checkbox"
            checked={options.smart_translation}
            onChange={(event) =>
              updateOptions("smart_translation", event.target.checked)
            }
          />
          智能上下文翻译
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={options.smart_subtitle_layout}
            onChange={(event) =>
              updateOptions("smart_subtitle_layout", event.target.checked)
            }
          />
          智能字幕换行排版
        </label>
        <label className="check">
          <input type="checkbox" checked={publishInPipeline} onChange={(event) => setPublishInPipeline(event.target.checked)} />
          渲染后打开 B 站创作中心并辅助上传填表
        </label>
      </div>
      {publishInPipeline && <div className="settings-grid compact publish-pipeline-settings">
        <label>投稿浏览器<select value={publishBrowser} onChange={(event) => setPublishBrowser(event.target.value)}><option value="chromium">Chromium</option><option value="msedge">Microsoft Edge</option></select></label>
        <label className="check"><input type="checkbox" checked={includeSourceLink} onChange={(event) => setIncludeSourceLink(event.target.checked)} />简介自动附原视频链接</label>
        <label className="check"><input type="checkbox" checked={closeAfterFill} onChange={(event) => setCloseAfterFill(event.target.checked)} />填表完成后关闭浏览器（可能中断上传）</label>
      </div>}
      {job && cues.length > 0 && (job.status === "completed" || (running && job.progress >= 48)) && (
        <section className="workspace-card result-preview">
          <header>
            <p className="eyebrow">文本处理结果</p>
            <h1>转写 / OCR / 翻译对照</h1>
            <p>
              {translationReady
                ? "每条字幕的来源（语音 / 画面文字 / 合并）与翻译结果逐条对照，可确认文本处理是否准确。"
                : "翻译处理中，当前显示原文预览，完成后自动更新译文。"}
            </p>
          </header>
          <div className="result-table">
            {cues.map((cue) => (
              <article key={cue.id} className="result-row">
                <span className={`src-kind ${cue.kind ?? "audio"}`}>
                  {cue.kind === "ocr" ? "画面文字" : cue.kind === "merged" ? "合并" : "语音"}
                </span>
                <div>
                  <p className="result-source" title={cue.source}>{cue.source}</p>
                  <p className={`result-translated ${translationReady ? "" : "pending"}`} title={cue.translated}>
                    {translationReady ? cue.translated : `${cue.translated}（翻译中…）`}
                  </p>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
      <footer className="card-footer">
        <button
          className="primary"
          onClick={start}
          disabled={running || (!selected && !sourceUrl.trim())}
        >
          {running ? "任务运行中" : "使用当前设置开始处理"}
        </button>
      </footer>
    </section>
  );
  const subtitleView = (
    <>
      <section className="media-panel">
        <div className="panel-toolbar">
          <div>
            <p className="eyebrow">字幕工作区</p>
            <h1>
              {preview === "rendered" && job?.result
                ? "中文字幕成片"
                : "原始视频"}
            </h1>
          </div>
          <div className="media-actions">
            <button
              className={preview === "source" ? "selected" : ""}
              onClick={() => setPreview("source")}
            >
              原片
            </button>
            <button
              className={preview === "rendered" ? "selected" : ""}
              disabled={!job?.result}
              onClick={() => setPreview("rendered")}
            >
              成片
            </button>
          </div>
        </div>
        <video
          ref={video}
          className="real-video"
          src={previewSrc}
          controls
          preload="metadata"
          onTimeUpdate={(event) => {
            const cue = cues.find(
              (item) =>
                event.currentTarget.currentTime >= item.start &&
                event.currentTarget.currentTime <= item.end,
            );
            setActiveCue(cue?.id ?? null);
          }}
        />
        <div className="timeline-wrap">
          <div className="time-ruler">
            <span>00:00</span>
            <span>{timecode(duration / 2)}</span>
            <span>{timecode(duration)}</span>
          </div>
          <div className="timeline">
            {cuePositions.length ? (
              cuePositions.map(({ cue, left, width }) => (
                <button
                  key={cue.id}
                  className={
                    activeCue === cue.id ? "cue-mark active" : "cue-mark"
                  }
                  style={{ left, width }}
                  onClick={() => seek(cue)}
                >
                  {cue.id + 1}
                </button>
              ))
            ) : (
              <span>转写完成后，真实字幕段会显示在此时间轴。</span>
            )}
          </div>
        </div>
      </section>
      <section className="workspace-card trim-card">
        <header>
          <p className="eyebrow">视频修改</p>
          <h1>删除、保留、静音、导出与重排</h1>
          <p>修改后字幕时间轴自动对齐并重新压制；修改基于当前任务，可连续操作。</p>
        </header>
        <div className="mod-op-row">
          {(
            [
              ["cut", "删除片段"],
              ["keep", "截取保留"],
              ["mute", "片段静音"],
            ] as const
          ).map(([value, label]) => (
            <label key={value} className="check">
              <input
                type="radio"
                name="mod-op"
                checked={modOp === value}
                onChange={() => setModOp(value)}
                disabled={running || !job?.result}
              />
              {label}
            </label>
          ))}
        </div>
        <div className="trim-controls">
          <input
            value={cutStart}
            placeholder="开始（如 0:05.0）"
            disabled={running || !job?.result}
            onChange={(event) => setCutStart(event.target.value)}
          />
          <span>至</span>
          <input
            value={cutEnd}
            placeholder="结束（如 0:10.0）"
            disabled={running || !job?.result}
            onChange={(event) => setCutEnd(event.target.value)}
          />
          <button onClick={addCutRange} disabled={running || !job?.result}>
            添加区间
          </button>
          <button
            className="primary small"
            onClick={() => void applyTrim()}
            disabled={running || !cutRanges.length}
          >
            {modOp === "cut" ? "删除选中区间" : modOp === "keep" ? "只保留选中区间" : "静音选中区间"}
          </button>
        </div>
        {cutRanges.length > 0 && (
          <div className="cut-list">
            {cutRanges.map((range, index) => (
              <span key={index} className="cut-chip">
                {formatTime(range[0])} – {formatTime(range[1])}
                <button onClick={() => removeCutRange(index)} title="移除此区间">✕</button>
              </span>
            ))}
          </div>
        )}
        <div className="trim-controls export-row">
          <b>导出片段</b>
          <input
            value={exportStart}
            placeholder="开始（如 0:10.0）"
            disabled={running || !job?.result}
            onChange={(event) => setExportStart(event.target.value)}
          />
          <span>至</span>
          <input
            value={exportEnd}
            placeholder="结束（如 0:20.0）"
            disabled={running || !job?.result}
            onChange={(event) => setExportEnd(event.target.value)}
          />
          <button onClick={() => void exportSegment()} disabled={running || !job?.result}>
            导出为新文件
          </button>
        </div>
        <div className="trim-controls reorder-row">
          <b>片段重排</b>
          <input
            value={reorderPoints}
            placeholder="分割点，如 0:10,0:20"
            disabled={running || !job?.result}
            onChange={(event) => setReorderPoints(event.target.value)}
          />
          <input
            value={reorderOrder}
            placeholder="新顺序，如 3,1,2"
            disabled={running || !job?.result}
            onChange={(event) => setReorderOrder(event.target.value)}
          />
          <button onClick={() => void applyReorder()} disabled={running || !job?.result}>
            应用重排
          </button>
        </div>
        <p className="trim-note">
          提示：修改会重写当前成片（可连续操作）；导出片段会另存新文件，不影响当前成片。修改前请先「保存」字幕。
        </p>
      </section>
      <section className="workspace-card subtitle-config">
        <header>
          <p className="eyebrow">字幕样式</p>
          <h1>排版与硬字幕成片</h1>
        </header>
        {mode === "basic" ? (
          <p className="metadata-notice">
            字幕样式当前使用推荐默认值（微软雅黑 24 号、中文单语、描边）。可在右上角切换到「高级模式」调整字体、字号与颜色。
          </p>
        ) : (
        <div className="settings-grid compact">
          <label>
            显示模式
            <select
              value={options.subtitle_display_mode}
              onChange={(event) =>
                updateOptions("subtitle_display_mode", event.target.value)
              }
            >
              <option value="translated">中文单语</option>
              <option value="bilingual-source-first">
                原文在上 + 中文在下
              </option>
              <option value="bilingual-translation-first">
                中文在上 + 原文在下
              </option>
            </select>
          </label>
          <label>
            字体
            <input
              value={options.font_name}
              onChange={(event) =>
                updateOptions("font_name", event.target.value)
              }
            />
          </label>
          <label>
            字号
            <input
              type="number"
              min="8"
              max="96"
              value={options.font_size}
              onChange={(event) =>
                updateOptions("font_size", Number(event.target.value || 24))
              }
            />
          </label>
          <label>
            字幕颜色
            <select
              value={options.subtitle_color}
              onChange={(event) =>
                updateOptions("subtitle_color", event.target.value)
              }
            >
              {["白色", "黄色", "青色", "绿色", "黑色"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            边框颜色
            <select
              value={options.subtitle_outline_color}
              onChange={(event) =>
                updateOptions("subtitle_outline_color", event.target.value)
              }
            >
              {["黑色", "白色", "灰色", "蓝色"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            效果
            <select
              value={options.subtitle_effect}
              onChange={(event) =>
                updateOptions("subtitle_effect", event.target.value)
              }
            >
              {["描边", "阴影", "描边+阴影", "无"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
        </div>
        )}
      </section>
      <section className="subtitle-editor">
        <div className="subtitle-toolbar">
          <div>
            <p className="eyebrow">字幕编辑</p>
            <div className="column-labels">
              <span>时间</span>
              <span>原文</span>
              <span>中文翻译</span>
            </div>
          </div>
          <div className="editor-actions">
            {dirty && <span className="dirty">未保存修改</span>}
            <button onClick={undo} disabled={!history.length || running}>
              撤销
            </button>
            <button onClick={redo} disabled={!future.length || running}>
              重做
            </button>
            <input
              className="offset-input"
              value={offsetValue}
              placeholder="偏移秒（如 0.5 / -0.5）"
              title="字幕与音频整体不同步时，输入正负秒数一次性平移所有字幕时间"
              disabled={running || !job?.result}
              onChange={(event) => setOffsetValue(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); applyOffset(); } }}
            />
            <button onClick={applyOffset} disabled={running || !job?.result || !cues.length}>
              整体偏移
            </button>
            <button
              onClick={() => void alignSubtitles()}
              disabled={running || !job?.result}
              title="有画面字幕（OCR）时，用帧级时间自动校正音频字幕的整体偏移"
            >
              自动对齐
            </button>
            <button onClick={save} disabled={!dirty || running}>
              保存
            </button>
            <button
              className="primary small"
              onClick={rerender}
              disabled={!job?.result || running || dirty}
            >
              重新渲染
            </button>
          </div>
        </div>
        {cues.length ? (
          <div className="cue-table">
            {cues.map((cue) => (
              <article
                key={cue.id}
                className={[
                  "cue-row",
                  activeCue === cue.id ? "active" : "",
                  cue.deleted ? "deleted" : "",
                ].join(" ")}
              >
                <div className="cue-time-edit">
                  <input
                    value={formatTime(cue.start)}
                    title="开始时间（分:秒，如 0:05.0）"
                    disabled={!job?.result || running || cue.deleted}
                    onChange={(event) => updateCueTime(cue.id, "start", event.target.value)}
                  />
                  <span>–</span>
                  <input
                    value={formatTime(cue.end)}
                    title="结束时间（分:秒，如 0:05.0）"
                    disabled={!job?.result || running || cue.deleted}
                    onChange={(event) => updateCueTime(cue.id, "end", event.target.value)}
                  />
                  <button className="cue-jump" onClick={() => seek(cue)} title="跳转到此字幕">
                    ▶
                  </button>
                </div>
                <p>{cue.source}</p>
                <textarea
                  value={cue.translated}
                  onFocus={() => setActiveCue(cue.id)}
                  onChange={(event) => updateCue(cue.id, event.target.value)}
                  disabled={!job?.result || running || cue.deleted}
                />
                <button
                  className={cue.deleted ? "cue-restore" : "cue-delete"}
                  onClick={() => toggleCueDeleted(cue.id)}
                  disabled={running || !job?.result}
                  title={cue.deleted ? "恢复这条字幕" : "删除这条字幕（此段不再显示字幕）"}
                >
                  {cue.deleted ? "恢复" : "删除"}
                </button>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-editor">
            <b>等待字幕结果</b>
            <span>
              开始处理后，这里会显示可播放、可编辑、可保存的真实字幕。
            </span>
          </div>
        )}
      </section>
    </>
  );
  const publishView = (
    <section className="workspace-card">
      <header>
        <p className="eyebrow">B 站发布辅助</p>
        <h1>准备文案，人工确认后投稿</h1>
        <p>工具只打开页面、选择视频并辅助填写；不会自动点击“立即投稿”。</p>
      </header>
      <div className="form-grid three">
        <label>
          浏览器
          <select
            value={publishBrowser}
            onChange={(event) => setPublishBrowser(event.target.value)}
          >
            <option value="chromium">Chromium</option>
            <option value="msedge">Microsoft Edge</option>
          </select>
        </label>
        <label>
          文案模板
          <select
            value={templateName}
            onChange={(event) => {
              if (
                templateDraftDirty &&
                !window.confirm(
                  `「模板内容」中有未保存的草稿修改。切换模板会丢失这些修改，确定继续吗？`,
                )
              )
                return;
              setTemplateName(event.target.value);
              setTemplateBody(
                templates.find((item) => item.name === event.target.value)
                  ?.body ?? "",
              );
            }}
          >
            {templates.map((item) => (
              <option key={item.name}>{item.name}</option>
            ))}
          </select>
        </label>
        <label>
          成品来源
          <select
            value={selectedPublishVideo}
            onChange={(event) => { setSelectedPublishVideo(event.target.value); setNativePublishToken(""); setNativePublishName(""); }}
          >
            <option value="">当前任务成片（如已完成）</option>
            {nativePublishToken && <option value="__native__">本地选择 · {nativePublishName}</option>}
            {outputs?.tasks.flatMap((task) =>
              task.files
                .filter((file) => file.kind === "成品视频")
                .map((file) => (
                  <option key={file.id} value={file.id}>
                    {file.name}
                  </option>
                )),
            )}
          </select>
        </label>
      </div>
      <div className="inline-actions">
        <button onClick={publishCheck}>登录检查</button>
        <button onClick={() => void loadPublishStatus()}>检查 Profile</button>
        <button onClick={() => void chooseNativePublishVideo()}>选择任意本地成片</button>
        <button onClick={generateDescription}>更新文案</button>
        <button onClick={() => openTemplateEditor()}>新建/编辑模板</button>
        <button onClick={deleteTemplate}>删除自定义模板</button>
      </div>
      {profileMessage && <p className="publish-status" role="status">{profileMessage}</p>}
      <div className="template-draft-bar">
        {templateDraftDirty && (
          <span className="dirty">
            草稿未保存：点击「更新文案」会优先使用当前草稿内容生成简介。
          </span>
        )}
        {templateDraftDirty && (
          <button onClick={() => setTemplateBody(savedTemplateBody)}>恢复模板内容</button>
        )}
        <button onClick={() => openTemplateEditor()}>新建/保存为模板</button>
      </div>
      <label>
        模板内容
        <textarea
          value={templateBody}
          onChange={(event) => setTemplateBody(event.target.value)}
          placeholder="模板支持 {source} 与 {custom_text}"
        />
      </label>
      <label>
        简介
        <textarea
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="生成后可继续人工修改"
        />
      </label>
      <div className="form-grid two">
        <label>
          标签（逗号分隔）
          <input
            value={tags}
            onChange={(event) => setTags(event.target.value)}
          />
        </label>
        <label>
          追加备注
          <input
            value={extraLine}
            onChange={(event) => setExtraLine(event.target.value)}
          />
        </label>
      </div>
      <div className="publish-warning">
        <label className="check">
          <input type="checkbox" checked={includeSourceLink} onChange={(event) => setIncludeSourceLink(event.target.checked)} />
          简介自动附原视频链接
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={closeAfterFill}
            onChange={(event) => setCloseAfterFill(event.target.checked)}
          />
          填表完成后关闭自动化浏览器（上传可能中断）
        </label>
        <span>投稿前请检查分区、创作声明、封面、简介、标签与平台提示。</span>
      </div>
      {(publishSession.active || publishSession.status !== "idle") && (
        <div className={`publish-session ${publishSession.status}`}>
          <div className="publish-session-head">
            <b>
              {publishSession.status === "waiting_review"
                ? "已到人工提交前"
                : publishSession.status === "failed"
                  ? "辅助失败"
                  : publishSession.status === "finished"
                    ? "辅助已结束"
                    : "B 站投稿辅助进行中"}
            </b>
            <span>{publishSession.active ? "运行中" : "已停止"}</span>
          </div>
          <p className="publish-session-message">{publishSession.message}</p>
          {publishSession.error && <p className="error">{publishSession.error}</p>}
          {publishSession.logs.length > 0 && (
            <div className="publish-session-logs">
              {publishSession.logs.slice(-6).map((line, index) => (
                <p key={`${line}-${index}`}>{line}</p>
              ))}
            </div>
          )}
        </div>
      )}
      <footer className="card-footer">
        <div className="publish-submit-hint">
          {!publishVideoReady && <span>请先选择一份可投稿的成片。</span>}
        </div>
        <button className="primary" onClick={publishAssist} disabled={!publishVideoReady} title={!publishVideoReady ? "请先选择成片" : "打开 B 站投稿辅助"}>
          打开投稿辅助
        </button>
        <button className="danger" onClick={closePublishBrowser}>
          关闭后台 B 站浏览器
        </button>
      </footer>
      {templateEditorOpen && <div className="modal-backdrop" role="presentation">
        <section className="template-modal" role="dialog" aria-modal="true" aria-label="编辑文案模板">
          <header><div><p className="eyebrow">文案模板</p><h2>新建或编辑模板</h2></div><button onClick={() => setTemplateEditorOpen(false)} aria-label="关闭模板编辑器">×</button></header>
          <label>模板名称<input value={templateDraftName} placeholder="例如：授权转载说明" onChange={(event) => setTemplateDraftName(event.target.value)} /></label>
          <label>模板内容<textarea value={templateBody} onChange={(event) => setTemplateBody(event.target.value)} /></label>
          <div className="inline-actions"><button onClick={() => insertTemplateVariable("{source}")}>{"{source}"}</button><button onClick={() => insertTemplateVariable("{custom_text}")}>{"{custom_text}"}</button></div>
          <div className="template-preview"><b>预览效果</b><p>{(templateBody || "填写模板内容后自动预览").replaceAll("{source}", sourceUrl || "https://example.com/video").replaceAll("{custom_text}", "自定义内容")}</p></div>
          <footer><button onClick={() => setTemplateEditorOpen(false)}>取消</button><button className="primary" onClick={saveTemplate}>保存模板</button></footer>
        </section>
      </div>}
    </section>
  );
  const viewedTask = outputs?.tasks.find((task) => task.id === viewedTaskId) ?? null;
  const filesView = (
    <section className="workspace-card files-view">
      <header>
        <p className="eyebrow">输出与文件管理</p>
        <h1>管理每次处理产生的文件</h1>
        <p>
          {outputs
            ? `${outputs.task_count} 个任务 · 共 ${outputs.total_size}`
            : "正在读取输出目录…"}
        </p>
      </header>
      <div className="inline-actions">
        <label className="path-input">
          输出目录
          <div className="output-dir-control">
            <input
              value={options.output_dir}
              onChange={(event) =>
                updateOptions("output_dir", event.target.value)
              }
            />
            <button onClick={() => void chooseOutputDir()}>选择文件夹</button>
          </div>
        </label>
        <button onClick={loadOutputs}>刷新</button>
        <button onClick={openOutput}>打开目录</button>
        <button
          className="danger"
          disabled={!selectedOutputPaths.length}
          onClick={deleteOutputs}
        >
          删除选中项
        </button>
      </div>
      <div className="storage-summary">
        {["下载/原视频", "成品视频", "字幕", "中间文件", "预览图", "其他"].map((kind) => <span key={kind}><b>{outputs?.category_totals[kind]?.label ?? "0 B"}</b>{kind}</span>)}
      </div>
      <section className="history-section">
        <header>
          <div>
            <b>任务历史</b>
            <small>{historyTotal ? `${historyScope === "current" ? "当前输出目录" : "全部目录"} · ${historyTotal} 条` : "暂无历史任务"}</small>
          </div>
          <label className="history-scope">
            范围
            <select value={historyScope} onChange={(event) => setHistoryScope(event.target.value as "current" | "all") }>
              <option value="current">当前输出目录</option>
              <option value="all">全部历史</option>
            </select>
          </label>
          <button onClick={() => void loadJobHistory()}>刷新</button>
          <button className="history-clear" onClick={() => void clearHistory()} disabled={!historyTotal}>清除记录</button>
        </header>
        {jobHistory.length > 0 && (
          <div className="history-table">
            {jobHistory.map((item) => (
              <article key={item.id} className="history-row">
                <span className={`history-status ${item.status}`}>
                  {item.status === "completed" ? "完成" : item.status === "failed" ? "失败" : item.status === "cancelled" ? "已取消" : item.status}
                </span>
                <b title={item.title}>{item.title}</b>
                <small>{item.created_at ? new Date(item.created_at * 1000).toLocaleString() : "—"}</small>
                <small>{item.finished_at && item.created_at ? `${Math.max(1, Math.round(item.finished_at - item.created_at))}s` : ""}</small>
                <button onClick={() => void openHistoryPath(item.output_dir || "")} disabled={!item.output_exists} title={item.output_exists ? "打开任务输出目录" : "输出目录已不存在"}>
                  打开输出
                </button>
                <button onClick={() => void openHistoryPath(item.rendered_video || "")} disabled={!item.rendered_exists} title={item.rendered_exists ? "打开成品视频" : "成品视频未生成或已删除"}>
                  成品
                </button>
              </article>
            ))}
          </div>
        )}
      </section>
      <p className="file-manager-legend">
        说明：「查看」按钮用于查看某个任务的文件；「☐ 删除」勾选后点上方「删除选中项」。点击任务名称仅查看，不会删除任何内容。
      </p>
      <div className="file-manager">
        <section className="task-pane">
          <header><b>1. 任务列表</b><small>点「查看」显示该任务文件</small></header>
          {outputs?.tasks.length ? outputs.tasks.map((task) => (
            <article key={task.id} className={viewedTaskId === task.id ? "task-row selected" : "task-row"}>
              <label className="delete-check" title="勾选后点上方“删除选中项”"><input type="checkbox" checked={selectedOutputPaths.includes(task.id)} onChange={(event) => setSelectedOutputPaths((values) => event.target.checked ? [...values, task.id] : values.filter((value) => value !== task.id))} /><small>删除</small></label>
              <div><b>{task.name}</b><small>{task.files.length} 个文件 · {new Date(task.modified_at * 1000).toLocaleString()}</small></div>
              <span>{task.size_label}</span>
              <button className="view-button" onClick={() => setViewedTaskId(task.id)}>查看</button>
              <button onClick={(event) => { event.stopPropagation(); void openOutputPath(task.id); }}>打开</button>
            </article>
          )) : <div className="empty-editor"><b>还没有输出任务</b><span>完成一次处理后，文件会显示在这里。</span></div>}
        </section>
        <section className="task-pane">
          <header><b>2. 文件列表</b><small>{viewedTask ? viewedTask.name : "尚未选择任务"}</small></header>
          {viewedTask ? viewedTask.files.map((file) => (
            <article key={file.id} className="file-row">
              <label className="delete-check" title="勾选后点上方“删除选中项”"><input type="checkbox" checked={selectedOutputPaths.includes(file.id)} onChange={(event) => setSelectedOutputPaths((values) => event.target.checked ? [...values, file.id] : values.filter((value) => value !== file.id))} /><small>删除</small></label>
              <span>{file.kind}</span>
              <b>{file.name}</b>
              <small>{file.size_label}</small>
              <button onClick={() => void openOutputPath(file.id)}>打开</button>
            </article>
          )) : <div className="empty-editor"><b>等待选择任务</b><span>点击左侧任务的「查看」按钮查看该任务的文件。</span></div>}
        </section>
      </div>
    </section>
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#workspace">
          <span>▶</span>
          <b>
            YouTube Bili
            <br />
            Localizer
          </b>
        </a>
        <div className="project-name">
          <strong>{job?.material.name ?? selected?.name ?? "等待素材"}</strong>
          <span>
            {running
              ? `${job?.stage} ${job?.progress}% · ${formatElapsed(job?.started_at ? (now - job.started_at * 1000) / 1000 : 0)}`
              : selected?.duration_seconds
                ? `${selected.duration_seconds.toFixed(1)} 秒`
                : "本地工作台"}
          </span>
        </div>
        <div className="top-actions">
          <span className={online ? "status online" : "status"}>
            {online ? "本地已连接" : "未连接"}
          </span>
          <button className="text-button" onClick={() => setSettingsOpen(true)}>
            设置
          </button>
          <button className="text-button" onClick={() => setShortcutHelpOpen(true)}>
            帮助
          </button>
          <button className="text-button" onClick={() => setInspectorOpen((value) => !value)}>
            检查器
          </button>
          <button
            className="text-button"
            onClick={toggleMode}
            title="切换基础/高级模式：高级模式显示全部处理参数"
          >
            {mode === "basic" ? "切到高级模式" : "切到基础模式"}
          </button>
          <button
            className="primary"
            onClick={running ? cancel : start}
            disabled={running ? !job : !selected && !sourceUrl.trim()}
          >
            {running ? "中断任务" : "开始处理"}
          </button>
        </div>
      </header>
      <input
        ref={fileInput}
        className="hidden-input"
        type="file"
        accept="video/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) importVideo(file);
          event.currentTarget.value = "";
        }}
      />
      <input
        ref={publishFileInput}
        className="hidden-input"
        type="file"
        accept="video/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void importPublishVideo(file);
          event.currentTarget.value = "";
        }}
      />
      <section
        className={`workbench ${inspectorOpen ? "inspector-open" : ""}`}
        id="workspace"
      >
        <nav className="side-nav" aria-label="工作区导航">
          <p className="eyebrow">工作区</p>
          {nav.map(([id, label, hint]) => (
            <button
              key={id}
              className={view === id ? "active" : ""}
              onClick={() => setView(id)}
            >
              <b>{label}</b>
              <small>{hint}</small>
            </button>
          ))}
          <div className="nav-status">
            <span>授权确认</span>
            <b>{authorized ? "已确认" : "待确认"}</b>
          </div>
        </nav>
        <section className="main-workspace">
          {view === "material" && materialView}
          {view === "process" && processView}
          {view === "subtitle" && subtitleView}
          {view === "publish" && publishView}
          {view === "files" && filesView}
        </section>
        {inspectorOpen && (
          <aside className="inspector">
            <div className="inspector-head">
              <div>
                <p className="eyebrow">任务检查器</p>
                <h2>{job?.stage ?? "等待任务"}</h2>
              </div>
              <button
                onClick={() => setInspectorOpen(false)}
                aria-label="收起检查器"
              >
                ×
              </button>
            </div>
            {job && (
              <>
                <div className="progress-area">
                  <b>{job.progress}%</b>
                  <small>{formatElapsed(job.finished_at && job.started_at ? job.finished_at - job.started_at : job.started_at ? (now - job.started_at * 1000) / 1000 : job.elapsed_seconds)}</small>
                  <div className="progress">
                    <i style={{ width: `${job.progress}%` }} />
                  </div>
                </div>
                {running && job.stage === "准备素材" && (
                  <p className="progress-hint">
                    正在下载视频：进度来自真实网络字节。若长时间停在 12% 以下不变，可能是网络较慢或被平台拦截，可到「设置」配置 Cookies 后重试。
                  </p>
                )}
                <dl className="job-options">
                  <div>
                    <dt>转写</dt>
                    <dd>{job.options.whisper_model_size}</dd>
                  </div>
                  <div>
                    <dt>翻译</dt>
                    <dd>{job.options.translator}</dd>
                  </div>
                  <div>
                    <dt>字幕源</dt>
                    <dd>{job.options.subtitle_source}</dd>
                  </div>
                  <div>
                    <dt>设备</dt>
                    <dd>
                      {job.device} · {job.compute_type}
                    </dd>
                  </div>
                </dl>
                <section className="compact-log">
                  <div className="log-head"><p className="eyebrow">{showFullLogs ? "完整日志" : "最近日志"}</p><div className="log-actions"><CopyButton text={buildDebugText()} label="复制调试信息" /><button onClick={() => setShowFullLogs((value) => !value)}>{showFullLogs ? "收起" : `查看全部 ${job.logs.length} 行`}</button></div></div>
                  {(showFullLogs ? job.logs : job.logs
                    .slice(-8)
                    .reverse()).map((line, index) => (
                      <p key={`${line}-${index}`}>{line}</p>
                    ))}
                </section>
                {job.error && (
                  <div className="error-block">
                    <p className="error">{job.error}</p>
                    <div className="error-actions">
                      <CopyButton text={job.error} label="复制错误" />
                      {cookieHintMatches(job.error) && (
                        <button className="cookie-hint-button" onClick={() => setSettingsOpen(true)}>
                          去设置 → YouTube Cookies 来源
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </aside>
        )}
      </section>
      {settingsOpen && (
        <div className="settings-drawer" role="dialog" aria-modal="true">
          <div className="drawer-head">
            <div>
              <p className="eyebrow">本地设置</p>
              <h2>密钥与浏览器权限</h2>
            </div>
            <button onClick={() => setSettingsOpen(false)}>×</button>
          </div>
          <p>密钥只会写入本机 `.env`，不会通过接口重新读取或显示。</p>
          <label>
            当前翻译器
            <select
              value={options.translator}
              onChange={(event) =>
                updateOptions("translator", event.target.value)
              }
            >
              <option value="deepseek">DeepSeek</option>
              <option value="openai">OpenAI</option>
            </select>
          </label>
          <label>
            API Key
            <input
              type="password"
              value={apiKey}
              placeholder="仅在需要更新时填写"
              onChange={(event) => setApiKey(event.target.value)}
            />
          </label>
          <label>
            YouTube Cookies 来源
            <select
              value={options.cookies_from_browser}
              disabled={!!options.cookies_file}
              onChange={(event) => {
                updateOptions("cookies_from_browser", event.target.value);
                if (event.target.value) updateOptions("cookies_file", "");
              }}
            >
            <option value="">不使用 Cookies</option>
            <option value="firefox">Firefox（推荐）</option>
            </select>
            {!!options.cookies_file && (
              <small className="drawer-note">
                当前已使用 cookies.txt 文件，浏览器来源已禁用；如需改用浏览器，请先清空下面的文件路径。
              </small>
            )}
            {!!options.cookies_file && cookiesFileValid === false && (
              <small className="drawer-note warn">
                警告：该 cookies 文件缺少登录凭据（未找到 SID/LOGIN_INFO），很可能是未登录时导出的，YouTube 下载会被拦截。请确认已登录后重新导出并保存。
              </small>
            )}
          </label>
          <label>
            Cookies 文件（cookies.txt）
            <div className="output-dir-control">
              <input
                value={options.cookies_file}
                placeholder="选择或输入 cookies.txt 路径"
                disabled={!!options.cookies_from_browser}
                onChange={(event) => {
                  updateOptions("cookies_file", event.target.value);
                  if (event.target.value) updateOptions("cookies_from_browser", "");
                }}
              />
              <button onClick={() => void chooseCookiesFile()} disabled={!!options.cookies_from_browser}>
                选择文件
              </button>
            </div>
            {!!options.cookies_from_browser && (
              <small className="drawer-note">
                已选择浏览器来源，请先改为「不使用 Cookies」后再设置文件。
              </small>
            )}
          </label>
          <p className="cookies-hint">
            提示：Chrome / Edge 130 及以上版本对 Cookies 做了应用绑定加密，无法直接读取，所以这里只保留 Firefox 选项；Chrome 用户请用浏览器插件（如 Get cookies.txt LOCALLY）导出 cookies.txt 后选择下面的文件。两种来源二选一。
          </p>
          <button className="primary" onClick={saveSettings}>
            保存本地设置
          </button>
        </div>
      )}
      {shortcutHelpOpen && (
        <div className="modal-backdrop" role="presentation">
          <section className="template-modal help-modal" role="dialog" aria-modal="true" aria-label="使用帮助">
            <header><div><p className="eyebrow">使用帮助</p><h2>开始前检查与快捷键</h2></div><button onClick={() => setShortcutHelpOpen(false)} aria-label="关闭帮助">×</button></header>
            <section className="help-section">
              <b>本机依赖</b>
              {diagnostics.length ? diagnostics.map((check) => <p key={check.id} className={check.available ? "help-ok" : "help-missing"}><strong>{check.available ? "✓" : "!"} {check.label}</strong> · {check.purpose}<br /><span>{check.message}</span></p>) : <p>正在检测 FFmpeg、Node.js 与 Tesseract…</p>}
            </section>
            <section className="help-section">
              <b>快捷键</b>
              <dl className="shortcut-list"><div><dt>Ctrl + O</dt><dd>选择本地视频（先确认授权）</dd></div><div><dt>Ctrl + L</dt><dd>回到素材区并聚焦视频链接</dd></div><div><dt>Ctrl + R</dt><dd>开始处理（输入框中不会触发）</dd></div><div><dt>Esc</dt><dd>请求中断正在运行的任务</dd></div></dl>
            </section>
            <footer><button className="primary" onClick={() => setShortcutHelpOpen(false)}>知道了</button></footer>
          </section>
        </div>
      )}
      <footer className="footer">
        <div className="footer-msg">
          <span>{message}</span>
          <CopyButton text={message} label="复制消息" />
        </div>
        <span>视频、字幕、浏览器 Profile 与密钥均留在本机</span>
      </footer>
      {ctxMenu && (
        <div className="ctx-menu" style={{ left: ctxMenu.x, top: ctxMenu.y }} onClick={(event) => event.stopPropagation()}>
          <button onClick={copySelection}>复制选中文本</button>
          <button onClick={selectAllText}>全选页面文本</button>
          <button onClick={() => setCtxMenu(null)}>取消</button>
        </div>
      )}
    </main>
  );
}

export default App;

import { useEffect, useMemo, useRef, useState } from "react";
import { stageAvailability, workflowPrimaryLabel } from "./workflowState";
import { videoLibraryCategory, videoLibraryMatches, type VideoLibraryCategory } from "./videoLibrary";

type View = "material" | "library" | "process" | "subtitle" | "publish" | "files";
type WorkflowStageName = "acquire" | "extract" | "translate" | "render" | "publish";
type WorkflowStage = {
  status: "pending" | "ready" | "running" | "completed" | "failed" | "cancelled" | "stale" | "interrupted";
  progress: number;
  error: string | null;
  started_at: number | null;
  finished_at: number | null;
  config_fingerprint: string | null;
};
type PreparationCheck = {
  id: string;
  label: string;
  purpose: string;
  status: "passed" | "warning" | "blocking" | "installing";
  message: string;
  action?: string | null;
  needs_recheck: boolean;
};
type Material = {
  id: string;
  name: string;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  authorized: boolean;
  is_demo: boolean;
  source_url?: string | null;
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
  log_cursor?: number;
  log_total?: number;
  error: string | null;
  error_code?: string | null;
  suggested_action?: string | null;
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
  workflow_version?: number;
  stages?: Record<WorkflowStageName, WorkflowStage>;
  checks?: PreparationCheck[];
  current_stage?: WorkflowStageName | null;
  next_stage?: WorkflowStageName | null;
  can_resume?: boolean;
  auto_run?: boolean;
  artifacts?: Record<string, { available: boolean; name: string | null; bytes: number }>;
  artifact_revision?: number;
  edit_state?: string;
  checkpoint_validation?: "pending" | "verified" | "invalid";
  subtitle_extraction?: {
    mode: string;
    ocr_status: "not-requested" | "completed" | "fallback" | "failed";
    message: string | null;
  };
  content_warnings?: string[];
};
type Options = {
  title: string;
  require_reuse_allowed: boolean;
  cookies_from_browser: string;
  cookies_file: string;
  youtube_po_token_mode: "auto" | "off";
  youtube_proxy: string;
  download_quality: "720p" | "1080p" | "original";
  max_seconds: number | null;
  subtitle_source: string;
  prefer_platform_subtitles: boolean;
  whisper_model_size: string;
  source_language: string;
  beam_size: number;
  ocr_interval: number;
  ocr_crop_ratio: number;
  ocr_min_chars: number;
  ocr_language: string;
  subtitle_margin_ratio: number;
  render_crf: number;
  render_encoder: "auto" | "cpu" | "nvidia";
  resource_profile: "background" | "balanced" | "maximum";
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
  description?: string;
  tags?: string[];
  publish_to_bilibili?: boolean;
  include_source_link?: boolean;
  bilibili_browser?: string;
  close_after_fill?: boolean;
};

function mergeRestoredOptions(current: Options, restored: Partial<Options> | null | undefined): Options {
  const merged = { ...current } as Record<string, unknown>;
  for (const [key, value] of Object.entries(restored ?? {})) {
    // Persisted jobs intentionally store sensitive/optional settings as null.
    // Keep the live local/default value instead of letting null escape into form state.
    if (value !== null && value !== undefined) merged[key] = value;
  }
  return merged as Options;
}
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
type RestorableJob = {
  id: string;
  title: string;
  status: string;
  next_stage: WorkflowStageName | null;
  updated_at: number;
  checkpoint_validation: "pending" | "verified" | "invalid";
};
type TestHistoryCandidate = {
  id: string;
  title: string;
  status: string;
  output_dir: string;
  reason: string;
  created_at: number | null;
};
type Diagnostic = {
  id: string;
  label: string;
  purpose: string;
  available: boolean;
  message: string;
  required?: boolean;
};
type Dependency = Diagnostic & {
  installable: boolean;
  size_hint: string;
  action_url: string | null;
};
type DependencyInstallJob = {
  id: string;
  dependency_id: string;
  status: "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
  progress: number;
  progress_kind: "indeterminate" | "determinate";
  message: string;
  logs: string[];
  error: string | null;
  cancel_requested: boolean;
};

const defaults: Options = {
  title: "",
  require_reuse_allowed: false,
  cookies_from_browser: "",
  cookies_file: "",
  youtube_po_token_mode: "auto",
  youtube_proxy: "",
  download_quality: "1080p",
  max_seconds: 10,
  subtitle_source: "audio",
  prefer_platform_subtitles: true,
  whisper_model_size: "small",
  source_language: "",
  beam_size: 5,
  ocr_interval: 0.5,
  ocr_crop_ratio: 0.3,
  ocr_min_chars: 3,
  ocr_language: "eng",
  subtitle_margin_ratio: 0.055,
  render_crf: 20,
  render_encoder: "auto",
  resource_profile: "balanced",
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
  ["library", "任务库", "待处理、可继续与成片"],
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
  const seconds = parsed * (unit === "小时" ? 3600 : unit === "分钟" ? 60 : 1);
  return Number.isInteger(seconds) && seconds <= 86_400 ? seconds : null;
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

const workflowStageMeta: Array<[WorkflowStageName, string, string]> = [
  ["acquire", "获取素材", "下载 URL 视频；本地文件会直接登记为可用素材。"],
  ["extract", "字幕提取", "提取音频，再按方案执行 Whisper、OCR 或合并。"],
  ["translate", "翻译", "校正源字幕，生成中文字幕和投稿标题标签。"],
  ["render", "渲染", "使用 FFmpeg 生成 SRT 与硬字幕成片。"],
  ["publish", "发布辅助", "可选；上传并填表，始终停在人工提交前。"],
];
const stageStatusLabel: Record<string, string> = {
  pending: "待执行", ready: "可执行", running: "运行中", completed: "已完成",
  failed: "失败", cancelled: "已取消", stale: "需重做", interrupted: "意外中断",
};

function FieldGuide({ summary, detail, badges = [] }: { summary: string; detail: string; badges?: string[] }) {
  return <span className="field-guide"><small>{summary}</small><span className="guide-badges">{badges.map((badge) => <i key={badge}>{badge}</i>)}</span><details><summary>？ 选择建议与影响</summary><p>{detail}</p></details></span>;
}
declare global {
  interface Window { pywebview?: { api?: NativeBridge } }
}

let csrfToken = "";

class ApiError extends Error {
  field?: string;
  code?: string;

  constructor(message: string, field?: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.field = field;
    this.code = code;
  }
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const form = options?.body instanceof FormData;
  const method = (options?.method ?? "GET").toUpperCase();
  const mutation = !["GET", "HEAD", "OPTIONS"].includes(method);
  const response = await fetch(path, {
    ...options,
    headers: form
      ? { ...(mutation && csrfToken ? { "X-YBL-CSRF": csrfToken } : {}), ...(options?.headers ?? {}) }
      : { "Content-Type": "application/json", ...(mutation && csrfToken ? { "X-YBL-CSRF": csrfToken } : {}), ...(options?.headers ?? {}) },
  });
  const payload = (await response.json()) as T & { error?: string; field?: string; code?: string };
  if (!response.ok) throw new ApiError(payload.error ?? "本地服务请求失败", payload.field, payload.code);
  return payload;
}

function App() {
  const fileInput = useRef<HTMLInputElement>(null);
  const publishFileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const jobLogCursor = useRef(0);
  const [view, setView] = useState<View>("material");
  const [materials, setMaterials] = useState<Material[]>([]);
  const [materialId, setMaterialId] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [authorized, setAuthorized] = useState(false);
  const [fullVideo, setFullVideo] = useState(false);
  const [durationValue, setDurationValue] = useState("10");
  const [durationUnit, setDurationUnit] = useState<"秒" | "分钟" | "小时">("秒");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [device, setDevice] = useState("cuda");
  const [computeType, setComputeType] = useState("float16");
  const [options, setOptions] = useState<Options>(defaults);
  const [appVersion, setAppVersion] = useState("0.2.6");
  const [demoPreview, setDemoPreview] = useState(false);
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
  const [dependenciesOpen, setDependenciesOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [message, setMessage] = useState("正在连接本地服务…");
  const [metadata, setMetadata] = useState<{
    duration: number | null;
    license: string | null;
    view_count: number | null;
    max_width: number | null;
    max_height: number | null;
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
  const [fullLogs, setFullLogs] = useState<string[]>([]);
  const [fullLogsTruncated, setFullLogsTruncated] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [readiness, setReadiness] = useState<string[]>([]);
  const [preflightWarnings, setPreflightWarnings] = useState<string[]>([]);
  const [preparingJob, setPreparingJob] = useState(false);
  const [outputs, setOutputs] = useState<Outputs | null>(null);
  const [selectedOutputPaths, setSelectedOutputPaths] = useState<string[]>([]);
  const [selectedOutputTaskIds, setSelectedOutputTaskIds] = useState<string[]>([]);
  const [selectedPublishVideo, setSelectedPublishVideo] = useState("");
  const [mode, setMode] = useState<"basic" | "advanced">(
    () => (localStorage.getItem("ybl-mode") as "basic" | "advanced") || "basic",
  );
  const [apiKeyConfigured, setApiKeyConfigured] = useState<boolean | null>(null);
  const [cookiesFileValid, setCookiesFileValid] = useState<boolean | null>(null);
  const [poTokenStatus, setPoTokenStatus] = useState<{ available: boolean; browser_path: string | null } | null>(null);
  const [publishSession, setPublishSession] = useState<{
    status: string;
    message: string;
    logs: string[];
    error: string | null;
    active: boolean;
  }>({ status: "idle", message: "", logs: [], error: null, active: false });
  const [viewedTaskId, setViewedTaskId] = useState<string | null>(null);
  const [jobHistory, setJobHistory] = useState<HistoryJob[]>([]);
  const [restorableJobs, setRestorableJobs] = useState<RestorableJob[]>([]);
  const [testHistoryCandidates, setTestHistoryCandidates] = useState<TestHistoryCandidate[]>([]);
  const [selectedTestHistoryIds, setSelectedTestHistoryIds] = useState<string[]>([]);
  const [testHistoryOpen, setTestHistoryOpen] = useState(false);
  const [historyScope, setHistoryScope] = useState<"current" | "all">("all");
  const [historyTotal, setHistoryTotal] = useState(0);
  const [libraryFilter, setLibraryFilter] = useState<"all" | VideoLibraryCategory>("all");
  const [libraryQuery, setLibraryQuery] = useState("");
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [dependencies, setDependencies] = useState<Dependency[]>([]);
  const [dependencyJob, setDependencyJob] = useState<DependencyInstallJob | null>(null);
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
  const [proxyConfigured, setProxyConfigured] = useState(false);
  const [proxyDirty, setProxyDirty] = useState(false);

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
    (demoPreview ? 10 : null) ??
    job?.material.duration_seconds ??
    selected?.duration_seconds ??
    metadata?.duration ??
    10;
  const previewSrc = demoPreview
    ? "/api/demo/rendered"
    : job
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
  const librarySourceJobs = useMemo(() => {
    const rows = [...jobHistory];
    if (job) {
      const current: HistoryJob = {
        id: job.id,
        title: job.options.title || job.material.name || "未命名任务",
        status: job.status,
        stage: job.stage,
        progress: job.progress,
        error: job.error,
        output_dir: job.result?.output_dir ?? null,
        rendered_video: job.result?.rendered_video ?? null,
        output_exists: Boolean(job.result?.output_dir),
        rendered_exists: Boolean(job.result?.rendered_video),
        created_at: job.started_at,
        finished_at: job.finished_at,
      };
      const currentIndex = rows.findIndex((item) => item.id === job.id);
      if (currentIndex >= 0) rows[currentIndex] = { ...rows[currentIndex], ...current };
      else rows.unshift(current);
    }
    return rows;
  }, [jobHistory, job]);
  const libraryJobs = useMemo(() => {
    const resumableIds = new Set(restorableJobs.map((item) => item.id));
    return librarySourceJobs.filter((item) => {
      const category = videoLibraryCategory(item, resumableIds.has(item.id));
      return (libraryFilter === "all" || category === libraryFilter)
        && videoLibraryMatches(item, libraryQuery);
    });
  }, [librarySourceJobs, restorableJobs, libraryFilter, libraryQuery]);
  const libraryCounts = useMemo(() => {
    const resumableIds = new Set(restorableJobs.map((item) => item.id));
    const counts: Record<VideoLibraryCategory, number> = { active: 0, attention: 0, completed: 0 };
    for (const item of librarySourceJobs) {
      counts[videoLibraryCategory(item, resumableIds.has(item.id))] += 1;
    }
    return counts;
  }, [librarySourceJobs, restorableJobs]);

  const updateOptions = <K extends keyof Options>(key: K, value: Options[K]) => {
    setFieldErrors((current) => {
      if (!current[String(key)]) return current;
      const next = { ...current };
      delete next[String(key)];
      return next;
    });
    setOptions((current) => ({ ...current, [key]: value }));
  };
  const reportApiError = (error: unknown, fallback: string) => {
    if (error instanceof ApiError && error.field) {
      setFieldErrors((current) => ({ ...current, [error.field as string]: error.message }));
    }
    return error instanceof Error ? error.message : fallback;
  };
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
      youtube_proxy_configured?: boolean;
      youtube_po_token?: { available: boolean; browser_path: string | null };
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
    setPoTokenStatus(settings.youtube_po_token ?? null);
    setApiKeyConfigured(Boolean(settings.deepseek_key_configured));
    setProxyConfigured(Boolean(settings.youtube_proxy_configured));
  };
  const loadRestorableJobs = async () => {
    const response = await api<{ jobs: RestorableJob[] }>("/api/jobs/restorable?limit=20");
    setRestorableJobs(response.jobs);
  };
  const loadRestoredJob = async (id: string) => {
    try {
      const restored = await api<Job>(`/api/jobs/${id}/load`, { method: "POST", body: "{}" });
      setJob(restored);
      setDemoPreview(false);
      setOptions((current) => ({
        ...mergeRestoredOptions(current, restored.options),
        cookies_file: current.cookies_file,
        youtube_proxy: current.youtube_proxy,
      }));
      setDevice(restored.device);
      setComputeType(restored.compute_type);
      setSourceUrl(restored.material.source_url ?? "");
      setAuthorized(restored.material.authorized);
      if (restored.material.source_url) {
        const restoredLimit = restored.options?.max_seconds;
        setFullVideo(restoredLimit === null);
        if (typeof restoredLimit === "number" && Number.isInteger(restoredLimit) && restoredLimit > 0) {
          setDurationValue(String(restoredLimit));
          setDurationUnit("秒");
        }
      }
      setDescription(restored.options.description ?? "");
      setTags((restored.options.tags ?? []).join(","));
      setPublishInPipeline(Boolean(restored.options.publish_to_bilibili));
      setIncludeSourceLink(restored.options.include_source_link ?? true);
      setPublishBrowser(restored.options.bilibili_browser ?? "chromium");
      setCloseAfterFill(Boolean(restored.options.close_after_fill));
      setPreview(restored.artifacts?.rendered_video?.available ? "rendered" : "source");
      await loadCues(restored.id);
      setView("process");
      const completedStageCount = Object.values(restored.stages ?? {}).filter((stage) => stage.status === "completed").length;
      setMessage(restored.edit_state === "legacy-ambiguous"
        ? "已载入旧任务并保留现有成片；后期版本顺序无法确认，已禁用重新渲染。"
        : restored.status === "draft" && completedStageCount === 0
          ? "任务草稿已载入。请运行准备检查；当前不会自动下载或占用大量电脑资源。"
        : restored.checkpoint_validation === "invalid"
          ? "已载入任务，但部分检查点无效；请先运行准备检查。"
          : "任务已载入。不会自动继续耗资源处理，请确认后再执行下一阶段。");
      await loadRestorableJobs();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "载入任务失败");
    }
  };
  const loadJobHistory = async () => {
    try {
      const query = new URLSearchParams({
        limit: "200",
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
  const loadDependencies = async () => {
    const response = await api<{
      dependencies: Dependency[];
      active_job: DependencyInstallJob | null;
    }>("/api/dependencies");
    setDependencies(response.dependencies);
    setDependencyJob((current) => response.active_job ?? (current && ["completed", "failed", "cancelled"].includes(current.status) ? current : null));
    setDiagnostics(response.dependencies.map(({ installable: _installable, size_hint: _size, action_url: _url, ...item }) => item));
  };
  const installDependency = async (dependency: Dependency) => {
    if (!window.confirm(`安装“${dependency.label}”吗？\n\n用途：${dependency.purpose}\n预计占用：${dependency.size_hint}\n\n安装过程需要联网，Windows 可能显示权限提示。`)) return;
    try {
      const response = await api<{ job: DependencyInstallJob }>(`/api/dependencies/${dependency.id}/install`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true }),
      });
      setDependencyJob(response.job);
      setDependenciesOpen(true);
      setMessage(response.job.message);
      if (response.job.status === "completed") await loadDependencies();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法开始安装");
    }
  };
  const cancelDependencyInstall = async () => {
    if (!dependencyJob || !["queued", "running", "cancelling"].includes(dependencyJob.status)) return;
    try {
      const response = await api<{ job: DependencyInstallJob }>(`/api/dependencies/jobs/${dependencyJob.id}/cancel`, {
        method: "POST",
        body: "{}",
      });
      setDependencyJob(response.job);
      setMessage(response.job.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "取消安装失败");
    }
  };
  const loadBootstrap = async () => {
    const bootstrap = await api<{
      version: string;
      csrf_token: string;
      defaults: Options;
      capabilities: Record<string, Omit<Diagnostic, "id" | "message">>;
    }>("/api/bootstrap");
    csrfToken = bootstrap.csrf_token;
    setAppVersion(bootstrap.version);
    setOptions((current) => ({ ...current, ...bootstrap.defaults }));
    setDiagnostics(Object.entries(bootstrap.capabilities).map(([id, item]) => ({
      id, ...item,
      message: item.available ? "已检测到。" : item.required ? "必需组件未安装。" : "可选组件未安装；不影响默认音频模式。",
    })));
  };
  const previewAuthorizedDemo = async () => {
    try {
      const data = await api<{ cues: Cue[]; translation_ready: boolean }>("/api/demo/cues");
      setDemoPreview(true);
      setJob(null);
      setCues(data.cues);
      setTranslationReady(true);
      setPreview("rendered");
      setView("subtitle");
      setMessage("正在预览仓库内预生成的授权演示结果；未启动下载、模型或 API。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法加载演示结果");
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
  const scanTestHistory = async () => {
    try {
      const response = await api<{ candidates: TestHistoryCandidate[] }>("/api/history/test-records/scan", {
        method: "POST", body: "{}",
      });
      setTestHistoryCandidates(response.candidates);
      setSelectedTestHistoryIds([]);
      setTestHistoryOpen(true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "扫描测试记录失败");
    }
  };
  const deleteSelectedTestHistory = async () => {
    if (!selectedTestHistoryIds.length) return;
    if (!window.confirm(`再次确认：只删除选中的 ${selectedTestHistoryIds.length} 条测试历史记录，不删除任何视频或输出目录。`)) return;
    try {
      const response = await api<{ message: string }>("/api/history/test-records/delete", {
        method: "POST",
        body: JSON.stringify({ ids: selectedTestHistoryIds, confirmed: true }),
      });
      setMessage(response.message);
      setTestHistoryOpen(false);
      await Promise.all([loadJobHistory(), loadRestorableJobs()]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除测试历史失败");
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
  const toggleFullLogs = async () => {
    if (showFullLogs) {
      setShowFullLogs(false);
      return;
    }
    if (!job) return;
    try {
      const response = await api<{ logs: string[]; total: number; truncated: boolean }>(`/api/jobs/${job.id}/logs?limit=5000`);
      setFullLogs(response.logs);
      setFullLogsTruncated(response.truncated);
      setShowFullLogs(true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取完整日志失败");
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
  const validateTaskFields = () => {
    const errors: Record<string, string> = {};
    if (sourceUrl.trim() && !fullVideo && secondsFromDuration(durationValue, durationUnit) === null) {
      errors.max_seconds = "读取长度必须换算为 1～86400 秒之间的整数。";
    }
    if (sourceUrl.trim().length > 2048) errors.source_url = "视频链接不能超过 2048 个字符。";
    if (options.title.length > 200) errors.title = "标题不能超过 200 个字符。";
    if ((description ?? "").length > 5000) errors.description = "简介不能超过 5000 个字符。";
    if (options.whisper_model_size.length > 128) errors.whisper_model_size = "模型名不能超过 128 个字符。";
    if (options.translate_model.length > 128) errors.translate_model = "模型名不能超过 128 个字符。";
    if (options.font_name.length > 100) errors.font_name = "字体名不能超过 100 个字符。";
    if (options.target_lang.length > 32) errors.target_lang = "目标语言不能超过 32 个字符。";
    const parsedTags = tags.split(",").map((value) => value.trim()).filter(Boolean);
    if (parsedTags.length > 20) errors.tags = "标签最多允许 20 个。";
    else if (parsedTags.some((value) => value.length > 30)) errors.tags = "每个标签不能超过 30 个字符。";
    setFieldErrors(errors);
    if (Object.keys(errors).length) {
      setMessage(Object.values(errors)[0]);
      return false;
    }
    return true;
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
    !!text && /Cookies|机器人|登录验证|拦截|PO Token|HTTP 403|浏览器验证/i.test(text);
  const checkReadiness = async () => {
    if (!validateTaskFields()) return false;
    try {
      const response = await api<{
        ready: boolean;
        blocking: Array<{ code: string; message: string }>;
        warnings: Array<{ code: string; message: string }>;
      }>("/api/preflight", {
        method: "POST",
        body: JSON.stringify({ source_url: sourceUrl, material_id: selected?.id, authorized, device, compute_type: computeType, source_height: metadata?.max_height, options: taskOptions() }),
      });
      setReadiness(response.blocking.map((item) => item.message));
      setPreflightWarnings(response.warnings.map((item) => item.message));
      setMessage(response.ready ? (response.warnings[0]?.message ?? "基础配置可以运行。") : response.blocking.map((item) => item.message).join("；"));
      return response.ready;
    } catch (error) {
      setMessage(reportApiError(error, "检查配置失败"));
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
    setOptions((current) => ({ ...current, download_quality: "1080p", render_encoder: "auto", resource_profile: "balanced", subtitle_source: "audio", prefer_platform_subtitles: true, whisper_model_size: "small", source_language: "", translator: "deepseek", target_lang: "zh-Hans", smart_translation: true, smart_subtitle_layout: true, font_name: "Microsoft YaHei", font_size: 24, subtitle_display_mode: "translated", subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "描边" }));
    setIncludeSourceLink(true);
    setMessage("已应用推荐设置：最高 1080p、自动优先 NVIDIA 渲染、CPU int8 转写与中文硬字幕。");
  };

  useEffect(() => {
    loadBootstrap()
      .then(() => Promise.all([
        loadMaterials(),
        loadTemplates(),
        loadSettings(),
        loadDependencies(),
        loadRestorableJobs(),
      ]))
      .then(() => {
        setOnline(true);
        setMessage("本地服务已就绪。选择素材后即可开始处理。");
      })
      .catch(() => setMessage("未连接本地服务，请启动工作台 EXE。"));
  }, []);
  useEffect(() => {
    if (!dependencyJob || !["queued", "running", "cancelling"].includes(dependencyJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const latest = await api<DependencyInstallJob>(`/api/dependencies/jobs/${dependencyJob.id}`);
        setDependencyJob(latest);
        setMessage(latest.message);
        if (["completed", "failed", "cancelled"].includes(latest.status)) {
          window.clearInterval(timer);
          await loadDependencies();
          if (latest.status === "completed" && job) await prepareJob();
        }
      } catch (error) {
        window.clearInterval(timer);
        setMessage(error instanceof Error ? error.message : "读取安装进度失败");
      }
    }, 800);
    return () => window.clearInterval(timer);
  }, [dependencyJob?.id, dependencyJob?.status]);
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
    setShowFullLogs(false);
    setFullLogs([]);
    setFullLogsTruncated(false);
  }, [job?.id]);
  useEffect(() => {
    if (!job || !running) return;
    jobLogCursor.current = job.log_cursor ?? 0;
    let inFlight = false;
    const timer = window.setInterval(async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const latest = await api<Job>(`/api/jobs/${job.id}?log_after=${jobLogCursor.current}&log_limit=100`);
        jobLogCursor.current = latest.log_cursor ?? jobLogCursor.current;
        setJob((current) => ({
          ...latest,
          logs: [...(current?.logs ?? []), ...latest.logs].slice(-2000),
        }));
        if (latest.status === "failed")
          setMessage(`处理失败：${latest.error ?? "请查看检查器日志"}`);
      } catch (error) {
        setMessage(
          `读取任务失败：${error instanceof Error ? error.message : "服务不可用"}`,
        );
      } finally {
        inFlight = false;
      }
    }, 1250);
    return () => window.clearInterval(timer);
  }, [job?.id, running]);
  useEffect(() => {
    if (view === "files") {
      loadOutputs().catch((error) => setMessage(String(error)));
      void loadJobHistory();
    }
    if (view === "library") void Promise.all([loadJobHistory(), loadRestorableJobs()]);
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
        event.preventDefault(); topAction(); return;
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
        max_width: number | null;
        max_height: number | null;
      }>("/api/metadata", {
        method: "POST",
        body: JSON.stringify({
          url: sourceUrl,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
          youtube_po_token_mode: options.youtube_po_token_mode,
          youtube_proxy: options.youtube_proxy,
          download_quality: options.download_quality,
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
      `YouTube 浏览器验证：${options.youtube_po_token_mode === "auto" ? "开启" : "关闭"}`,
      `YouTube 代理：${options.youtube_proxy ? "已配置（地址已隐藏）" : "未配置"}`,
      `下载画质：${options.download_quality}`,
      `字幕渲染：${options.render_encoder}`,
      `资源模式：${options.resource_profile}`,
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
    if (!validateTaskFields()) return;
    if (
      url &&
      options.download_quality === "original" &&
      (metadata?.max_height ?? 0) >= 2160 &&
      !window.confirm(
        `该素材最高为 ${metadata?.max_width ?? "?"}×${metadata?.max_height ?? "?"}。原始画质会显著增加显存、CPU 和磁盘占用。\n\n建议改为 1080p。仍要继续吗？`,
      )
    ) return;
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
      setDemoPreview(false);
      setCues([]);
      setPreview("source");
      setView("process");
      setMessage("任务草稿已保存。请确认处理方案，然后运行开始前准备检查；当前不会下载视频。 ");
    } catch (error) {
      setMessage(`无法启动：${reportApiError(error, "未知错误")}`);
    }
  };
  const saveWorkflowOptions = async () => {
    if (!job || running) return job;
    if (!validateTaskFields()) return null;
    try {
      const updated = await api<Job & { stale_stages?: string[] }>(`/api/jobs/${job.id}/options`, {
        method: "PATCH",
        body: JSON.stringify({ device, compute_type: computeType, options: taskOptions() }),
      });
      setJob(updated);
      const stale = updated.stale_stages ?? [];
      setMessage(stale.length ? `方案已保存；${stale.map((name) => workflowStageMeta.find(([id]) => id === name)?.[1] ?? name).join("、")}需要重新执行。` : "方案已保存，请运行准备检查。 ");
      return updated;
    } catch (error) {
      setMessage(reportApiError(error, "保存处理方案失败"));
      return null;
    }
  };
  const prepareJob = async () => {
    if (!job || running || preparingJob) return;
    const saved = await saveWorkflowOptions();
    if (!saved) return;
    setPreparingJob(true);
    try {
      const prepared = await api<Job>(`/api/jobs/${job.id}/prepare`, { method: "POST", body: "{}" });
      setJob(prepared);
      const blockers = (prepared.checks ?? []).filter((item) => item.status === "blocking");
      setMessage(blockers.length ? `准备检查发现 ${blockers.length} 个阻断项，请按清单修正。` : "开始前准备已通过。你可以逐阶段运行，或一键运行到渲染。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "准备检查失败");
    } finally { setPreparingJob(false); }
  };
  const runStage = async (stage: WorkflowStageName) => {
    if (!job || running) return;
    try {
      const latest = await api<Job>(`/api/jobs/${job.id}/stages/${stage}/run`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true);
      setMessage(`已开始：${workflowStageMeta.find(([id]) => id === stage)?.[1] ?? stage}`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法启动该阶段"); }
  };
  const openStageResult = (stage: WorkflowStageName) => {
    if (!job) return;
    if (stage === "publish") { setView("publish"); return; }
    if (stage === "render") setPreview("rendered");
    else setPreview("source");
    if (["extract", "translate", "render"].includes(stage)) void loadCues(job.id);
    setView("subtitle");
  };
  const runAll = async () => {
    if (!job || running) return;
    try {
      const latest = await api<Job>(`/api/jobs/${job.id}/run-all`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true); setMessage("已按同一套阶段顺序运行；任一步失败都会停在该检查点。 ");
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法运行全部阶段"); }
  };
  const resumeJob = async () => {
    if (!job || running) return;
    try {
      const latest = await api<Job>(`/api/jobs/${job.id}/resume`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true); setMessage("已从最后一个有效检查点继续。 ");
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法恢复任务"); }
  };
  const topAction = () => {
    if (running) return void cancel();
    if (!job) return void start();
    if (!job.checks?.length || (job.checks ?? []).some((item) => item.status === "blocking")) return void prepareJob();
    if (job.can_resume) return void resumeJob();
    if (job.next_stage) return void runStage(job.next_stage);
    setView("subtitle");
  };
  const topActionLabel = preparingJob ? "正在检查准备…" : workflowPrimaryLabel({
    running, hasJob: Boolean(job), hasChecks: Boolean(job?.checks?.length),
    hasBlocking: (job?.checks ?? []).some((item) => item.status === "blocking"),
    canResume: Boolean(job?.can_resume), nextStage: job?.next_stage ?? null,
    stageLabel: (stage) => workflowStageMeta.find(([id]) => id === stage)?.[1] ?? "下一阶段",
  });
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
      const savedJob = await api<Job>(`/api/jobs/${job.id}/cues`, {
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
      setJob(savedJob);
      setHistory([]);
      setFuture([]);
      setMessage("字幕已保存；渲染检查点已标记为需要重做，可直接点击“重新渲染”。 ");
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
          ...(proxyDirty ? { youtube_proxy: options.youtube_proxy } : {}),
        }),
      });
      setApiKey("");
      if (proxyDirty) setProxyConfigured(Boolean(options.youtube_proxy));
      setProxyDirty(false);
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
          <button className="secondary small" onClick={() => void previewAuthorizedDemo()}>
            直接预览演示结果
          </button>
        </div>
      )}
      {diagnostics.some((check) => check.required && !check.available) && (
        <section className="diagnostics-banner" aria-label="首次使用依赖检查">
          <div>
            <b>开始前还需安装 {diagnostics.filter((check) => check.required && !check.available).map((check) => check.label).join("、")}</b>
            <span>这里只列出默认流程的必需组件；Tesseract 等 OCR 组件按所选模式检查。</span>
          </div>
          <button className="text-button" onClick={() => setDependenciesOpen(true)}>打开依赖中心</button>
        </section>
      )}
      <div className="form-grid two">
        <label>
          视频链接
          <input
            id="source-url"
            value={sourceUrl}
            maxLength={2048}
            placeholder="粘贴 YouTube / B 站视频链接"
            onChange={(event) => {
              setSourceUrl(event.target.value);
              setFieldErrors((current) => ({ ...current, source_url: "" }));
              setDemoPreview(false);
              setMetadata(null);
              setMetadataNotice("");
              setMetadataState("idle");
            }}
            onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void readMetadata(); } }}
            disabled={running}
          />
          {fieldErrors.source_url && <small className="field-error">{fieldErrors.source_url}</small>}
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
            保存素材并配置方案
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
              去设置 → YouTube 访问设置
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
          <span>
            {metadata.max_width && metadata.max_height
              ? `最高 ${metadata.max_width} × ${metadata.max_height}`
              : "分辨率未知"}
          </span>
        </div>
      )}
      {metadata && options.download_quality === "original" && (metadata.max_height ?? 0) >= 2160 && (
        <p className="metadata-notice warning">
          这是 4K 素材。原始画质处理会明显增加负载；普通使用建议改为 1080p。
        </p>
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
            保存素材并配置方案
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
            maxLength={200}
            onChange={(event) => updateOptions("title", event.target.value)}
          />
          {fieldErrors.title && <small className="field-error">{fieldErrors.title}</small>}
        </label>
        <label>
          URL 读取长度
          <div className="duration-control">
            <label className="check"><input type="checkbox" checked={fullVideo} disabled={!sourceUrl.trim()} onChange={(event) => setFullVideo(event.target.checked)} />完整视频</label>
            <input type="number" min={durationUnit === "秒" ? 1 : 1 / 60} max={durationUnit === "小时" ? 24 : durationUnit === "分钟" ? 1440 : 86400} step={durationUnit === "秒" ? 1 : durationUnit === "分钟" ? 1 / 60 : 1 / 3600} value={durationValue} disabled={!sourceUrl.trim() || fullVideo} onChange={(event) => { setDurationValue(event.target.value); setFieldErrors((current) => ({ ...current, max_seconds: "" })); }} />
            <select value={durationUnit} disabled={!sourceUrl.trim() || fullVideo} onChange={(event) => setDurationUnit(event.target.value as "秒" | "分钟" | "小时")}><option>秒</option><option>分钟</option><option>小时</option></select>
          </div>
          {fieldErrors.max_seconds && <small className="field-error">{fieldErrors.max_seconds}</small>}
        </label>
        <label>
          URL 下载画质
          <select
            value={options.download_quality}
            disabled={!sourceUrl.trim() || running}
            onChange={(event) => updateOptions("download_quality", event.target.value as Options["download_quality"])}
          >
            <option value="720p">720p（低负载）</option>
            <option value="1080p">1080p（推荐）</option>
            <option value="original">原始画质（可能为 4K）</option>
          </select>
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
        我声明自己拥有处理及发布该视频所需的权利或许可证
        <small className="check-hint">程序只能记录这项声明，无法代替版权或身份核验。</small>
      </label>
      <div className="card-footer task-tools">
        <button className="secondary" onClick={() => void previewAuthorizedDemo()}>
          预览授权演示结果
        </button>
        <button onClick={applyRecommended} disabled={running}>推荐设置</button>
        <button onClick={() => void checkReadiness()} disabled={running}>检查配置</button>
        {readiness.length > 0 && <span className="readiness-error">{readiness.join("；")}</span>}
        {!readiness.length && preflightWarnings.length > 0 && <span className="readiness-warning">{preflightWarnings.join("；")}</span>}
      </div>
    </section>
  );
  const preparationBlocking = (job?.checks ?? []).some((item) => item.status === "blocking");
  const preparationReady = Boolean(job?.checks?.length) && !preparationBlocking;
  const handleCheckAction = (check: PreparationCheck) => {
    const action = check.action ?? "";
    if (/安装|模型|FFmpeg|Tesseract/.test(action)) {
      const dependency = dependencies.find((item) => item.id === check.id || (check.id === "ffprobe" && item.id === "ffmpeg"));
      setDependenciesOpen(true);
      if (dependency?.installable) void installDependency(dependency);
    }
    else if (/设置|API/.test(action)) setSettingsOpen(true);
    else if (/素材/.test(action)) setView("material");
    else setMode("advanced");
  };
  const workflowPanel = job ? (
    <section className="workflow-panel" aria-label="分阶段任务工作流">
      <header className="workflow-head">
        <div>
          <p className="eyebrow">任务工作流</p>
          <h2>先检查，再逐阶段运行</h2>
          <p>已完成阶段会保留检查点。取消或关闭应用后，不会丢掉之前的有效产物。</p>
        </div>
        <div className="workflow-actions">
          <button onClick={() => void saveWorkflowOptions()} disabled={running || preparingJob}>保存方案</button>
          <button className="primary" onClick={() => void prepareJob()} disabled={running || preparingJob} title={running ? "当前阶段运行中" : undefined}>
            {preparingJob ? "正在检查链接与环境…" : job.checks?.length ? "重新检查准备" : "检查准备"}
          </button>
          <button onClick={() => void runAll()} disabled={running || !preparationReady} title={!preparationReady ? "准备检查全部通过后才能运行" : undefined}>
            一键运行到渲染
          </button>
        </div>
      </header>
      <ol className="workflow-stepper">
        <li className="done"><span>1</span><b>素材确认</b></li>
        <li className="done"><span>2</span><b>方案配置</b></li>
        <li className={preparationReady ? "done" : job.checks?.length ? "attention" : ""}><span>3</span><b>环境准备</b></li>
        {workflowStageMeta.slice(0, 4).map(([id, label], index) => (
          <li key={id} className={job.stages?.[id]?.status === "completed" ? "done" : job.current_stage === id ? "active" : ["failed", "stale", "interrupted"].includes(job.stages?.[id]?.status ?? "") ? "attention" : ""}>
            <span>{index + 4}</span><b>{label}</b>
          </li>
        ))}
      </ol>
      <section className="preparation-centre">
        <div className="section-title">
          <div><b>开始前准备</b><small>模型、工具、API、设备和磁盘问题全部在下载前显示</small></div>
          {job.checks?.length ? <span className={preparationBlocking ? "blocking" : "passed"}>{preparationBlocking ? "存在阻断项" : "全部可运行"}</span> : <span>尚未检查</span>}
        </div>
        {job.checks?.length ? (
          <div className="check-list">
            {job.checks.map((check) => (
              <article key={check.id} className={`prepare-check ${check.status}`}>
                <span className="check-state" aria-hidden="true">{check.status === "passed" ? "✓" : check.status === "warning" ? "!" : check.status === "installing" ? "…" : "×"}</span>
                <div><b>{check.label}</b><small>{check.purpose}</small><p>{check.message}</p></div>
                {check.action && check.status !== "passed" && <button onClick={() => handleCheckAction(check)}>{check.action}</button>}
              </article>
            ))}
          </div>
        ) : <p className="empty-checks">保存当前方案后点击“检查准备”。检查不会下载视频，也不会自动安装任何组件。</p>}
      </section>
      <section className="stage-list">
        {workflowStageMeta.map(([id, label, purpose], index) => {
          const state = job.stages?.[id];
          const availability = stageAvailability(id, Object.fromEntries(workflowStageMeta.map(([name]) => [name, job.stages?.[name]?.status])), preparationReady, running, publishInPipeline);
          const enabled = availability.enabled;
          const disabledReason = availability.reason;
          return (
            <article key={id} className={`stage-card ${state?.status ?? "pending"}`}>
              <div className="stage-index">{index + 1}</div>
              <div className="stage-copy"><div><b>{label}</b><span className={`stage-status ${state?.status ?? "pending"}`}>{stageStatusLabel[state?.status ?? "pending"]}</span></div><p>{purpose}</p>{state?.error && <small className="stage-error">{state.error}</small>}</div>
              <div className="stage-action">
                {state?.status === "completed" && id !== "publish" ? <span className="checkpoint">检查点已保存</span> : null}
                <button onClick={() => state?.status === "running" ? void cancel() : state?.status === "completed" ? openStageResult(id) : void runStage(id)} disabled={state?.status === "running" || state?.status === "completed" ? false : !enabled} title={!["running", "completed"].includes(state?.status ?? "") && !enabled ? disabledReason : undefined}>
                  {state?.status === "running" ? "取消当前阶段" : state?.status === "completed" ? "打开结果" : ["failed", "cancelled", "stale", "interrupted"].includes(state?.status ?? "") ? "重试" : "开始"}
                </button>
                {!["running", "completed"].includes(state?.status ?? "") && !enabled && disabledReason && <small>{disabledReason}</small>}
              </div>
            </article>
          );
        })}
      </section>
      {job.can_resume && <div className="resume-banner"><div><b>检测到上次运行被中断</b><span>已完成阶段不会重做；翻译会复用已保存批次，其他未完成阶段从头执行。</span></div><button onClick={() => void resumeJob()} disabled={!preparationReady || running}>从 {workflowStageMeta.find(([id]) => id === job.next_stage)?.[1] ?? "检查点"}继续</button></div>}
    </section>
  ) : null;
  const processView = (
    <>
    {workflowPanel}
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
      {job?.subtitle_extraction?.ocr_status === "fallback" && (
        <p className="metadata-notice warning">
          本次 OCR 没有成功，实际字幕来自 Whisper 音频转写。原因：{job.subtitle_extraction.message || "未找到可信画面字幕"}
        </p>
      )}
      {job?.subtitle_extraction?.ocr_status === "completed" && (
        <p className="metadata-notice success">
          OCR 已实际参与本次字幕提取；模式：{job.subtitle_extraction.mode || "OCR"}。
        </p>
      )}
      {job?.content_warnings?.map((warning) => (
        <p className="metadata-notice warning" key={warning}>
          人工复核提醒：{warning}
        </p>
      ))}
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
            <FieldGuide summary="决定源字幕来自声音、画面文字，或两者合并。" detail="一般视频推荐仅音频；画面已有清晰外语字幕时可选 OCR；合并模式更慢且需要额外安装 Tesseract。" badges={options.subtitle_source === "audio" ? ["推荐"] : options.subtitle_source.includes("ocr") || options.subtitle_source === "merged" ? ["需要额外安装", "高负载"] : []} />
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
            <FieldGuide summary="把源字幕翻译为目标语言。" detail="DeepSeek/OpenAI 会调用在线 API，可能产生少量费用；不翻译只适合保留原文或调试提取结果。" badges={options.translator === "none" ? [] : ["可能产生费用"]} />
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
          <label>
            字幕渲染
            <select
              value={options.render_encoder}
              onChange={(event) => updateOptions("render_encoder", event.target.value as Options["render_encoder"])}
            >
              <option value="auto">自动（NVIDIA 优先）</option>
              <option value="nvidia">NVIDIA NVENC</option>
              <option value="cpu">CPU（兼容模式）</option>
            </select>
            <FieldGuide summary="决定最终视频由显卡还是 CPU 编码。" detail="自动模式会优先使用可用的 NVIDIA 编码器；CPU 兼容性最好但较慢。准备检查会在下载前验证。" badges={["推荐：自动"]} />
          </label>
          <label>
            电脑资源占用
            <select
              value={options.resource_profile}
              onChange={(event) => updateOptions("resource_profile", event.target.value as Options["resource_profile"])}
            >
              <option value="background">后台（最流畅，处理较慢）</option>
              <option value="balanced">均衡（推荐）</option>
              <option value="maximum">极速（可能影响其他软件）</option>
            </select>
            <FieldGuide summary="限制下载带宽和 CPU 线程，避免处理视频时整台电脑变卡。" detail="后台模式优先保证其他软件流畅；均衡模式默认保留一部分算力和网络；极速模式会尽量使用全部资源。切换模式不会让已完成阶段失效。" badges={options.resource_profile === "balanced" ? ["推荐"] : options.resource_profile === "maximum" ? ["高负载"] : ["低占用"]} />
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
          <FieldGuide summary="模型越大通常更准确，也越慢、越占显存和磁盘。" detail="small 是桌面使用的平衡选择。所选模型必须在准备中心明确安装，转写阶段不会自动联网下载。" badges={["需要预先安装"]} />
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
            maxLength={128}
            placeholder="使用默认模型"
            onChange={(event) =>
              updateOptions("translate_model", event.target.value)
            }
          />
          {fieldErrors.translate_model && <small className="field-error">{fieldErrors.translate_model}</small>}
        </label>
        <label>
          目标语言
          <input
            value={options.target_lang}
            maxLength={32}
            onChange={(event) =>
              updateOptions("target_lang", event.target.value)
            }
          />
          {fieldErrors.target_lang && <small className="field-error">{fieldErrors.target_lang}</small>}
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
          <FieldGuide summary="选择语音转写使用显卡还是处理器。" detail="CUDA 通常更快但需要 NVIDIA 显卡和正确驱动；CPU 更通用，准备检查会纠正不兼容精度。" badges={device === "cuda" ? ["高性能"] : ["兼容"]} />
        </label>
        <label>
          电脑资源占用
          <select
            value={options.resource_profile}
            onChange={(event) => updateOptions("resource_profile", event.target.value as Options["resource_profile"])}
          >
            <option value="background">后台（最流畅）</option>
            <option value="balanced">均衡（推荐）</option>
            <option value="maximum">极速（可能卡顿）</option>
          </select>
          <FieldGuide summary="统一控制下载、Whisper 与 FFmpeg 的资源上限。" detail="后台约限制下载到 2 MiB/s，并减少 CPU 线程；均衡约 6 MiB/s；极速不主动限速或限线程。Windows 下后台与均衡还会降低重型子进程优先级。" badges={options.resource_profile === "maximum" ? ["高负载"] : options.resource_profile === "balanced" ? ["推荐"] : ["低占用"]} />
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
          <FieldGuide summary="控制转写速度、显存和数值精度。" detail="CUDA 推荐 float16；CPU 推荐 int8。错误组合会在下载前作为阻断项显示。" badges={computeType === "float16" ? ["CUDA 推荐"] : computeType === "int8" ? ["CPU 推荐"] : []} />
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
            <option value={0.25}>0.25（短字幕，最慢）</option>
            <option value={0.5}>0.5（推荐）</option>
            <option value={1}>1（更快）</option>
            <option value={2}>2（更快）</option>
          </select>
          <FieldGuide summary="决定多久读取一次画面，默认 0.5 秒可覆盖多数短字幕。" detail="字幕一闪而过时使用 0.25 秒；普通字幕推荐 0.5 秒；1～2 秒会更快，但可能漏掉短字幕。" badges={options.ocr_interval <= 0.25 ? ["高负载"] : options.ocr_interval === 0.5 ? ["推荐"] : ["可能漏短字幕"]} />
        </label>
        <label>
          OCR 语言
          <select
            value={options.ocr_language}
            onChange={(event) => updateOptions("ocr_language", event.target.value)}
          >
            <option value="eng">英文（eng）</option>
            <option value="chi_sim">简体中文（chi_sim）</option>
            <option value="eng+chi_sim">英文 + 简体中文</option>
            <option value="jpn">日文（jpn）</option>
            <option value="kor">韩文（kor）</option>
          </select>
          <FieldGuide summary="必须和画面中已有字幕的语言一致。" detail="准备检查会在下载前确认对应 Tesseract 语言包是否安装；缺少语言包时不会开始下载。" badges={["下载前检查"]} />
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
          字幕渲染编码器
          <select
            value={options.render_encoder}
            onChange={(event) => updateOptions("render_encoder", event.target.value as Options["render_encoder"])}
          >
            <option value="auto">自动（NVIDIA 优先）</option>
            <option value="nvidia">NVIDIA NVENC</option>
            <option value="cpu">CPU libx264</option>
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
            checked={options.prefer_platform_subtitles}
            onChange={(event) =>
              updateOptions("prefer_platform_subtitles", event.target.checked)
            }
          />
          优先使用 YouTube 字幕（更快；没有时自动转写）
        </label>
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
                  {cue.kind === "ocr" ? "画面文字" : cue.kind === "merged" ? "合并" : cue.kind === "platform" ? "平台字幕" : "语音"}
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
        <button className="secondary" onClick={() => void previewAuthorizedDemo()}>
          预览授权演示结果
        </button>
        <button
          className="primary"
          onClick={() => job ? void prepareJob() : void start()}
          disabled={running || preparingJob || (!job && !selected && !sourceUrl.trim())}
        >
          {running ? "当前阶段运行中" : job ? "保存方案并检查准备" : "保存素材并配置方案"}
        </button>
      </footer>
    </section>
    </>
  );
  const subtitleView = (
    <>
      <section className="media-panel">
        <div className="panel-toolbar">
          <div>
            <p className="eyebrow">字幕工作区</p>
            <h1>
              {demoPreview
                ? "授权演示成片（只读）"
                : preview === "rendered" && job?.result
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
              disabled={!job?.result && !demoPreview}
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
              maxLength={100}
              onChange={(event) =>
                updateOptions("font_name", event.target.value)
              }
            />
            {fieldErrors.font_name && <small className="field-error">{fieldErrors.font_name}</small>}
          </label>
          <label>
            字号
            <input
              type="number"
              min="8"
              max="96"
              step="1"
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
              disabled={!job?.result || running || dirty || job?.edit_state === "legacy-ambiguous"}
              title={job?.edit_state === "legacy-ambiguous" ? "旧任务的后期版本顺序无法确认；为保护现有成片，已禁用重新渲染" : "使用当前活动视频和字幕重新渲染"}
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
          maxLength={5000}
          onChange={(event) => setTemplateBody(event.target.value)}
          placeholder="模板支持 {source} 与 {custom_text}"
        />
      </label>
      <label>
        简介
        <textarea
          value={description}
          maxLength={5000}
          onChange={(event) => { setDescription(event.target.value); setFieldErrors((current) => ({ ...current, description: "" })); }}
          placeholder="生成后可继续人工修改"
        />
        {fieldErrors.description && <small className="field-error">{fieldErrors.description}</small>}
      </label>
      <div className="form-grid two">
        <label>
          标签（逗号分隔）
          <input
            value={tags}
            onChange={(event) => { setTags(event.target.value); setFieldErrors((current) => ({ ...current, tags: "" })); }}
          />
          {fieldErrors.tags && <small className="field-error">{fieldErrors.tags}</small>}
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
  const libraryView = (
    <section className="workspace-card library-view">
      <header className="library-header">
        <div>
          <p className="eyebrow">视频任务库</p>
          <h1>所有视频都在这里</h1>
          <p>汇总待处理、处理中、可继续和已完成任务。同一时间只运行一个耗资源阶段。</p>
        </div>
        <div className="library-header-actions">
          <button onClick={() => void Promise.all([loadJobHistory(), loadRestorableJobs()])}>刷新</button>
          <button className="primary" onClick={() => setView("material")}>添加视频</button>
        </div>
      </header>
      <div className="library-summary" aria-label="任务数量摘要">
        <button className={libraryFilter === "all" ? "selected" : ""} onClick={() => setLibraryFilter("all")}>
          <b>{librarySourceJobs.length}</b><span>全部</span>
        </button>
        <button className={libraryFilter === "active" ? "selected active" : "active"} onClick={() => setLibraryFilter("active")}>
          <b>{libraryCounts.active}</b><span>处理中</span>
        </button>
        <button className={libraryFilter === "attention" ? "selected attention" : "attention"} onClick={() => setLibraryFilter("attention")}>
          <b>{libraryCounts.attention}</b><span>待处理 / 可继续</span>
        </button>
        <button className={libraryFilter === "completed" ? "selected completed" : "completed"} onClick={() => setLibraryFilter("completed")}>
          <b>{libraryCounts.completed}</b><span>已完成</span>
        </button>
      </div>
      <div className="library-tools">
        <label>
          搜索任务
          <input value={libraryQuery} onChange={(event) => setLibraryQuery(event.target.value)} placeholder="输入视频标题、阶段或错误信息" />
        </label>
        <label>
          显示范围
          <select value={historyScope} onChange={(event) => setHistoryScope(event.target.value as "current" | "all")}>
            <option value="all">全部输出目录</option>
            <option value="current">当前输出目录</option>
          </select>
        </label>
        <small>{historyTotal > 200 ? `共 ${historyTotal} 条，当前显示最近 200 条` : `共 ${historyTotal} 条任务记录`}</small>
      </div>
      <div className="video-library-list" aria-live="polite">
        {libraryJobs.length ? libraryJobs.map((item) => {
          const resumable = restorableJobs.some((candidate) => candidate.id === item.id);
          const category = videoLibraryCategory(item, resumable);
          const isCurrent = job?.id === item.id;
          const statusText = category === "active"
            ? item.status === "cancelling" ? "正在取消" : "处理中"
            : category === "completed"
              ? "已完成"
              : resumable ? "可继续" : item.status === "failed" ? "失败" : item.status === "cancelled" ? "已取消" : "待处理";
          return (
            <article className={`video-library-row ${category}`} key={item.id}>
              <div className="video-library-icon" aria-hidden="true">▶</div>
              <div className="video-library-main">
                <div className="video-library-title">
                  <b title={item.title}>{item.title}</b>
                  {isCurrent && <span className="current-task">当前任务</span>}
                </div>
                <div className="video-library-meta">
                  <span className={`library-status ${category}`}>{statusText}</span>
                  <span>{item.stage || "等待配置"}</span>
                  <span>{item.created_at ? new Date(item.created_at * 1000).toLocaleString() : "时间未知"}</span>
                </div>
                {(category === "active" || (item.progress > 0 && item.progress < 100)) && (
                  <div className="library-progress" aria-label={`处理进度 ${item.progress}%`}><i style={{ width: `${item.progress}%` }} /></div>
                )}
                {item.error && <small className="library-error" title={item.error}>{item.error}</small>}
              </div>
              <div className="video-library-actions">
                <button className={resumable || category === "active" ? "primary small" : ""} onClick={() => isCurrent && running ? setView("process") : void loadRestoredJob(item.id)}>
                  {isCurrent && running ? "查看进度" : resumable ? "继续处理" : category === "completed" ? "打开任务" : "检查任务"}
                </button>
                <button onClick={() => void openHistoryPath(item.rendered_video || "")} disabled={!item.rendered_exists} title={item.rendered_exists ? "打开最终成片" : "尚未生成成片"}>成片</button>
                <button onClick={() => void openHistoryPath(item.output_dir || "")} disabled={!item.output_exists} title={item.output_exists ? "打开该任务的输出目录" : "输出目录不存在"}>文件</button>
              </div>
            </article>
          );
        }) : (
          <div className="empty-editor library-empty">
            <b>{libraryQuery || libraryFilter !== "all" ? "没有匹配的任务" : "任务库还是空的"}</b>
            <span>{libraryQuery || libraryFilter !== "all" ? "试试清除搜索或切换到“全部”。" : "添加一个链接或本地视频后，任务会出现在这里。"}</span>
            {!libraryQuery && libraryFilter === "all" && <button className="primary" onClick={() => setView("material")}>添加第一个视频</button>}
          </div>
        )}
      </div>
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
          <button onClick={() => void scanTestHistory()}>检查测试记录</button>
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
                <button onClick={() => void loadRestoredJob(item.id)} title="载入任务状态、字幕和当前产物；不会自动继续处理">
                  载入任务
                </button>
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
            Localizer <small>v{appVersion}</small>
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
          <button className="text-button" onClick={() => setDependenciesOpen(true)}>
            依赖
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
            onClick={topAction}
            disabled={preparingJob || (running ? !job : !job && !selected && !sourceUrl.trim())}
          >
            {topActionLabel}
          </button>
        </div>
      </header>
      {restorableJobs.length > 0 && !running && (
        <section className="restore-banner" aria-live="polite">
          <div>
            <b>发现 {restorableJobs.length} 个可继续任务</b>
            <span>载入只会检查该任务的产物，不会自动开始下载、转写或渲染。</span>
          </div>
          <select aria-label="选择可继续任务" defaultValue="" onChange={(event) => { if (event.target.value) void loadRestoredJob(event.target.value); event.currentTarget.value = ""; }}>
            <option value="" disabled>选择并载入…</option>
            {restorableJobs.map((item) => (
              <option key={item.id} value={item.id}>{item.title} · {item.next_stage ? stageStatusLabel.pending + " " + item.next_stage : item.status}</option>
            ))}
          </select>
        </section>
      )}
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
          {view === "library" && libraryView}
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
                    正在准备 YouTube 媒体：首次自动浏览器验证可能需要数秒，并短暂启动最小化的 Chrome/Edge；随后进度来自真实下载字节。
                  </p>
                )}
                {job.stages && (
                  <div className="inspector-stages">
                    {workflowStageMeta.map(([id, label]) => (
                      <div key={id} className={job.stages?.[id]?.status ?? "pending"}>
                        <span>{job.stages?.[id]?.status === "completed" ? "✓" : job.current_stage === id ? "●" : "○"}</span>
                        <b>{label}</b>
                        <small>{stageStatusLabel[job.stages?.[id]?.status ?? "pending"]}</small>
                      </div>
                    ))}
                  </div>
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
                  {job.subtitle_extraction?.mode && (
                    <div>
                      <dt>实际提取</dt>
                      <dd>{job.subtitle_extraction.mode}</dd>
                    </div>
                  )}
                  <div>
                    <dt>设备</dt>
                    <dd>
                      {job.device} · {job.compute_type}
                    </dd>
                  </div>
                  <div>
                    <dt>下载画质</dt>
                    <dd>{job.options.download_quality}</dd>
                  </div>
                  <div>
                    <dt>字幕渲染</dt>
                    <dd>{job.options.render_encoder}</dd>
                  </div>
                </dl>
                <section className="compact-log">
                  <div className="log-head"><p className="eyebrow">{showFullLogs ? "完整日志" : "最近日志"}</p><div className="log-actions"><CopyButton text={buildDebugText()} label="复制调试信息" /><button onClick={() => void toggleFullLogs()}>{showFullLogs ? "收起" : `查看完整日志${job.log_total ? `（${job.log_total} 行）` : ""}`}</button></div></div>
                  {showFullLogs && fullLogsTruncated && <p>日志较长，仅显示最近 5000 行。</p>}
                  {(showFullLogs ? fullLogs : job.logs
                    .slice(-8)
                    .reverse()).map((line, index) => (
                      <p key={`${line}-${index}`}>{line}</p>
                    ))}
                </section>
                {job.error && (
                  <div className="error-block">
                    <p className="error">{job.error}</p>
                    {job.suggested_action && <p className="error-suggestion">建议：{job.suggested_action}</p>}
                    <div className="error-actions">
                      <CopyButton text={job.error} label="复制错误" />
                      {cookieHintMatches(job.error) && (
                        <button className="cookie-hint-button" onClick={() => setSettingsOpen(true)}>
                          去设置 → YouTube 访问设置
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
          <label>
            YouTube 自动浏览器验证
            <select
              value={options.youtube_po_token_mode}
              onChange={(event) => updateOptions("youtube_po_token_mode", event.target.value as "auto" | "off")}
            >
              <option value="auto">开启（推荐，自动获取 PO Token）</option>
              <option value="off">关闭（仅使用 yt-dlp 常规模式）</option>
            </select>
            <small className={`drawer-note ${poTokenStatus && !poTokenStatus.available ? "warn" : ""}`}>
              {poTokenStatus?.available
                ? `验证组件已就绪${poTokenStatus.browser_path ? "，已检测到 Chrome/Edge" : "；运行时将自动查找 Chrome/Edge"}。首次请求可能短暂启动一个最小化浏览器。`
                : "当前未检测到验证组件；程序会退回常规模式。公开发行版应通过“修复安装”补齐该组件。"}
            </small>
          </label>
          <label>
            YouTube 代理（可选，仅当前运行）
            <input
              type="password"
              value={options.youtube_proxy}
              autoComplete="off"
              placeholder={proxyConfigured && !proxyDirty ? "本机已配置；输入新地址可覆盖，留空不会改动" : "例如 http://127.0.0.1:7890 或 socks5://127.0.0.1:1080"}
              onChange={(event) => { updateOptions("youtube_proxy", event.target.value); setProxyDirty(true); }}
            />
            <small className="drawer-note">
              同一地址用于元数据、视频、封面和浏览器验证；只保存在本机设置，不会写入任务历史或日志。
            </small>
          </label>
          <button className="primary" onClick={saveSettings}>
            保存本地设置并应用本次访问参数
          </button>
        </div>
      )}
      {testHistoryOpen && (
        <div className="modal-backdrop" role="presentation">
          <section className="template-modal test-history-modal" role="dialog" aria-modal="true" aria-label="测试历史清理">
            <header>
              <div><p className="eyebrow">安全清理</p><h2>pytest 测试历史记录</h2></div>
              <button onClick={() => setTestHistoryOpen(false)} aria-label="关闭">×</button>
            </header>
            <p>这里只列出输出目录严格位于 pytest 临时目录中的记录。删除仅影响历史列表，不会删除视频、字幕或目录。</p>
            <div className="test-history-list">
              {testHistoryCandidates.length ? testHistoryCandidates.map((item) => (
                <label key={item.id}>
                  <input type="checkbox" checked={selectedTestHistoryIds.includes(item.id)} onChange={(event) => setSelectedTestHistoryIds((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))} />
                  <span><b>{item.title}</b><small>{item.reason}</small><code>{item.output_dir}</code></span>
                </label>
              )) : <div className="empty-editor"><b>没有符合规则的测试记录</b><span>普通任务不会出现在这里。</span></div>}
            </div>
            <footer>
              <button onClick={() => setTestHistoryOpen(false)}>取消</button>
              <button className="danger" disabled={!selectedTestHistoryIds.length} onClick={() => void deleteSelectedTestHistory()}>删除选中的历史记录</button>
            </footer>
          </section>
        </div>
      )}
      {dependenciesOpen && (
        <div className="modal-backdrop" role="presentation">
          <section className="template-modal dependency-modal" role="dialog" aria-modal="true" aria-label="依赖中心">
            <header>
              <div><p className="eyebrow">首次使用</p><h2>依赖中心</h2></div>
              <button onClick={() => setDependenciesOpen(false)} aria-label="关闭依赖中心">×</button>
            </header>
            <p className="dependency-intro">默认流程只强制需要 FFmpeg。模型、OCR 与投稿浏览器按你实际使用的功能安装；已经安装的组件不会重复下载。</p>
            <div className="dependency-list">
              {dependencies.map((dependency) => {
                const active = dependencyJob?.dependency_id === dependency.id && ["queued", "running", "cancelling"].includes(dependencyJob.status);
                return (
                  <article className={`dependency-row ${dependency.available ? "available" : "missing"}`} key={dependency.id}>
                    <span className="dependency-state" aria-hidden="true">{dependency.available ? "✓" : dependency.required ? "!" : "＋"}</span>
                    <div>
                      <div className="dependency-title"><b>{dependency.label}</b><small>{dependency.required ? "默认流程必需" : "按需安装"} · {dependency.size_hint}</small></div>
                      <p>{dependency.purpose}</p>
                      <small>{active ? dependencyJob?.message : dependency.message}</small>
                      {active && <div className={`dependency-progress ${dependencyJob?.progress_kind === "indeterminate" ? "indeterminate" : ""}`}><i style={dependencyJob?.progress_kind === "determinate" ? { width: `${dependencyJob?.progress ?? 0}%` } : undefined} /></div>}
                      {dependencyJob?.dependency_id === dependency.id && dependencyJob.status === "failed" && (
                        <p className="dependency-error">{dependencyJob.error}</p>
                      )}
                    </div>
                    <div className="dependency-actions">
                      {dependency.available ? <span>可用</span> : dependency.installable ? (
                        <button className="primary small" disabled={Boolean(dependencyJob && ["queued", "running", "cancelling"].includes(dependencyJob.status))} onClick={() => void installDependency(dependency)}>
                          {active ? dependencyJob?.status === "cancelling" ? "取消中" : "安装中" : dependencyJob?.dependency_id === dependency.id && ["failed", "cancelled"].includes(dependencyJob.status) ? "重试" : "安装"}
                        </button>
                      ) : dependency.action_url ? (
                        <a href={dependency.action_url} target="_blank" rel="noreferrer">安装说明</a>
                      ) : <span>需手动安装</span>}
                    </div>
                  </article>
                );
              })}
            </div>
            {dependencyJob && dependencyJob.logs.length > 0 && (
              <details className="dependency-logs"><summary>安装日志</summary>{dependencyJob.logs.map((line, index) => <p key={`${index}-${line}`}>{line}</p>)}</details>
            )}
            {dependencyJob && ["queued", "running", "cancelling"].includes(dependencyJob.status) && (
              <div className="dependency-cancel-row">
                <span>{dependencyJob.status === "cancelling" ? "正在结束当前安装步骤…" : "安装期间仍可安全取消；外部安装进程会一并关闭。"}</span>
                <button className="danger" disabled={dependencyJob.status === "cancelling"} onClick={() => void cancelDependencyInstall()}>取消安装</button>
              </div>
            )}
            <footer>
              <span className="dependency-note">组件安装到当前 Windows 用户环境，不会写入项目仓库。</span>
              <button onClick={() => void loadDependencies()}>重新检测</button>
              <button className="primary" onClick={() => setDependenciesOpen(false)}>完成</button>
            </footer>
          </section>
        </div>
      )}
      {shortcutHelpOpen && (
        <div className="modal-backdrop" role="presentation">
          <section className="template-modal help-modal" role="dialog" aria-modal="true" aria-label="使用帮助">
            <header><div><p className="eyebrow">使用帮助</p><h2>开始前检查与快捷键</h2></div><button onClick={() => setShortcutHelpOpen(false)} aria-label="关闭帮助">×</button></header>
            <section className="help-section">
              <b>本机依赖</b>
              {diagnostics.length ? diagnostics.map((check) => <p key={check.id} className={check.available ? "help-ok" : "help-missing"}><strong>{check.available ? "✓" : "!"} {check.label}</strong> · {check.purpose}<br /><span>{check.message}</span></p>) : <p>正在检测本机组件…</p>}
              <button onClick={() => { setShortcutHelpOpen(false); setDependenciesOpen(true); }}>打开依赖中心</button>
            </section>
            <section className="help-section">
              <b>快捷键</b>
              <dl className="shortcut-list"><div><dt>Ctrl + O</dt><dd>选择本地视频（先确认授权）</dd></div><div><dt>Ctrl + L</dt><dd>回到素材区并聚焦视频链接</dd></div><div><dt>Ctrl + R</dt><dd>执行顶部主操作（检查准备 / 继续下一阶段）</dd></div><div><dt>Esc</dt><dd>只中断当前运行阶段，保留已完成检查点</dd></div></dl>
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

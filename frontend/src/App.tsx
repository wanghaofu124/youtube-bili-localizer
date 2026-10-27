import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { publishStageButton, sameTaskOption, stageAvailability, workflowPrimaryLabel, type StageName } from "./workflowState";
import { retainVisibleVideoLibrarySelection, selectableVideoLibraryIds, videoLibraryCategory, videoLibraryMatches, type VideoLibraryCategory } from "./videoLibrary";
import { cueNeedsReview, replaceLiteral } from "./subtitleEditor";
import { SetupModal, TranslationPresetFields, type TranslationPreset } from "./SetupModal";
import { TutorialModal, type TutorialAction } from "./TutorialModal";

type View = "material" | "library" | "process" | "subtitle" | "publish" | "files";
const CURRENT_TUTORIAL_VERSION = 1;
const TUTORIAL_STORAGE_KEY = "ybl-onboarding";
const CURRENT_SETUP_VERSION = 1;
const SETUP_STORAGE_KEY = "ybl-first-setup";
const readSetupDismissed = () => {
  try {
    const raw = localStorage.getItem(SETUP_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) as { dismissedVersion?: unknown } : null;
    return parsed?.dismissedVersion === CURRENT_SETUP_VERSION;
  } catch {
    return false;
  }
};
const readTutorialDismissed = () => {
  try {
    const raw = localStorage.getItem(TUTORIAL_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) as { dismissedVersion?: unknown } : null;
    return parsed?.dismissedVersion === CURRENT_TUTORIAL_VERSION;
  } catch {
    return false;
  }
};
type WorkflowStageName = StageName;
type RecoveryAction = {
  id: "retry_extract_with_lower_resource";
  stage: WorkflowStageName;
  requires_confirmation: boolean;
  impact: string;
};
type WorkflowStage = {
  status: "pending" | "ready" | "running" | "completed" | "failed" | "cancelled" | "stale" | "interrupted";
  progress: number;
  error: string | null;
  started_at: number | null;
  finished_at: number | null;
  config_fingerprint: string | null;
  error_code?: string | null;
  recovery?: RecoveryAction | null;
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
  issues?: string[];
  needs_review?: boolean;
  candidate_id?: string;
  confirmed_weak?: boolean;
  confirmed_source_edit?: boolean;
  translation_source?: string;
  source_stale?: boolean;
};
type VoiceReview = {
  status: "unavailable" | "pending" | "running" | "paused" | "completed" | "failed";
  channel?: "mix" | "left" | "right";
  baseline_kind?: string;
  candidates: Array<{ id: string; start: number; end: number; status: string; accepted_words?: number }>;
  remaining: number;
  recovered_words?: Array<{ candidate_id?: string; text?: string; weak?: boolean; adoptable?: boolean }>;
  actual_decode_seconds: number;
  error?: string | null;
  windows?: Array<{ id: string; start: number; end: number; status?: string }>;
  result_candidates?: Array<{ id: string; window_id: string; start: number; end: number; tier: "strong" | "weak" | "diagnostic"; adoptable?: boolean; words?: Array<{ text?: string; start?: number; end?: number }>; status?: string; confirmed_weak?: boolean }>;
  diagnostics?: Array<Record<string, unknown>>;
  resource_profile?: "background" | "balanced" | "maximum";
  cpu_threads?: number;
};
type CueBaseMeta = { revision: number; base_fingerprint: string; base_ids: number[] };
type DraftSnapshot = { cues: Cue[] };
type TranslationCandidate = { cueId: number; source: string; originalTranslation: string; translation: string; provider: string; model: string };
type CompactionCandidate = { cue_id: number; original: string; candidate: string; error?: string | null; still_overflowing: boolean };
const cueIssueLabels: Record<string, string> = {
  "layout-overflow": "超出画面",
  "dense-text": "字数偏多",
  "reading-speed": "阅读偏快",
  "short-duration": "时长过短",
  "invalid-time": "时间异常",
  overlap: "字幕重叠",
  "empty-source": "原文为空",
  "empty-translation": "译文为空",
};
type LlmCost = {
  status: "no_requests" | "estimated" | "range" | "partial" | "unknown";
  min_cny: string;
  max_cny: string;
  priced_calls: number;
  unpriced_calls: number;
  incomplete_calls: number;
  usage_unknown_requests: number;
  request_count_unknown_calls: number;
  pricing_sources: string[];
};
type LlmUsage = {
  schema_version: number;
  scope?: string;
  logical_calls: number;
  http_requests: number;
  sdk_retry_requests: number;
  thinking_fallback_requests: number;
  failed_calls: number;
  calls_without_usage: number;
  duration_ms_total: number;
  cache_reused_items: number;
  split_events: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  prompt_cache_hit_tokens: number | null;
  reasoning_tokens: number | null;
  cost?: LlmCost;
  recorded_since?: number;
  session_count?: number;
  token_field_calls: Record<string, number>;
  by_kind: Record<string, LlmUsage>;
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
  recovery?: RecoveryAction | null;
  result: {
    output_dir: string;
    source_srt: string;
    translated_srt: string;
    rendered_video: string;
    dub_video?: string;
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
  publish_copy?: { title: string; tags: string[]; has_metadata?: boolean };
  subtitle_extraction?: {
    mode: string;
    ocr_status: "not-requested" | "completed" | "fallback" | "failed";
    message: string | null;
  };
  content_warnings?: string[];
  quality_proposal?: { width: number | null; height: number | null; baseline_width?: number | null; baseline_height?: number | null; quality: string; requires_consent?: boolean; fingerprint?: string } | null;
  quality_consent?: boolean;
  llm_usage?: LlmUsage | null;
  llm_usage_total?: LlmUsage | null;
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
  audio_channel: "mix" | "left" | "right";
  beam_size: number;
  asr_hint: string;
  asr_vad_mode: "standard" | "sensitive";
  ocr_interval: number;
  ocr_crop_ratio: number;
  ocr_min_chars: number;
  ocr_language: string;
  subtitle_margin_ratio: number;
  render_crf: number;
  dub_enabled?: boolean;
  dub_mix?: "duck" | "dub_only";
  render_encoder: "auto" | "cpu" | "nvidia";
  resource_profile: "background" | "balanced" | "maximum";
  translator: string;
  target_lang: string;
  translate_model: string;
  deepseek_thinking: "disabled" | "enabled" | "auto";
  llm_pricing_mode: "auto" | "custom" | "off";
  llm_custom_pricing: Record<string, string>;
  smart_translation: boolean;
  smart_subtitle_layout: boolean;
  font_name: string;
  font_size: number;
  subtitle_display_mode: string;
  subtitle_preset: string;
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
  title?: string;
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
  dub_video?: string | null;
  dub_exists?: boolean;
  created_at: number | null;
  finished_at: number | null;
  llm_usage_total?: LlmUsage | null;
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
type HistoryDeletePlan = {
  mode: "record_only" | "record_and_outputs";
  job_ids: string[];
  tasks: Array<{ id: string; title: string; status: string; output_dir: string; files: Array<{ path: string; name: string; size: number }>; bytes: number; disposition?: "verified_files" | "record_only_required"; reason?: string }>;
  task_count: number;
  file_count: number;
  bytes: number;
  verified_files_count?: number;
  record_only_required_count?: number;
  fingerprint: string;
};
type OutputDeletePlan = {
  root: string;
  paths: string[];
  targets: Array<{ id: string; name: string; kind: "task" | "file"; record_will_remove: boolean; file_count: number; bytes: number }>;
  target_count: number;
  file_count: number;
  bytes: number;
  fingerprint: string;
};
type DeleteItemResult = { id: string; status: "deleted" | "failed" | "partial"; record_removed: boolean; bytes: number; error: string | null };
type PendingDelete =
  | { source: "history"; plan: HistoryDeletePlan; mode: "record_only" | "record_and_outputs" }
  | { source: "outputs"; plan: OutputDeletePlan };
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
  state?: "ready" | "missing" | "stopped" | "blocked" | "unavailable";
  action_label?: string | null;
  depends_on?: string | null;
  installer_available?: boolean;
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

const sourceLanguageOptions = [
  ["en", "英语（en）"], ["zh", "中文（zh）"], ["ja", "日语（ja）"], ["ko", "韩语（ko）"],
  ["fr", "法语（fr）"], ["de", "德语（de）"], ["es", "西班牙语（es）"], ["ru", "俄语（ru）"],
] as const;
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
  audio_channel: "mix",
  beam_size: 5,
  asr_hint: "",
  asr_vad_mode: "standard",
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
  deepseek_thinking: "disabled",
  llm_pricing_mode: "auto",
  llm_custom_pricing: {},
  smart_translation: true,
  smart_subtitle_layout: true,
  font_name: "Microsoft YaHei",
  font_size: 24,
  subtitle_display_mode: "translated",
  subtitle_preset: "soft",
  subtitle_color: "白色",
  subtitle_outline_color: "黑色",
  subtitle_effect: "描边",
  output_dir: "outputs/workbench_demo",
};
type NavIconName = "material" | "library" | "process" | "subtitle" | "publish" | "files" | "save" | "link" | "rerender" | "download" | "voice" | "translate" | "render" | "upload" | "check" | "running" | "error" | "pause" | "ear" | "speed" | "text" | "video" | "subtitle-file" | "audio" | "cover" | "file" | "book";
const nav: Array<[View, string, string, NavIconName]> = [
  ["material", "素材", "链接、本地视频与授权", "material"],
  ["library", "任务库", "待处理、可继续与成片", "library"],
  ["process", "处理", "识别、翻译与推理", "process"],
  ["subtitle", "字幕", "预览、编辑与渲染", "subtitle"],
  ["publish", "发布", "B 站与抖音投稿辅助", "publish"],
  ["files", "文件", "输出与空间管理", "files"],
];
function NavIcon({ name, className = "nav-icon" }: { name: NavIconName; className?: string }) {
  const paths: Record<NavIconName, ReactNode> = {
    material: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="m9 4 1.5-2h3L15 4M8 11l3 3 2-2 3 3" /></>,
    library: <><path d="M4 6.5h6l1.5 2H20v10.5H4z" /><path d="M4 6.5V5a2 2 0 0 1 2-2h4l1.5 2H20a2 2 0 0 1 2 2v1.5" /></>,
    process: <><path d="M5 5h14M5 12h14M5 19h14" /><circle cx="9" cy="5" r="2" /><circle cx="15" cy="12" r="2" /><circle cx="11" cy="19" r="2" /></>,
    subtitle: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 9h10M7 13h7M7 17h4" /></>,
    publish: <><path d="M12 16V4M8 8l4-4 4 4" /><path d="M5 13v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" /></>,
    files: <><path d="M3 6a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><path d="M3 9h18" /></>,
    save: <><path d="M5 3h12l3 3v15H4V3z" /><path d="M8 3v6h7V3M8 21v-7h8v7" /></>,
    link: <><path d="m9 15 6-6" /><path d="m7 17-1 1a3 3 0 0 1-4-4l3-3a3 3 0 0 1 4 0" /><path d="m17 7 1-1a3 3 0 0 1 4 4l-3 3a3 3 0 0 1-4 0" /></>,
    rerender: <><path d="M20 11a8 8 0 0 0-14-5L4 8" /><path d="M4 4v4h4M4 13a8 8 0 0 0 14 5l2-2" /><path d="M20 20v-4h-4" /></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5" /><path d="M4 20h16" /></>,
    voice: <><path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" /><path d="M19 11a7 7 0 0 1-14 0M12 18v3M8 21h8" /></>,
    translate: <><path d="M4 5h8M8 3v2M5 5c.5 4 2.5 7 6 9M11 5c-.6 2.4-1.8 4.2-3.4 5.8M14 12h6M17 9l-4 12M15 17h5" /></>,
    render: <><path d="M4 5h16v14H4z" /><path d="m10 9 5 3-5 3z" /></>,
    upload: <><path d="M12 16V4M8 8l4-4 4 4" /><path d="M5 13v5a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    running: <><path d="M4 12h3M10 12h4M17 12h3" /><circle cx="8" cy="12" r="1" /><circle cx="16" cy="12" r="1" /></>,
    error: <><circle cx="12" cy="12" r="9" /><path d="M12 7v6M12 16v1" /></>,
    pause: <><path d="M8 5v14M16 5v14" /></>,
    ear: <><path d="M18 13c0 3-3 3-3 6M15 19c0 1-1 2-2 2" /><path d="M7 10a5 5 0 1 1 9 3c-1 1-3 1-3 4" /><path d="M7 10V8a5 5 0 0 1 10-1" /></>,
    speed: <><path d="M4 16a8 8 0 1 1 16 0" /><path d="m12 12 4-3M6 19h12" /></>,
    text: <><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h8M8 16h5" /></>,
    video: <><rect x="3" y="5" width="14" height="14" rx="2" /><path d="m17 10 4-2v8l-4-2z" /></>,
    "subtitle-file": <><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M7 10h10M7 14h7" /></>,
    audio: <><path d="M5 9v6M9 6v12M13 3v18M17 7v10M21 10v4" /></>,
    cover: <><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8" cy="9" r="1" /><path d="m4 17 5-5 3 3 2-2 6 5" /></>,
    file: <><path d="M6 3h8l4 4v14H6z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21z" /><path d="M4 5.5v15M8 7h8M8 11h8" /></>,
  };
  return <svg className={className} aria-hidden="true" focusable="false" viewBox="0 0 24 24">{paths[name]}</svg>;
}
const timecode = (seconds: number, precise = false) => {
  const safe = precise
    ? Math.round(Math.max(0, seconds) * 10) / 10
    : Math.floor(Math.max(0, seconds));
  const totalMinutes = Math.floor(safe / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  const rest = safe - totalMinutes * 60;
  const wholeSeconds = Math.floor(rest);
  const base = hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`
    : String(minutes).padStart(2, "0");
  return `${base}:${precise ? rest.toFixed(1).padStart(4, "0") : String(wholeSeconds).padStart(2, "0")}`;
};

const parseTime = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const rawParts = trimmed.split(":");
  if (rawParts.some((item) => !item.trim())) return null;
  const parts = rawParts.map((item) => Number(item));
  if (!parts.length || parts.some((item) => Number.isNaN(item))) return null;
  let seconds = 0;
  if (parts.length === 3) seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
  else if (parts.length === 2) seconds = parts[0] * 60 + parts[1];
  else if (parts.length === 1) seconds = parts[0];
  return seconds >= 0 && Number.isFinite(seconds) ? seconds : null;
};
const formatTime = (seconds: number) => {
  const value = Math.round(Math.max(0, seconds) * 10) / 10;
  const minutes = Math.floor(value / 60);
  const rest = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
};

const truncateLabel = (text: string, max = 42) => {
  const value = (text || "").trim();
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(1, max - 1))}…`;
};

const formatOutputTaskStamp = (modifiedAt: number) => {
  if (!modifiedAt) return "";
  try {
    return new Date(modifiedAt * 1000).toLocaleString(undefined, {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
};

const outputTaskDisplayTitle = (task: OutputTask, fallbackTitles?: Map<string, string>) => {
  const fromApi = (task.title || "").trim();
  if (fromApi && fromApi !== task.name) return fromApi;
  const fromHistory = fallbackTitles?.get(task.name)?.trim();
  if (fromHistory) return fromHistory;
  return fromApi || task.name;
};

const formatPublishSourceLabel = (task: OutputTask, file: OutputFile, fallbackTitles?: Map<string, string>) => {
  const title = truncateLabel(outputTaskDisplayTitle(task, fallbackTitles), 36);
  const stamp = formatOutputTaskStamp(task.modified_at);
  const bits = [title];
  if (stamp) bits.push(stamp);
  if (file.name && file.name.toLowerCase() !== "rendered.mp4") bits.push(file.name);
  return bits.join(" · ");
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
// 后端的 VoxCPM 刷新接口定下来后只改这里。
const VOXCPM_CHECK_URL = "/api/dub/recheck";
const dubStageMeta: [WorkflowStageName, string, string] = ["dub", "配音", "用 VoxCPM 生成中文配音，另存 *_dub.mp4；字幕版成片照常保留。"];
const whisperModelOrder = ["tiny", "base", "small", "medium", "large-v3"] as const;
const stageStatusLabel: Record<string, string> = {
  pending: "待执行", ready: "可执行", running: "运行中", completed: "已完成",
  failed: "失败", cancelled: "已取消", stale: "需重做", interrupted: "意外中断",
};
const fileIconName = (kind: string, name = ""): NavIconName => {
  const value = `${kind} ${name}`.toLowerCase();
  if (value.includes("视频") || /\.(mp4|mkv|webm|mov)$/i.test(name)) return "video";
  if (value.includes("字幕") || /\.(srt|ass|vtt)$/i.test(name)) return "subtitle-file";
  if (value.includes("音频") || /\.(wav|mp3|m4a|aac)$/i.test(name)) return "audio";
  if (value.includes("预览") || value.includes("封面") || /\.(png|jpe?g|webp)$/i.test(name)) return "cover";
  return "file";
};

function FieldGuide({ summary, detail, badges = [] }: { summary: string; detail: string; badges?: string[] }) {
  return <span className="field-guide"><small>{summary}</small><span className="guide-badges">{badges.map((badge) => <i key={badge}>{badge}</i>)}</span><details><summary>？ 选择建议与影响</summary><p>{detail}</p></details></span>;
}

function AsrRecognitionFields({ options, onHint, onVad, hintError }: {
  options: Options;
  onHint: (value: string) => void;
  onVad: (value: Options["asr_vad_mode"]) => void;
  hintError?: string;
}) {
  const ocrOnly = options.subtitle_source === "ocr";
  return <>
    <label>
      人声检测
      <select value={options.asr_vad_mode || "standard"} disabled={ocrOnly} onChange={(event) => onVad(event.target.value as Options["asr_vad_mode"])}>
        <option value="standard">标准</option>
        <option value="sensitive">敏感（户外噪声）</option>
      </select>
      <FieldGuide summary="标准去掉长静音后再补回像有人声、但还没识别到的区间。" detail="敏感会更早把风声、水声里的说话当成语音，并在句首多留一点。风声和水声也可能被收进来。纯 OCR 不用这项。改完后需要重新提取字幕。" badges={options.asr_vad_mode === "sensitive" ? ["可能多收噪声"] : ["当前行为"]} />
    </label>
    <label className="settings-span-2">
      识别提示
      <input value={options.asr_hint || ""} disabled={ocrOnly} maxLength={240} placeholder="可选，例如 tarpon, ballyhoo, pompano jig" onChange={(event) => onHint(event.target.value)} />
      {hintError && <small className="field-error">{hintError}</small>}
      <FieldGuide summary="留空时，只从标题、简介和标签里取专名，不用先听片。" detail="手写词会追加在自动专名后面，用来偏向鱼名、地名这类听不清就猜错的词。使用平台字幕时不会发给 Whisper。改完后需要重新提取字幕。" badges={["可选"]} />
    </label>
  </>;
}

function ActivityBars() {
  return <span className="activity-bars" aria-hidden="true"><i /><i /><i /><i /><i /></span>;
}

function describeTaskError(message: string, fallbackTitle = "任务处理未完成", fallbackHint = "查看技术详情，确认原因后从失败阶段继续。"): { title: string; hint: string } {
  if (/WinError 32|另一个程序正在使用|being used by another process|Unable to rename file/i.test(message)) {
    return { title: "文件写入失败", hint: "目标文件可能正在被其他程序占用。关闭占用文件的程序后，再从失败阶段继续。" };
  }
  if (/WinError 10060|连接尝试失败|youtube_network_timeout|媒体流时连接超时|timed out|Timeout/i.test(message)) {
    return { title: "下载连接超时", hint: "YouTube 媒体 CDN 无响应。检查代理/VPN，或稍后重试「获取素材」。" };
  }
  if (/Sign in to confirm|not a bot|机器人验证|登录要求/i.test(message)) {
    return { title: "视频访问被平台拦截", hint: "请检查 YouTube 访问设置或稍后重试。" };
  }
  if (/403 Forbidden|HTTP Error 403|HTTP 403/i.test(message)) {
    return { title: "视频暂时无法访问", hint: "请检查链接、登录状态或网络连接。" };
  }
  if (/No such file|WinError 2|找不到指定的文件/i.test(message)) {
    return { title: "找不到素材文件", hint: "原文件可能已移动或删除，请检查素材位置。" };
  }
  if (/out of memory|显存不足|CUDA OOM/i.test(message)) {
    return { title: "处理资源不足", hint: "可尝试较小模型、低负载设置或 CPU 模式。" };
  }
  return { title: fallbackTitle, hint: fallbackHint };
}

function TaskErrorSummary({ message, fallbackTitle, fallbackHint }: { message: string; fallbackTitle?: string; fallbackHint?: string }) {
  const { title, hint } = describeTaskError(message, fallbackTitle, fallbackHint);
  return <div className="task-error-summary">
    <div className="task-error-overview"><b>{title}</b><span>{hint}</span></div>
    <details><summary>查看详情</summary><pre>{message}</pre></details>
  </div>;
}

function mostlyCjk(text: string): boolean {
  const chars = [...text].filter((character) => character.trim());
  if (!chars.length) return false;
  const cjk = chars.filter((character) => character >= "\u3400" && character <= "\u9fff").length;
  return cjk / chars.length > 0.35;
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

type ExternalAiDisclosure = {
  provider: string;
  endpoint_origin: string;
  model: string;
  data_kinds: string;
  operation: string;
  fee_notice: string;
};

function externalAiConfirmation(disclosure: ExternalAiDisclosure): boolean {
  return window.confirm([
    `即将进行${disclosure.operation}`,
    `服务商：${disclosure.provider}`,
    `端点：${disclosure.endpoint_origin}`,
    `模型：${disclosure.model}`,
    `发送内容：${disclosure.data_kinds}`,
    disclosure.fee_notice,
    "确认后仅允许当前请求发送；取消不会启动或修改任务。",
  ].join("\n"));
}

type BatchEntry = { url: string; title: string; duration?: number | null; error?: string; selected: boolean };
type SubscribedChannel = { id: string; url: string; title: string; added_at?: string };
type WeeklyHotVideo = {
  url: string;
  title: string;
  duration: number | null;
  upload_date: string | null;
  timestamp: number | null;
  channel_title: string;
  channel_url: string;
  channel_id: string;
  thumbnail_url?: string | null;
  view_count?: number | null;
};
type WeeklyChannelResult = { id: string; url: string; title: string; ok: boolean; error: string | null; count: number };
type WeeklyHot = {
  enabled: boolean;
  channels: SubscribedChannel[];
  videos: WeeklyHotVideo[];
  channel_results: WeeklyChannelResult[];
  refreshed_at: string | null;
  stale: boolean;
  refreshing?: boolean;
  window_days: number;
  top_k: number;
  fetch_count: number;
  show_thumbnails: boolean;
  cache_ttl_hours: number;
  channel_progress?: { done: number; total: number } | null;
  sort_mode?: "views" | "recent";
};
type SiteHot = {
  category: string;
  categories: { id: string; label: string }[];
  region: string;
  regions: { id: string; label: string }[];
  cached_region: string | null;
  videos: WeeklyHotVideo[];
  refreshed_at: string | null;
  stale: boolean;
  refreshing?: boolean;
  progress?: { done: number; total: number; category: string } | null;
  error: string | null;
  query_errors: string[];
  sort_mode?: "views";
};
function siteNeedsAutoRefresh(data: SiteHot): boolean {
  const neverFetched = data.videos.length === 0 && data.refreshed_at == null;
  return data.stale || neverFetched;
}

type SiteDurationBand = "all" | "voice" | "mid" | "longer";

function siteDurationMatches(duration: number | null, band: SiteDurationBand): boolean {
  if (band === "all") return true;
  if (typeof duration !== "number") return false;
  const minutes = duration / 60;
  if (band === "voice") return minutes >= 8 && minutes <= 25;
  if (band === "mid") return minutes > 25 && minutes <= 45;
  return minutes > 45;
}

const SITE_DURATION_BANDS: { id: SiteDurationBand; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "voice", label: "8–25 分钟" },
  { id: "mid", label: "25–45 分钟" },
  { id: "longer", label: "更长" },
];

function thumbnailCandidates(src: string): string[] {
  const id = src.match(/\/vi(?:_webp)?\/([A-Za-z0-9_-]{11})\//)?.[1] ?? "";
  // Signed hq720 links expire, and hqdefault/mqdefault stall on this network.
  // sddefault, the first frame, and the small default image answer.
  if (!id) return src ? [src] : [];
  return [
    `https://i.ytimg.com/vi/${id}/sddefault.jpg`,
    `https://i.ytimg.com/vi/${id}/0.jpg`,
    `https://i.ytimg.com/vi/${id}/default.jpg`,
  ];
}

function WeeklyThumb({ src }: { src: string }) {
  const candidates = thumbnailCandidates(src);
  const [index, setIndex] = useState(0);
  if (index >= candidates.length) return <span className="weekly-hot-thumb is-placeholder" aria-hidden="true" />;
  return <img className="weekly-hot-thumb" src={candidates[index]} alt="" onError={() => setIndex((value) => value + 1)} />;
}
const formatUploadDay = (value: string | null): string | null => {
  if (!value || !/^\d{8}$/.test(value)) return null;
  return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}`;
};
const formatDaysAgo = (timestamp: number | null): string | null => {
  if (!timestamp) return null;
  const upload = new Date(timestamp * 1000);
  const now = new Date();
  const days = Math.round((Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()) - Date.UTC(upload.getUTCFullYear(), upload.getUTCMonth(), upload.getUTCDate())) / 86400000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  return `${days} 天前`;
};
const formatClock = (seconds: number): string => {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remain = total % 60;
  const clock = `${minutes}:${String(remain).padStart(2, "0")}`;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remain).padStart(2, "0")}` : clock;
};
type BatchItem = { id: string; url: string; title: string; job_id: string | null; status: "pending" | "running" | "done" | "failed" | "cancelled"; error: string | null; stage: string | null; progress: number | null };
type BatchQueue = { items: BatchItem[]; active: boolean };

type GlossaryEntry = { source: string; target: string; note: string; aliases: string[] };
type GlossaryRow = { key: number; source: string; target: string; aliases: string; note: string; dirty: boolean; saved: boolean };
let glossaryKey = 0;
const toGlossaryRow = (entry: Partial<GlossaryEntry>, dirty = false): GlossaryRow => ({
  key: ++glossaryKey, source: entry.source ?? "", target: entry.target ?? "", aliases: (entry.aliases ?? []).join(", "), note: entry.note ?? "", dirty, saved: !dirty,
});
const fromGlossaryRow = (row: GlossaryRow): GlossaryEntry => ({
  source: row.source.trim(), target: row.target.trim(), note: row.note.trim(), aliases: row.aliases.split(/[,，]/).map((alias) => alias.trim()).filter(Boolean),
});
const glossaryMatchKey = (source: string) => source.trim().split(/\s+/).join(" ").toLowerCase();
const parseCsvLine = (line: string) => {
  const cells: string[] = [];
  let cell = "", quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted && ch === '"' && line[i + 1] === '"') { cell += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { cells.push(cell); cell = ""; }
    else cell += ch;
  }
  return [...cells, cell];
};
const parseGlossaryCsv = (text: string): GlossaryEntry[] => {
  const [head, ...rows] = text.trim().split(/\r?\n/).map(parseCsvLine);
  const cell = (row: string[], name: string) => (row[head.indexOf(name)] ?? "").trim();
  return rows.map((row) => ({ source: cell(row, "source"), target: cell(row, "target"), note: cell(row, "note"), aliases: cell(row, "aliases").split(";").map((alias) => alias.trim()).filter(Boolean) }));
};
const csvCell = (value: string) => /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
const glossaryCsv = (entries: GlossaryEntry[]) => ["source,target,aliases,note", ...entries.map((entry) => [entry.source, entry.target, entry.aliases.join(";"), entry.note].map(csvCell).join(","))].join("\r\n");
const downloadText = (name: string, text: string, type: string) => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const form = options?.body instanceof FormData;
  const method = (options?.method ?? "GET").toUpperCase();
  const mutation = !["GET", "HEAD", "OPTIONS"].includes(method);
  const request = (body?: BodyInit | null): RequestInit => ({
    ...options,
    ...(body === undefined ? {} : { body }),
    headers: form
      ? { ...(mutation && csrfToken ? { "X-YBL-CSRF": csrfToken } : {}), ...(options?.headers ?? {}) }
      : { "Content-Type": "application/json", ...(mutation && csrfToken ? { "X-YBL-CSRF": csrfToken } : {}), ...(options?.headers ?? {}) },
  });
  let response = await fetch(path, request());
  let payload = (await response.json()) as T & { error?: string; field?: string; code?: string; consent_id?: string; disclosure?: ExternalAiDisclosure };
  if (response.status === 409 && payload.code === "external_ai_consent_required" && payload.consent_id && payload.disclosure && !form && mutation) {
    if (!externalAiConfirmation(payload.disclosure)) throw new ApiError("已取消在线 AI 请求。", undefined, "external_ai_consent_cancelled");
    let original: unknown;
    try { original = options?.body === undefined ? {} : JSON.parse(String(options.body)); } catch { original = null; }
    if (!original || typeof original !== "object" || Array.isArray(original)) throw new ApiError("请求无法附加在线 AI 确认。", undefined, "external_ai_consent_invalid");
    const retryBody = JSON.stringify({ ...(original as Record<string, unknown>), external_ai_consent: { consent_id: payload.consent_id, approved: true } });
    response = await fetch(path, request(retryBody));
    payload = (await response.json()) as T & { error?: string; field?: string; code?: string };
  }
  if (!response.ok) throw new ApiError(payload.error ?? "本地服务请求失败", payload.field, payload.code);
  return payload;
}

function DeepseekThinkingChoice({ mode, model, onChange }: { mode: Options["deepseek_thinking"]; model: string; onChange: (mode: Options["deepseek_thinking"]) => void }) {
  const reasoner = model.toLowerCase().split(/[/:]/).pop() === "deepseek-reasoner";
  return <label>
    翻译速度
    <select value={mode} onChange={(event) => onChange(event.target.value as Options["deepseek_thinking"])}>
      <option value="disabled">快速（默认）</option>
      <option value="enabled">保留思考</option>
      <option value="auto">接口默认</option>
    </select>
    <small className="drawer-note">{reasoner ? "deepseek-reasoner 会保留思考。" : "快速模式关闭额外思考；效果有差异时可切回。"} 已完成批次继续复用，只影响新请求。兼容接口不支持时沿用其默认模式。</small>
  </label>;
}

function costLabel(usage: LlmUsage | null | undefined): string {
  const cost = usage?.cost;
  if (!cost) return "暂无费用记录";
  if (cost.status === "unknown") return "待核对";
  const money = (text: string) => {
    const value = Number(text);
    return value > 0 && value < 0.0001 ? "<¥0.0001" : `¥${value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}`;
  };
  const amount = cost.min_cny === cost.max_cny ? money(cost.min_cny) : `${money(cost.min_cny)}–${money(cost.max_cny)}`;
  return cost.status === "partial" ? `已知 ${amount}，部分待核对` : amount;
}

function LlmPricingChoice({ options, onChange }: { options: Options; onChange: (key: "llm_pricing_mode" | "llm_custom_pricing", value: Options["llm_pricing_mode"] | Record<string, string>) => void }) {
  const custom = options.llm_custom_pricing ?? {};
  const change = (key: string, value: string) => onChange("llm_custom_pricing", { ...custom, [key]: value });
  return <details className="llm-pricing-settings">
    <summary>费用估算设置</summary>
    <p>官方 DS 使用内置人民币单价；其他接口可填写账单单价。仅影响之后的请求。</p>
    <label>计价方式<select value={options.llm_pricing_mode ?? "auto"} onChange={(event) => onChange("llm_pricing_mode", event.target.value as Options["llm_pricing_mode"])}>
      <option value="auto">自动识别官方价格</option><option value="custom">填写自定义单价</option><option value="off">只记录用量</option>
    </select></label>
    {options.llm_pricing_mode === "custom" && <>
      <label>服务<select value={custom.provider ?? ""} onChange={(event) => change("provider", event.target.value)}><option value="">请选择</option><option value="deepseek">DeepSeek / 兼容接口</option><option value="openai">OpenAI</option></select></label>
      <label>接口地址<input value={custom.endpoint_origin ?? ""} placeholder="https://api.deepseek.com" onChange={(event) => change("endpoint_origin", event.target.value)} /></label>
      <label>计价模型<input value={custom.model ?? ""} placeholder="填写准确模型名（包括默认模型）" maxLength={128} onChange={(event) => change("model", event.target.value)} /></label>
      {[["cache_hit_input_cny_per_million", "缓存输入"], ["cache_miss_input_cny_per_million", "普通输入"], ["output_cny_per_million", "输出"]].map(([key, label]) => <label key={key}>{label} · 元 / 百万 Token<input type="number" min="0" max="1000000" step="any" value={custom[key] ?? ""} onChange={(event) => change(key, event.target.value)} /></label>)}
      <small>服务、接口和模型完全匹配才会采用这些价格；不另算思考 Token。</small>
    </>}
  </details>;
}

function LlmUsagePanel({ usage, total }: { usage?: LlmUsage | null; total?: LlmUsage | null }) {
  const count = (row: LlmUsage) => row.total_tokens == null ? "未返回" : `${row.total_tokens.toLocaleString()}${(row.token_field_calls.total_tokens ?? 0) < row.logical_calls ? "+" : ""}`;
  const kinds: Record<string, string> = { source_correction: "源文校对", translation: "字幕翻译", review: "译文复审", formatting: "字幕精简", metadata: "标题标签", segmentation: "语义断句", single_translation: "单条重译" };
  const seconds = Math.round((usage?.duration_ms_total ?? 0) / 1000);
  const summary = total ?? usage;
  const unknown = summary?.cost;
  const row = (label: string, value: LlmUsage, key: string, className = "") => <tr key={key} className={className}><th scope="row">{label}</th><td>{value.http_requests}{value.cost?.request_count_unknown_calls ? "+?" : ""}</td><td>{count(value)}</td><td>{costLabel(value)}</td></tr>;
  return <section className="llm-usage" aria-label="翻译费用">
    <b>翻译费用 · 人民币估算</b>
    <div className="llm-cost-table"><table><thead><tr><th>环节</th><th>请求</th><th>Token</th><th>估算费用</th></tr></thead><tbody>
      {usage && Object.entries(usage.by_kind).map(([kind, value]) => row(kinds[kind] ?? kind, value, kind))}
      {usage && row("本轮", usage, "current", "llm-cost-total")}
      {total && row("任务累计", total, "total", "llm-cost-total")}
    </tbody></table></div>
    {!usage && !total && <p>暂无费用记录。</p>}
    <p>累计包含重试和单条重译；{total?.recorded_since ? `从 ${new Date(total.recorded_since * 1000).toLocaleDateString()} 开始记录。` : "旧任务未记录的费用无法补算。"}</p>
    {unknown && (unknown.unpriced_calls > 0 || unknown.incomplete_calls > 0 || unknown.usage_unknown_requests > 0 || unknown.request_count_unknown_calls > 0) && <p>价格未匹配 {unknown.unpriced_calls} 次调用 · 用量不完整 {unknown.incomplete_calls} 次 · 另有 {unknown.usage_unknown_requests} 次请求未返回用量。待核对部分未计入金额。</p>}
    <details><summary>用量与价格说明</summary>
      <p>本轮重试 / 兼容回退 {usage?.sdk_retry_requests ?? 0} / {usage?.thinking_fallback_requests ?? 0} 次 · 复用 {usage?.cache_reused_items ?? 0} 条已有结果（不增加 API 费用）。</p>
      <p>本轮调用累计 {Math.floor(seconds / 60)} 分 {seconds % 60} 秒；并行调用的耗时会累加。</p>
      <p>官方 DS 内置单价更新于 2026-10-11，按请求开始时的缓存和峰谷价估算。未返回缓存拆分时显示费用区间。</p>
      {unknown?.pricing_sources.includes("custom") && <p>部分调用使用你填写的自定义人民币单价。</p>}
      <p>用量来自接口返回；估算金额不代表实际扣费，最终以服务商账单为准。</p>
    </details>
  </section>;
}

function App() {
  const fileInput = useRef<HTMLInputElement>(null);
  const publishFileInput = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const subtitlePlaybackEnd = useRef<number | null>(null);
  const trimPlaybackEnd = useRef<number | null>(null);
  const pendingPlayback = useRef<{ start: number; end: number; jobId: string; generation: number } | null>(null);
  const timelinePlayhead = useRef<HTMLDivElement>(null);
  const timelineCurrentTime = useRef<HTMLSpanElement>(null);
  const jobLogCursor = useRef(0);
  const historyRequestSeq = useRef(0);
  const publishHydrateKey = useRef("");
  const installTargetJobId = useRef<string | null>(null);
  const prepareSpecificJobRef = useRef<(jobId: string) => Promise<void>>(async () => undefined);
  const schemeDirtyRef = useRef(false);
  const [view, setView] = useState<View>("material");
  const [materials, setMaterials] = useState<Material[]>([]);
  const [materialId, setMaterialId] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [weekly, setWeekly] = useState<WeeklyHot | null>(null);
  const [weeklyBusy, setWeeklyBusy] = useState(false);
  const [weeklyError, setWeeklyError] = useState("");
  const [channelDraft, setChannelDraft] = useState("");
  const [channelSaving, setChannelSaving] = useState(false);
  const weeklyBusyRef = useRef(false);
  const [hotTab, setHotTab] = useState<"site" | "channels">("site");
  const [site, setSite] = useState<SiteHot | null>(null);
  const [siteCategory, setSiteCategory] = useState("mix");
  const [siteDurationBand, setSiteDurationBand] = useState<SiteDurationBand>("all");
  const [siteBusy, setSiteBusy] = useState(false);
  const [siteError, setSiteError] = useState("");
  const siteBusyRef = useRef(false);
  const siteCategoryRef = useRef("mix");
  const siteRegionRef = useRef("US");
  const siteAutoTriedRef = useRef(new Set<string>());
  const [materialSourceTab, setMaterialSourceTab] = useState<"url" | "local">("url");
  const sourceRevision = useRef(0);
  const [authorized, setAuthorized] = useState(false);
  const [fullVideo, setFullVideo] = useState(false);
  const [durationValue, setDurationValue] = useState("10");
  const [durationUnit, setDurationUnit] = useState<"秒" | "分钟" | "小时">("秒");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [device, setDevice] = useState("cuda");
  const [computeType, setComputeType] = useState("float16");
  const [options, setOptions] = useState<Options>(defaults);
  const [batchMode, setBatchMode] = useState(false);
  const [batchText, setBatchText] = useState("");
  const [batchEntries, setBatchEntries] = useState<BatchEntry[]>([]);
  const [batchQualityConsent, setBatchQualityConsent] = useState(false);
  const [batchDub, setBatchDub] = useState(false);
  const [voxcpm, setVoxcpm] = useState<{ voxcpm_available?: boolean; voxcpm_reason?: string; voxcpm_help?: string }>({});
  const [checkingVoxcpm, setCheckingVoxcpm] = useState(false);
  const recheckVoxcpm = async () => {
    setCheckingVoxcpm(true);
    try {
      const { voxcpm_available, voxcpm_reason, voxcpm_help } = await api<typeof voxcpm>(VOXCPM_CHECK_URL, { method: "POST" });
      setVoxcpm({ voxcpm_available, voxcpm_reason, voxcpm_help });
    } finally {
      setCheckingVoxcpm(false);
    }
  };
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchQueue, setBatchQueue] = useState<BatchItem[]>([]);
  const [batchSummaryDismissed, setBatchSummaryDismissed] = useState(false);
  const [appVersion, setAppVersion] = useState("");
  const [demoPreview, setDemoPreview] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const jobRef = useRef<Job | null>(job);
  jobRef.current = job;
  const selectedJobIdRef = useRef<string | null>(job?.id ?? null);
  const [cues, setCues] = useState<Cue[]>([]);
  const [cueBaseMeta, setCueBaseMeta] = useState<CueBaseMeta | null>(null);
  const cueOwner = useRef<string | null>(null);
  const cueGeneration = useRef(0);
  const [cuesReady, setCuesReady] = useState(false);
  const [voiceReview, setVoiceReview] = useState<VoiceReview | null>(null);
  const [voiceReviewOpen, setVoiceReviewOpen] = useState(false);
  const [voiceReviewBusy, setVoiceReviewBusy] = useState(false);
  const [voiceCandidateText, setVoiceCandidateText] = useState<Record<string, string>>({});
  const [voiceCandidateConfirmed, setVoiceCandidateConfirmed] = useState<Record<string, boolean>>({});
  const [voiceTranslationBusy, setVoiceTranslationBusy] = useState<string | null>(null);
  const [savingCues, setSavingCues] = useState(false);
  const savingCuesRef = useRef(false);
  const [history, setHistory] = useState<DraftSnapshot[]>([]);
  const [future, setFuture] = useState<DraftSnapshot[]>([]);
  const [activeCue, setActiveCue] = useState<number | null>(null);
  const [selectedCue, setSelectedCue] = useState<number | null>(null);
  const [cueFilter, setCueFilter] = useState<"all" | "review">("all");
  const [bulkSelectedCueIds, setBulkSelectedCueIds] = useState<number[]>([]);
  const [replaceField, setReplaceField] = useState<"source" | "translated">("translated");
  const [replaceScope, setReplaceScope] = useState<"all" | "selected">("all");
  const [replaceNeedle, setReplaceNeedle] = useState("");
  const [replaceValue, setReplaceValue] = useState("");
  const [replaceCaseSensitive, setReplaceCaseSensitive] = useState(false);
  const [translationCandidate, setTranslationCandidate] = useState<TranslationCandidate | null>(null);
  const [compactionCandidates, setCompactionCandidates] = useState<CompactionCandidate[]>([]);
  const [retranslatingCueId, setRetranslatingCueId] = useState<number | null>(null);
  const [compactingCues, setCompactingCues] = useState(false);
  const [preview, setPreview] = useState<"source" | "rendered" | "dub">("source");
  const dubSettings = (
    <section className="settings-section dub-section">
      <header className="settings-group-heading"><NavIcon name="voice" className="app-icon" /><span><b>配音</b><small>可选：渲染后用 VoxCPM 生成中文配音，另存 *_dub.mp4</small></span></header>
      <div className={`settings-grid settings-fold-body compact ${voxcpm.voxcpm_available ? "" : "is-disabled"}`}>
        <label className="check settings-check">
          <input type="checkbox" checked={!!options.dub_enabled} disabled={!voxcpm.voxcpm_available} onChange={(event) => updateOptions("dub_enabled", event.target.checked)} />
          生成配音版（*_dub.mp4）
        </label>
        <label>
          混音模式
          <select value={options.dub_mix ?? "duck"} disabled={!voxcpm.voxcpm_available} onChange={(event) => updateOptions("dub_mix", event.target.value as Options["dub_mix"])}>
            <option value="duck">压低原声、保留背景（推荐）</option>
            <option value="dub_only">只留配音</option>
          </select>
        </label>
      {voxcpm.voxcpm_available
        ? <small className="check-hint">字幕版成片照常保留，配音版单独输出。</small>
        : <div className="dub-unavailable"><b>VoxCPM 不可用：</b>{voxcpm.voxcpm_reason || "未检测到 VoxCPM。"}{voxcpm.voxcpm_help && <span className="dub-help">{voxcpm.voxcpm_help}</span>}</div>}
        <button type="button" className="settings-link-button" onClick={() => void recheckVoxcpm()} disabled={checkingVoxcpm}>{checkingVoxcpm ? "检测中…" : "重新检测"}</button>
      </div>
    </section>
  );
  const [mediaNonce, setMediaNonce] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const [online, setOnline] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [glossaryRows, setGlossaryRows] = useState<GlossaryRow[]>([]);
  const [glossaryRemoved, setGlossaryRemoved] = useState(0);
  const [glossaryQuery, setGlossaryQuery] = useState("");
  const [glossaryCount, setGlossaryCount] = useState<number | null>(null);
  const [glossaryNote, setGlossaryNote] = useState("");
  const glossaryFileRef = useRef<HTMLInputElement>(null);
  useEffect(() => { api<{ entries: GlossaryEntry[] }>("/api/glossary").then((result) => setGlossaryCount(result.entries.length)).catch(() => undefined); }, []);
  const glossaryUnsaved = glossaryRows.filter((row) => row.dirty).length + glossaryRemoved;
  const glossaryDuplicates = new Set(glossaryRows.map((row) => glossaryMatchKey(row.source)).filter((key, index, keys) => key && keys.indexOf(key) !== index));
  const loadGlossary = (entries: GlossaryEntry[]) => {
    setGlossaryRows(entries.map((entry) => toGlossaryRow(entry)));
    setGlossaryCount(entries.length);
    setGlossaryRemoved(0);
  };
  const openGlossary = async () => {
    loadGlossary((await api<{ entries: GlossaryEntry[] }>("/api/glossary")).entries);
    setGlossaryQuery("");
    setGlossaryNote("");
    setGlossaryOpen(true);
  };
  const closeGlossary = () => {
    if (glossaryUnsaved && !window.confirm("术语表有未保存的改动，确定放弃吗？")) return;
    setGlossaryOpen(false);
  };
  const saveGlossary = async () => {
    const entries = glossaryRows.map(fromGlossaryRow);
    const clearing = !entries.some((entry) => entry.source.trim());
    if (clearing && !window.confirm("确认清空术语表？此操作会删除全部术语。")) return;
    loadGlossary((await api<{ entries: GlossaryEntry[] }>("/api/glossary", { method: "POST", body: JSON.stringify({ entries, ...(clearing ? { confirmed: true } : {}) }) })).entries);
    setGlossaryNote(clearing ? "已清空术语表。" : "已保存。已完成翻译的任务再次载入时会提示重新翻译。");
  };
  const clearGlossary = async () => {
    if (!window.confirm("确认清空术语表？此操作会删除全部术语。")) return;
    loadGlossary((await api<{ entries: GlossaryEntry[] }>("/api/glossary", { method: "POST", body: JSON.stringify({ entries: [], confirmed: true }) })).entries);
    setGlossaryNote("已清空术语表。");
  };
  const updateGlossaryRow = (key: number, patch: Partial<GlossaryRow>) => setGlossaryRows((rows) => rows.map((row) => row.key === key ? { ...row, ...patch, dirty: true } : row));
  const removeGlossaryRow = (row: GlossaryRow) => {
    setGlossaryRows((rows) => rows.filter((item) => item.key !== row.key));
    if (row.saved) setGlossaryRemoved((count) => count + 1);
  };
  const importGlossary = async (file: File) => {
    let entries: GlossaryEntry[];
    try {
      const text = (await file.text()).replace(/^\uFEFF/, "");
      const data = file.name.toLowerCase().endsWith(".csv") ? parseGlossaryCsv(text) : JSON.parse(text);
      entries = Array.isArray(data) ? data : data.entries;
      if (!Array.isArray(entries)) throw new Error("缺少 entries 列表");
    } catch (error) {
      setGlossaryNote(`导入失败：${file.name} 不是有效的 JSON / CSV（${error}）`);
      return;
    }
    const valid = entries.filter((entry) => String(entry.source ?? "").trim());
    if (valid.length === 0) {
      if (!window.confirm("导入内容为空，确认清空术语表？")) return;
      loadGlossary((await api<{ entries: GlossaryEntry[] }>("/api/glossary", { method: "POST", body: JSON.stringify({ entries: [], confirmed: true }) })).entries);
      setGlossaryNote(`已导入 ${file.name}：术语表已清空。`);
      return;
    }
    const rows = [...glossaryRows];

    let replaced = 0, added = 0;
    for (const entry of entries) {
      const key = glossaryMatchKey(String(entry.source ?? ""));
      if (!key) continue;
      const row = toGlossaryRow(fromGlossaryRow(toGlossaryRow(entry)), true);
      const index = rows.findIndex((item) => glossaryMatchKey(item.source) === key);
      if (index >= 0) { rows[index] = { ...row, key: rows[index].key, source: rows[index].source, saved: rows[index].saved }; replaced++; }
      else { rows.push(row); added++; }
    }
    setGlossaryRows(rows);
    setGlossaryQuery("");
    setGlossaryNote(`已导入 ${file.name}：覆盖 ${replaced} 条，新增 ${added} 条。点「保存术语表」后才会生效。`);
  };
  const exportGlossary = (format: "json" | "csv") => {
    const entries = glossaryRows.map(fromGlossaryRow).filter((entry) => entry.source);
    if (format === "csv") downloadText("glossary.csv", "\uFEFF" + glossaryCsv(entries), "text/csv;charset=utf-8");
    else downloadText("glossary.json", JSON.stringify(entries, null, 2), "application/json");
  };
  const [shortcutHelpOpen, setShortcutHelpOpen] = useState(false);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [preferencesLoading, setPreferencesLoading] = useState(true);
  const [dismissedVersion, setDismissedVersion] = useState<number | null>(null);
  const [sessionDismissed, setSessionDismissed] = useState(readTutorialDismissed);
  const tutorialDismissed = sessionDismissed || dismissedVersion === CURRENT_TUTORIAL_VERSION;
  const [setupLoading, setSetupLoading] = useState(true);
  const [setupDismissedVersion, setSetupDismissedVersion] = useState<number | null>(null);
  const [setupSessionDismissed, setSetupSessionDismissed] = useState(readSetupDismissed);
  const [setupKey, setSetupKey] = useState("");
  const [setupPreset, setSetupPreset] = useState("deepseek");
  const [setupBase, setSetupBase] = useState("");
  const [setupModel, setSetupModel] = useState("");
  const [translationPresets, setTranslationPresets] = useState<TranslationPreset[]>([]);
  const [translationPreset, setTranslationPreset] = useState("");
  const [serviceBaseUrl, setServiceBaseUrl] = useState("");
  const [presetDirty, setPresetDirty] = useState(false);
  const [setupBusy, setSetupBusy] = useState(false);
  const [setupActive, setSetupActive] = useState<"ffmpeg" | "whisper" | "">("");
  const [setupProgress, setSetupProgress] = useState("");
  const [setupError, setSetupError] = useState("");
  const setupDismissed = setupSessionDismissed || setupDismissedVersion === CURRENT_SETUP_VERSION;
  const tutorialTriggerRef = useRef<HTMLButtonElement>(null);
  const [dependenciesOpen, setDependenciesOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const savingSettingsRef = useRef(false);
  const [exportingDiagnostics, setExportingDiagnostics] = useState(false);
  const [diagnosticsNote, setDiagnosticsNote] = useState("");
  const [message, setMessage] = useState("正在连接本地服务…");
  const [metadata, setMetadata] = useState<{
    duration: number | null;
    license: string | null;
    view_count: number | null;
    max_width: number | null;
    max_height: number | null;
    quality_proposal?: { width: number | null; height: number | null; quality: string; requires_consent?: boolean; fingerprint?: string } | null;
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
  const [publishPlatform, setPublishPlatform] = useState<"bilibili" | "douyin">(
    () => localStorage.getItem("ybl-publish-platform") === "douyin" ? "douyin" : "bilibili",
  );
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
  const preparingJobRef = useRef(preparingJob);
  preparingJobRef.current = preparingJob;
  const [outputs, setOutputs] = useState<Outputs | null>(null);
  const [selectedOutputPaths, setSelectedOutputPaths] = useState<string[]>([]);
  const [selectedOutputTaskIds, setSelectedOutputTaskIds] = useState<string[]>([]);
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [deleteResults, setDeleteResults] = useState<DeleteItemResult[] | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [historyDeleteError, setHistoryDeleteError] = useState("");
  const [libraryBatchMode, setLibraryBatchMode] = useState(false);
  const [expandedCostId, setExpandedCostId] = useState<string | null>(null);
  const [materialDraftToken, setMaterialDraftToken] = useState(0);
  const [selectedPublishVideo, setSelectedPublishVideo] = useState("");
  const [mode, setMode] = useState<"basic" | "advanced">(
    () => (localStorage.getItem("ybl-mode") as "basic" | "advanced") || "basic",
  );
  const [apiKeyConfigured, setApiKeyConfigured] = useState<boolean | null>(null);
  const [openaiKeyConfigured, setOpenaiKeyConfigured] = useState<boolean | null>(null);
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
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyLoadingMore, setHistoryLoadingMore] = useState(false);
  const [historyCategoryCounts, setHistoryCategoryCounts] = useState<Record<VideoLibraryCategory, number>>({ active: 0, attention: 0, completed: 0 });
  const [libraryFilter, setLibraryFilter] = useState<"all" | VideoLibraryCategory>("all");
  const [libraryQuery, setLibraryQuery] = useState("");
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [dependencies, setDependencies] = useState<Dependency[]>([]);
  const onlineTranslateModel = useRef("");
  const dependencySignature = useRef("");
  const dependencyAvailability = useRef<Record<string, boolean>>({});
  const autoPrepareEnvironmentKey = useRef("");
  const ocrLanguageRef = useRef(options.ocr_language || "eng");
  const [dependencyJob, setDependencyJob] = useState<DependencyInstallJob | null>(null);
  const [ollamaStartBusy, setOllamaStartBusy] = useState(false);
  const [tesseractInstallerBusy, setTesseractInstallerBusy] = useState(false);
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
  const ollamaRuntimeDependency = dependencies.find((item) => item.id === "ollama-runtime");
  const localModelTag = options.translate_model || "qwen3.5:4b";
  const knownLocalModels: Record<string, string> = {
    "qwen3.5:4b": "ollama-qwen35-4b",
    "qwen3.5:9b": "ollama-qwen35-9b",
    "translategemma:12b": "ollama-translategemma-12b",
  };
  const localModelDependencyId = knownLocalModels[options.translate_model] ?? "";
  const selectedLocalModelDependency = dependencies.find((item) => item.id === localModelDependencyId);
  const materialSourceReady = materialSourceTab === "url" ? Boolean(sourceUrl.trim()) : Boolean(selected);
  const running = ["queued", "running", "preparing", "cancelling"].includes(
    job?.status ?? "",
  );
  const runningRef = useRef(running);
  runningRef.current = running;
  const dirty = history.length > 0;
  const reviewCueCount = useMemo(() => cues.filter((cue) => cueNeedsReview(cue)).length, [cues]);
  const visibleCueRows = useMemo(
    () => cues.map((cue, index) => ({ cue, index })).filter(({ cue }) =>
      cueFilter === "all" || cueNeedsReview(cue),
    ),
    [cues, cueFilter],
  );
  const selectedBulkCues = useMemo(
    () => cues.filter((cue) => bulkSelectedCueIds.includes(cue.id) && !cue.deleted),
    [cues, bulkSelectedCueIds],
  );
  const selectedReviewCues = useMemo(
    () => selectedBulkCues.filter((cue) => cueNeedsReview(cue)),
    [selectedBulkCues],
  );
  const replacePreviews = useMemo(() => {
    if (!replaceNeedle) return [];
    const selectedIds = new Set(bulkSelectedCueIds);
    return cues.flatMap((cue) => {
      if (cue.deleted || (replaceScope === "selected" && !selectedIds.has(cue.id))) return [];
      const before = cue[replaceField];
      const result = replaceLiteral(before, replaceNeedle, replaceValue, replaceCaseSensitive);
      return result.count ? [{ cueId: cue.id, before, after: result.value, count: result.count }] : [];
    });
  }, [cues, replaceNeedle, replaceValue, replaceField, replaceScope, replaceCaseSensitive, bulkSelectedCueIds]);
  const taskModelLabel = job
    ? `${job.options.translator === "ollama" ? "Ollama 本地模型" : job.options.translator === "openai" ? "OpenAI" : job.options.translator === "deepseek" ? "DeepSeek" : "未配置翻译模型"} · ${job.options.translate_model || (["deepseek", "openai"].includes(job.options.translator) ? "服务默认模型" : job.options.translator === "ollama" ? "任务模型" : "无")}`
    : "当前任务模型";
  const taskModelIsOnline = job?.options.translator === "deepseek" || job?.options.translator === "openai";
  const publishVideoReady = Boolean(
    nativePublishToken || selectedPublishVideo || job?.result?.rendered_video || job?.artifacts?.rendered_video?.available,
  );
  const savedTemplateBody =
    templates.find((item) => item.name === templateName)?.body ?? "";
  const templateDraftDirty = templateBody !== savedTemplateBody;
  const duration =
    mediaDuration ??
    (demoPreview ? 10 : null) ??
    job?.material.duration_seconds ??
    selected?.duration_seconds ??
    metadata?.duration ??
    10;
  const previewSrc = demoPreview
    ? `/api/demo/rendered?v=${mediaNonce || "demo"}`
    : job
    ? `/api/jobs/${job.id}/media/${preview === "dub" ? "dub" : preview === "rendered" && job.result ? "rendered" : "source"}?v=${mediaNonce || job.stages?.render?.finished_at || "0"}`
    : selected
      ? `/api/materials/${selected.id}/media`
      : "";
  const cuePositions = useMemo(
    () =>
      cues.map((cue) => ({
        cue,
        left: `${(cue.start / duration) * 100}%`,
        width: `${Math.max(0.25, ((cue.end - cue.start) / duration) * 100)}%`,
      })),
    [cues, duration],
  );
  const librarySourceJobs = useMemo(() => {
    const rows = [...jobHistory];
    if (job) {
      const backend = rows.find((item) => item.id === job.id);
      const current: HistoryJob = {
        id: job.id,
        title: job.options.title || job.material.name || "未命名任务",
        status: job.status,
        stage: job.stage,
        progress: job.progress,
        error: job.error,
        output_dir: job.result?.output_dir ?? backend?.output_dir ?? null,
        rendered_video: job.result?.rendered_video ?? backend?.rendered_video ?? null,
        // Prefer backend filesystem probes; local path strings alone can outlive deleted folders.
        output_exists: backend?.output_exists ?? Boolean(job.result?.output_dir),
        rendered_exists: backend?.rendered_exists ?? Boolean(job.result?.rendered_video),
        created_at: job.started_at ?? backend?.created_at ?? null,
        finished_at: job.finished_at ?? backend?.finished_at ?? null,
      };
      const currentIndex = rows.findIndex((item) => item.id === job.id);
      if (currentIndex >= 0) rows[currentIndex] = { ...rows[currentIndex], ...current, output_exists: backend?.output_exists ?? current.output_exists, rendered_exists: backend?.rendered_exists ?? current.rendered_exists };
      else rows.unshift(current);
    }
    return rows;
  }, [jobHistory, job]);
  const outputTaskTitles = useMemo(
    () => new Map(
      jobHistory
        .filter((item) => item.output_dir)
        .map((item) => [item.output_dir!.split(/[\\/]/).pop() ?? "", item.title]),
    ),
    [jobHistory],
  );
  const libraryJobs = useMemo(() => {
    const resumableIds = new Set(restorableJobs.map((item) => item.id));
    return librarySourceJobs.filter((item) => {
      const category = videoLibraryCategory(item, resumableIds.has(item.id));
      return (libraryFilter === "all" || category === libraryFilter)
        && videoLibraryMatches(item, libraryQuery);
    });
  }, [librarySourceJobs, restorableJobs, libraryFilter, libraryQuery]);
  const libraryCounts = historyCategoryCounts;
  const visibleLibraryTaskIds = useMemo(
    () => selectableVideoLibraryIds(libraryJobs),
    [libraryJobs],
  );
  const selectablePage = visibleLibraryTaskIds.slice(0, 200);
  const allVisibleLibraryTasksSelected = selectablePage.length > 0
    && selectablePage.every((id) => selectedOutputTaskIds.includes(id));
  useEffect(() => {
    // A filter/search change must not leave hidden task IDs armed for deletion.
    setSelectedOutputTaskIds((current) => retainVisibleVideoLibrarySelection(current, libraryJobs));
  }, [visibleLibraryTaskIds]);

  const updateOptions = <K extends keyof Options>(key: K, value: Options[K]) => {
    setFieldErrors((current) => {
      if (!current[String(key)]) return current;
      const next = { ...current };
      delete next[String(key)];
      return next;
    });
    setOptions((current) => ({ ...current, [key]: value }));
  };
  const changeTranslator = (value: string) => {
    const localModels = ["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"];
    if (value === "ollama" && options.translator !== "ollama") {
      onlineTranslateModel.current = localModels.includes(options.translate_model) ? "" : options.translate_model;
      updateOptions("translate_model", "qwen3.5:4b");
      setApiKey("");
      setTranslationPreset("ollama");
      setPresetDirty(true);
    } else if (value !== "ollama" && options.translator === "ollama") {
      updateOptions("translate_model", onlineTranslateModel.current);
      if (value === "deepseek" || value === "openai") {
        setTranslationPreset(value);
        setPresetDirty(true);
      }
    } else if (value === "deepseek" || value === "openai") {
      const selected = translationPresets.find((item) => item.id === translationPreset);
      if (selected?.engine !== value) {
        setTranslationPreset(value);
        setPresetDirty(true);
      }
    }
    updateOptions("translator", value);
  };
  const selectTranslationPreset = (id: string) => {
    const preset = translationPresets.find((item) => item.id === id);
    if (!preset) return;
    const localModels = ["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"];
    if (preset.engine === "ollama" && options.translator !== "ollama") {
      onlineTranslateModel.current = localModels.includes(options.translate_model) ? "" : options.translate_model;
    }
    setTranslationPreset(id);
    setServiceBaseUrl(preset.base_url);
    setPresetDirty(true);
    setApiKey("");
    updateOptions("translator", preset.engine);
    updateOptions("translate_model", preset.model || preset.model_choices[0]?.id || "");
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
  const loadTemplates = async (platform: "bilibili" | "douyin" = publishPlatform) => {
    const response = await api<{ templates: Template[]; platform?: string }>(
      `/api/templates?platform=${encodeURIComponent(platform)}`,
    );
    const nextTemplates = response.templates;
    setTemplates(nextTemplates);
    const storedName = localStorage.getItem(`ybl-publish-template-${platform}`) || "";
    const preferred =
      (storedName && nextTemplates.find((item) => item.name === storedName)?.name)
      || (platform === "douyin" ? "中文字幕版" : "授权本地化");
    const resolved =
      nextTemplates.find((item) => item.name === preferred)?.name
      || nextTemplates[0]?.name
      || preferred;
    setTemplateName(resolved);
    setTemplateBody(nextTemplates.find((item) => item.name === resolved)?.body ?? "");
    return nextTemplates;
  };
  const loadSettings = async () => {
    const settings = await api<{
      cookies_from_browser?: string;
      cookies_file?: string;
      cookies_file_valid?: boolean | null;
      deepseek_key_configured?: boolean;
      openai_key_configured?: boolean;
      translation_presets?: TranslationPreset[];
      translation_preset?: string;
      translation_base_url?: string;
      translate_model?: string;
      default_output_dir?: string;
      youtube_proxy_configured?: boolean;
      youtube_po_token?: { available: boolean; browser_path: string | null };
      voxcpm_available?: boolean;
      voxcpm_reason?: string;
      voxcpm_help?: string;
    }>("/api/settings");
    setVoxcpm(settings);
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
    setOpenaiKeyConfigured(Boolean(settings.openai_key_configured));
    if (settings.translation_presets?.length) setTranslationPresets(settings.translation_presets);
    if (settings.translation_preset) setTranslationPreset(settings.translation_preset);
    if (settings.translation_base_url) setServiceBaseUrl(settings.translation_base_url);
    if (settings.translate_model) {
      setOptions((current) => current.translate_model ? current : { ...current, translate_model: settings.translate_model! });
    }
    setProxyConfigured(Boolean(settings.youtube_proxy_configured));
  };
  const loadWeekly = async () => {
    try {
      const data = await api<WeeklyHot>("/api/subscriptions/weekly");
      setWeekly(data);
      return data;
    } catch (error) {
      setWeeklyError(error instanceof Error ? error.message : "读取本周热片失败");
      return null;
    }
  };
  const refreshWeekly = async (channelId?: string) => {
    if (weeklyBusyRef.current) return;
    weeklyBusyRef.current = true;
    setWeeklyBusy(true);
    setWeeklyError("");
    try {
      const data = await api<WeeklyHot>("/api/subscriptions/weekly/refresh", {
        method: "POST",
        body: JSON.stringify({
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
          youtube_proxy: options.youtube_proxy,
          youtube_po_token_mode: options.youtube_po_token_mode,
          channel_id: channelId || undefined,
        }),
      });
      setWeekly(data);
      if (data.refreshing) setWeeklyError("正在刷新，请稍候再试。");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "刷新失败";
      setWeeklyError(detail);
      setMessage(`本周热片刷新失败：${detail}`);
    } finally {
      weeklyBusyRef.current = false;
      setWeeklyBusy(false);
    }
  };
  const saveWeeklyPrefs = async (patch: Partial<Pick<WeeklyHot, "window_days" | "top_k" | "fetch_count" | "cache_ttl_hours" | "show_thumbnails">>) => {
    setChannelSaving(true);
    try {
      const data = await api<WeeklyHot>("/api/subscriptions/settings", { method: "POST", body: JSON.stringify(patch) });
      setWeekly(data);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存列表设置失败");
    } finally {
      setChannelSaving(false);
    }
  };
  const setWeeklyEnabled = async (enabled: boolean) => {
    setChannelSaving(true);
    try {
      const data = await api<WeeklyHot>("/api/subscriptions/enabled", { method: "POST", body: JSON.stringify({ enabled }) });
      setWeekly(data);
      if (enabled && data.channels.length > 0 && data.stale) void refreshWeekly();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存关注开关失败");
    } finally {
      setChannelSaving(false);
    }
  };
  const addSubscribedChannel = async () => {
    const url = channelDraft.trim();
    if (!url) return;
    setChannelSaving(true);
    try {
      const data = await api<WeeklyHot>("/api/subscriptions", { method: "POST", body: JSON.stringify({ url }) });
      setChannelDraft("");
      setWeekly(data);
      setMessage("已添加频道。");
      if (data.enabled) void refreshWeekly();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "添加频道失败");
    } finally {
      setChannelSaving(false);
    }
  };
  const removeSubscribedChannel = async (id: string) => {
    setChannelSaving(true);
    try {
      setWeekly(await api<WeeklyHot>("/api/subscriptions/remove", { method: "POST", body: JSON.stringify({ id }) }));
      setMessage("已取消关注。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "移除频道失败");
    } finally {
      setChannelSaving(false);
    }
  };
  const loadSite = async (category = siteCategoryRef.current) => {
    try {
      const data = await api<SiteHot>(`/api/trending?category=${encodeURIComponent(category)}`);
      if (siteCategoryRef.current !== category) return data;
      siteRegionRef.current = data.region;
      setSite(data);
      setSiteCategory(data.category);
      return data;
    } catch (error) {
      setSiteError(error instanceof Error ? error.message : "读取热片失败");
      return null;
    }
  };
  const siteAutoKey = (category: string, region: string) => `${category}\n${region}`;
  const claimSiteAutoRefresh = (data: SiteHot | null | undefined) => {
    if (!data || !siteNeedsAutoRefresh(data)) return false;
    const key = siteAutoKey(data.category, data.region);
    if (siteAutoTriedRef.current.has(key)) return false;
    siteAutoTriedRef.current.add(key);
    return true;
  };
  const refreshSite = async () => {
    if (siteBusyRef.current) return;
    const category = siteCategoryRef.current;
    const requestedRegion = siteRegionRef.current;
    siteBusyRef.current = true;
    setSiteBusy(true);
    setSiteError("");
    try {
      const data = await api<SiteHot>("/api/trending/refresh", {
        method: "POST",
        body: JSON.stringify({
          category,
          ...(site ? { region: requestedRegion } : {}),
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
          youtube_proxy: options.youtube_proxy,
          youtube_po_token_mode: options.youtube_po_token_mode,
        }),
      });
      if (data.refreshing) siteAutoTriedRef.current.delete(siteAutoKey(category, requestedRegion));
      if (siteCategoryRef.current === category && siteRegionRef.current === requestedRegion) {
        siteRegionRef.current = data.region;
        setSite(data);
        setSiteCategory(data.category);
      }
      if (data.refreshing) setSiteError("正在刷新，请稍候再试。");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "刷新失败";
      setSiteError(detail);
      setMessage(`热片刷新失败：${detail}`);
    } finally {
      siteBusyRef.current = false;
      setSiteBusy(false);
      const nextCategory = siteCategoryRef.current;
      const nextRegion = siteRegionRef.current;
      if (nextCategory !== category || nextRegion !== requestedRegion) {
        void loadSite(nextCategory).then((data) => {
          if (siteCategoryRef.current !== nextCategory || siteRegionRef.current !== nextRegion || siteBusyRef.current) return;
          if (claimSiteAutoRefresh(data)) void refreshSite();
        });
      }
    }
  };
  const selectSiteCategory = (id: string) => {
    siteCategoryRef.current = id;
    setSiteCategory(id);
    setSiteError("");
    void loadSite(id).then((data) => {
      if (!data || siteCategoryRef.current !== id || siteBusyRef.current) return;
      if (claimSiteAutoRefresh(data)) void refreshSite();
    });
  };
  const saveSiteRegion = async (region: string) => {
    siteRegionRef.current = region;
    setSite((current) => current ? { ...current, region } : current);
    try {
      const data = await api<SiteHot>("/api/trending/region", {
        method: "POST",
        body: JSON.stringify({ region, category: siteCategoryRef.current }),
      });
      if (siteCategoryRef.current === data.category) {
        siteRegionRef.current = data.region;
        setSite(data);
      }
      if (siteCategoryRef.current !== data.category || siteRegionRef.current !== data.region || siteBusyRef.current) return;
      if (claimSiteAutoRefresh(data)) void refreshSite();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存地区失败");
    }
  };
  const loadRestorableJobs = async () => {
    const response = await api<{ jobs: RestorableJob[] }>("/api/jobs/restorable?limit=20");
    setRestorableJobs(response.jobs);
  };
  const loadRestoredJob = async (id: string) => {
    if ((dirty || materialDraftToken > 0) && !window.confirm(dirty ? "字幕还有未保存修改。载入其他任务会清空当前草稿，确定继续吗？" : "素材页还有未保存的链接/本地选择。载入任务会覆盖这些内容，确定继续吗？")) {
      return;
    }
    const generation = ++cueGeneration.current;
    selectedJobIdRef.current = id;
    pendingPlayback.current = null;
    cueOwner.current = null;
    setCues([]); setHistory([]); setFuture([]); setCuesReady(false);
    try {
      const restored = await api<Job>(`/api/jobs/${id}/load`, { method: "POST", body: "{}" });
      if (generation !== cueGeneration.current || selectedJobIdRef.current !== id) return;
      setJob(restored);
      setDemoPreview(false);
      setOptions((current) => ({
        ...mergeRestoredOptions(current, restored.options),
        cookies_file: current.cookies_file,
        youtube_proxy: current.youtube_proxy,
      }));
      setDevice(restored.device);
      setComputeType(restored.compute_type);
      sourceRevision.current += 1;
      setMetadata(null);
      setMetadataNotice("");
      setMetadataState("idle");
      setSourceUrl(restored.material.source_url ?? "");
      setMaterialId(restored.material.source_url ? "" : restored.material.id);
      setMaterialSourceTab(restored.material.source_url ? "url" : "local");
      setAuthorized(restored.material.authorized);
      setMaterialDraftToken(0);
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
      setSelectedCue(null);
      const cuesLoaded = await loadCues(restored.id, generation);
      setView("process");
      const completedStageCount = Object.values(restored.stages ?? {}).filter((stage) => stage.status === "completed").length;
      if (!cuesLoaded) return;
      setMessage(restored.edit_state === "legacy-ambiguous"
        ? "已载入旧任务并保留现有成片；后期版本顺序无法确认，已禁用重新渲染。"
        : !(restored.checks?.length)
          ? restored.status === "completed" && !restored.next_stage
            ? "已载入已完成任务，可查看字幕与成片；如需重新处理，请先检查当前环境与配置。"
            : "任务已载入。由于应用已重启，请重新运行准备检查，确认当前环境与配置后再继续。"
        : restored.status === "draft" && completedStageCount === 0
          ? "任务草稿已载入。请运行准备检查；当前不会自动下载或占用大量电脑资源。"
        : restored.checkpoint_validation === "invalid"
          ? "已载入任务，但部分检查点无效；请先运行准备检查。"
          : "任务已载入。不会自动继续耗资源处理，请确认后再执行下一阶段。");
      await loadRestorableJobs();
    } catch (error) {
      if (selectedJobIdRef.current === id) {
        selectedJobIdRef.current = null;
        cueOwner.current = null;
        setCuesReady(false);
      }
      setMessage(error instanceof Error ? error.message : "载入任务失败");
    }
  };
  const parseBatch = async () => {
    setBatchBusy(true);
    try {
      const result = await api<{ entries: Omit<BatchEntry, "selected">[] }>("/api/batch/parse", { method: "POST", body: JSON.stringify({ text: batchText, options: taskOptions() }) });
      setBatchEntries(result.entries.map((entry) => ({ ...entry, selected: !entry.error })));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBatchBusy(false);
    }
  };

  const startBatch = async () => {
    if (!authorized) {
      setRightsHint(true);
      setMessage("请先确认拥有处理和转载授权。");
      return;
    }
    setBatchBusy(true);
    try {
      const result = await api<BatchQueue>("/api/batch/start", {
        method: "POST",
        body: JSON.stringify({
          entries: batchEntries.filter((entry) => entry.selected).map(({ url, title }) => ({ url, title })),
          authorized, device, compute_type: computeType, options: { ...taskOptions(), dub_enabled: batchDub }, quality_consent: batchQualityConsent,
        }),
      });
      setBatchQueue(result.items);
      setBatchSummaryDismissed(false);
      setBatchEntries([]);
      setBatchText("");
      setView("library");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBatchBusy(false);
    }
  };

  const retryBatch = async (itemId?: string) => {
    try {
      const result = await api<BatchQueue>("/api/batch/retry", { method: "POST", body: JSON.stringify({ item_id: itemId ?? null }) });
      setBatchQueue(result.items);
      setBatchSummaryDismissed(false);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const cancelBatch = async () => {
    try {
      const result = await api<BatchQueue>("/api/batch/cancel", { method: "POST", body: "{}" });
      setBatchQueue(result.items);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const cancelBatchItem = async (jobId: string) => {
    try {
      await api<Job>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: "{}" });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  };

  useEffect(() => {
    if (view !== "library") return;
    const load = () => { api<BatchQueue>("/api/batch").then((result) => { setBatchQueue(result.items); if (result.active) setBatchSummaryDismissed(false); }).catch(() => undefined); };
    load();
    const timer = window.setInterval(load, 3000);
    return () => window.clearInterval(timer);
  }, [view]);

  const loadJobHistory = async (append = false) => {
    const requestId = ++historyRequestSeq.current;
    if (append) setHistoryLoadingMore(true);
    else setHistoryLoadingMore(false);
    try {
      const query = new URLSearchParams({
        limit: "200",
        offset: append ? String(jobHistory.length) : "0",
        scope: historyScope,
        output_dir: options.output_dir,
      });
      if (view === "library") {
        if (libraryQuery.trim()) query.set("q", libraryQuery.trim());
        if (libraryFilter !== "all") query.set("category", libraryFilter);
      }
      const response = await api<{ jobs: HistoryJob[]; total: number; has_more?: boolean; counts?: Record<VideoLibraryCategory, number> }>(`/api/history/jobs?${query}`);
      if (requestId !== historyRequestSeq.current) return;
      setJobHistory((current) => append ? [...current, ...response.jobs] : response.jobs);
      setHistoryTotal(response.total);
      setHistoryHasMore(response.has_more ?? (append ? jobHistory.length + response.jobs.length < response.total : response.jobs.length < response.total));
      if (response.counts) setHistoryCategoryCounts(response.counts);
    } catch {
      /* history is best-effort */
    } finally {
      if (append && requestId === historyRequestSeq.current) setHistoryLoadingMore(false);
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
  const loadDependencies = async (force = false) => {
    const requestedJobId = jobRef.current?.id ?? null;
    const response = await api<{
      dependencies: Dependency[];
      active_job: DependencyInstallJob | null;
    }>(`/api/dependencies?ocr_language=${encodeURIComponent(ocrLanguageRef.current)}${force ? "&refresh=1" : ""}`);
    const previousAvailability = dependencyAvailability.current;
    const newlyAvailable = response.dependencies
      .filter((item) => previousAvailability[item.id] === false && item.available)
      .map((item) => item.id);
    dependencyAvailability.current = Object.fromEntries(response.dependencies.map((item) => [item.id, item.available]));
    const nextSignature = JSON.stringify(response.dependencies.map((item) => [item.id, item.available, item.installable, item.state, item.message]));
    if (nextSignature !== dependencySignature.current) {
      dependencySignature.current = nextSignature;
      setDependencies(response.dependencies);
      setDiagnostics(response.dependencies.map(({ installable: _installable, size_hint: _size, action_url: _url, ...item }) => item));
    }
    const currentJob = jobRef.current;
    const resolvedBlockers = newlyAvailable.filter((id) => currentJob?.checks?.some((check) => check.id === id && check.status === "blocking"));
    if (currentJob && resolvedBlockers.length) {
      const recoveryKey = `${currentJob.id}:${resolvedBlockers.sort().join(",")}`;
      if (autoPrepareEnvironmentKey.current !== recoveryKey) {
        autoPrepareEnvironmentKey.current = recoveryKey;
        const installTarget = installTargetJobId.current;
        if (requestedJobId !== currentJob.id || (installTarget && installTarget !== currentJob.id)) {
          setMessage("本机环境已更新，但当前任务与开始安装时的任务选择不同。请在当前任务中点击“重新检查准备”后继续。");
        } else if (!runningRef.current && !preparingJobRef.current && !schemeDirtyRef.current && jobRef.current?.id === currentJob.id) {
          setMessage("检测到本机依赖或模型已安装，正在刷新当前任务的准备检查…");
          void prepareSpecificJobRef.current(currentJob.id);
        } else {
          setMessage("检测到本机环境已更新。保存当前方案后点击“重新检查准备”即可使用，无需重启应用。 ");
        }
      }
    }
    setDependencyJob((current) => {
      const active = response.active_job;
      if (!active) return current && ["completed", "failed", "cancelled"].includes(current.status) ? current : null;
      if (current?.id === active.id && current.status === active.status && current.progress === active.progress && current.message === active.message) return current;
      return active;
    });
    return response.dependencies;
  };
  const startOllama = async () => {
    if (ollamaStartBusy) return;
    setOllamaStartBusy(true);
    try {
      const started = await api<{ state: "ready" | "starting"; message: string }>("/api/dependencies/ollama/start", {
        method: "POST",
        body: "{}",
      });
      setMessage(started.message);
      if (started.state === "ready") {
        await loadDependencies(true);
        return;
      }
      for (let attempt = 0; attempt < 12; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        const latest = await loadDependencies(true);
        const runtime = latest.find((item) => item.id === "ollama-runtime");
        if (runtime?.state === "ready") {
          if (!jobRef.current?.checks?.some((check) => check.id === "ollama-runtime" && check.status === "blocking")) {
            setMessage("Ollama 已连接。 ");
          }
          return;
        }
        if (runtime?.message.startsWith("Ollama 启动失败")) {
          setMessage(runtime.message);
          return;
        }
      }
      setMessage("已启动 Ollama，但服务仍未就绪。请稍后重新检查；若仍未连接，请检查 Ollama 状态。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法启动 Ollama。 ");
    } finally {
      setOllamaStartBusy(false);
    }
  };
  const startTesseractInstaller = async () => {
    if (tesseractInstallerBusy) return;
    setTesseractInstallerBusy(true);
    try {
      const response = await api<{ message: string }>("/api/dependencies/tesseract/start", { method: "POST", body: "{}" });
      setMessage(response.message);
      for (let attempt = 0; attempt < 24; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 5000));
        const latest = await loadDependencies(true);
        if (latest.find((item) => item.id === "tesseract")?.available) {
          setMessage("已检测到 Tesseract，可以继续使用。 ");
          return;
        }
      }
      setMessage("安装器已启动，但暂未检测到 Tesseract。完成安装后可再次点击重新检测。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法启动 Tesseract 安装器");
    } finally {
      setTesseractInstallerBusy(false);
    }
  };
  const installDependency = async (dependency: Dependency) => {
    const action = dependency.action_label === "下载模型" ? "下载" : "安装";
    if (!window.confirm(`${action}“${dependency.label}”吗？\n\n用途：${dependency.purpose}\n预计占用：${dependency.size_hint}\n\n过程需要联网；模型下载可取消，取消不会删除 Ollama 已保存的数据。`)) return;
    installTargetJobId.current = job?.id ?? "__no-task-selected__";
    try {
      const response = await api<{ job: DependencyInstallJob }>(`/api/dependencies/${dependency.id}/install`, {
        method: "POST",
        body: JSON.stringify({ confirmed: true, ocr_language: options.ocr_language || "eng" }),
      });
      setDependencyJob(response.job);
      setDependenciesOpen(true);
      setMessage(response.job.message);
      if (response.job.status === "completed") {
        await loadDependencies(true);
        installTargetJobId.current = null;
      }
    } catch (error) {
      installTargetJobId.current = null;
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
      translation_presets?: TranslationPreset[];
      translation_preset?: string;
      translation_base_url?: string;
    }>("/api/bootstrap");
    csrfToken = bootstrap.csrf_token;
    setAppVersion(bootstrap.version);
    setOptions((current) => ({ ...current, ...bootstrap.defaults }));
    if (bootstrap.translation_presets?.length) setTranslationPresets(bootstrap.translation_presets);
    if (bootstrap.translation_preset) setTranslationPreset(bootstrap.translation_preset);
    if (bootstrap.translation_base_url) setServiceBaseUrl(bootstrap.translation_base_url);
    const initialPreset = bootstrap.translation_presets?.find((item) => item.id === (bootstrap.translation_preset || "deepseek")) ?? bootstrap.translation_presets?.[0];
    if (initialPreset) {
      setSetupPreset(initialPreset.id);
      setSetupBase(initialPreset.base_url);
      setSetupModel(bootstrap.defaults.translate_model || initialPreset.model || initialPreset.model_choices[0]?.id || "");
    }
    setDiagnostics(Object.entries(bootstrap.capabilities).map(([id, item]) => ({
      id, ...item,
      message: item.available ? "已检测到。" : item.required ? "必需组件未安装。" : "可选组件未安装；不影响默认音频模式。",
    })));
  };
  const loadTutorialPreferences = async () => {
    try {
      const preferences = await api<{ dismissedVersion: number }>("/api/tutorial-preferences");
      setDismissedVersion(preferences.dismissedVersion);
    } catch {
      setSessionDismissed(readTutorialDismissed());
    } finally {
      setPreferencesLoading(false);
    }
  };
  const loadSetupPreferences = async () => {
    try {
      const preferences = await api<{ dismissedVersion: number }>("/api/setup-preferences");
      setSetupDismissedVersion(preferences.dismissedVersion);
    } catch {
      setSetupSessionDismissed(readSetupDismissed());
    } finally {
      setSetupLoading(false);
    }
  };
  const dismissSetup = () => {
    setSetupDismissedVersion(CURRENT_SETUP_VERSION);
    setSetupSessionDismissed(true);
    try { localStorage.setItem(SETUP_STORAGE_KEY, JSON.stringify({ dismissedVersion: CURRENT_SETUP_VERSION })); } catch { /* Keep the dismissal for this session when storage is unavailable. */ }
    if (!csrfToken) return;
    void api<{ saved: boolean }>("/api/setup-preferences", {
      method: "POST",
      body: JSON.stringify({ dismissedVersion: CURRENT_SETUP_VERSION }),
    }).catch(() => setMessage("准备页偏好未能保存，下次启动可能再次显示。"));
  };
  const waitForSetupInstall = async (jobId: string) => {
    for (;;) {
      const latest = await api<DependencyInstallJob>(`/api/dependencies/jobs/${jobId}`);
      setSetupProgress(latest.message);
      if (latest.status === "completed") return;
      if (latest.status === "failed" || latest.status === "cancelled") {
        throw new Error(latest.error || latest.message || "安装没有完成");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
  };
  const installSetupPiece = async (id: "ffmpeg" | "whisper-small") => {
    setSetupActive(id === "ffmpeg" ? "ffmpeg" : "whisper");
    setSetupProgress(id === "ffmpeg" ? "正在下载 FFmpeg…" : "正在下载语音模型…");
    const response = await api<{ job: DependencyInstallJob }>(`/api/dependencies/${id}/install`, {
      method: "POST",
      body: JSON.stringify({ confirmed: true, ocr_language: "eng" }),
    });
    if (response.job.status === "failed" || response.job.status === "cancelled") {
      throw new Error(response.job.error || response.job.message || "安装没有完成");
    }
    if (response.job.status !== "completed") await waitForSetupInstall(response.job.id);
  };
  const finishSetup = async () => {
    if (setupBusy) return;
    setSetupBusy(true);
    setSetupError("");
    try {
      let rows = await loadDependencies(true);
      if (!rows.some((item) => item.id === "ffmpeg" && item.available)) {
        await installSetupPiece("ffmpeg");
        rows = await loadDependencies(true);
      }
      if (!rows.some((item) => item.id === "whisper-small" && item.available)) {
        await installSetupPiece("whisper-small");
        rows = await loadDependencies(true);
      }
      if (!rows.some((item) => item.id === "ffmpeg" && item.available) || !rows.some((item) => item.id === "whisper-small" && item.available)) {
        throw new Error("组件还没就绪，可以再点一次补齐。");
      }
      const preset = translationPresets.find((item) => item.id === setupPreset);
      if (!preset) throw new Error("翻译服务列表还没载入，请稍后再试。");
      const model = setupModel.trim() || preset.model || preset.model_choices[0]?.id || "";
      const base = preset.base_editable ? setupBase.trim() : preset.base_url;
      if (preset.needs_key && !preset.key_optional && !setupKey.trim()) {
        throw new Error("请先填写密钥。还没有密钥时，可以点「以后再看」。");
      }
      if (preset.base_editable && !base) throw new Error("请填写接口地址。");
      if (!model) throw new Error("请填写模型名。");
      await api("/api/settings", {
        method: "POST",
        body: JSON.stringify({
          translator: preset.engine,
          translation_preset: preset.id,
          api_key: setupKey.trim(),
          base_url: base,
          translate_model: model,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
        }),
      });
      updateOptions("translator", preset.engine);
      updateOptions("translate_model", model);
      setTranslationPreset(preset.id);
      setServiceBaseUrl(base);
      setSetupKey("");
      if (preset.engine === "deepseek") setApiKeyConfigured(true);
      if (preset.engine === "openai") setOpenaiKeyConfigured(true);
      dismissSetup();
      setMessage("这台电脑已准备好。");
    } catch (error) {
      setSetupError(error instanceof Error ? error.message : "准备没有完成");
    } finally {
      setSetupBusy(false);
      setSetupActive("");
    }
  };
  const selectSetupPreset = (id: string) => {
    const preset = translationPresets.find((item) => item.id === id);
    setSetupPreset(id);
    setSetupError("");
    if (!preset) return;
    setSetupBase(preset.base_url);
    setSetupModel(preset.model || preset.model_choices[0]?.id || "");
    setSetupKey("");
  };
  const openKeyPage = (presetId: string) => {
    void api("/api/setup/open-key-page", { method: "POST", body: JSON.stringify({ preset: presetId }) }).catch((error) => {
      setSetupError(error instanceof Error ? error.message : "无法打开密钥页面");
    });
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
      const response = await api<{ message: string; deleted_ids?: string[] }>("/api/history/clear", {
        method: "POST",
        body: JSON.stringify({ confirmed: true, scope: historyScope, output_dir: options.output_dir }),
      });
      for (const id of response.deleted_ids ?? []) clearLoadedTaskState(id);
      setSelectedOutputTaskIds((current) => current.filter((id) => !(response.deleted_ids ?? []).includes(id)));
      setMessage(response.message);
      await Promise.all([loadJobHistory(), loadRestorableJobs()]);
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
  const clearLoadedTaskState = (id: string) => {
    if (job?.id !== id) return;
    cueGeneration.current += 1;
    selectedJobIdRef.current = null;
    cueOwner.current = null;
    setCuesReady(false);
    setJob(null);
    setCues([]);
    setHistory([]);
    setFuture([]);
    setActiveCue(null);
    setSelectedCue(null);
    setPreview("source");
    setInspectorOpen(false);
    setView("library");
  };
  const deleteSelectedHistoryTasks = async (mode: "record_only" | "record_and_outputs") => {
    if (selectedOutputTaskIds.length > 200) {
      setHistoryDeleteError("批量删除每次最多选择 200 个任务，请减少选择后重试。 ");
      return;
    }
    const ids = selectedOutputTaskIds.filter((id) => libraryJobs.some((item) => item.id === id && !["queued", "running", "preparing", "cancelling"].includes(item.status)));
    if (!ids.length) {
      setHistoryDeleteError("请至少选择一条可删除的历史任务。");
      return;
    }
    setDeleteBusy(true);
    setHistoryDeleteError("正在生成删除预览…");
    try {
      const plan = await api<HistoryDeletePlan>("/api/history/jobs/delete-preview", {
        method: "POST", body: JSON.stringify({ ids, mode }),
      });
      setDeleteResults(null);
      setPendingDelete({ source: "history", plan, mode });
      setHistoryDeleteError("");
    } catch (error) {
      setHistoryDeleteError(error instanceof Error ? error.message : "读取删除预览失败");
    } finally {
      setDeleteBusy(false);
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
    const taskIds = new Set(snapshot.tasks.map((task) => task.id));
    const fileIds = new Set(snapshot.tasks.find((task) => task.id === viewedTaskId)?.files.map((file) => file.id) ?? []);
    setSelectedOutputPaths((current) => current.filter((id) => taskIds.has(id) || fileIds.has(id)));
    setViewedTaskId((current) => snapshot.tasks.some((task) => task.id === current) ? current : null);
  };
  const loadCues = async (id: string, generation = cueGeneration.current): Promise<boolean> => {
    try {
      const data = await api<{ cues: Cue[]; ready?: boolean; translation_ready?: boolean; revision?: number; base_fingerprint?: string; base_ids?: number[] }>(
        `/api/jobs/${id}/cues`,
      );
      if (generation !== cueGeneration.current || selectedJobIdRef.current !== id) return false;
      const ready = data.ready ?? true;
      cueOwner.current = ready ? id : null;
      setCues(data.cues ?? []);
      if (typeof data.revision === "number" && data.base_fingerprint && Array.isArray(data.base_ids)) setCueBaseMeta({ revision: data.revision, base_fingerprint: data.base_fingerprint, base_ids: data.base_ids });
      setCuesReady(ready);
      setTranslationReady(data.translation_ready ?? true);
      setHistory([]);
      setFuture([]);
      setBulkSelectedCueIds([]);
      setTranslationCandidate(null);
      setVoiceCandidateText({}); setVoiceCandidateConfirmed({});
      setCompactionCandidates([]);
      return true;
    } catch (error) {
      if (generation !== cueGeneration.current || selectedJobIdRef.current !== id) return false;
      cueOwner.current = null;
      setCues([]); setCueBaseMeta(null); setHistory([]); setFuture([]); setCuesReady(false);
      setMessage(error instanceof Error ? `字幕加载失败：${error.message}` : "字幕加载失败，请重试");
      return false;
    }
  };
  const loadVoiceReview = async (id: string) => {
    try {
      const data = await api<VoiceReview>(`/api/jobs/${id}/voice-review`);
      if (selectedJobIdRef.current === id) setVoiceReview(data);
    } catch {
      if (selectedJobIdRef.current === id) setVoiceReview(null);
    }
  };
  const prepareVoiceReview = async () => {
    if (!job || dirty || running) {
      setMessage(dirty ? "请先保存字幕草稿，再准备语音复核。" : "当前任务正在运行，请稍后再试。");
      return;
    }
    setVoiceReviewBusy(true);
    const requestJobId = job.id;
    const requestGeneration = cueGeneration.current;
    try {
      const data = await api<VoiceReview>(`/api/jobs/${job.id}/voice-review/prepare`, { method: "POST", body: "{}" });
      if (requestGeneration !== cueGeneration.current || selectedJobIdRef.current !== requestJobId) return;
      setVoiceReview(data); setVoiceReviewOpen(true);
    } catch (error) { setMessage(error instanceof Error ? error.message : "准备语音复核失败"); }
    finally { setVoiceReviewBusy(false); }
  };
  const startVoiceReview = async () => {
    if (!job || dirty || running) { setMessage("请先保存字幕草稿，并等待当前任务完成。"); return; }
    setVoiceReviewBusy(true);
    const requestJobId = job.id;
    const requestGeneration = cueGeneration.current;
    try {
      const data = await api<VoiceReview>(`/api/jobs/${job.id}/voice-review/${voiceReview?.status === "paused" ? "resume" : "start"}`, { method: "POST", body: "{}" });
      if (requestGeneration !== cueGeneration.current || selectedJobIdRef.current !== requestJobId) return;
      setVoiceReview(data);
    } catch (error) { setMessage(error instanceof Error ? error.message : "启动语音复核失败"); }
    finally { setVoiceReviewBusy(false); }
  };
  const cancelVoiceReview = async () => {
    if (!job) return;
    const requestJobId = job.id;
    const requestGeneration = cueGeneration.current;
    try {
      const data = await api<VoiceReview>(`/api/jobs/${job.id}/voice-review/cancel`, { method: "POST", body: "{}" });
      if (requestGeneration === cueGeneration.current && selectedJobIdRef.current === requestJobId) setVoiceReview(data);
    } catch (error) { setMessage(error instanceof Error ? error.message : "暂停语音复核失败"); }
  };
  const adoptVoiceCandidate = (candidate: NonNullable<VoiceReview["result_candidates"]>[number]) => {
    const source = (candidate.words ?? []).map((word) => word.text ?? "").join("").trim();
    const translated = (voiceCandidateText[candidate.id] ?? "").trim();
    if (!source) { setMessage("候选没有可用原文。"); return; }
    const confirmedWeak = candidate.tier === "weak" && voiceCandidateConfirmed[candidate.id] === true;
    if (candidate.tier === "weak" && !confirmedWeak) { setMessage("弱候选需要先确认已重听。"); return; }
    if (candidate.tier === "diagnostic") return;
    if (cues.some((cue) => !cue.deleted && !cue.candidate_id && candidate.start < cue.end && candidate.end > cue.start)) {
      setMessage("候选与现有字幕时间重叠，请先手动调整时间。"); return;
    }
    setHistory((items) => [...items, { cues }]);
    setFuture([]);
    setCues((items) => [...items, { id: Math.max(-1, ...items.map((cue) => cue.id)) + 1, start: candidate.start, end: candidate.end, source, translated, needs_review: true, candidate_id: candidate.id, confirmed_weak: confirmedWeak, confirmed_source_edit: false, translation_source: source, source_stale: false }].sort((a, b) => a.start - b.start));
    setMessage(translated ? "候选已加入本地字幕草稿，保存后才会写入字幕文件。" : "候选已加入草稿，请先填写中文译文后再保存。");
  };
  const generateVoiceTranslation = async (candidate: NonNullable<VoiceReview["result_candidates"]>[number]) => {
    if (!job) return;
    const source = (candidate.words ?? []).map((word) => word.text ?? "").join("").trim();
    if (!source) { setMessage("候选没有可翻译的原文。"); return; }
    const sourceHash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
    const hash = Array.from(new Uint8Array(sourceHash)).map((value) => value.toString(16).padStart(2, "0")).join("");
    const paid = ["deepseek", "openai"].includes(options.translator);
    if (paid && !window.confirm("这次候选翻译可能产生 API 费用，确认继续吗？")) return;
    const requestJobId = job.id; const requestGeneration = cueGeneration.current;
    setVoiceTranslationBusy(candidate.id);
    try {
      const result = await api<{ source: string; translated: string; source_hash: string }>(`/api/jobs/${requestJobId}/voice-review/candidates/${encodeURIComponent(candidate.id)}/translate`, { method: "POST", body: JSON.stringify({ source_text: source, source_hash: hash, confirmed_paid: paid }) });
      if (requestGeneration !== cueGeneration.current || selectedJobIdRef.current !== requestJobId || result.source_hash !== hash) return;
      setVoiceCandidateText((items) => ({ ...items, [candidate.id]: result.translated }));
      setMessage("候选译文已返回，请核对后再加入草稿。");
    } catch (error) { setMessage(error instanceof Error ? error.message : "候选翻译失败"); }
    finally { setVoiceTranslationBusy((current) => current === candidate.id ? null : current); }
  };
  useEffect(() => {
    if (!job?.id) { setVoiceReview(null); return; }
    void loadVoiceReview(job.id);
    const timer = window.setInterval(() => {
      if (selectedJobIdRef.current === job.id) void loadVoiceReview(job.id);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.id]);
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
    const urlSourceActive = job ? Boolean(job.material.source_url) : materialSourceTab === "url";
    const maxSeconds = urlSourceActive && !fullVideo
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
    const urlSourceActive = job ? Boolean(job.material.source_url) : materialSourceTab === "url";
    if (urlSourceActive && !fullVideo && secondsFromDuration(durationValue, durationUnit) === null) {
      errors.max_seconds = "处理范围须为 1～86400 秒之间的整数。";
    }
    if (sourceUrl.trim().length > 2048) errors.source_url = "视频链接不能超过 2048 个字符。";
    if (options.title.length > 200) errors.title = "标题不能超过 200 个字符。";
    if ((description ?? "").length > 5000) errors.description = "简介不能超过 5000 个字符。";
    if (options.whisper_model_size.length > 128) errors.whisper_model_size = "模型名不能超过 128 个字符。";
    if ((options.asr_hint || "").length > 240) errors.asr_hint = "识别提示不能超过 240 个字符。";
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
      const status = await api<{ message: string }>(`/api/publish/status?browser=${publishBrowser}&platform=${publishPlatform}`);
      setProfileMessage(status.message);
    } catch (error) {
      setProfileMessage(error instanceof Error ? error.message : "无法读取投稿浏览器状态");
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
    if (job) {
      await prepareJob();
      return;
    }
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
    ocrLanguageRef.current = options.ocr_language || "eng";
    void loadDependencies(true).catch(() => undefined);
  }, [options.ocr_language]);
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
  const dismissTutorial = () => {
    setSessionDismissed(true);
    setDismissedVersion(CURRENT_TUTORIAL_VERSION);
    try { localStorage.setItem(TUTORIAL_STORAGE_KEY, JSON.stringify({ dismissedVersion: CURRENT_TUTORIAL_VERSION })); } catch { /* Keep the dismissal for this session when storage is unavailable. */ }
    if (!csrfToken) return;
    void api<{ saved: boolean }>("/api/tutorial-preferences", {
      method: "POST",
      body: JSON.stringify({ dismissedVersion: CURRENT_TUTORIAL_VERSION }),
    }).catch(() => setMessage("偏好未能保存，下次启动可能再次显示。"));
  };
  const closeTutorial = () => {
    setTutorialOpen(false);
    window.setTimeout(() => tutorialTriggerRef.current?.focus(), 0);
  };
  const openTutorial = () => {
    if (settingsOpen || dependenciesOpen || shortcutHelpOpen || templateEditorOpen) return;
    setTutorialOpen(true);
  };
  const handleTutorialAction = (action: TutorialAction) => {
    if (action === "dependencies") {
      setView("process");
      window.setTimeout(() => setDependenciesOpen(true), 0);
      return;
    }
    if (action === "settings") {
      setView("process");
      window.setTimeout(() => setSettingsOpen(true), 0);
      return;
    }
    setView(action);
  };
  const applyRecommended = () => {
    setDevice("cpu");
    setComputeType("int8");
    setOptions((current) => ({ ...current, download_quality: "1080p", render_encoder: "auto", resource_profile: "balanced", subtitle_source: "audio", prefer_platform_subtitles: true, whisper_model_size: "small", source_language: "", audio_channel: "mix", asr_hint: "", asr_vad_mode: "standard", translator: "deepseek", translate_model: "", deepseek_thinking: "disabled", target_lang: "zh-Hans", smart_translation: true, smart_subtitle_layout: true, font_name: "Microsoft YaHei", font_size: 24, subtitle_display_mode: "translated", subtitle_preset: "soft", subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "描边" }));
    setIncludeSourceLink(true);
    setMessage("已应用推荐设置：最高 1080p、自动优先 NVIDIA 渲染、CPU int8 转写与中文硬字幕。");
  };

  useEffect(() => {
    loadBootstrap()
      .then(() => {
        void loadTutorialPreferences();
        void loadSetupPreferences();
        return Promise.all([
          loadMaterials(),
          loadTemplates(),
          loadSettings(),
          loadDependencies(),
          loadRestorableJobs(),
          loadWeekly(),
          loadSite(),
        ]);
      })
      .then((results) => {
        setOnline(true);
        setMessage("本地服务已就绪。选择素材后即可开始处理。");
        const weeklyData = results[5] as WeeklyHot | null;
        const siteData = results[6] as SiteHot | null;
        if (weeklyData?.enabled && weeklyData.channels.length > 0 && weeklyData.stale) void refreshWeekly();
        if (siteData && siteCategoryRef.current === siteData.category && !siteBusyRef.current && claimSiteAutoRefresh(siteData)) void refreshSite();
      })
      .catch(() => setMessage("未连接本地服务，请启动工作台 EXE。"));
  }, []);
  useEffect(() => {
    let inFlight = false;
    const refresh = async (force = false) => {
      if (inFlight || document.visibilityState !== "visible") return;
      inFlight = true;
      try { await loadDependencies(force); } catch { /* Keep the last known local environment status. */ }
      finally { inFlight = false; }
    };
    const onFocus = () => void refresh(true);
    const onVisibility = () => { if (document.visibilityState === "visible") void refresh(true); };
    const timer = window.setInterval(() => void refresh(false), 5000);
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
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
          await loadDependencies(true);
          installTargetJobId.current = null;
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
      setMediaNonce(Date.now());
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
    if (view === "library") void loadRestorableJobs();
    if (view === "publish") {
      void loadPublishSession();
      void loadOutputs().catch(() => undefined);
      void loadJobHistory();
    }
  }, [view, historyScope, options.output_dir]);
  useEffect(() => {
    if (view !== "library") return;
    const timer = window.setTimeout(() => void loadJobHistory(), 250);
    return () => window.clearTimeout(timer);
  }, [view, historyScope, options.output_dir, libraryQuery, libraryFilter]);
  useEffect(() => {
    if (view !== "publish" || !publishVideoReady) return;
    const hydrateKey = [
      job?.id ?? "",
      selectedPublishVideo || "__current__",
      nativePublishToken || "",
      job?.artifacts?.publish_metadata?.available ? "meta" : "none",
      job?.result?.rendered_video || "",
    ].join("|");
    if (publishHydrateKey.current === hydrateKey) return;
    publishHydrateKey.current = hydrateKey;
    if (job?.publish_copy?.has_metadata || (job?.publish_copy?.title && job.publish_copy.tags?.length)) {
      applyPublishCopy(job.publish_copy);
    }
    void loadPublishCopy({ force: false });
  }, [view, publishVideoReady, job?.id, job?.publish_copy?.title, selectedPublishVideo, nativePublishToken]);
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
      if (tutorialOpen) return;
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
  }, [running, tutorialOpen, sourceUrl, selected?.id, authorized, device, computeType, options, description, tags, publishInPipeline, includeSourceLink, fullVideo, durationValue, durationUnit]);

  const confirmDiscardSubtitles = (action: string) =>
    !dirty ||
    window.confirm(
      `当前字幕还有未保存的修改（已编辑 ${history.length} 次）。\n${action}，未保存的内容将被清空。\n\n确定继续吗？`,
    );
  const resetForSourceSwitch = () => {
    sourceRevision.current += 1;
    setJob(null);
    // Do not carry a previous task's strict Creative Commons filter into a
    // different URL/local source.  The filter is an explicit per-source
    // choice; inheriting it silently makes ordinary authorized videos fail at
    // acquire with no relation to the new material.  Users can re-enable it
    // after entering the new URL when they specifically require CC metadata.
    updateOptions("require_reuse_allowed", false);
    setCues([]);
    setHistory([]);
    setFuture([]);
    setActiveCue(null);
    setSelectedCue(null);
    setPreview("source");
    setDemoPreview(false);
    setMetadata(null);
    setMetadataNotice("");
    setMetadataState("idle");
  };
  const readMetadata = async (urlOverride?: string) => {
    const url = (typeof urlOverride === "string" ? urlOverride : sourceUrl).trim();
    if (!url) {
      setMetadataState("error");
      setMetadataNotice("请先粘贴完整的视频链接，再读取标题与时长。");
      return;
    }
    const revision = ++sourceRevision.current;
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
          url,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
          youtube_po_token_mode: options.youtube_po_token_mode,
          youtube_proxy: options.youtube_proxy,
          download_quality: options.download_quality,
        }),
      });
      if (revision !== sourceRevision.current) return;
      setMetadata(value);
      if (value.title) updateOptions("title", value.title);
      setMetadataState("success");
      setMetadataNotice("已读取视频信息，标题已自动填入下方字段。");
      setMessage("已读取视频信息。");
    } catch (error) {
      if (revision !== sourceRevision.current) return;
      const detail = error instanceof Error ? error.message : "未知错误";
      setMetadataState("error");
      setMetadataNotice(`读取失败：${detail}`);
      setMessage(`读取失败：${detail}`);
    }
  };
  const fillSourceFromHot = (url: string) => {
    if (running) {
      setMessage("当前阶段运行中，请先取消或等待完成后再切换素材。");
      return;
    }
    if (job && !window.confirm(`修改素材链接会离开当前任务；原任务可在任务库重新载入。${dirty ? "当前未保存的字幕修改会丢失。" : ""}\n\n确定继续吗？`)) return;
    if (!job && dirty && !confirmDiscardSubtitles("改用这条视频会清空当前未保存的字幕修改")) return;
    if (job || selected) resetForSourceSwitch();
    else sourceRevision.current += 1;
    if (job || selected) setMessage("已切换到新链接草稿；原任务可在任务库重新载入。");
    setMaterialSourceTab("url");
    setMaterialId("");
    setSourceUrl(url);
    setMaterialDraftToken((value) => value + 1);
    setFieldErrors((current) => ({ ...current, source_url: "" }));
    setMetadata(null);
    setMetadataNotice("");
    setMetadataState("idle");
    window.setTimeout(() => document.getElementById("source-url")?.scrollIntoView({ behavior: "smooth", block: "center" }), 0);
    void readMetadata(url);
  };
  const batchHasUrl = (url: string) => {
    const lines = batchText.split(/\r?\n/).map((line) => line.trim());
    return lines.includes(url) || batchEntries.some((entry) => entry.url === url);
  };
  const addHotToBatch = (entry: WeeklyHotVideo) => {
    if (running || batchBusy) {
      setMessage("当前阶段运行中，请先取消或等待完成后再加入批量。");
      return;
    }
    const page = document.scrollingElement;
    const pageTop = page?.scrollTop ?? 0;
    const list = document.querySelector(".weekly-hot-list");
    const listTop = list instanceof HTMLElement ? list.scrollTop : 0;
    setMaterialSourceTab("url");
    setBatchText((current) => {
      const lines = current.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      if (lines.includes(entry.url)) return current;
      const trimmed = current.replace(/\s+$/, "");
      return trimmed ? `${trimmed}\n${entry.url}` : entry.url;
    });
    setBatchEntries((current) => {
      if (current.length === 0) return current;
      if (current.some((row) => row.url === entry.url)) {
        return current.map((row) => row.url === entry.url ? { ...row, selected: true, title: row.title || entry.title } : row);
      }
      return [...current, { url: entry.url, title: entry.title, duration: entry.duration, selected: true }];
    });
    setMessage("已加入批量链接。");
    window.requestAnimationFrame(() => {
      if (page) page.scrollTop = pageTop;
      if (list instanceof HTMLElement) list.scrollTop = listTop;
    });
  };
  const openOriginalVideo = (url: string) => {
    void api("/api/open-url", { method: "POST", body: JSON.stringify({ url }) }).catch((error) => {
      setMessage(error instanceof Error ? error.message : "无法打开原链接");
    });
  };
  const importVideo = async (file: File) => {
    if (running) return setMessage("当前阶段运行中，请先取消或等待完成后再切换素材。");
    if (!authorized) {
      setMessage("请先勾选上方「我确认拥有处理、转载或发布该视频的授权/许可证」复选框，再导入本地视频。");
      document.getElementById("rights-check")?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    if (!confirmDiscardSubtitles("切换到本地视频")) return;
    const body = new FormData();
    body.append("video", file);
    body.append("authorized", "true");
    try {
      const material = await api<Material>("/api/materials", {
        method: "POST",
        body,
      });
      await loadMaterials();
      resetForSourceSwitch();
      setMaterialId(material.id);
      setSourceUrl("");
      setMaterialSourceTab("local");
      setMaterialDraftToken((value) => value + 1);
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
      publishHydrateKey.current = "";
      void generateDescription({ silent: true });
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
  const start = async () => {
    const url = materialSourceTab === "url" ? sourceUrl.trim() : "";
    if ((materialSourceTab === "url" ? !url : !selected) || running) return;
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
          ...(url ? { source_url: url } : { material_id: selected?.id }),
          authorized,
          device,
          compute_type: computeType,
          options: taskOptions(),
        }),
      });
      cueGeneration.current += 1;
      selectedJobIdRef.current = created.id;
      cueOwner.current = null;
      setCuesReady(false);
      setJob(created);
      setOptions((current) => mergeRestoredOptions(current, created.options));
      setDemoPreview(false);
      setCues([]);
      setHistory([]);
      setFuture([]);
      setSelectedCue(null);
      setPreview("source");
      setMaterialDraftToken(0);
      setView("process");
      setMessage("任务草稿已保存。请确认处理方案，然后运行开始前准备检查；当前不会下载视频。 ");
    } catch (error) {
      setMessage(`无法启动：${reportApiError(error, "未知错误")}`);
    }
  };
  const checksReady = (current: Job | null | undefined) =>
    Boolean(current?.checks?.length) && !(current?.checks ?? []).some((item) => item.status === "blocking");
  const patchJobOptions = async (optionPatch: Partial<Options>, successMessage?: string) => {
    if (!job || running) return job;
    if (!validateTaskFields()) return null;
    try {
      const updated = await api<Job & { stale_stages?: string[] }>(`/api/jobs/${job.id}/options`, {
        method: "PATCH",
        body: JSON.stringify({
          device,
          compute_type: computeType,
          options: { ...taskOptions(), ...optionPatch },
        }),
      });
      setJob(updated);
      setOptions((current) => mergeRestoredOptions(current, updated.options));
      if (successMessage) {
        setMessage(successMessage);
      } else {
        const stale = updated.stale_stages ?? [];
        setMessage(stale.length ? `方案已保存；${stale.map((name) => [...workflowStageMeta, dubStageMeta].find(([id]) => id === name)?.[1] ?? name).join("、")}需要重新执行。` : "方案已保存，请运行准备检查。 ");
      }
      return updated;
    } catch (error) {
      setMessage(reportApiError(error, "保存处理方案失败"));
      return null;
    }
  };
  const saveWorkflowOptions = async () => patchJobOptions({});
  const persistRequireReuseAllowed = async (checked: boolean) => {
    updateOptions("require_reuse_allowed", checked);
    const summary = checked
      ? "已开启仅接受可复用 CC 视频。获取素材将使用新设置。"
      : "已关闭 CC 限制。当前任务会立刻使用新设置，可重新获取素材。";
    if (!job || running) {
      setMessage(summary);
      return;
    }
    await patchJobOptions({ require_reuse_allowed: checked }, summary);
  };
  const prepareJob = async () => {
    if (!job || running || preparingJob) return;
    const saved = await saveWorkflowOptions();
    if (!saved) return;
    preparingJobRef.current = true;
    setPreparingJob(true);
    try {
      const prepared = await api<Job>(`/api/jobs/${job.id}/prepare`, { method: "POST", body: "{}" });
      setJob(prepared);
      const blockers = (prepared.checks ?? []).filter((item) => item.status === "blocking");
      setMessage(blockers.length ? `准备检查发现 ${blockers.length} 个阻断项，请按清单修正。` : "开始前准备已通过。你可以逐阶段运行，或一键运行到渲染。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "准备检查失败");
    } finally { preparingJobRef.current = false; setPreparingJob(false); }
  };
  const confirmQualityProposal = async () => {
    if (!job?.quality_proposal?.fingerprint) return;
    try {
      const confirmed = await api<Job>(`/api/jobs/${job.id}/quality/confirm`, {
        method: "POST",
        body: JSON.stringify({ fingerprint: job.quality_proposal.fingerprint }),
      });
      setJob(confirmed);
      setMessage("画质方案已确认，可以开始获取素材。 ");
    } catch (error) {
      setMessage(reportApiError(error, "确认画质方案失败"));
    }
  };
  prepareSpecificJobRef.current = async (jobId: string) => {
    if (jobRef.current?.id !== jobId || runningRef.current || preparingJobRef.current || schemeDirtyRef.current) return;
    preparingJobRef.current = true;
    setPreparingJob(true);
    try {
      const prepared = await api<Job>(`/api/jobs/${jobId}/prepare`, { method: "POST", body: "{}" });
      if (jobRef.current?.id !== jobId) return;
      setJob(prepared);
      const blockers = (prepared.checks ?? []).filter((item) => item.status === "blocking");
      setMessage(blockers.length ? `本机环境已更新；当前任务仍有 ${blockers.length} 个准备阻断项，请查看清单。` : "本机环境已更新，当前任务准备检查已刷新，可以继续，无需重启应用。");
    } catch (error) {
      if (jobRef.current?.id === jobId) setMessage(error instanceof Error ? error.message : "刷新当前任务准备检查失败。");
    } finally {
      preparingJobRef.current = false;
      setPreparingJob(false);
    }
  };
  const runRecoveryAction = async (
    recovery: RecoveryAction,
    change: { device?: string; compute_type?: string; options?: Partial<Options> },
  ) => {
    if (!job || running || preparingJob) return;
    if (schemeDirtyRef.current) {
      setMessage("当前处理方案有未保存修改。先保存并检查准备，再执行恢复操作。 ");
      return;
    }
    const stageLabel = workflowStageMeta.find(([id]) => id === recovery.stage)?.[1] ?? recovery.stage;
    if (recovery.requires_confirmation && !window.confirm(`${recovery.impact}\n\n确认后仅重新执行“${stageLabel}”；其他有效阶段结果会保留。继续吗？`)) return;
    preparingJobRef.current = true;
    setPreparingJob(true);
    try {
      const updated = await api<Job>(`/api/jobs/${job.id}/options`, {
        method: "PATCH",
        body: JSON.stringify({
          device: change.device ?? job.device,
          compute_type: change.compute_type ?? job.compute_type,
          options: change.options ?? {},
        }),
      });
      setJob(updated);
      setOptions((current) => mergeRestoredOptions(current, updated.options));
      setDevice(updated.device);
      setComputeType(updated.compute_type);
      const prepared = await api<Job>(`/api/jobs/${updated.id}/prepare`, { method: "POST", body: "{}" });
      setJob(prepared);
      const blockers = (prepared.checks ?? []).filter((item) => item.status === "blocking");
      if (blockers.length) {
        setMessage(`恢复设置已保存，但准备检查仍有 ${blockers.length} 个阻断项。解决后可从“${stageLabel}”继续。`);
        return;
      }
      const started = await api<Job>(`/api/jobs/${prepared.id}/stages/${recovery.stage}/run`, { method: "POST", body: "{}" });
      setJob(started);
      setInspectorOpen(true);
      setView("process");
      setMessage(`已按确认后的设置，仅重新执行“${stageLabel}”。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "恢复操作未能启动；已有检查点仍保留。 ");
    } finally {
      preparingJobRef.current = false;
      setPreparingJob(false);
    }
  };
  const runStage = async (stage: WorkflowStageName) => {
    if (!job || running) return;
    const saved = await saveWorkflowOptions();
    if (!saved) return;
    if (!checksReady(saved)) {
      setMessage("方案已同步到当前任务。请先重新检查准备，再运行该阶段。");
      return;
    }
    try {
      const latest = await api<Job>(`/api/jobs/${saved.id}/stages/${stage}/run`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true);
      setMessage(`已开始：${workflowStageMeta.find(([id]) => id === stage)?.[1] ?? stage}`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法启动该阶段"); }
  };
  const openStageResult = (stage: WorkflowStageName) => {
    if (!job) return;
    if (stage === "publish") { setView("publish"); return; }
    if (stage === "render") {
      setMediaNonce(Date.now());
      setPreview("rendered");
    } else setPreview(stage === "dub" ? "dub" : "source");
    if (["extract", "translate", "render"].includes(stage)) void loadCues(job.id);
    setView("subtitle");
  };
  const runAll = async () => {
    if (!job || running) return;
    const saved = await saveWorkflowOptions();
    if (!saved) {
      return;
    }
    if (!checksReady(saved)) {
      setMessage("方案已同步到当前任务。请先重新检查准备，再一键运行。");
      return;
    }
    try {
      const latest = await api<Job>(`/api/jobs/${saved.id}/run-all`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true); setMessage("已按同一套阶段顺序运行；任一步失败都会停在该检查点。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法运行全部阶段");
    }
  };
  const resumeJob = async () => {
    if (!job || running) return;
    const saved = await saveWorkflowOptions();
    if (!saved) return;
    if (!checksReady(saved)) {
      setMessage("方案已同步到当前任务。请先重新检查准备，再从检查点继续。");
      return;
    }
    try {
      const latest = await api<Job>(`/api/jobs/${saved.id}/resume`, { method: "POST", body: "{}" });
      setJob(latest); setInspectorOpen(true); setMessage("已从最后一个有效检查点继续。 ");
    } catch (error) { setMessage(error instanceof Error ? error.message : "无法恢复任务"); }
  };
  const topAction = () => {
    if (running) return void cancel();
    if (!job) return void start();
    if (job.status === "completed" && !job.next_stage) return setView("subtitle");
    if (!job.checks?.length || (job.checks ?? []).some((item) => item.status === "blocking")) return void prepareJob();
    if (job.can_resume) return void resumeJob();
    if (job.next_stage === "publish") return setView("publish");
    if (job.next_stage) return void runStage(job.next_stage);
    setView("subtitle");
  };
  const topActionLabel = preparingJob ? "正在检查准备…" : job?.status === "completed" && !job.next_stage
    ? "查看字幕成片" : job?.next_stage === "publish" && !running && job.checks?.length && !(job.checks ?? []).some((item) => item.status === "blocking")
    ? "查看投稿辅助" : workflowPrimaryLabel({
    running, hasJob: Boolean(job), hasChecks: Boolean(job?.checks?.length),
    hasBlocking: (job?.checks ?? []).some((item) => item.status === "blocking"),
    canResume: Boolean(job?.can_resume), nextStage: job?.next_stage ?? null,
    stageLabel: (stage) => [...workflowStageMeta, dubStageMeta].find(([id]) => id === stage)?.[1] ?? "下一阶段",
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
    if (savingCuesRef.current) return;
    if (!job || !dirty || cueOwner.current !== job.id || !cuesReady) {
      if (job && cueOwner.current !== job.id) setMessage("当前任务字幕尚未加载完成，暂不能保存。");
      return;
    }
    for (const cue of cues) {
      if (cue.candidate_id && !cue.deleted && !cue.translated.trim()) {
        setMessage(`候选字幕 ${cue.candidate_id} 还没有中文译文，请先填写后再保存。`);
        return;
      }
      if (cue.candidate_id && !cue.deleted && cue.source_stale && cue.confirmed_source_edit !== true) {
        setMessage(`候选字幕 ${cue.candidate_id} 的原文已修改，请先核对原文、修改中文或重新生成译文。`);
        return;
      }
      if (!cue.deleted && cue.start >= cue.end) {
        setMessage(`第 ${cue.id + 1} 条字幕时间无效：开始时间必须小于结束时间。`);
        return;
      }
    }
    savingCuesRef.current = true;
    setSavingCues(true);
    try {
      const saveJobId = job.id;
      const saveGeneration = cueGeneration.current;
      const candidateRows = cues.filter((cue) => cue.candidate_id).map((cue) => ({
        candidate_id: cue.candidate_id,
        source: cue.source,
        translated: cue.translated,
        start: cue.start,
        end: cue.end,
        confirmed_weak: cue.confirmed_weak === true,
        confirmed_source_edit: cue.confirmed_source_edit === true,
        translation_source: cue.translation_source,
        source_stale: cue.source_stale === true,
      }));
      const baseCues = cues.filter((cue) => !cue.candidate_id);
      const savedJob = await api<Job>(`/api/jobs/${job.id}/cues`, {
        method: "PUT",
        body: JSON.stringify({
          base_revision: cueBaseMeta?.revision,
          base_fingerprint: cueBaseMeta?.base_fingerprint,
          base_ids: cueBaseMeta?.base_ids,
          candidate_rows: candidateRows,
          cues: baseCues.map((cue) => ({
            start: cue.start,
            end: cue.end,
            source: cue.source,
            translated: cue.translated,
            deleted: Boolean(cue.deleted),
            needs_review: Boolean(cue.needs_review),
          })),
        }),
      });
      if (saveGeneration !== cueGeneration.current || selectedJobIdRef.current !== saveJobId) return;
      setJob(savedJob);
      setHistory([]);
      setFuture([]);
      setTranslationCandidate(null);
      setCompactionCandidates([]);
      await loadCues(saveJobId, saveGeneration);
      setMessage("字幕已保存；渲染检查点已标记为需要重做，可直接点击“重新渲染”。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存字幕失败");
    } finally {
      savingCuesRef.current = false;
      setSavingCues(false);
    }
  };
  const rerender = async () => {
    if (!job) return;
    if (dirty) { setMessage("字幕有未保存修改，请先保存后再重新渲染。 "); return; }
    try {
      if (schemeDirty) {
        const saved = await saveWorkflowOptions();
        if (!saved) return;
        const prepared = await api<Job>(`/api/jobs/${job.id}/prepare`, { method: "POST", body: "{}" });
        setJob(prepared);
        if ((prepared.checks ?? []).some((item) => item.status === "blocking")) {
          setMessage("渲染前准备检查未通过，请先修正阻断项。 ");
          return;
        }
      }
      setJob(
        await api<Job>(`/api/jobs/${job.id}/rerender`, {
          method: "POST",
          body: JSON.stringify({ dirty }),
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
  const setTrimPointFromPlayhead = (field: "cutStart" | "cutEnd" | "exportStart" | "exportEnd") => {
    const currentTime = video.current?.currentTime;
    if (!Number.isFinite(currentTime)) {
      setMessage("请先打开视频并播放到需要的位置。");
      return;
    }
    const value = formatTime(currentTime ?? 0);
    if (field === "cutStart") setCutStart(value);
    else if (field === "cutEnd") setCutEnd(value);
    else if (field === "exportStart") setExportStart(value);
    else setExportEnd(value);
  };
  const previewTrimRange = (start: number, end: number) => {
    const player = video.current;
    if (!player) {
      setMessage("当前没有可预览的视频。");
      return;
    }
    // Review timestamps use the current/source video clock, never the rendered
    // subtitle preview which may have different cuts or offsets.
    if ((preview !== "source" || player.readyState < 1) && job) {
      pendingPlayback.current = { start, end, jobId: job.id, generation: cueGeneration.current };
      setPreview("source");
      return;
    }
    subtitlePlaybackEnd.current = null;
    trimPlaybackEnd.current = end;
    player.currentTime = start;
    void player.play().catch(() => {
      trimPlaybackEnd.current = null;
      setMessage("无法播放此区间，请先点击视频播放一次。");
    });
    setMessage(`正在预览 ${formatTime(start)} – ${formatTime(end)}。`);
  };
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
  const updateCueDraft = (id: number, patch: Partial<Pick<Cue, "source" | "translated" | "needs_review" | "confirmed_source_edit" | "translation_source" | "source_stale">>) => {
    setHistory((value) => [...value, { cues }]);
    setFuture([]);
    setCues((value) =>
      value.map((cue) => (cue.id === id ? {
        ...cue,
        ...patch,
        ...(cue.candidate_id && patch.source !== undefined
          ? { confirmed_source_edit: false, source_stale: patch.source !== cue.translation_source }
          : {}),
        ...(cue.candidate_id && patch.translated !== undefined
          ? { confirmed_source_edit: true, source_stale: false, translation_source: patch.source ?? cue.source }
          : {}),
      } : cue)),
    );
  };
  const updateCue = (id: number, translated: string) => updateCueDraft(id, { translated });
  const toggleCueReview = (cue: Cue) => updateCueDraft(cue.id, { needs_review: !cue.needs_review });
  const toggleBulkCue = (id: number) => setBulkSelectedCueIds((current) =>
    current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
  );
  const selectReviewCuesForBatch = () => setBulkSelectedCueIds(
    visibleCueRows.filter(({ cue }) => cueNeedsReview(cue) && !cue.deleted).map(({ cue }) => cue.id),
  );
  const moveEditorSelection = (direction: -1 | 1) => {
    const navigable = visibleCueRows.filter(({ cue }) => !cue.deleted);
    if (!navigable.length) return;
    const currentId = selectedCue ?? activeCue;
    const currentIndex = navigable.findIndex(({ cue }) => cue.id === currentId);
    const nextIndex = currentIndex < 0
      ? (direction > 0 ? 0 : navigable.length - 1)
      : Math.max(0, Math.min(navigable.length - 1, currentIndex + direction));
    const next = navigable[nextIndex].cue;
    setActiveCue(next.id);
    setSelectedCue(next.id);
    document.querySelector<HTMLElement>(`[data-cue-id="${next.id}"]`)?.scrollIntoView({ block: "nearest" });
  };
  const generateCueTranslation = async (cue: Cue) => {
    if (!job || running || retranslatingCueId !== null || compactingCues) return;
    const requestJobId = job.id;
    const requestGeneration = cueGeneration.current;
    const modelUseNote = taskModelIsOnline
      ? "这条字幕及相邻上下文会发送到在线服务，可能产生 API 费用。"
      : "会调用本机模型，不会自动覆盖原译文。";
    if (!window.confirm(`使用 ${taskModelLabel} 为第 ${cue.id + 1} 条生成重译候选。\n\n${modelUseNote}\n\n继续吗？`)) return;
    setRetranslatingCueId(cue.id);
    setTranslationCandidate(null);
    try {
      if (cue.candidate_id) {
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(cue.source));
        const sourceHash = Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
        const paid = ["deepseek", "openai"].includes(options.translator);
        if (paid && !window.confirm("这次候选翻译可能产生 API 费用，确认继续吗？")) return;
        const result = await api<{ translated: string; source: string; source_hash: string; provider: string; model: string }>(
          `/api/jobs/${requestJobId}/voice-review/candidates/${encodeURIComponent(cue.candidate_id)}/translate`,
          { method: "POST", body: JSON.stringify({ source_text: cue.source, source_hash: sourceHash, confirmed_paid: paid }) },
        );
        if (requestGeneration !== cueGeneration.current || selectedJobIdRef.current !== requestJobId || result.source_hash !== sourceHash) return;
        setTranslationCandidate({ cueId: cue.id, source: cue.source, originalTranslation: cue.translated, translation: result.translated, provider: result.provider, model: result.model });
        return;
      }
      const result = await api<{ cue_id: number; translation: string; provider: string; model: string }>(
        `/api/jobs/${requestJobId}/cues/${cue.id}/retranslate`,
        {
          method: "POST",
          body: JSON.stringify({
            source: cue.source,
            context: [-1, 1].flatMap((offset) => {
              const neighbor = cues.find((item) => item.id === cue.id + offset);
              return neighbor ? [{ cue_id: neighbor.id, source: neighbor.source, translated: neighbor.translated }] : [];
            }),
          }),
        },
      );
      if (requestGeneration !== cueGeneration.current || selectedJobIdRef.current !== requestJobId) return;
      setTranslationCandidate({
        cueId: cue.id,
        source: cue.source,
        originalTranslation: cue.translated,
        translation: result.translation,
        provider: result.provider,
        model: result.model,
      });
      setMessage(`第 ${cue.id + 1} 条译文候选已生成；核对后再采用并保存。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "单条重译失败；原字幕和草稿已保留。");
    } finally {
      if (requestGeneration === cueGeneration.current && selectedJobIdRef.current === requestJobId) setRetranslatingCueId(null);
    }
  };
  const applyTranslationCandidate = () => {
    if (!translationCandidate) return;
    const current = cues.find((cue) => cue.id === translationCandidate.cueId);
    if (!current || current.source !== translationCandidate.source || current.translated !== translationCandidate.originalTranslation) {
      setMessage("原文或译文已变化，请重新生成这条译文候选。");
      return;
    }
    updateCueDraft(current.id, {
      source: translationCandidate.source,
      translated: translationCandidate.translation,
    });
    setTranslationCandidate(null);
    setMessage("译文候选已放入草稿；保存字幕后才会写入任务。");
  };
  const generateCompactionCandidates = async () => {
    if (!job || running || compactingCues || retranslatingCueId !== null) return;
    const cueIds = selectedReviewCues.map((cue) => cue.id);
    if (!cueIds.length) {
      setMessage("请先勾选待复核的问题字幕条目。");
      return;
    }
    if (cueIds.length > 50) {
      setMessage("一次最多为 50 条字幕生成精简建议，请缩小选择范围。");
      return;
    }
    const modelUseNote = taskModelIsOnline
      ? "选中的字幕文本会发送到在线服务，可能产生 API 费用。"
      : "会调用本机模型，不会自动替换原译文。";
    if (!window.confirm(`使用 ${taskModelLabel} 为 ${cueIds.length} 条字幕逐条生成精简候选（最多 ${cueIds.length} 次请求）。\n\n${modelUseNote}\n\n继续吗？`)) return;
    setCompactingCues(true);
    setCompactionCandidates([]);
    try {
      const result = await api<{ provider: string; model: string; request_count: number; candidates: CompactionCandidate[] }>(
        `/api/jobs/${job.id}/cues/compact`,
        {
          method: "POST",
          body: JSON.stringify({
            cue_ids: cueIds,
            cues: selectedReviewCues.map((cue) => ({ cue_id: cue.id, source: cue.source, translated: cue.translated })),
          }),
        },
      );
      setCompactionCandidates(result.candidates);
      const failed = result.candidates.filter((item) => item.error).length;
      setMessage(`${result.provider} · ${result.model} 已处理 ${result.request_count} 条；生成 ${result.candidates.length - failed} 条精简候选，失败条目保留原译文。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "精简建议生成失败；当前字幕草稿已保留。");
    } finally {
      setCompactingCues(false);
    }
  };
  const applyCompactionCandidate = (candidate: CompactionCandidate) => {
    if (candidate.error || candidate.candidate === candidate.original) return;
    const current = cues.find((cue) => cue.id === candidate.cue_id);
    if (!current || current.translated !== candidate.original) {
      setMessage(`第 ${candidate.cue_id + 1} 条译文已变化，请重新生成精简候选。`);
      return;
    }
    updateCueDraft(current.id, { translated: candidate.candidate });
    setCompactionCandidates((items) => items.filter((item) => item.cue_id !== candidate.cue_id));
  };
  const applyAllCompactionCandidates = () => {
    const applicable = compactionCandidates.filter((candidate) => {
      const current = cues.find((cue) => cue.id === candidate.cue_id);
      return !candidate.error && candidate.candidate !== candidate.original && current?.translated === candidate.original;
    });
    if (!applicable.length) return;
    setHistory((value) => [...value, { cues }]);
    setFuture([]);
    const byId = new Map(applicable.map((candidate) => [candidate.cue_id, candidate.candidate]));
    setCues((current) => current.map((cue) => byId.has(cue.id) ? { ...cue, translated: byId.get(cue.id)! } : cue));
    setCompactionCandidates((items) => items.filter((item) => !byId.has(item.cue_id)));
    setMessage(`已采用 ${applicable.length} 条精简译文候选；保存字幕后才会写入任务。`);
  };
  const applyLiteralReplacement = () => {
    if (!replacePreviews.length) return;
    const previewById = new Map(replacePreviews.map((preview) => [preview.cueId, preview.after]));
    setHistory((value) => [...value, { cues }]);
    setFuture([]);
    setCues((current) => current.map((cue) => previewById.has(cue.id)
      ? { ...cue, [replaceField]: previewById.get(cue.id)! }
      : cue,
    ));
    setMessage(`已替换 ${replacePreviews.length} 条字幕中的 ${replacePreviews.reduce((sum, item) => sum + item.count, 0)} 处文本；可撤销，保存后写入任务。`);
  };
  const updateCueTime = (id: number, field: "start" | "end", value: string) => {
    const parsed = parseTime(value);
    if (parsed === null) {
      setMessage(`时间格式无效（如 0:05.0 或 1:30）：${value}`);
      return;
    }
    setHistory((value) => [...value, { cues }]);
    setFuture([]);
    setCues((current) =>
      current.map((cue) => (cue.id === id ? { ...cue, [field]: parsed } : cue)),
    );
  };
  const toggleCueDeleted = (id: number) => {
    setHistory((value) => [...value, { cues }]);
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
  const shiftAllCues = (offset: number) => {
    if (!job || !cues.length) return;
    if (!Number.isFinite(offset) || offset === 0) {
      setMessage("偏移秒数必须是非零数字。");
      return;
    }
    const offsetTicks = Math.round(offset * 100);
    const duration = job.material.duration_seconds;
    const maxVideoTicks = duration ? Math.max(0, Math.floor(duration * 100)) : Number.POSITIVE_INFINITY;
    setHistory((value) => [...value, { cues }]);
    setFuture([]);
    setCues((current) =>
      current.map((cue) => {
        const startTicks = Math.round(cue.start * 100);
        const originalEndTicks = Math.max(startTicks + 10, Math.round(cue.end * 100));
        const cueDurationTicks = Math.min(maxVideoTicks, originalEndTicks - startTicks);
        const latestStartTicks = Number.isFinite(maxVideoTicks)
          ? Math.max(0, maxVideoTicks - cueDurationTicks)
          : Number.POSITIVE_INFINITY;
        const shiftedStartTicks = Math.min(
          latestStartTicks,
          Math.max(0, startTicks + offsetTicks),
        );
        const shiftedEndTicks = Math.min(maxVideoTicks, shiftedStartTicks + cueDurationTicks);
        return {
          ...cue,
          start: shiftedStartTicks / 100,
          end: shiftedEndTicks / 100,
        };
      }),
    );
    const direction = offset > 0 ? "延后" : "提前";
    const seconds = String(Number((Math.abs(offsetTicks) / 100).toFixed(2)));
    setMessage(`已将全部字幕${direction} ${seconds} 秒，每条时长保持不变；可撤销，点击「保存字幕」后生效。`);
  };
  const applyOffset = (direction: -1 | 1) => {
    const amount = Math.abs(Number(offsetValue));
    if (!Number.isFinite(amount) || amount === 0) {
      setMessage("请输入大于 0 的秒数，再选择提前或延后。");
      return;
    }
    shiftAllCues(amount * direction);
    setOffsetValue("");
  };
  const shiftByStep = (direction: -1 | 1) => shiftAllCues(0.5 * direction);
  const undo = () => {
    const previous = history.at(-1);
    if (!previous) return;
    setFuture((value) => [{ cues }, ...value]);
    setCues(previous.cues);
    setHistory((value) => value.slice(0, -1));
  };
  const redo = () => {
    const next = future[0];
    if (!next) return;
    setHistory((value) => [...value, { cues }]);
    setCues(next.cues);
    setFuture((value) => value.slice(1));
  };
  const seek = (cue: Cue) => {
    setActiveCue(cue.id);
    setSelectedCue(cue.id);
    if (video.current) {
      trimPlaybackEnd.current = null;
      subtitlePlaybackEnd.current = cue.end;
      video.current.currentTime = cue.start;
      video.current.play();
    }
  };
  useEffect(() => {
    const onSubtitleKeyDown = (event: KeyboardEvent) => {
      if (tutorialOpen) return;
      if (view !== "subtitle") return;
      const target = event.target as HTMLElement | null;
      const typing = Boolean(target?.isContentEditable || target?.closest("input, textarea, select, [contenteditable='true']"));
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (!running) void save();
        return;
      }
      if (typing || running) return;
      if (event.altKey && event.key === "ArrowUp") {
        event.preventDefault();
        moveEditorSelection(-1);
      } else if (event.altKey && event.key === "ArrowDown") {
        event.preventDefault();
        moveEditorSelection(1);
      } else if (event.code === "Space" && !event.altKey && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        const player = video.current;
        if (player && !player.paused) {
          subtitlePlaybackEnd.current = null;
          trimPlaybackEnd.current = null;
          player.pause();
          return;
        }
        const currentId = selectedCue ?? activeCue;
        const currentCue = cues.find((cue) => cue.id === currentId && !cue.deleted)
          ?? visibleCueRows.find(({ cue }) => !cue.deleted)?.cue;
        if (currentCue) seek(currentCue);
      }
    };
    window.addEventListener("keydown", onSubtitleKeyDown);
    return () => window.removeEventListener("keydown", onSubtitleKeyDown);
  }, [view, tutorialOpen, running, selectedCue, activeCue, cues, visibleCueRows, save]);
  const saveSettings = async () => {
    if (savingSettingsRef.current) return;
    savingSettingsRef.current = true;
    setSavingSettings(true);
    try {
      const selected = translationPresets.find((item) => item.id === translationPreset);
      const presetMatches = selected?.engine === options.translator;
      const saved = await api<{ translator?: string; translate_model?: string; translation_preset?: string; translation_base_url?: string }>("/api/settings", {
        method: "POST",
        body: JSON.stringify({
          translator: options.translator,
          api_key: apiKey,
          cookies_from_browser: options.cookies_from_browser,
          cookies_file: options.cookies_file,
          ...(proxyDirty ? { youtube_proxy: options.youtube_proxy } : {}),
          ...(presetDirty && presetMatches ? {
            translation_preset: selected.id,
            base_url: selected.base_editable ? serviceBaseUrl : selected.base_url,
            translate_model: options.translate_model,
          } : {}),
        }),
      });
      if (saved.translator) updateOptions("translator", saved.translator);
      if (saved.translate_model !== undefined && saved.translation_preset) updateOptions("translate_model", saved.translate_model);
      if (saved.translation_preset) setTranslationPreset(saved.translation_preset);
      if (saved.translation_base_url) setServiceBaseUrl(saved.translation_base_url);
      if (saved.translator === "deepseek") setApiKeyConfigured(true);
      if (saved.translator === "openai") setOpenaiKeyConfigured(true);
      setPresetDirty(false);
      setApiKey("");
      if (proxyDirty) setProxyConfigured(Boolean(options.youtube_proxy));
      setProxyDirty(false);
      setSettingsOpen(false);
      setMessage("本地设置已保存，密钥未回显。 ");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存设置失败");
    } finally {
      savingSettingsRef.current = false;
      setSavingSettings(false);
    }
  };
  const exportDiagnostics = async () => {
    if (exportingDiagnostics) return;
    setExportingDiagnostics(true);
    setDiagnosticsNote("");
    try {
      const result = await api<{ path: string }>("/api/diagnostics/export", { method: "POST", body: "{}" });
      const note = `诊断包已导出：${result.path}`;
      setDiagnosticsNote(note);
      setMessage(note);
    } catch (error) {
      const note = error instanceof Error ? error.message : "导出诊断包失败";
      setDiagnosticsNote(note);
      setMessage(note);
    } finally {
      setExportingDiagnostics(false);
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
  const generateDescription = async (opts?: { silent?: boolean; sourceUrlOverride?: string }) => {
    try {
      const response = await api<{ description: string }>(
        "/api/publish/description",
        {
          method: "POST",
          body: JSON.stringify({
            platform: publishPlatform,
            template: templateName,
            source_url: opts?.sourceUrlOverride ?? sourceUrl,
            include_source_link: publishPlatform === "douyin" ? false : includeSourceLink,
            custom_text: templateBody,
            template_body: templateBody,
            extra_lines: extraLine ? [extraLine] : [],
          }),
        },
      );
      setDescription(response.description);
      if (!opts?.silent) {
        setMessage(
          publishPlatform === "douyin"
            ? "已按抖音模板生成正文（不含原视频链接）。"
            : "已按当前模板内容生成简介（草稿内容优先于已保存模板）。",
        );
      }
      return response.description;
    } catch (error) {
      if (!opts?.silent) {
        setMessage(error instanceof Error ? error.message : "生成简介失败");
      }
      return null;
    }
  };
  const applyPublishCopy = (copy: { title?: string; tags?: string[]; source_url?: string }, opts?: { force?: boolean }) => {
    const force = Boolean(opts?.force);
    const nextTitle = (copy.title || "").trim();
    if (nextTitle && (force || !options.title.trim() || !mostlyCjk(options.title))) {
      updateOptions("title", nextTitle);
    }
    if ((copy.tags?.length ?? 0) > 0 && (force || !tags.trim())) {
      setTags((copy.tags ?? []).join(", "));
    }
    if (copy.source_url && !sourceUrl.trim()) {
      setSourceUrl(copy.source_url);
    }
  };
  const loadPublishCopy = async (opts?: { generate?: boolean; force?: boolean }) => {
    if (!publishVideoReady && !job?.result && !job?.id) {
      setMessage("请先选择成片，再加载投稿文案。");
      return null;
    }
    try {
      const response = await api<{
        title: string;
        tags: string[];
        has_metadata: boolean;
        generated: boolean;
        source_url: string;
        video_name: string;
      }>("/api/publish/copy", {
        method: "POST",
        body: JSON.stringify({
          job_id: !nativePublishToken && !selectedPublishVideo && job?.id ? job.id : "",
          video_id: selectedPublishVideo === "__native__" ? "" : selectedPublishVideo,
          native_video_token: nativePublishToken,
          output_dir: options.output_dir,
          generate: Boolean(opts?.generate),
        }),
      });
      applyPublishCopy(response, { force: opts?.force || opts?.generate });
      if (!description.trim() || opts?.force || opts?.generate) {
        await generateDescription({
          silent: true,
          sourceUrlOverride: response.source_url || sourceUrl,
        });
      }
      setMessage(
        opts?.generate
          ? `已根据字幕重新生成投稿标题/标签：${response.title || response.video_name}`
          : response.has_metadata
            ? `已载入投稿文案：${response.title || response.video_name}`
            : `已载入基础投稿信息：${response.title || response.video_name}`,
      );
      return response;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "加载投稿文案失败");
      return null;
    }
  };
  const saveTemplate = async () => {
    const name = templateDraftName.trim();
    if (!name) return setMessage("请填写模板名称。");
    try {
      await api("/api/templates", {
        method: "POST",
        body: JSON.stringify({
          name,
          body: templateBody || description,
          platform: publishPlatform,
        }),
      });
      await loadTemplates(publishPlatform);
      setTemplateName(name);
      localStorage.setItem(`ybl-publish-template-${publishPlatform}`, name);
      setTemplateEditorOpen(false);
      setMessage(publishPlatform === "douyin" ? "抖音模板已保存到本机。" : "B 站模板已保存到本机。");
    } catch (error) {
      setMessage(String(error));
    }
  };
  const openTemplateEditor = (name = "") => {
    const current = templates.find((item) => item.name === (name || templateName));
    const builtinDefault = publishPlatform === "douyin" ? "中文字幕版" : "授权本地化";
    setTemplateDraftName(name || (templateName === builtinDefault ? "" : templateName));
    if (name || !templateDraftDirty) {
      setTemplateBody(current?.body ?? templateBody);
    }
    setTemplateEditorOpen(true);
  };
  const insertTemplateVariable = (variable: string) => setTemplateBody((value) => `${value}${value && !value.endsWith(" ") ? " " : ""}${variable}`);
  const deleteTemplate = async () => {
    if (!window.confirm(`删除自定义模板“${templateName}”吗？`)) return;
    try {
      await api(
        `/api/templates/${encodeURIComponent(templateName)}?platform=${encodeURIComponent(publishPlatform)}`,
        { method: "DELETE" },
      );
      const next = await loadTemplates(publishPlatform);
      const fallback = publishPlatform === "douyin" ? "中文字幕版" : "授权本地化";
      const resolved = next.find((item) => item.name === fallback)?.name || next[0]?.name || fallback;
      setTemplateName(resolved);
      localStorage.setItem(`ybl-publish-template-${publishPlatform}`, resolved);
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
  const openDouyinPublisher = async () => {
    try {
      const response = await api<{ message: string }>("/api/publish/douyin/open", {
        method: "POST",
        body: "{}",
      });
      setMessage(response.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "打开抖音创作者服务平台失败");
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
      let nextDescription = description;
      if (!nextDescription.trim()) {
        nextDescription = (await generateDescription({ silent: true })) ?? "";
      }
      const response = await api<{ message: string }>("/api/publish/assist", {
        method: "POST",
        body: JSON.stringify({
          job_id: !nativePublishToken && !selectedPublishVideo && job?.id ? job.id : "",
          video_id: selectedPublishVideo === "__native__" ? "" : selectedPublishVideo,
          native_video_token: nativePublishToken,
          output_dir: options.output_dir,
          confirmed: true,
          browser: publishBrowser,
          title: options.title,
          description: nextDescription,
          tags: tags
            .split(",")
            .map((value) => value.trim())
            .filter(Boolean),
          include_source_link: includeSourceLink,
          template: templateName,
          template_body: templateBody,
          extra_lines: extraLine ? [extraLine] : [],
          close_after_fill: closeAfterFill,
          platform: "bilibili",
        }),
      });
      setMessage(response.message);
      void loadPublishSession();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "发布辅助失败");
    }
  };
  const publishDouyinAssist = async () => {
    if (!publishVideoReady) {
      setMessage("请先选择当前任务成片、输出目录中的成片，或任意本地成片，再启动抖音投稿辅助。");
      return;
    }
    if (publishSession.active) {
      setMessage("已有投稿辅助正在运行，请先关闭当前自动化浏览器。");
      return;
    }
    if (!window.confirm("将把所选成片上传到抖音创作者中心，并尝试填写标题、正文和话题。应用不会点击最终发布；请在抖音页面检查后手动发布。继续吗？")) return;
    try {
      let nextDescription = description;
      if (!nextDescription.trim()) {
        nextDescription = (await generateDescription({ silent: true })) ?? "";
      }
      const response = await api<{ message: string }>("/api/publish/douyin/assist", {
        method: "POST",
        body: JSON.stringify({
          job_id: !nativePublishToken && !selectedPublishVideo && job?.id ? job.id : "",
          video_id: selectedPublishVideo === "__native__" ? "" : selectedPublishVideo,
          native_video_token: nativePublishToken,
          output_dir: options.output_dir,
          confirmed: true,
          browser: publishBrowser,
          title: options.title,
          description: nextDescription,
          tags: tags.split(",").map((value) => value.trim()).filter(Boolean),
          include_source_link: false,
          template: templateName,
          template_body: templateBody,
          extra_lines: extraLine ? [extraLine] : [],
          platform: "douyin",
        }),
      });
      setMessage(response.message);
      void loadPublishSession();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "抖音投稿辅助失败");
    }
  };
  const closePublishBrowser = async () => {
    const platformLabel = publishPlatform === "douyin" ? "抖音" : "B 站";
    if (!window.confirm(`关闭${platformLabel}自动化浏览器会中断上传。确认关闭吗？`))
      return;
    try {
      const response = await api<{ closed: number }>(
        "/api/publish/close-browser",
        { method: "POST", body: JSON.stringify({ confirmed: true, platform: publishPlatform }) },
      );
      setMessage(`已关闭 ${response.closed} 个后台浏览器进程。`);
    } catch (error) {
      setMessage(String(error));
    }
  };
  const deleteOutputs = async () => {
    if (!outputs || !selectedOutputPaths.length) return;
    try {
      const plan = await api<OutputDeletePlan>("/api/outputs/delete-preview", {
        method: "POST", body: JSON.stringify({ output_dir: outputs.root, paths: selectedOutputPaths }),
      });
      setDeleteResults(null);
      setPendingDelete({ source: "outputs", plan });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "读取删除预览失败");
    }
  };
  const confirmPendingDelete = async () => {
    if (!pendingDelete || deleteBusy) return;
    if (pendingDelete.source === "history" && pendingDelete.mode === "record_and_outputs" && (pendingDelete.plan.record_only_required_count ?? 0) > 0) {
      if (!window.confirm(`其中 ${pendingDelete.plan.record_only_required_count} 条记录只能删除历史记录，不能删除磁盘文件。确认仅删除这些记录，并删除已验证输出吗？`)) return;
    }
    setDeleteBusy(true);
    try {
      const response = pendingDelete.source === "history"
        ? await api<{ results: DeleteItemResult[] }>("/api/history/jobs/delete", {
            method: "POST", body: JSON.stringify({ ids: pendingDelete.plan.job_ids, mode: pendingDelete.mode, fingerprint: pendingDelete.plan.fingerprint, confirmed: true }),
          })
        : await api<{ results: DeleteItemResult[] }>("/api/outputs/delete", {
            method: "POST", body: JSON.stringify({ output_dir: pendingDelete.plan.root, paths: pendingDelete.plan.paths, fingerprint: pendingDelete.plan.fingerprint, confirmed: true }),
          });
      const succeeded = response.results.filter((item) => item.status === "deleted").map((item) => item.id);
      succeeded.forEach((id) => clearLoadedTaskState(pendingDelete.source === "outputs" ? id.split("/")[0] : id));
      if (pendingDelete.source === "history") {
        setSelectedOutputTaskIds((current) => current.filter((id) => !succeeded.includes(id)));
        if (succeeded.length === response.results.length) setLibraryBatchMode(false);
      } else {
        setSelectedOutputPaths((current) => current.filter((id) => !succeeded.includes(id)));
      }
      if (selectedPublishVideo && succeeded.some((id) => selectedPublishVideo.includes(id))) setSelectedPublishVideo("");
      setDeleteResults(response.results);
      setMessage(`已删除 ${succeeded.length} 项；${response.results.length - succeeded.length} 项未完成。详情见结果清单。`);
      await Promise.all([loadJobHistory(), loadRestorableJobs(), loadOutputs().catch(() => undefined)]);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除失败；请刷新预览后再试");
      setPendingDelete(null);
    } finally { setDeleteBusy(false); }
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
      setOutputs(null);
      setSelectedOutputPaths([]);
      setViewedTaskId(null);
      setMessage(`输出目录已设为：${selection.path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "选择输出目录失败");
    }
  };
  const [rightsHint, setRightsHint] = useState(false);
  const [weeklyField, setWeeklyField] = useState<{ key: string; value: string } | null>(null);
  useEffect(() => {
    if (!weeklyBusy) return;
    const timer = window.setInterval(() => {
      void api<WeeklyHot>("/api/subscriptions/weekly").then((data) => {
        if (!weeklyBusyRef.current) return;
        setWeekly((current) => current ? { ...current, channel_progress: data.channel_progress, refreshing: data.refreshing } : data);
      }).catch(() => undefined);
    }, 700);
    return () => window.clearInterval(timer);
  }, [weeklyBusy]);
  useEffect(() => {
    if (!siteBusy) return;
    const category = siteCategoryRef.current;
    const timer = window.setInterval(() => {
      void api<SiteHot>(`/api/trending?category=${encodeURIComponent(category)}`).then((data) => {
        if (!siteBusyRef.current || siteCategoryRef.current !== category) return;
        setSite((current) => current && current.category === category
          ? { ...current, progress: data.progress, refreshing: data.refreshing }
          : current);
      }).catch(() => undefined);
    }, 700);
    return () => window.clearInterval(timer);
  }, [siteBusy]);
  const chooseNativeMaterial = async () => {
    if (running) return setMessage("当前阶段运行中，请先取消或等待完成后再切换素材。");
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
      if (!confirmDiscardSubtitles("切换到本地视频")) return;
      await loadMaterials();
      resetForSourceSwitch();
      setMaterialId(selection.material_id);
      setSourceUrl("");
      setMaterialSourceTab("local");
      setMaterialDraftToken((value) => value + 1);
      updateOptions("title", selection.name.replace(/\.[^.]+$/, ""));
      setMessage(`已导入本地视频（直接使用原路径，不会复制到工作区）：${selection.name}。`);
    } catch (error) {
      setMessage(`导入本地视频失败：${error instanceof Error ? error.message : String(error)}`);
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
      publishHydrateKey.current = "";
      void generateDescription({ silent: true });
      setMessage(`已选择成品视频：${detail.name}。`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "选择成品视频失败"); }
  };

  const weeklyFailures = weekly?.channel_results.filter((item) => !item.ok) ?? [];
  const canRefreshWeekly = Boolean(weekly?.enabled && weekly.channels.length);
  const weeklyRankedByViews = weekly?.sort_mode === "views";
  const weeklyTitle = weeklyRankedByViews || !weekly?.videos.length ? "本周热片" : "本周新片";
  const weeklyWindow = weekly?.window_days ?? 7;
  const weeklyBusyLabel = weekly?.channel_progress ? `正在刷新 ${weekly.channel_progress.done}/${weekly.channel_progress.total}` : "正在刷新…";
  const weeklySummary = !weekly
    ? "读取关注列表…"
    : !weekly.enabled
      ? "未开启 · 展开查看说明"
      : weekly.channels.length === 0
        ? "还没有频道 · 展开查看说明"
        : weeklyBusy
          ? weeklyBusyLabel
          : weeklyRankedByViews
            ? `${weekly.videos.length} 条近 ${weeklyWindow} 天热片`
            : `${weekly.videos.length} 条近 ${weeklyWindow} 天新片`;
  const siteLabel = site?.categories.find((item) => item.id === siteCategory)?.label ?? "综合";
  const siteBusyLabel = site?.progress && site.progress.category === siteCategory
    ? `正在刷新 ${site.progress.done}/${site.progress.total}`
    : "正在刷新…";
  const siteSummary = !site
    ? "正在读取…"
    : siteBusy
      ? siteBusyLabel
      : "按本周播放量搜索";
  const weeklyHasChannels = Boolean(weekly?.enabled && weekly.channels.length > 0);
  const siteVideos = (site && site.category === siteCategory ? site.videos : []).filter((entry) => siteDurationMatches(entry.duration, siteDurationBand));
  const siteRegions = site?.regions ?? [{ id: "US", label: "美国" }, { id: "GB", label: "英国" }, { id: "", label: "不限" }];
  const siteFailedQueries = (site?.query_errors ?? []).map((item) => item.split(":")[0]).filter(Boolean);
  const siteReady = Boolean(site && site.category === siteCategory);
  const weeklyNumber = (key: "window_days" | "top_k" | "fetch_count" | "cache_ttl_hours", label: string, min: number, max: number) => (
    <label>
      {label}
      <input
        type="number"
        min={min}
        max={max}
        value={weeklyField?.key === key ? weeklyField.value : String(weekly?.[key] ?? "")}
        disabled={channelSaving || !weekly}
        onFocus={() => setWeeklyField({ key, value: String(weekly?.[key] ?? "") })}
        onChange={(event) => setWeeklyField({ key, value: event.target.value })}
        onBlur={() => {
          const raw = weeklyField?.key === key ? weeklyField.value : String(weekly?.[key] ?? "");
          setWeeklyField(null);
          const number = Number(raw);
          if (weekly && Number.isFinite(number) && number !== weekly[key]) void saveWeeklyPrefs({ [key]: number });
        }}
      />
    </label>
  );
  const hotVideoRow = (entry: WeeklyHotVideo, when: string | null, showViews: boolean, row?: { rank?: number; showStatLabels?: boolean }) => {
    const filled = !batchMode && sourceUrl === entry.url;
    const queued = batchMode && batchHasUrl(entry.url);
    const showStatLabels = row?.showStatLabels === true;
    const showRankColumn = row !== undefined && "rank" in row;
    return (
      <div className={`video-library-row${showRankColumn ? " has-rank" : ""}`} key={entry.url}>
        {showRankColumn && <span className="weekly-hot-rank">{row.rank ?? ""}</span>}
        {weekly?.show_thumbnails !== false && (
          <div className="weekly-hot-thumb-wrap">
            {entry.thumbnail_url
              ? <WeeklyThumb src={entry.thumbnail_url} />
              : <span className="weekly-hot-thumb is-placeholder" aria-hidden="true" />}
            {typeof entry.duration === "number" && <span className="weekly-hot-duration">{formatClock(entry.duration)}</span>}
          </div>
        )}
        <div className="video-library-main">
          <p className="weekly-hot-eyebrow">{entry.channel_title || "频道"}</p>
          <div className="video-library-title"><b title={entry.title}>{entry.title}</b></div>
          <div className="weekly-hot-stats" aria-label="视频数据">
            {showViews && typeof entry.view_count === "number" && (
              <div className="weekly-hot-stat views">
                {showStatLabels && <span className="weekly-hot-stat-label">播放量</span>}
                <span className="weekly-hot-stat-value">{entry.view_count.toLocaleString()}</span>
              </div>
            )}
            {typeof entry.duration === "number" && (
              <div className="weekly-hot-stat">
                {showStatLabels && <span className="weekly-hot-stat-label">时长</span>}
                <span className="weekly-hot-stat-value">{formatClock(entry.duration)}</span>
              </div>
            )}
            {when && (
              <div className="weekly-hot-stat">
                {showStatLabels && <span className="weekly-hot-stat-label">上传</span>}
                <span className="weekly-hot-stat-value">{when}</span>
              </div>
            )}
          </div>
        </div>
        <div className="video-library-actions">
          <button type="button" className="small open-original" onMouseDown={(event) => event.preventDefault()} onClick={() => openOriginalVideo(entry.url)}>原链接</button>
          <button type="button" className="small use-video" onMouseDown={(event) => event.preventDefault()} onClick={() => batchMode ? addHotToBatch(entry) : fillSourceFromHot(entry.url)} disabled={running || batchBusy || filled || queued}>{batchMode ? (queued ? "已加入" : "加入批量") : (filled ? "已填入" : "用这条")}</button>
        </div>
      </div>
    );
  };
  const weeklyHotCard = (
    <details className={`workspace-card weekly-hot-card${weekly?.show_thumbnails === false ? " no-thumbs" : ""}`}>
      <summary>
        <b>{hotTab === "channels" ? weeklyTitle : "本周热片"}</b>
        <span>{hotTab === "site" ? siteSummary : weeklySummary}</span>
      </summary>
      <div className="weekly-hot-body">
        <div className="weekly-hot-toolbar">
          {weeklyHasChannels && (
            <div className="weekly-hot-tabs" role="tablist">
              <button type="button" className={hotTab === "site" ? "selected" : ""} aria-selected={hotTab === "site"} onClick={() => setHotTab("site")}>本周搜索</button>
              <button type="button" className={hotTab === "channels" ? "selected" : ""} aria-selected={hotTab === "channels"} onClick={() => setHotTab("channels")}>关注频道</button>
            </div>
          )}
          {hotTab === "site" ? (
            <div className="weekly-hot-tools">
              <button type="button" onClick={() => void refreshSite()} disabled={siteBusy} aria-busy={siteBusy}>
                {siteBusy ? siteBusyLabel : "刷新"}
              </button>
              <label className="weekly-hot-region">
                搜索地区
                <select
                  aria-label="搜索地区"
                  value={site?.region ?? "US"}
                  disabled={siteBusy || !site}
                  onChange={(event) => void saveSiteRegion(event.target.value)}
                >
                  {siteRegions.map((item) => <option key={item.id || "any"} value={item.id}>{item.label}</option>)}
                </select>
              </label>
              {siteReady && site?.refreshed_at && <small>上次获取 {new Date(site.refreshed_at).toLocaleString()}</small>}
              {!weeklyHasChannels && (
                <button type="button" className="weekly-hot-quiet" onClick={() => setHotTab("channels")}>关注频道</button>
              )}
            </div>
          ) : weeklyHasChannels ? (
            <div className="weekly-hot-tools">
              <button type="button" onClick={() => void refreshWeekly()} disabled={!canRefreshWeekly || weeklyBusy} aria-busy={weeklyBusy}>
                {weeklyBusy ? weeklyBusyLabel : weeklyFailures.length ? "全部重试" : "刷新"}
              </button>
              {weekly?.refreshed_at && <small>上次获取 {new Date(weekly.refreshed_at).toLocaleString()}</small>}
            </div>
          ) : (
            <div className="weekly-hot-tools">
              <button type="button" className="weekly-hot-quiet" onClick={() => setHotTab("site")}>返回热片</button>
            </div>
          )}
        </div>
        {hotTab === "site" ? (
          <>
            <div className="weekly-hot-cats" role="tablist" aria-label="分类">
              {(site?.categories ?? []).map((item) => (
                <button type="button" key={item.id} className={item.id === siteCategory ? "selected" : ""} aria-selected={item.id === siteCategory} onClick={() => selectSiteCategory(item.id)}>
                  {item.label}
                </button>
              ))}
            </div>
            <div className="weekly-hot-cats weekly-hot-bands" role="tablist" aria-label="时长">
              {SITE_DURATION_BANDS.map((item) => (
                <button type="button" key={item.id} className={item.id === siteDurationBand ? "selected" : ""} aria-selected={item.id === siteDurationBand} onClick={() => setSiteDurationBand(item.id)}>
                  {item.label}
                </button>
              ))}
            </div>
            {siteReady && site?.category === "mix" && (
              <p className="weekly-hot-note">综合是科技、游戏、科普和编程。音乐、影视、生活、资讯在各自的分类里。</p>
            )}
            {!site && <div className="empty-editor"><b>正在读取热片</b><span>稍候即可查看近一周播放量较高的视频。</span></div>}
            {siteError && <p className="metadata-notice error" role="status">{siteError}</p>}
            {siteReady && site?.error && <p className="metadata-notice error" role="status">{site.error}</p>}
            {siteReady && site?.refreshed_at && site.cached_region !== site.region && !siteBusy && (
              <p className="metadata-notice warning" role="status">地区已改，下面暂时仍是上次的列表。</p>
            )}
            {siteReady && siteFailedQueries.length > 0 && !site?.error && (
              <p className="metadata-notice warning" role="status">部分搜索词没有拉到：{siteFailedQueries.join("、")}。下面是这次拿到的结果。</p>
            )}
            {site && !siteReady && !siteError && (
              <div className="empty-editor"><b>正在读取{siteLabel}</b><span>稍候。</span></div>
            )}
            {siteReady && site && site.videos.length === 0 && !siteBusy && (
              <div className="empty-editor">
                <b>{site.error ? "没有拿到视频" : "还没有列表"}</b>
                <span>{site.error ? "上次如果有列表会留在这里。" : "会按本周播放量自动获取。"}</span>
              </div>
            )}
            {siteReady && site && site.videos.length > 0 && siteVideos.length === 0 && !siteBusy && (
              <div className="empty-editor">
                <b>这个时长里没有片子</b>
                <span>换一个时长，或看全部。</span>
              </div>
            )}
            {siteBusy && siteReady && (site?.videos.length ?? 0) === 0 && (
              <div className="empty-editor">
                <b>正在搜索本周热门</b>
                <span>按这个分类的词搜索近一周视频，去掉 Shorts 和直播后按播放量排列。</span>
              </div>
            )}
            {siteReady && siteVideos.length > 0 && (
              <div className="weekly-hot-list">
                {siteVideos.map((entry, index) => hotVideoRow(entry, formatDaysAgo(entry.timestamp), true, {
                  rank: index < 3 ? index + 1 : undefined,
                  showStatLabels: index === 0,
                }))}
              </div>
            )}
          </>
        ) : (
          <>
            {!weekly && <div className="empty-editor"><b>正在读取关注列表</b><span>稍候即可查看本周热片。</span></div>}
            {weeklyError && <p className="metadata-notice error" role="status">{weeklyError}</p>}
            {weeklyFailures.map((item) => (
              <p className="metadata-notice warning weekly-hot-failure" role="status" key={item.id}>
                <span>{item.title || item.url}：{item.error || "读取失败"}</span>
                <button type="button" onClick={() => void refreshWeekly(item.id)} disabled={weeklyBusy}>重试</button>
              </p>
            ))}
            {weekly && !weekly.enabled && (
              <div className="empty-editor">
                <b>关注频道未开启</b>
                <span>到设置里打开「启用周热」，添加 YouTube 频道后，这里会列出近 {weeklyWindow} 天的片子。</span>
                <button type="button" onClick={() => setSettingsOpen(true)}>打开设置</button>
              </div>
            )}
            {weekly?.enabled && weekly.channels.length === 0 && (
              <div className="empty-editor">
                <b>还没有关注的频道</b>
                <span>在设置 → 关注频道里粘贴 https://www.youtube.com/@handle 或 /channel/UC… 链接。</span>
                <button type="button" onClick={() => setSettingsOpen(true)}>去添加频道</button>
              </div>
            )}
            {weekly?.enabled && weekly.channels.length > 0 && weekly.videos.length === 0 && !weeklyBusy && (
              <div className="empty-editor">
                <b>{weeklyFailures.length ? "没有拿到新片" : `这 ${weeklyWindow} 天没有新片`}</b>
                <span>{weeklyFailures.length ? "失败的频道列在上面，可以重试该频道，或点「全部重试」。成功的频道会留在列表里。" : `已关注的频道在近 ${weeklyWindow} 天内没有带上传时间的视频。没有日期的不会出现在这里。`}</span>
              </div>
            )}
            {weeklyBusy && (weekly?.videos.length ?? 0) === 0 && canRefreshWeekly && (
              <div className="empty-editor">
                <b>正在获取最近上传</b>
                <span>每个频道读取最近约 {weekly?.fetch_count ?? 20} 条，只保留 {weeklyWindow} 天内、能读到上传时间的视频。</span>
              </div>
            )}
            {weekly?.videos.map((entry, index) => hotVideoRow(entry, formatUploadDay(entry.upload_date), weeklyRankedByViews, { showStatLabels: index === 0 }))}
            {weekly?.enabled && weekly.channels.length > 0 && (
              <p className="weekly-hot-note">{weeklyRankedByViews
                ? "按播放量从高到低，播放量相同则较新的在前。数字来自频道页，没有逐条再打开视频。"
                : "按上传时间从新到旧。播放量缺失时用这个顺序，也不逐条重抓。日期来自频道页的近似时间，可能有几小时偏差；太旧或没有日期的不会列入。"}</p>
            )}
          </>
        )}
      </div>
    </details>
  );
  const ffmpegReady = dependencies.some((item) => item.id === "ffmpeg" && item.available);
  const whisperReady = dependencies.some((item) => item.id === "whisper-small" && item.available);
  const setupTranslationReady = apiKeyConfigured === true || openaiKeyConfigured === true || options.translator === "ollama";
  const setupOpen = !setupLoading && !setupDismissed && dependencies.length > 0 && apiKeyConfigured !== null && openaiKeyConfigured !== null && (!ffmpegReady || !whisperReady || !setupTranslationReady);
  const materialView = (
    <section className="workspace-card material-view">
      <header>
        <p className="eyebrow">素材来源</p>
        <h1>添加素材</h1>
        <p>选择视频链接或本地文件；仅处理你已获授权的内容。</p>
      </header>
      {!preferencesLoading && !setupOpen && !tutorialDismissed && (
        <div className="welcome-banner tutorial-welcome">
          <div>
            <b>第一次使用？先看新手教程</b>
            <span>用几分钟了解素材、依赖、字幕校对、渲染和投稿辅助。教程不会启动任务或下载。</span>
          </div>
          <button className="primary small" onClick={() => { dismissTutorial(); openTutorial(); }}>开始教程</button>
          <button className="secondary small" onClick={dismissTutorial}>以后再看</button>
        </div>
      )}
      {!setupOpen && apiKeyConfigured === false && options.translator === "deepseek" && (
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
      {!setupOpen && diagnostics.some((check) => check.required && !check.available) && (
        <section className="diagnostics-banner" aria-label="首次使用依赖检查">
          <div>
            <b>开始前还需安装 {diagnostics.filter((check) => check.required && !check.available).map((check) => check.label).join("、")}</b>
            <span>这里只列出默认流程的必需组件；Tesseract 等 OCR 组件按所选模式检查。</span>
          </div>
          <button className="text-button" onClick={() => setDependenciesOpen(true)}>打开依赖中心</button>
        </section>
      )}
      <div className="source-tabs" role="tablist" aria-label="素材来源">
        <button type="button" role="tab" aria-selected={materialSourceTab === "url"} aria-controls="material-url-panel" className={materialSourceTab === "url" ? "selected" : ""} onClick={() => setMaterialSourceTab("url")}>视频链接</button>
        <button type="button" role="tab" aria-selected={materialSourceTab === "local"} aria-controls="material-local-panel" className={materialSourceTab === "local" ? "selected" : ""} onClick={() => setMaterialSourceTab("local")}>本地文件</button>
      </div>
      <div className="import-methods">
        {materialSourceTab === "url" && <section className="material-source material-url" id="material-url-panel" role="tabpanel" aria-label="视频链接导入">
          <p className="source-intro">粘贴 YouTube 或 B 站地址，读取标题与时长后再确认处理范围。</p>
      <label className="check"><input type="checkbox" checked={batchMode} disabled={running} onChange={(event) => setBatchMode(event.target.checked)} />批量（每行一个链接，或粘贴播放列表链接）</label>
      {!batchMode && (
      <div className="form-grid two">
        <label>
          视频链接
          <input
            id="source-url"
            value={sourceUrl}
            maxLength={2048}
            placeholder="粘贴 YouTube / B 站视频链接"
            onChange={(event) => {
              if (job && !window.confirm(`修改素材链接会离开当前任务；原任务可在任务库重新载入。${dirty ? "当前未保存的字幕修改会丢失。" : ""}\n\n确定继续吗？`)) return;
              if (job || selected) resetForSourceSwitch();
              else sourceRevision.current += 1;
              if (job || selected) setMessage("已切换到新链接草稿；原任务可在任务库重新载入。");
              if (job || selected || metadataState === "success") updateOptions("title", "");
              setMaterialId("");
              setSourceUrl(event.target.value);
              setMaterialDraftToken((value) => value + 1);
              setFieldErrors((current) => ({ ...current, source_url: "" }));
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
            onClick={() => void readMetadata()}
            disabled={running || metadataState === "loading"}
            aria-busy={metadataState === "loading"}
          >
            {metadataState === "loading" && <ActivityBars />}
            {metadataState !== "loading" && <NavIcon name="link" className="app-icon" />}
            {metadataState === "loading" ? "正在读取…" : "读取标题与时长"}
          </button>
        </div>
      </div>
      )}
      {batchMode && (
        <div className="form-grid">
          <label>
            视频链接（每行一个）
            <textarea id="batch-urls" value={batchText} rows={5} maxLength={20000} placeholder="每行一个 YouTube / B 站链接，或一个播放列表链接。热片里可点「加入批量」。" onChange={(event) => setBatchText(event.target.value)} />
          </label>
          <div className="inline-actions">
            <button onClick={() => void parseBatch()} disabled={batchBusy || !batchText.trim()} aria-busy={batchBusy}>
              {batchBusy ? <ActivityBars /> : <NavIcon name="link" className="app-icon" />}
              {batchBusy ? "正在解析…" : "解析"}
            </button>
          </div>
          {batchEntries.length > 0 && (
            <div className="batch-parse">
              <div className="batch-panel-head">
                <h2>解析结果<small>已选 {batchEntries.filter((entry) => entry.selected).length} / {batchEntries.length}</small></h2>
              </div>
              {batchEntries.map((entry, index) => (
                <label className={`video-library-row ${entry.error ? "attention" : ""}`} key={`${entry.url}-${index}`}>
                  <input type="checkbox" checked={entry.selected} disabled={!!entry.error} onChange={(event) => setBatchEntries((current) => current.map((row, i) => i === index ? { ...row, selected: event.target.checked } : row))} />
                  <div className="video-library-main">
                    <div className="video-library-title"><b title={entry.url}>{entry.title || entry.url}</b></div>
                    <div className="video-library-meta">
                      {entry.error ? <><span className="library-status attention">解析失败</span><span>{entry.error}</span></> : entry.duration ? <span>时长 {Math.round(entry.duration)} 秒</span> : null}
                    </div>
                  </div>
                  <span />
                </label>
              ))}
              <div className="batch-parse-foot">
                <label className="check">
                  <input type="checkbox" checked={batchQualityConsent} onChange={(event) => setBatchQualityConsent(event.target.checked)} />
                  可用画质低于所选上限时，按可用画质继续（整批共用）
                </label>
                {voxcpm.voxcpm_available && (
                  <label className="check">
                    <input type="checkbox" checked={batchDub} onChange={(event) => setBatchDub(event.target.checked)} />
                    同时生成配音版（整批共用）
                  </label>
                )}
                <button className="primary" onClick={() => void startBatch()} disabled={batchBusy || !batchEntries.some((entry) => entry.selected)}>加入批量并开始</button>
              </div>
            </div>
          )}
          {batchEntries.length > 0 && <small className="check-hint">批量沿用当前处理设置；只处理到压制字幕，不会自动投稿。</small>}
        </div>
      )}
      {metadataState !== "idle" && (
        <div className="metadata-notice-wrap">
          {metadataState === "error" && /ERROR:|WinError|Traceback|OSError|HTTP Error|FileNotFoundError/i.test(metadataNotice)
            ? <TaskErrorSummary message={metadataNotice} fallbackTitle="视频信息读取失败" fallbackHint="请检查链接或访问设置，再重新读取。" />
            : <p className={`metadata-notice ${metadataState}`} role="status">{metadataNotice}</p>}
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
      {metadata?.max_height && options.download_quality !== "original" && metadata.max_height < (options.download_quality === "1080p" ? 1080 : 720) && (
        <p className="metadata-notice warning" role="status">
          源站最高为 {metadata.max_width ?? "?"}×{metadata.max_height}。低于当前画质选择；开始前会要求确认，应用不会静默降档。
        </p>
      )}
      {job?.quality_proposal?.requires_consent && !job.quality_consent && (
        <div className="metadata-notice warning" role="alert">
          <p>服务器拟使用 {job.quality_proposal.width ?? "?"}×{job.quality_proposal.height ?? "?"}，{(job.quality_proposal.baseline_width ?? 0) > (job.quality_proposal.width ?? 0) || (job.quality_proposal.baseline_height ?? 0) > (job.quality_proposal.height ?? 0) ? `低于此前检查到的 ${job.quality_proposal.baseline_width ?? "?"}×${job.quality_proposal.baseline_height ?? "?"}` : `低于 ${job.quality_proposal.quality} 档位要求`}。未确认前不会下载低画质。</p>
          <button className="primary small" onClick={() => void confirmQualityProposal()} disabled={running || preparingJob}>确认此画质方案</button>
        </div>
      )}
        </section>}
        {materialSourceTab === "local" && <section className="material-source material-local" id="material-local-panel" role="tabpanel" aria-label="本地视频导入">
          <p className="source-intro">选择已导入视频，或从电脑中选择新视频。桌面版直接使用原路径。</p>
      <div className="form-grid two">
        <label>
          已导入视频
          <select
            value={materialId}
            onChange={(event) => {
              if (dirty && !confirmDiscardSubtitles("切换素材会清空当前未保存的字幕修改")) return;
              resetForSourceSwitch();
              setMaterialId(event.target.value);
              setSourceUrl("");
              updateOptions("title", materials.find((item) => item.id === event.target.value)?.name.replace(/\.[^.]+$/, "") ?? "");
              setMaterialDraftToken((value) => value + 1);
              setMessage("已切换本地素材；原任务可在任务库重新载入。");
            }}
            disabled={running}
          >
            <option value="">未选择视频</option>
            {job && materialId && !selected && <option value={materialId}>当前历史任务 · {job.material.name}</option>}
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
        </div>
      </div>
      <p className="material-hint">
        桌面版直接使用本地视频原路径，不占用双倍磁盘；处理期间请勿移动或删除原文件。
      </p>
        </section>}
      </div>
      <div className="material-detail-heading"><b>处理设置与授权</b><span>先确认这一步，再进入处理方案</span></div>
      <div className={`form-grid three material-settings ${materialSourceTab === "local" ? "local-settings" : ""}`}>
        <label>
          标题
          <input
            value={options.title}
            maxLength={200}
            onChange={(event) => updateOptions("title", event.target.value)}
          />
          {fieldErrors.title && <small className="field-error">{fieldErrors.title}</small>}
        </label>
        {materialSourceTab === "url" && <fieldset className="material-range">
          <legend>处理范围</legend>
          <label className="check"><input type="radio" name="material-range" checked={fullVideo} onChange={() => setFullVideo(true)} />完整视频</label>
          <div className="range-limited">
            <label className="check"><input type="radio" name="material-range" checked={!fullVideo} onChange={() => setFullVideo(false)} />仅处理前</label>
            <input aria-label="处理前多少时长" type="number" min={durationUnit === "秒" ? 1 : 1 / 60} max={durationUnit === "小时" ? 24 : durationUnit === "分钟" ? 1440 : 86400} step={durationUnit === "秒" ? 1 : durationUnit === "分钟" ? 1 / 60 : 1 / 3600} value={durationValue} disabled={fullVideo} onChange={(event) => { setDurationValue(event.target.value); setFieldErrors((current) => ({ ...current, max_seconds: "" })); }} />
            <select aria-label="处理范围单位" value={durationUnit} disabled={fullVideo} onChange={(event) => setDurationUnit(event.target.value as "秒" | "分钟" | "小时")}><option>秒</option><option>分钟</option><option>小时</option></select>
          </div>
          <small>仅处理开头片段，便于快速测试识别、翻译和字幕效果。</small>
          {fieldErrors.max_seconds && <small className="field-error">{fieldErrors.max_seconds}</small>}
        </fieldset>}
        {materialSourceTab === "url" && <label>
          下载画质
          <select
            value={options.download_quality}
            disabled={!sourceUrl.trim() || running}
            onChange={(event) => updateOptions("download_quality", event.target.value as Options["download_quality"])}
          >
            <option value="720p">最高 720p（低负载）</option>
            <option value="1080p">最高 1080p（推荐）</option>
            <option value="original">原始画质（最高可用，不静默降档）</option>
          </select>
        </label>}
        {materialSourceTab === "url" && <label className="check cc-reuse-check">
          <input
            type="checkbox"
            checked={options.require_reuse_allowed}
            onChange={(event) => void persistRequireReuseAllowed(event.target.checked)}
            disabled={!sourceUrl.trim() || running}
          />
          <span>仅接受可复用 CC 视频</span>
          <small className="check-hint">
            {job
              ? `当前任务：${job.options.require_reuse_allowed ? "仅接受可复用 CC 视频" : "未启用 CC 限制"}`
              : "大多数视频无此标记，勾选后可能无法下载"}
            {job && job.options.require_reuse_allowed !== options.require_reuse_allowed ? "（正在同步到任务）" : ""}
          </small>
        </label>}
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
        <button className="primary material-submit" onClick={start} disabled={running || !materialSourceReady}>保存素材并配置方案 →</button>
      </div>
    </section>
  );
  const preparationBlocking = (job?.checks ?? []).some((item) => item.status === "blocking");
  const preparationReady = Boolean(job?.checks?.length) && !preparationBlocking;
  const schemeDirty = Boolean(job && (
    device !== job.device || computeType !== job.compute_type ||
    Object.entries(taskOptions()).some(([key, value]) =>
      !["cookies_file", "youtube_proxy"].includes(key) &&
      key in job.options && !sameTaskOption(key, value, job.options[key as keyof Options]),
    )
  ));
  schemeDirtyRef.current = schemeDirty;
  const smallerInstalledWhisperModels = job?.recovery?.id === "retry_extract_with_lower_resource"
    ? dependencies
      .filter((item) => item.available && item.id.startsWith("whisper-"))
      .map((item) => item.id.slice("whisper-".length))
      .filter((model): model is typeof whisperModelOrder[number] =>
        whisperModelOrder.includes(model as typeof whisperModelOrder[number]) &&
        whisperModelOrder.indexOf(model as typeof whisperModelOrder[number]) < whisperModelOrder.indexOf(job.options.whisper_model_size as typeof whisperModelOrder[number]),
      )
      .sort((left, right) => whisperModelOrder.indexOf(right) - whisperModelOrder.indexOf(left))
    : [];
  const handleCheckAction = (check: PreparationCheck) => {
    const action = check.action ?? "";
    if (/安装|下载|模型|FFmpeg|Tesseract|依赖中心|Ollama/.test(action)) {
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
          <p className={schemeDirty ? "scheme-warning" : undefined}>{schemeDirty ? "当前方案有未保存的修改。阶段状态仍对应上次保存的方案；先保存并检查准备。" : "已完成阶段会保留检查点。取消或关闭应用后，不会丢掉之前的有效产物。"}</p>
          {job.quality_proposal?.requires_consent && !job.quality_consent && (
            <div className="metadata-notice warning" role="alert">
              <p>获取素材拟使用 {job.quality_proposal.width ?? "?"}×{job.quality_proposal.height ?? "?"}，{(job.quality_proposal.baseline_width ?? 0) > (job.quality_proposal.width ?? 0) || (job.quality_proposal.baseline_height ?? 0) > (job.quality_proposal.height ?? 0) ? `低于此前检查到的 ${job.quality_proposal.baseline_width ?? "?"}×${job.quality_proposal.baseline_height ?? "?"}` : `低于 ${job.quality_proposal.quality} 档位要求`}。请明确确认。</p>
              <button className="primary small" onClick={() => void confirmQualityProposal()} disabled={running || preparingJob}>确认画质方案</button>
            </div>
          )}
        </div>
        <div className="workflow-actions">
          <button onClick={() => void saveWorkflowOptions()} disabled={running || preparingJob}><NavIcon name="save" className="app-icon" />{schemeDirty ? "保存修改" : "保存方案"}</button>
          <button className="primary" onClick={() => void prepareJob()} disabled={running || preparingJob} aria-busy={preparingJob} title={running ? "当前阶段运行中" : undefined}>
            {preparingJob && <ActivityBars />}
            {preparingJob ? "正在检查素材与环境…" : job.checks?.length ? "重新检查准备" : "检查准备"}
          </button>
          <button
            onClick={() => {
              if (running || !preparationReady || schemeDirty) {
                const reason = schemeDirty
                  ? "先保存修改并重新检查准备"
                  : !preparationReady
                    ? "准备检查全部通过后才能运行"
                    : "当前阶段运行中";
                setMessage(reason);
                return;
              }
              void runAll();
            }}
            disabled={false}
            aria-disabled={running || !preparationReady || schemeDirty}
            title={schemeDirty ? "先保存修改并重新检查准备" : !preparationReady ? "准备检查全部通过后才能运行" : undefined}
          >
            <NavIcon name="render" className="app-icon" />一键运行到渲染
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
          <div><b>开始前准备</b><small>模型、工具、API、设备和磁盘问题在运行前显示</small></div>
          {job.checks?.length ? <span className={preparationBlocking ? "blocking" : "passed"}>{preparationBlocking ? "存在阻断项" : "全部可运行"}</span> : <span>尚未检查</span>}
        </div>
        {job.checks?.length ? (
          <div className="check-list">
            {job.checks.map((check) => (
              <article key={check.id} className={`prepare-check ${check.status}`}>
                <span className="check-state" aria-hidden="true"><NavIcon name={check.status === "passed" ? "check" : check.status === "warning" ? "error" : check.status === "installing" ? "running" : "error"} className="app-icon" /></span>
                <div><b>{check.label}</b><small>{check.purpose}</small><p>{check.message}</p></div>
                {check.id === "ollama-runtime" && ollamaRuntimeDependency?.state === "stopped" ? (
                  <button disabled={ollamaStartBusy} onClick={() => void startOllama()}>
                    {ollamaStartBusy ? "启动中…" : "启动 Ollama"}
                  </button>
                ) : check.action && check.status !== "passed" ? <button onClick={() => handleCheckAction(check)}>{check.action}</button> : null}
              </article>
            ))}
          </div>
        ) : <p className="empty-checks">保存当前方案后点击“检查准备”。检查不会下载视频，也不会自动安装任何组件。</p>}
      </section>
      <section className="stage-list">
        {(job.stages?.dub ? [...workflowStageMeta.slice(0, 4), dubStageMeta, ...workflowStageMeta.slice(4)] : workflowStageMeta).map(([id, label, purpose], index) => {
          const state = job.stages?.[id];
          const availability = stageAvailability(id, Object.fromEntries(Object.entries(job.stages ?? {}).map(([name, stage]) => [name, stage.status])), preparationReady && !schemeDirty, running, publishInPipeline);
          const publishButton = id === "publish"
            ? publishStageButton({
                publishEnabled: publishInPipeline,
                renderStatus: job.stages?.render?.status,
                publishStatus: state?.status,
              })
            : null;
          const publishRun = publishButton?.action === "run";
          const enabled = publishButton ? publishButton.enabled && (!publishRun || (preparationReady && !schemeDirty && !running)) : availability.enabled;
          const disabledReason = schemeDirty && (state?.status !== "completed") ? "先保存修改并重新检查准备" : publishButton && !publishButton.enabled ? publishButton.reason : publishRun && !preparationReady ? "先通过准备检查" : publishRun && running ? "另一个阶段正在运行" : publishButton ? publishButton.reason : availability.reason;
          const buttonLabel = state?.status === "running"
            ? "取消当前阶段"
            : publishButton
              ? publishButton.label
              : state?.status === "completed"
                ? id === "extract"
                  ? "重新提取字幕"
                  : id === "translate"
                    ? "重新翻译"
                    : id === "render"
                      ? "重新渲染"
                      : id === "dub"
                        ? "重新配音"
                        : "打开结果"
                : ["failed", "cancelled", "stale", "interrupted"].includes(state?.status ?? "")
                  ? "重试"
                  : "开始";
          const onStageClick = () => {
            if (state?.status === "running") {
              void cancel();
              return;
            }
            if (!enabled) {
              if (disabledReason) setMessage(disabledReason);
              return;
            }
            // Completed extract/translate/render: re-run that stage (burn-in needs
            // re-render after external subtitle updates).
            if (
              state?.status === "completed" &&
              (id === "extract" || id === "translate" || id === "render" || id === "dub")
            ) {
              void runStage(id);
              return;
            }
            if (publishButton?.action === "open" || (id !== "publish" && state?.status === "completed")) {
              openStageResult(id);
              return;
            }
            if (id === "publish" && ["interrupted", "failed", "cancelled"].includes(state?.status ?? "") &&
                !window.confirm("投稿辅助可能已经上传视频。请先在 B 站创作中心核对草稿，确认不会重复投稿后再重试。")) return;
            void runStage(id);
          };
          const artifactKey = { acquire: "raw_video", extract: "source_srt", translate: "translated_srt", render: "rendered_video", dub: "dub_video", publish: "" }[id];
          const artifact = artifactKey ? job.artifacts?.[artifactKey] : null;
          const stageIcon: NavIconName = id === "acquire" ? "download" : id === "extract" ? "voice" : id === "translate" ? "translate" : id === "render" ? "render" : id === "dub" ? "voice" : "upload";
          const statusIcon: NavIconName = state?.status === "completed" ? "check" : state?.status === "running" ? "running" : ["failed", "stale", "interrupted"].includes(state?.status ?? "") ? "error" : state?.status === "cancelled" ? "pause" : stageIcon;
          const elapsed = state?.started_at ? formatElapsed(Math.max(0, (state.finished_at ?? now / 1000) - state.started_at)) : null;
          return (
            <article key={id} className={`stage-card ${state?.status ?? "pending"}`}>
              <div className="stage-index">{index + 1}</div>
              <div className="stage-copy"><div><b><NavIcon name={stageIcon} className="app-icon" />{label}</b><span className={`stage-status ${state?.status ?? "pending"}`}><NavIcon name={statusIcon} className="app-icon" />{state?.status === "running" && <ActivityBars />}{stageStatusLabel[state?.status ?? "pending"]}</span></div><p>{purpose}</p>{(state?.status === "running" || state?.status === "completed") && <small className="stage-metrics">{state.status === "running" ? `${Math.round(state.progress)}% · ` : ""}{elapsed ?? ""}{artifact?.available && artifact.name ? ` · ${artifact.name}` : ""}</small>}{state?.error && <TaskErrorSummary message={state.error} />}</div>
              <div className="stage-action">
                {state?.status === "completed" && id !== "publish" ? <span className="checkpoint">检查点已保存</span> : null}
                <button
                  onClick={onStageClick}
                  disabled={false}
                  aria-disabled={state?.status === "running" ? false : !enabled}
                  title={!enabled && disabledReason ? disabledReason : undefined}
                >
                  <NavIcon name={state?.status === "running" ? "pause" : state?.status === "completed" ? (id === "render" ? "rerender" : "check") : stageIcon} className="app-icon" />{buttonLabel}
                </button>
                {!["running", "completed"].includes(state?.status ?? "") && !enabled && disabledReason && <small>{disabledReason}</small>}
              </div>
            </article>
          );
        })}
      </section>
      {job.can_resume && <div className="resume-banner"><div><b>检测到上次运行被中断</b><span>已完成阶段不会重做；翻译会复用已保存批次，其他未完成阶段从头执行。</span></div><button onClick={() => void resumeJob()} disabled={!preparationReady || schemeDirty || running}>从 {workflowStageMeta.find(([id]) => id === job.next_stage)?.[1] ?? "检查点"}继续</button></div>}
    </section>
  ) : null;
  const localTranslationPanel = options.translator !== "ollama" ? null : (
    <section className="local-translation-panel" aria-label="本地翻译模型状态">
      <div>
        <b>本地翻译 · {localModelTag}</b>
        <p>
          {ollamaRuntimeDependency?.state === "ready"
            ? !options.translate_model
              ? "先选一个模型，或填写本机已经有的模型名。不会一次下载全部模型。"
            : !localModelDependencyId
              ? `使用本机已有的 ${localModelTag}。这里不会再下载其他模型。`
            : selectedLocalModelDependency?.available
              ? "这个模型已在本机。翻译内容只发给 Ollama，不需要 API Key，也不会下载其他模型。"
              : selectedLocalModelDependency?.installable
                ? "只差这一个模型。点下载只会拉取当前选中的模型。"
                : selectedLocalModelDependency?.state === "blocked"
                  ? selectedLocalModelDependency.message
                  : "正在检查模型状态…"
            : ollamaRuntimeDependency?.state === "stopped"
              ? "Ollama 已安装但未运行。点击“启动 Ollama”后，应用会自动重新检测。"
              : ollamaRuntimeDependency?.state === "unavailable"
                ? ollamaRuntimeDependency.message
              : ollamaRuntimeDependency?.state === "missing"
                ? "这台电脑还没有 Ollama。先安装并启动 Ollama，再回到这里下载翻译模型。"
                : "正在检查 Ollama；此功能需要单独安装本地运行环境和模型，不会打进 EXE。"}
        </p>
        {localModelTag === "qwen3.5:9b" && <small>9B 模型文件约 6.6 GB，运行时会额外占用内存。</small>}
        {localModelTag === "translategemma:12b" && <small>专用多语言翻译模型，模型文件约 8.1 GB，运行时会占用更多显存和内存；当前逐条字幕翻译，不执行通用模型语义切句与源文校对。</small>}
      </div>
      <div className="local-translation-actions">
        {ollamaRuntimeDependency?.state === "missing" && ollamaRuntimeDependency.action_url && (
          <a className="secondary small" href={ollamaRuntimeDependency.action_url} target="_blank" rel="noreferrer">下载 Ollama</a>
        )}
        {ollamaRuntimeDependency?.state === "stopped" && (
          <button className="primary small" disabled={ollamaStartBusy} onClick={() => void startOllama()}>
            {ollamaStartBusy ? "启动中…" : "启动 Ollama"}
          </button>
        )}
        {ollamaRuntimeDependency?.state === "ready" && !selectedLocalModelDependency?.available && selectedLocalModelDependency?.installable && (
          <button className="primary small" disabled={Boolean(dependencyJob && ["queued", "running", "cancelling"].includes(dependencyJob.status))} onClick={() => void installDependency(selectedLocalModelDependency)}>
            {dependencyJob?.dependency_id === selectedLocalModelDependency.id && ["queued", "running"].includes(dependencyJob.status) ? "下载中…" : "下载所选模型"}
          </button>
        )}
        <button className="small" onClick={() => setDependenciesOpen(true)}>查看依赖状态</button>
      </div>
    </section>
  );
  const processView = (
    <>
    <section className="workspace-card process-view">
      <header className="process-header">
        <p className="eyebrow">处理配置</p>
        <h1>字幕、翻译与推理</h1>
        <p>先选择字幕来源、Whisper 模型和翻译方式；声道、精度与 OCR 细节可在高级设置的分类中展开。</p>
        <button className="process-mode-toggle" onClick={toggleMode}>{mode === "basic" ? "高级设置 ›" : "收起高级设置 ‹"}</button>
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
      {localTranslationPanel}
      {mode === "basic" ? (
        <div className="settings-grid basic-settings-grid">
          <section className="settings-section basic-primary-section"><div className="settings-group-heading"><NavIcon name="voice" className="app-icon" /><span><b>字幕提取</b><small>字幕来源与语音识别方式</small></span></div>
          <label className="settings-span-2">
            字幕来源
            <select
              value={options.subtitle_source}
              onChange={(event) =>
                updateOptions("subtitle_source", event.target.value)
              }
            >
              <option value="auto">自动：平台/语音优先，空结果再尝试 OCR</option>
              <option value="merged">音频 + OCR 合并</option>
              <option value="audio">仅音频转写</option>
              <option value="ocr">仅画面字幕 OCR</option>
            </select>
            <FieldGuide summary="自动模式优先读取平台字幕/语音；只有没有可用字幕时才尝试 OCR。" detail="仅音频不会读取画面文字；仅 OCR 不会调用 Whisper，OCR 失败会明确报错。合并模式更慢且需要 Tesseract。" badges={options.subtitle_source === "audio" ? ["推荐"] : options.subtitle_source === "ocr" || options.subtitle_source === "merged" ? ["需要额外安装", "高负载"] : []} />
          </label>
          <label>
            Whisper 模型
            <select value={options.whisper_model_size} disabled={options.subtitle_source === "ocr"} onChange={(event) => updateOptions("whisper_model_size", event.target.value)}>
              {["tiny", "base", "small", "medium", "large-v3"].map((value) => <option key={value}>{value}</option>)}
            </select>
            <FieldGuide summary={options.subtitle_source === "ocr" ? "纯 OCR 不使用 Whisper 模型。" : "模型越大通常越准确，也越慢、越占显存和磁盘。"} detail="所选模型必须在准备中心明确安装，转写阶段不会自动联网下载。" badges={options.subtitle_source === "ocr" ? [] : ["需要预先安装"]} />
          </label>
          <label>
            源语言
            <select value={options.source_language} onChange={(event) => updateOptions("source_language", event.target.value)}>
              <option value="">自动检测</option>
              {sourceLanguageOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <AsrRecognitionFields options={options} hintError={fieldErrors.asr_hint} onHint={(value) => updateOptions("asr_hint", value)} onVad={(value) => updateOptions("asr_vad_mode", value)} />
           {(job?.material.source_url || (!job && sourceUrl.trim())) && ["auto", "merged"].includes(options.subtitle_source) && <label className="check settings-check">
             <input type="checkbox" checked={options.prefer_platform_subtitles} onChange={(event) => updateOptions("prefer_platform_subtitles", event.target.checked)} />
             优先使用 YouTube 字幕（更快；没有时自动转写）
           </label>}
          </section>
          <section className="settings-section basic-primary-section"><div className="settings-group-heading"><NavIcon name="translate" className="app-icon" /><span><b>翻译服务与模型</b><small>在线服务、本地模型与目标语言</small></span></div>
          <label>
            翻译器
            <select
              value={options.translator}
              onChange={(event) => changeTranslator(event.target.value)}
            >
              <option value="deepseek">DeepSeek（在线）</option>
              <option value="openai">OpenAI 兼容接口（含 Kimi、智谱、通义、本机服务）</option>
              <option value="ollama">Ollama（本机）</option>
              <option value="none">不翻译</option>
            </select>
            <FieldGuide summary="把源字幕翻译为目标语言。" detail={options.translator === "ollama" ? "Ollama 在本机运行，不需要 API Key。推荐模型里选一个，或填写本机已经有的模型名；不会把几个模型一起下载。" : options.translator === "openai" ? "这一项走 OpenAI 兼容接口。具体厂商、本机地址、模型名和密钥在接口设置里选择，并按那里的配置要求填写。" : "DeepSeek 会调用在线 API，可能产生费用。其他厂商和本机服务在接口设置里选择。不翻译只适合保留原文或调试提取结果。"} badges={options.translator === "none" ? [] : options.translator === "ollama" ? ["本机运行", "只下载所选"] : ["可能产生费用"]} />
          </label>
           <label>
             翻译模型
             {options.translator === "ollama" ? (
               <select value={["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"].includes(options.translate_model) ? options.translate_model : "__custom__"} onChange={(event) => updateOptions("translate_model", event.target.value === "__custom__" ? "" : event.target.value)}>
                 <option value="qwen3.5:4b">Qwen 3.5 4B · 约 3.4 GB（选这一个即可）</option>
                 <option value="qwen3.5:9b">Qwen 3.5 9B · 约 6.6 GB（选这一个即可）</option>
                 <option value="translategemma:12b">TranslateGemma 12B · 约 8.1 GB（选这一个即可）</option>
                 <option value="__custom__">本机已经有的其他模型</option>
               </select>
             ) : (
               <input value={options.translate_model} disabled={options.translator === "none"} maxLength={128} placeholder="留空使用服务默认模型" onChange={(event) => updateOptions("translate_model", event.target.value)} />
             )}
             {fieldErrors.translate_model && <small className="field-error">{fieldErrors.translate_model}</small>}
             {options.translator === "ollama" && !["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"].includes(options.translate_model) && (
               <input value={options.translate_model} maxLength={128} placeholder="例如 llama3.1:8b，只填这一个" onChange={(event) => updateOptions("translate_model", event.target.value)} />
             )}
             {options.translator === "ollama" && <FieldGuide summary="只使用你选中的那一个模型。" detail={localModelTag === "translategemma:12b" ? "TranslateGemma 专门用于多语言翻译。选它不会把另外两个模型一起下载。" : "推荐项里选一个，或填写本机 Ollama 已经有的名字。点下载时只拉取当前这一个。"} badges={["不会一次下载全部", "已有模型可直接填"]} />}
           </label>
           <label>
             目标语言
             <input value={options.target_lang} disabled={options.translator === "none"} maxLength={32} placeholder="例如 zh-Hans（简体中文）" onChange={(event) => updateOptions("target_lang", event.target.value)} />
             {fieldErrors.target_lang && <small className="field-error">{fieldErrors.target_lang}</small>}
           </label>
          {options.translator === "deepseek" && <DeepseekThinkingChoice mode={options.deepseek_thinking} model={options.translate_model} onChange={(mode) => updateOptions("deepseek_thinking", mode)} />}
           <label className="check settings-check">
             <input type="checkbox" checked={options.smart_translation} onChange={(event) => updateOptions("smart_translation", event.target.checked)} />
             智能上下文翻译
           </label>
           <div className="settings-entry-row">
             <button type="button" className="settings-inline-link" onClick={() => { setMode("advanced"); window.setTimeout(() => { document.getElementById("advanced-translation-settings")?.scrollIntoView({ behavior: "smooth", block: "start" }); }, 0); }}>模型与费用设置 →</button>
            {(["deepseek", "openai"].includes(options.translator)) && <button type="button" className="settings-link-button" onClick={() => setSettingsOpen(true)}><NavIcon name="link" className="app-icon" />打开接口设置与 API 密钥</button>}
            <button type="button" className="settings-link-button" onClick={() => void openGlossary()}><NavIcon name="book" className="app-icon" />术语表{glossaryCount === null ? "" : `（${glossaryCount} 条）`} · 编辑</button>
           </div>
          </section>
          <section className="settings-section"><div className="settings-group-heading"><NavIcon name="download" className="app-icon" /><span><b>性能与转写</b><small>资源占用与处理负载</small></span></div>
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
          </section>
          <section className="settings-section"><div className="settings-group-heading"><NavIcon name="text" className="app-icon" /><span><b>字幕样式</b><small>排版、字体、字号与颜色</small></span></div>
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
            双语样式
            <select
              value={options.subtitle_preset}
              disabled={options.subtitle_display_mode === "translated"}
              onChange={(event) => updateOptions("subtitle_preset", event.target.value)}
            >
              <option value="soft">弱化原文（小一号、浅灰）</option>
              <option value="compact">小字原文（更小、更灰）</option>
              <option value="equal">两行一致</option>
            </select>
          </label>
          <label>
            字体
            <input
              value={options.font_name}
              maxLength={100}
              onChange={(event) => updateOptions("font_name", event.target.value)}
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
              onChange={(event) => updateOptions("font_size", Number(event.target.value || 24))}
            />
          </label>
          <label>
            字幕颜色
            <select
              value={options.subtitle_color}
              onChange={(event) => updateOptions("subtitle_color", event.target.value)}
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
              onChange={(event) => updateOptions("subtitle_outline_color", event.target.value)}
            >
              {["黑色", "白色", "灰色", "蓝色"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            字幕效果
            <select
              value={options.subtitle_effect}
              onChange={(event) => updateOptions("subtitle_effect", event.target.value)}
            >
              {["描边", "阴影", "描边+阴影", "无"].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
            <FieldGuide summary="控制硬字幕的描边与阴影。" detail="改完样式后需重新渲染成片才会生效；已完成的渲染检查点会提示重做。" />
          </label>
           <label className="check settings-check">
             <input type="checkbox" checked={options.smart_subtitle_layout} onChange={(event) => updateOptions("smart_subtitle_layout", event.target.checked)} />
             智能字幕换行排版
           </label>
          </section>
          <section className="settings-section"><div className="settings-group-heading"><NavIcon name="render" className="app-icon" /><span><b>输出与渲染</b><small>渲染编码器与输出目录</small></span></div>
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
           <div className="settings-group-summary"><span>当前输出目录：<b title={options.output_dir}>{options.output_dir}</b></span><button type="button" className="settings-inline-link" onClick={() => { setMode("advanced"); window.setTimeout(() => { document.getElementById("advanced-output-settings")?.scrollIntoView({ behavior: "smooth", block: "start" }); }, 0); }}>输出目录与编码设置 →</button></div>
          </section>
        </div>
      ) : (
      <div className="settings-grid">
        <section className="settings-section"><header className="settings-group-heading"><NavIcon name="voice" className="app-icon" /><span><b>字幕提取</b><small>来源、Whisper、语言与声道</small></span></header><div className="settings-grid settings-fold-body">
        <label>
          字幕来源
          <select
            value={options.subtitle_source}
            onChange={(event) =>
              updateOptions("subtitle_source", event.target.value)
            }
          >
            <option value="auto">自动：平台/语音优先，空结果再尝试 OCR</option>
            <option value="merged">音频 + OCR 合并</option>
            <option value="audio">仅音频转写</option>
            <option value="ocr">仅画面字幕 OCR</option>
          </select>
        </label>
        <label>
          Whisper 模型
          <select
            value={options.whisper_model_size}
            disabled={options.subtitle_source === "ocr"}
            onChange={(event) =>
              updateOptions("whisper_model_size", event.target.value)
            }
          >
            {["tiny", "base", "small", "medium", "large-v3"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
          <FieldGuide summary={options.subtitle_source === "ocr" ? "纯 OCR 不使用 Whisper 模型。" : "模型越大通常更准确，也越慢、越占显存和磁盘。"} detail="small 是桌面使用的平衡选择。所选模型必须在准备中心明确安装，转写阶段不会自动联网下载。" badges={options.subtitle_source === "ocr" ? [] : ["需要预先安装"]} />
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
            {sourceLanguageOptions.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label>
          语音声道
          <select
            value={options.audio_channel}
            disabled={options.subtitle_source === "ocr"}
            onChange={(event) => updateOptions("audio_channel", event.target.value as Options["audio_channel"])}
          >
            <option value="mix">混合（默认）</option>
            <option value="left">左声道（仅双声道）</option>
            <option value="right">右声道（仅双声道）</option>
          </select>
          <FieldGuide summary="默认保留混合声道；左右声道只适用于明确的双声道音频。" detail="系统不会根据音量自动判断哪一侧有人声，也不会静默丢弃另一声道。" />
          </label>
          <AsrRecognitionFields options={options} hintError={fieldErrors.asr_hint} onHint={(value) => updateOptions("asr_hint", value)} onVad={(value) => updateOptions("asr_vad_mode", value)} />
         {(job?.material.source_url || (!job && sourceUrl.trim())) && ["auto", "merged"].includes(options.subtitle_source) && <label className="check settings-check">
           <input type="checkbox" checked={options.prefer_platform_subtitles} onChange={(event) => updateOptions("prefer_platform_subtitles", event.target.checked)} />
           优先使用 YouTube 字幕（更快；没有时自动转写）
         </label>}
        <details className="settings-subfold">
          <summary><span><b>OCR 细节</b><small>画面字幕的采样、语言与识别区域</small></span><i>{options.subtitle_source === "audio" ? "当前未启用" : "展开/收起"}</i></summary>
          <div className="settings-subfold-body">
            <p className="settings-subfold-note">{options.subtitle_source === "audio" ? "当前来源为仅音频，OCR 参数暂不参与；切换到自动、合并或仅 OCR 后会继续使用已保存的值。" : "只在来源可能读取画面字幕时生效；参数保留在本机任务设置中。"}</p>
            <div className="settings-grid settings-subfold-grid">
              <label>
                OCR 帧间隔（秒）
                <select disabled={options.subtitle_source === "audio"} value={options.ocr_interval} onChange={(event) => updateOptions("ocr_interval", Number(event.target.value))}>
                  <option value={0.25}>0.25（短字幕，最慢）</option><option value={0.5}>0.5（推荐）</option><option value={1}>1（更快）</option><option value={2}>2（更快）</option>
                </select>
                <FieldGuide summary="决定多久读取一次画面，默认 0.5 秒可覆盖多数短字幕。" detail="字幕一闪而过时使用 0.25 秒；普通字幕推荐 0.5 秒；1～2 秒会更快，但可能漏掉短字幕。" badges={options.ocr_interval <= 0.25 ? ["高负载"] : options.ocr_interval === 0.5 ? ["推荐"] : ["可能漏短字幕"]} />
              </label>
              <label>
                OCR 语言
                <select disabled={options.subtitle_source === "audio"} value={options.ocr_language} onChange={(event) => updateOptions("ocr_language", event.target.value)}>
                  <option value="eng">英文（eng）</option><option value="chi_sim">简体中文（chi_sim）</option><option value="eng+chi_sim">英文 + 简体中文</option><option value="jpn">日文（jpn）</option><option value="kor">韩文（kor）</option>
                </select>
                <FieldGuide summary="必须和画面中已有字幕的语言一致。" detail="准备检查会在下载前确认对应 Tesseract 语言包是否安装；缺少语言包时不会开始下载。" badges={["下载前检查"]} />
              </label>
              <label>
                OCR 识别区域
                <select disabled={options.subtitle_source === "audio"} value={options.ocr_crop_ratio} onChange={(event) => updateOptions("ocr_crop_ratio", Number(event.target.value))}>
                  <option value={0.2}>仅底部 20%</option><option value={0.3}>底部 30%（推荐）</option><option value={0.5}>下半屏</option><option value={1}>全画面</option>
                </select>
              </label>
              <label>
                OCR 最小字符数
                <input disabled={options.subtitle_source === "audio"} type="number" min="1" max="20" value={options.ocr_min_chars} onChange={(event) => updateOptions("ocr_min_chars", Number(event.target.value || 3))} />
              </label>
            </div>
          </div>
        </details>
        </div></section>
        <section className="settings-section"><header className="settings-group-heading" id="advanced-translation-settings"><NavIcon name="translate" className="app-icon" /><span><b>翻译服务与模型</b><small>服务、模型、目标语言与请求计价</small></span></header><div className="settings-grid settings-fold-body">
        <label>
          翻译器
          <select
            value={options.translator}
            onChange={(event) => changeTranslator(event.target.value)}
          >
            <option value="deepseek">DeepSeek（在线）</option>
            <option value="openai">OpenAI 兼容接口（含 Kimi、智谱、通义、本机服务）</option>
            <option value="ollama">Ollama（本机）</option>
            <option value="none">不翻译</option>
          </select>
        </label>
        <label>
          翻译模型
          {options.translator === "ollama" ? (
            <select value={["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"].includes(options.translate_model) ? options.translate_model : "__custom__"} onChange={(event) => updateOptions("translate_model", event.target.value === "__custom__" ? "" : event.target.value)}>
              <option value="qwen3.5:4b">Qwen 3.5 4B · 约 3.4 GB（选这一个即可）</option>
              <option value="qwen3.5:9b">Qwen 3.5 9B · 约 6.6 GB（选这一个即可）</option>
              <option value="translategemma:12b">TranslateGemma 12B · 约 8.1 GB（选这一个即可）</option>
              <option value="__custom__">本机已经有的其他模型</option>
            </select>
          ) : (
            <input
              value={options.translate_model}
              disabled={options.translator === "none"}
              maxLength={128}
              placeholder="使用默认模型"
              onChange={(event) => updateOptions("translate_model", event.target.value)}
            />
          )}
          {options.translator === "ollama" && !["qwen3.5:4b", "qwen3.5:9b", "translategemma:12b"].includes(options.translate_model) && (
            <input value={options.translate_model} maxLength={128} placeholder="例如 llama3.1:8b，只填这一个" onChange={(event) => updateOptions("translate_model", event.target.value)} />
          )}
          {fieldErrors.translate_model && <small className="field-error">{fieldErrors.translate_model}</small>}
        </label>
        {options.translator === "deepseek" && (
          <DeepseekThinkingChoice mode={options.deepseek_thinking} model={options.translate_model} onChange={(mode) => updateOptions("deepseek_thinking", mode)} />
        )}
        <label>
          目标语言
          <input value={options.target_lang} disabled={options.translator === "none"} maxLength={32} placeholder="例如 zh-Hans（简体中文）" onChange={(event) => updateOptions("target_lang", event.target.value)} />
          {fieldErrors.target_lang && <small className="field-error">{fieldErrors.target_lang}</small>}
        </label>
        <label className="check settings-check">
          <input type="checkbox" checked={options.smart_translation} onChange={(event) => updateOptions("smart_translation", event.target.checked)} />
          智能上下文翻译
        </label>
        {["deepseek", "openai"].includes(options.translator) && <LlmPricingChoice options={options} onChange={(key, value) => {
          if (key === "llm_pricing_mode") updateOptions(key, value as Options["llm_pricing_mode"]);
          else updateOptions(key, value as Record<string, string>);
        }} />}
        <button type="button" className="settings-link-button" onClick={() => void openGlossary()}><NavIcon name="book" className="app-icon" />术语表{glossaryCount === null ? "" : `（${glossaryCount} 条）`} · 编辑</button>
        {["deepseek", "openai"].includes(options.translator) && <button type="button" className="settings-link-button" onClick={() => setSettingsOpen(true)}><NavIcon name="link" className="app-icon" />打开接口设置与 API 密钥</button>}
        </div></section>
        <section className="settings-section"><header className="settings-group-heading"><NavIcon name="download" className="app-icon" /><span><b>性能与转写</b><small>资源、设备、精度与转写速度</small></span></header><div className="settings-grid settings-fold-body">
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
            disabled={options.subtitle_source === "ocr"}
            onChange={(event) => updateOptions("beam_size", Number(event.target.value))}
          >
            <option value={1}>1（快）</option>
            <option value={5}>5（推荐）</option>
            <option value={10}>10（更准更慢）</option>
          </select>
        </label>
        </div></section>
        <section className="settings-section"><header className="settings-group-heading"><NavIcon name="text" className="app-icon" /><span><b>字幕样式</b><small>排版、字体、字号、颜色与位置</small></span></header><div className="settings-grid settings-fold-body">
        <label>
          字幕底部位置
          <select value={options.subtitle_margin_ratio} onChange={(event) => updateOptions("subtitle_margin_ratio", Number(event.target.value))}>
            <option value={0.03}>3%（贴近底部）</option>
            <option value={0.055}>5.5%（推荐）</option>
            <option value={0.1}>10%（偏上）</option>
            <option value={0.15}>15%（更高）</option>
          </select>
        </label>
        <label>
          字幕排版
          <select
            value={options.subtitle_display_mode}
            onChange={(event) => updateOptions("subtitle_display_mode", event.target.value)}
          >
            <option value="translated">中文单语</option>
            <option value="bilingual-source-first">原文在上 + 中文在下</option>
            <option value="bilingual-translation-first">中文在上 + 原文在下</option>
          </select>
        </label>
        <label>
          双语样式
          <select
            value={options.subtitle_preset}
            disabled={options.subtitle_display_mode === "translated"}
            onChange={(event) => updateOptions("subtitle_preset", event.target.value)}
          >
            <option value="soft">弱化原文（小一号、浅灰）</option>
            <option value="compact">小字原文（更小、更灰）</option>
            <option value="equal">两行一致</option>
          </select>
        </label>
        <label>
          字体
          <input
            value={options.font_name}
            maxLength={100}
            onChange={(event) => updateOptions("font_name", event.target.value)}
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
            onChange={(event) => updateOptions("font_size", Number(event.target.value || 24))}
          />
        </label>
        <label>
          字幕颜色
          <select
            value={options.subtitle_color}
            onChange={(event) => updateOptions("subtitle_color", event.target.value)}
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
            onChange={(event) => updateOptions("subtitle_outline_color", event.target.value)}
          >
            {["黑色", "白色", "灰色", "蓝色"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          字幕效果
          <select
            value={options.subtitle_effect}
            onChange={(event) => updateOptions("subtitle_effect", event.target.value)}
          >
            {["描边", "阴影", "描边+阴影", "无"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
         <label className="check settings-check">
           <input type="checkbox" checked={options.smart_subtitle_layout} onChange={(event) => updateOptions("smart_subtitle_layout", event.target.checked)} />
           智能字幕换行排版
         </label>
        </div></section>
        <section className="settings-section" id="advanced-output-settings"><header className="settings-group-heading"><NavIcon name="render" className="app-icon" /><span><b>输出与渲染</b><small>渲染质量、编码器和输出目录</small></span></header><div className="settings-grid settings-fold-body">
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
              onChange={(event) => { updateOptions("output_dir", event.target.value); setOutputs(null); setSelectedOutputPaths([]); setViewedTaskId(null); }}
            />
            <button onClick={() => void chooseOutputDir()} disabled={running}>
              选择文件夹
            </button>
          </div>
        </label>
        </div></section>
      </div>
      )}
      {dubSettings}
      <section className="settings-section post-process-section">
        <header className="settings-group-heading"><NavIcon name="upload" className="app-icon" /><span><b>处理后操作</b><small>渲染完成后打开投稿辅助</small></span></header>
        <div className="settings-grid settings-fold-body compact publish-pipeline-settings">
          <label className="check settings-check"><input type="checkbox" checked={publishInPipeline} onChange={(event) => setPublishInPipeline(event.target.checked)} />渲染后打开 B 站创作中心并辅助上传填表</label>
          {publishInPipeline && <>
            <label>投稿浏览器<select value={publishBrowser} onChange={(event) => setPublishBrowser(event.target.value)}><option value="chromium">Chromium</option><option value="msedge">Microsoft Edge</option></select></label>
            <label className="check settings-check"><input type="checkbox" checked={includeSourceLink} onChange={(event) => setIncludeSourceLink(event.target.checked)} />简介自动附原视频链接</label>
            <label className="check settings-check"><input type="checkbox" checked={closeAfterFill} onChange={(event) => setCloseAfterFill(event.target.checked)} />填表完成后关闭浏览器（可能中断上传）</label>
          </>}
        </div>
      </section>
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
          disabled={running || preparingJob || (!job && !materialSourceReady)}
        >
          {running ? "当前阶段运行中" : job ? "保存方案并检查准备" : "保存素材并配置方案"}
        </button>
      </footer>
    </section>
    {workflowPanel}
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
                : preview === "dub"
                ? "中文配音版"
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
              onClick={() => {
                setMediaNonce(Date.now());
                setPreview("rendered");
              }}
            >
              成片
            </button>
            {job?.artifacts?.dub_video?.available && (
              <>
                <button className={preview === "dub" ? "selected" : ""} onClick={() => { setMediaNonce(Date.now()); setPreview("dub"); }}>配音版</button>
                <button onClick={() => void openHistoryPath(job.result?.dub_video || "")}>打开配音版</button>
              </>
            )}
          </div>
        </div>
        <video
          ref={video}
          className="real-video"
          src={previewSrc}
          controls
          preload="metadata"
          onLoadStart={() => {
            subtitlePlaybackEnd.current = null;
            setMediaDuration(null);
            if (timelinePlayhead.current) timelinePlayhead.current.style.left = "0%";
            if (timelineCurrentTime.current) timelineCurrentTime.current.textContent = "00:00";
          }}
          onLoadedMetadata={(event) => {
            subtitlePlaybackEnd.current = null;
            trimPlaybackEnd.current = null;
            const actual = event.currentTarget.duration;
            setMediaDuration(Number.isFinite(actual) && actual > 0 ? actual : null);
            const pending = pendingPlayback.current;
            if (pending && job && pending.jobId === job.id && pending.generation === cueGeneration.current) {
              pendingPlayback.current = null;
              event.currentTarget.currentTime = pending.start;
              trimPlaybackEnd.current = pending.end;
              void event.currentTarget.play().catch(() => setMessage("无法播放此区间，请先点击视频播放一次。"));
            }
          }}
          onTimeUpdate={(event) => {
            const boundedEnd = subtitlePlaybackEnd.current;
            if (boundedEnd !== null && event.currentTarget.currentTime >= boundedEnd) {
              subtitlePlaybackEnd.current = null;
              event.currentTarget.pause();
              event.currentTarget.currentTime = boundedEnd;
            }
            const trimEnd = trimPlaybackEnd.current;
            if (trimEnd !== null && event.currentTarget.currentTime >= trimEnd) {
              trimPlaybackEnd.current = null;
              event.currentTarget.pause();
              event.currentTarget.currentTime = trimEnd;
            }
            if (timelinePlayhead.current) {
              timelinePlayhead.current.style.left = `${Math.min(100, Math.max(0, (event.currentTarget.currentTime / (event.currentTarget.duration || duration)) * 100))}%`;
            }
            if (timelineCurrentTime.current) timelineCurrentTime.current.textContent = timecode(event.currentTarget.currentTime);
            const cue = cues.find(
              (item) =>
                event.currentTarget.currentTime >= item.start &&
                event.currentTarget.currentTime <= item.end,
            );
            setActiveCue(cue?.id ?? null);
          }}
        />
        <div className="timeline-wrap">
          <div className="time-ruler" aria-label="视频时间刻度">
            {previewSrc ? [0, 0.25, 0.5, 0.75, 1].map((position) => (
              <span key={position} className={`ruler-tick ${position === 0 ? "first" : position === 1 ? "last" : ""}`} style={{ left: `${position * 100}%` }}>
                {timecode(duration * position, duration < 40)}
              </span>
            )) : <span className="ruler-placeholder">等待视频素材</span>}
          </div>
          <div className={`timeline ${cuePositions.length ? "has-cues" : "is-empty"}`}>
            <span className="timeline-track-name">字幕轨 <small>{cues.length ? `${cues.length} 段` : ""}</small></span>
            {previewSrc && <span className="timeline-current"><span ref={timelineCurrentTime}>00:00</span> / {timecode(duration)}</span>}
            {previewSrc && <div className="timeline-playhead" ref={timelinePlayhead} aria-hidden="true" />}
            {previewSrc && duration > 0 && cutRanges.map(([start, end], index) => (
              <span
                key={`trim-range-${index}`}
                className={`trim-range-mark trim-range-mark-${modOp}`}
                style={{ left: `${Math.max(0, start / duration) * 100}%`, width: `${Math.max(0, Math.min(duration, end) - Math.max(0, start)) / duration * 100}%` }}
                title={`${modOp === "cut" ? "待删除" : modOp === "keep" ? "待保留" : "待静音"}：${formatTime(start)}–${formatTime(end)}`}
                aria-label={`已选区间 ${formatTime(start)} 至 ${formatTime(end)}`}
              />
            ))}
            {cuePositions.length ? (
              cuePositions.map(({ cue, left, width }) => (
                <button
                  key={cue.id}
                  className={["cue-mark", selectedCue === cue.id ? "selected" : "", activeCue === cue.id ? "current" : ""].join(" ")}
                  style={{ left, width }}
                  onClick={() => seek(cue)}
                  title={`${formatTime(cue.start)}–${formatTime(cue.end)}：${cue.translated || cue.source}`}
                  aria-label={`跳转到第 ${cue.id + 1} 段字幕，${formatTime(cue.start)}`}
                  aria-pressed={selectedCue === cue.id}
                />
              ))
            ) : (
              <span className="timeline-empty">{previewSrc ? "字幕提取完成后，段落会显示在这条轨道上。" : "选择视频后，这里会显示真实字幕轨道。"}</span>
            )}
          </div>
        </div>
      </section>
      <details className="workspace-card trim-card">
        <summary className="studio-drawer-summary"><span><b>剪辑视频</b><small>选时间段，再选择删除、保留或静音</small></span><span>展开</span></summary>
        <p className="studio-drawer-note">播放到片段起止位置，点击“设为开始 / 结束”；添加后可在上方时间轴确认并单独试听。</p>
        <div className="trim-action-title">1. 选择要做的操作</div>
        <div className="mod-op-row">
          {(
            [
              ["cut", "删除选中片段", "从成片中移除"],
              ["keep", "只保留选中片段", "其余部分会移除"],
              ["mute", "将选中片段静音", "画面保留，声音静音"],
            ] as const
          ).map(([value, label, detail]) => (
            <label key={value} className={`check trim-action ${modOp === value ? "active" : ""}`}>
              <input
                type="radio"
                name="mod-op"
                checked={modOp === value}
                onChange={() => setModOp(value)}
                disabled={running || !job?.result}
              />
              <span><b>{label}</b><small>{detail}</small></span>
            </label>
          ))}
        </div>
        <div className="trim-action-title">2. 选取时间段</div>
        <div className="trim-controls trim-range-controls">
          <div className="trim-time-field"><span>开始时间</span>
            <input
              value={cutStart}
              placeholder="例如 00:05.0"
              aria-label="剪辑开始时间"
              disabled={running || !job?.result}
              onChange={(event) => setCutStart(event.target.value)}
            />
            <button onClick={() => setTrimPointFromPlayhead("cutStart")} disabled={running || !job?.result} title="使用当前播放位置作为开始时间">设为开始</button>
          </div>
          <span className="trim-range-separator">至</span>
          <div className="trim-time-field"><span>结束时间</span>
            <input
              value={cutEnd}
              placeholder="例如 00:10.0"
              aria-label="剪辑结束时间"
              disabled={running || !job?.result}
              onChange={(event) => setCutEnd(event.target.value)}
            />
            <button onClick={() => setTrimPointFromPlayhead("cutEnd")} disabled={running || !job?.result} title="使用当前播放位置作为结束时间">设为结束</button>
          </div>
          <button className="trim-add-range" onClick={addCutRange} disabled={running || !job?.result}>
            添加时间段
          </button>
        </div>
        {cutRanges.length > 0 ? (
          <div className="cut-list">
            <b className="trim-selection-label">已选 {cutRanges.length} 段</b>
            {cutRanges.map((range, index) => (
              <span key={index} className={`cut-chip cut-chip-${modOp}`}>
                {formatTime(range[0])} – {formatTime(range[1])}
                <button onClick={() => previewTrimRange(range[0], range[1])} title="试听此时间段">试听</button>
                <button onClick={() => removeCutRange(index)} title="移除此时间段">移除</button>
              </span>
            ))}
          </div>
        ) : (
          <p className="trim-empty-selection">还没有选时间段。可以手动输入，也可以播放视频后用当前播放位置设定。</p>
        )}
        <div className="trim-apply-row">
          <button
            className="primary small"
            onClick={() => void applyTrim()}
            disabled={running || !cutRanges.length}
          >
            {modOp === "cut" ? "应用：删除这些片段" : modOp === "keep" ? "应用：只保留这些片段" : "应用：静音这些片段"}
          </button>
          {cutRanges.length > 0 && <button onClick={() => setCutRanges([])} disabled={running}>清空选择</button>}
        </div>
        <details className="trim-advanced-tools">
          <summary>更多剪辑工具：导出片段、调整顺序</summary>
          <div className="trim-controls export-row">
            <b>导出一段为新视频</b>
            <div className="trim-time-field"><span>开始时间</span>
              <input aria-label="导出片段开始时间" value={exportStart} placeholder="例如 00:10.0" disabled={running || !job?.result} onChange={(event) => setExportStart(event.target.value)} />
              <button onClick={() => setTrimPointFromPlayhead("exportStart")} disabled={running || !job?.result}>设为开始</button>
            </div>
            <div className="trim-time-field"><span>结束时间</span>
              <input aria-label="导出片段结束时间" value={exportEnd} placeholder="例如 00:20.0" disabled={running || !job?.result} onChange={(event) => setExportEnd(event.target.value)} />
              <button onClick={() => setTrimPointFromPlayhead("exportEnd")} disabled={running || !job?.result}>设为结束</button>
            </div>
            <button onClick={() => void exportSegment()} disabled={running || !job?.result}>导出新文件</button>
          </div>
          <div className="trim-controls reorder-row">
            <b>调整片段顺序</b>
            <input value={reorderPoints} placeholder="分割点，如 00:10,00:20" disabled={running || !job?.result} onChange={(event) => setReorderPoints(event.target.value)} />
            <input value={reorderOrder} placeholder="新顺序，如 3,1,2" disabled={running || !job?.result} onChange={(event) => setReorderOrder(event.target.value)} />
            <button onClick={() => void applyReorder()} disabled={running || !job?.result}>应用新顺序</button>
            <small>先按分割点切成几段，再按编号填写播放顺序。</small>
          </div>
        </details>
        <p className="trim-note">应用删除、保留或静音后会生成新的成片并同步字幕时间；导出会另存文件。修改前请先保存字幕。</p>
      </details>
      <details className="workspace-card subtitle-config" open>
        <summary className="studio-drawer-summary"><span><b>字幕样式</b><small>与处理页同步；改完后需重新渲染</small></span><span>展开</span></summary>
        <header className="subtitle-config-header">
          <p className="eyebrow">字幕样式</p>
          <h1>排版与硬字幕成片</h1>
          <p>字体、字号与颜色也可在「处理」页直接设置。</p>
        </header>
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
            双语样式
            <select
              value={options.subtitle_preset}
              disabled={options.subtitle_display_mode === "translated"}
              onChange={(event) => updateOptions("subtitle_preset", event.target.value)}
            >
              <option value="soft">弱化原文（小一号、浅灰）</option>
              <option value="compact">小字原文（更小、更灰）</option>
              <option value="equal">两行一致</option>
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
      </details>
      <section className="voice-review-card workspace-card">
        <header className="voice-review-header">
          <div>
            <p className="eyebrow">语音复核</p>
            <h2>检查字幕未覆盖的说话区间</h2>
            <small>
              {voiceReview?.status === "unavailable"
                ? "完成字幕提取后可准备复核队列。"
                : `剩余 ${voiceReview?.remaining ?? 0} 个区间 · 候选 ${voiceReview?.recovered_words?.length ?? 0} 个词 · 本轮最多 120 秒实际音频`}
            </small>
          </div>
          <button className="secondary small" onClick={() => setVoiceReviewOpen((value) => !value)}>
            {voiceReviewOpen ? "收起" : "展开"}
          </button>
        </header>
        {voiceReviewOpen && <div className="voice-review-body">
          <p className="review-rule-note">队列只表示待核查时间段，不代表这些区间一定有人声；恢复结果仍会标记待复核。</p>
          <div className="voice-review-actions">
            <button onClick={() => void prepareVoiceReview()} disabled={voiceReviewBusy || dirty || running}>
              {voiceReviewBusy ? "处理中…" : "准备区间"}
            </button>
            <button className="primary small" onClick={() => void startVoiceReview()} disabled={voiceReviewBusy || dirty || running || !voiceReview || voiceReview.status === "completed" || voiceReview.status === "unavailable"}>
              {voiceReview?.status === "paused" ? "继续下一轮" : "开始复核"}
            </button>
            <button onClick={() => void cancelVoiceReview()} disabled={voiceReview?.status !== "running"}>暂停</button>
            <span className="review-rule-note">声道：{voiceReview?.channel === "left" ? "左" : voiceReview?.channel === "right" ? "右" : "混合"}</span>
            {voiceReview?.cpu_threads && <span className="review-rule-note">资源：{voiceReview.resource_profile === "background" ? "省资源" : voiceReview.resource_profile === "maximum" ? "性能优先" : "均衡"} · CPU 最多 {voiceReview.cpu_threads} 线程</span>}
          </div>
          {voiceReview?.error && <p className="field-error">{voiceReview.error}</p>}
          <div className="voice-review-list">
            {(voiceReview?.candidates ?? []).slice(0, 24).map((candidate) => (
              <div className="voice-review-row" key={candidate.id}>
                <b>{candidate.id}</b>
                <span>{formatTime(candidate.start)}–{formatTime(candidate.end)}</span>
                <span>{candidate.status === "done" ? `已完成${candidate.accepted_words ? ` · ${candidate.accepted_words} 词` : ""}` : candidate.status === "running" ? "处理中" : "待处理"}</span>
                <button className="small" onClick={() => previewTrimRange(candidate.start, candidate.end)}>重听</button>
              </div>
            ))}
          </div>
          {(voiceReview?.result_candidates ?? []).filter((candidate) => candidate.status !== "committed").map((candidate) => {
            const source = (candidate.words ?? []).map((word) => word.text ?? "").join("").trim();
            return <article className="voice-candidate-row" key={candidate.id}>
              <div><b>{candidate.tier === "weak" ? "弱候选" : "候选"} · {candidate.id}</b><small>{formatTime(candidate.start)}–{formatTime(candidate.end)} · {source || "无可用原文"}</small></div>
              <input value={voiceCandidateText[candidate.id] ?? ""} placeholder="填写中文译文" onChange={(event) => setVoiceCandidateText((items) => ({ ...items, [candidate.id]: event.target.value }))} />
              {candidate.tier === "weak" && <label><input type="checkbox" checked={voiceCandidateConfirmed[candidate.id] === true} onChange={(event) => setVoiceCandidateConfirmed((items) => ({ ...items, [candidate.id]: event.target.checked }))} />我已重听并确认</label>}
              <button className="small" onClick={() => void generateVoiceTranslation(candidate)} disabled={voiceTranslationBusy === candidate.id}>{voiceTranslationBusy === candidate.id ? "翻译中…" : "生成译文候选"}</button>
              <button className="small" onClick={() => adoptVoiceCandidate(candidate)} disabled={candidate.tier === "diagnostic"}>加入字幕草稿</button>
            </article>;
          })}
        </div>}
      </section>
      <section className="subtitle-editor">
        <div className="subtitle-toolbar">
          <div className="subtitle-toolbar-heading">
            <p className="eyebrow">字幕编辑</p>
            <h2>字幕段落 <small>{cues.length} 段</small></h2>
          </div>
          <div className="editor-actions">
            {dirty && <span className="dirty">未保存：先保存再渲染</span>}
            <button onClick={undo} disabled={!history.length || running}>
              <NavIcon name="rerender" className="app-icon" />撤销
            </button>
            <button onClick={redo} disabled={!future.length || running}>
              <NavIcon name="rerender" className="app-icon" />重做
            </button>
            <button onClick={save} disabled={savingCues || !dirty || running || retranslatingCueId !== null || compactingCues || !job || cueOwner.current !== job.id || !cuesReady} aria-busy={savingCues}>
              <NavIcon name="save" className="app-icon" />保存字幕
            </button>
            <button
              className="primary small"
              onClick={rerender}
              disabled={!job?.result || running || retranslatingCueId !== null || compactingCues || dirty || job?.edit_state === "legacy-ambiguous"}
              title={job?.edit_state === "legacy-ambiguous" ? "旧任务的后期版本顺序无法确认；为保护现有成片，已禁用重新渲染" : dirty ? "先保存当前字幕修改，再重新渲染" : "使用当前活动视频和字幕重新渲染"}
            >
              <NavIcon name="rerender" className="app-icon" />重新渲染
            </button>
          </div>
          <div className="editor-timing-actions">
            <div className="editor-group-label"><NavIcon name="speed" className="app-icon" /><b>时间同步</b><span>调整整份字幕的时间</span></div>
            <div className="editor-timing-main">
              <strong>字幕和声音没对齐？</strong>
              <button onClick={() => shiftByStep(-1)} disabled={running || !job?.result || !cues.length}>
                <NavIcon name="speed" className="app-icon" />字幕提前 0.5 秒
              </button>
              <button onClick={() => shiftByStep(1)} disabled={running || !job?.result || !cues.length}>
                <NavIcon name="speed" className="app-icon" />字幕延后 0.5 秒
              </button>
              <span className="editor-timing-note">字幕太晚就点“提前”，太早就点“延后”。每次移动全部字幕 0.5 秒。</span>
            </div>
            <details className="editor-timing-more">
              <summary>精确调整 / 自动对齐</summary>
              <div className="editor-timing-details">
                <div className="editor-timing-exact">
                  <label htmlFor="subtitle-exact-offset">相差秒数</label>
                  <input
                    id="subtitle-exact-offset"
                    className="offset-input"
                    inputMode="decimal"
                    value={offsetValue}
                    placeholder="例如 1.25"
                    disabled={running || !job?.result}
                    onChange={(event) => setOffsetValue(event.target.value)}
                  />
                  <button onClick={() => applyOffset(-1)} disabled={running || !job?.result || !cues.length}>
                    精确提前
                  </button>
                  <button onClick={() => applyOffset(1)} disabled={running || !job?.result || !cues.length}>
                    精确延后
                  </button>
                  <span>输入秒数后选择方向。</span>
                </div>
                <div className="editor-timing-auto">
                  <button
                    onClick={() => void alignSubtitles()}
                    disabled={running || !job?.result}
                    title="根据视频画面里清晰可见的字幕，估算并调整整份字幕的时间"
                  >
                    按画面字幕自动调整
                  </button>
                  <span>仅适用于视频画面本身带有清晰可见字幕的情况。</span>
                </div>
              </div>
            </details>
          </div>
          <div className="review-toolbar">
            <div className="editor-group-label"><NavIcon name="ear" className="app-icon" /><b>问题筛选</b></div>
            <label className="review-filter">
              字幕
              <select value={cueFilter} onChange={(event) => setCueFilter(event.target.value as "all" | "review")}>
                <option value="all">全部字幕（{cues.filter((cue) => !cue.deleted).length}）</option>
                <option value="review">待复核（{reviewCueCount}）</option>
              </select>
            </label>
            <span className="review-queue-count">{reviewCueCount ? `待复核 ${reviewCueCount} 条` : "目前没有待复核字幕"}</span>
            {reviewCueCount > 0 && <button onClick={selectReviewCuesForBatch}>选中全部待复核</button>}
            {selectedBulkCues.length > 0 && (
              <div className="review-batch-actions">
                <span>已选 {selectedBulkCues.length} 条</span>
                {selectedReviewCues.length > 0 && <button
                  onClick={() => void generateCompactionCandidates()}
                  disabled={running || compactingCues || retranslatingCueId !== null}
                  title={`为所选 ${selectedReviewCues.length} 条待复核字幕生成精简候选`}
                >
                  {compactingCues ? "正在生成…" : "生成精简建议"}
                </button>}
              </div>
            )}
            <span className="review-rule-note" title="待复核包括自动检查发现的问题和你手动标记的字幕。">自动发现的问题 + 手动标记</span>
            {dirty && <span className="review-stale-note">当前问题标签按已保存字幕检查；保存后更新。</span>}
            <button className="review-shortcut-help" onClick={() => setShortcutHelpOpen(true)}>快捷键说明</button>
          </div>
          <details className="subtitle-batch-tools">
            <summary><NavIcon name="text" className="app-icon" />批量操作 <span>搜索替换、字幕精简与候选处理</span></summary>
            <div className="replace-controls">
              <label>文本字段<select value={replaceField} onChange={(event) => setReplaceField(event.target.value as "source" | "translated")}><option value="source">原文</option><option value="translated">中文翻译</option></select></label>
              <label>应用范围<select value={replaceScope} onChange={(event) => setReplaceScope(event.target.value as "all" | "selected")}><option value="all">全部字幕</option><option value="selected">仅已选字幕（{selectedBulkCues.length}）</option></select></label>
              <label>查找<input value={replaceNeedle} onChange={(event) => setReplaceNeedle(event.target.value)} placeholder="要查找的文字" /></label>
              <label>替换为<input value={replaceValue} onChange={(event) => setReplaceValue(event.target.value)} placeholder="替换后的文字" /></label>
              <label className="replace-case"><input type="checkbox" checked={replaceCaseSensitive} onChange={(event) => setReplaceCaseSensitive(event.target.checked)} />区分大小写</label>
              <button onClick={applyLiteralReplacement} disabled={!replacePreviews.length || running}>应用替换</button>
            </div>
            <div className="replace-preview-head"><b>预览：{replacePreviews.length} 条受影响，{replacePreviews.reduce((sum, item) => sum + item.count, 0)} 处替换</b><small>只替换文本，不改时间和另一语言字段。</small></div>
            {replacePreviews.length > 0 && <div className="replace-preview-list">{replacePreviews.slice(0, 30).map((preview) => <div key={preview.cueId}><b>#{preview.cueId + 1} · {preview.count} 处</b><span>{preview.before}</span><span aria-hidden="true">→</span><span>{preview.after}</span></div>)}{replacePreviews.length > 30 && <small>另有 {replacePreviews.length - 30} 条命中未展开。</small>}</div>}
          </details>
          {compactionCandidates.length > 0 && <section className="model-candidates" aria-label="字幕精简候选">
            <header><div><b>字幕精简候选</b><small>核对后逐条采用或全部采用；时间轴保持不变。</small></div><button onClick={applyAllCompactionCandidates} disabled={!compactionCandidates.some((candidate) => !candidate.error && candidate.candidate !== candidate.original)}>全部采用</button></header>
            {compactionCandidates.map((candidate) => <article key={candidate.cue_id} className="model-candidate-row">
              <b>#{candidate.cue_id + 1}</b>
              <div><small>原译文</small><p>{candidate.original}</p></div>
              <div><small>精简候选</small><p>{candidate.error ? `生成失败：${candidate.error}` : candidate.candidate}{candidate.still_overflowing && !candidate.error ? "（仍超出容量，可继续手动修改）" : ""}</p></div>
              <button onClick={() => applyCompactionCandidate(candidate)} disabled={Boolean(candidate.error) || candidate.candidate === candidate.original}>采用</button>
            </article>)}
          </section>}
          <div className="column-labels">
            <span>选中 / #</span>
            <span>时间</span>
            <span>原文</span>
            <span>中文翻译</span>
            <span>操作</span>
          </div>
        </div>
        {cues.length ? (
          <div className="cue-table">
            {visibleCueRows.map(({ cue, index }) => (
              <article
                key={cue.id}
                data-cue-id={cue.id}
                className={[
                  "cue-row",
                  activeCue === cue.id ? "current" : "",
                  selectedCue === cue.id ? "selected" : "",
                  cue.deleted ? "deleted" : "",
                ].join(" ")}
              >
                <div className="cue-select-index">
                  <input type="checkbox" aria-label={`选择第 ${index + 1} 段字幕`} checked={bulkSelectedCueIds.includes(cue.id)} onChange={() => toggleBulkCue(cue.id)} disabled={cue.deleted} />
                  <button className="cue-index" onClick={() => seek(cue)} title={`播放第 ${index + 1} 段字幕`}>{String(index + 1).padStart(2, "0")}</button>
                </div>
                <div className="cue-time-edit">
                  <input
                    value={formatTime(cue.start)}
                    aria-label={`第 ${index + 1} 段开始时间`}
                    title="开始时间（分:秒，如 0:05.0）"
                    disabled={!job?.result || running || cue.deleted}
                    onChange={(event) => updateCueTime(cue.id, "start", event.target.value)}
                  />
                  <span>–</span>
                  <input
                    value={formatTime(cue.end)}
                    aria-label={`第 ${index + 1} 段结束时间`}
                    title="结束时间（分:秒，如 0:05.0）"
                    disabled={!job?.result || running || cue.deleted}
                    onChange={(event) => updateCueTime(cue.id, "end", event.target.value)}
                  />
                  <button className="cue-jump" onClick={() => seek(cue)} title="跳转到此字幕">
                    ▶
                  </button>
                </div>
                <div className="cue-text-cell">
                  <textarea
                    value={cue.source}
                    aria-label={`第 ${index + 1} 段原文`}
                    onFocus={() => { setActiveCue(cue.id); setSelectedCue(cue.id); }}
                    onChange={(event) => updateCueDraft(cue.id, { source: event.target.value })}
                    disabled={!job?.result || running || cue.deleted}
                  />
                  {cue.candidate_id && <label className="candidate-source-confirm"><input type="checkbox" checked={cue.confirmed_source_edit === true} onChange={(event) => updateCueDraft(cue.id, { confirmed_source_edit: event.target.checked, source_stale: event.target.checked ? false : cue.source !== cue.translation_source, ...(event.target.checked ? { translation_source: cue.source } : {}) })} />已核对修改后的原文</label>}
                  {cue.issues?.length ? <div className="cue-issue-tags">{cue.issues.map((issue) => <span key={issue}><NavIcon name={issue === "reading-speed" || issue === "short-duration" ? "speed" : issue === "layout-overflow" || issue === "dense-text" || issue === "empty-source" || issue === "empty-translation" ? "text" : "error"} className="app-icon" />{cueIssueLabels[issue] ?? issue}</span>)}</div> : null}
                </div>
                <div className="cue-text-cell">
                  <textarea
                    value={cue.translated}
                    aria-label={`第 ${index + 1} 段中文翻译`}
                    onFocus={() => { setActiveCue(cue.id); setSelectedCue(cue.id); }}
                    onChange={(event) => updateCue(cue.id, event.target.value)}
                    disabled={!job?.result || running || cue.deleted}
                  />
                  {cue.needs_review && <span className="manual-review-tag"><NavIcon name="ear" className="app-icon" />人工待复核</span>}
                  {translationCandidate?.cueId === cue.id && <div className="single-translation-candidate"><b>{translationCandidate.provider} · {translationCandidate.model} 候选</b><p>{translationCandidate.translation}</p><div><button onClick={applyTranslationCandidate}>采用候选</button><button onClick={() => setTranslationCandidate(null)}>放弃</button></div></div>}
                </div>
                <div className="cue-row-actions">
                  <button className={cue.needs_review ? "review-flag active" : "review-flag"} onClick={() => toggleCueReview(cue)} aria-pressed={Boolean(cue.needs_review)} disabled={running || cue.deleted} title="把这条字幕加入待复核列表"><NavIcon name="ear" className="app-icon" />{cue.needs_review ? "取消标记" : "标记复核"}</button>
                  <details className="cue-more-actions">
                    <summary>更多</summary>
                    <div>
                      <button onClick={() => void generateCueTranslation(cue)} disabled={running || cue.deleted || retranslatingCueId !== null || compactingCues || !["deepseek", "openai", "ollama"].includes(job?.options.translator ?? "")} title="使用当前任务模型生成单条译文候选">{retranslatingCueId === cue.id ? "生成中…" : "单条重译"}</button>
                      <button
                        className={cue.deleted ? "cue-restore" : "cue-delete"}
                        onClick={() => toggleCueDeleted(cue.id)}
                        disabled={running || !job?.result}
                        title={cue.deleted ? "恢复这条字幕" : "删除这条字幕（此段不再显示字幕）"}
                      >{cue.deleted ? "恢复字幕" : "删除字幕"}</button>
                    </div>
                  </details>
                </div>
              </article>
            ))}
            {cueFilter === "review" && visibleCueRows.length === 0 && <div className="empty-editor"><b>没有待复核字幕</b><span>自动检查未发现问题，也没有人工标记。</span></div>}
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
  const publishSelectedTask = outputs?.tasks.find((task) => task.files.some((file) => file.id === selectedPublishVideo));
  const publishSelectedFile = publishSelectedTask?.files.find((file) => file.id === selectedPublishVideo);
  const publishVideoName = selectedPublishVideo === "__native__"
    ? nativePublishName
    : publishSelectedFile && publishSelectedTask
      ? formatPublishSourceLabel(publishSelectedTask, publishSelectedFile, outputTaskTitles)
      : (job?.options.title || job?.material.name || job?.result?.rendered_video?.split(/[\\/]/).pop() || (publishVideoReady ? "当前任务成片" : "尚未选择成片"));
  const currentTaskPublishLabel = truncateLabel(job?.options.title || job?.material.name || "当前任务成片", 40);
  const douyinTags = tags
    .split(",")
    .map((tag) => tag.trim().replace(/^#+/, ""))
    .filter(Boolean)
    .map((tag) => `#${tag}`)
    .join(" ");
  const douyinPublishCopy = [
    options.title.trim() ? `标题：${options.title.trim()}` : "",
    description.trim() ? `正文：${description.trim()}` : "",
    douyinTags ? `话题：${douyinTags}` : "",
  ].filter(Boolean).join("\n\n");
  const publishView = (
    <section className="workspace-card publish-view">
      <header>
        <p className="eyebrow">发布辅助</p>
        <h1>准备文案，人工确认后投稿</h1>
        <p>选择投稿平台。应用可上传成片并辅助填写投稿信息，最后由你检查并手动发布。</p>
      </header>

      <div className="publish-platform-choice">
        <label>投稿平台
          <select
            value={publishPlatform}
            onChange={(event) => {
              const platform = event.target.value as "bilibili" | "douyin";
              if (templateDraftDirty && !window.confirm("切换平台会加载该平台的文案模板，未保存的模板草稿可能丢失。确定继续吗？")) {
                return;
              }
              setPublishPlatform(platform);
              localStorage.setItem("ybl-publish-platform", platform);
              void loadTemplates(platform).then(() => {
                setDescription("");
                setMessage(
                  platform === "douyin"
                    ? "已切换到抖音文案模板（默认不含原视频链接）。"
                    : "已切换到 B 站文案模板（可附原视频链接）。",
                );
              });
            }}
          >
            <option value="bilibili">哔哩哔哩（B 站）</option>
            <option value="douyin">抖音</option>
          </select>
        </label>
      </div>

      <section className="publish-section">
        <div className="publish-section-head"><b>01 · 选定成片</b><small>当前任务成片或任意本地视频</small></div>
        <div className="publish-file-strip"><span className={publishVideoReady ? "ready" : ""}>▶</span><div><b title={publishVideoName}>{publishVideoName}</b><small>{publishVideoReady ? "成片已就绪 · 投稿前仍需人工预览" : "请先选择一份可投稿的成片"}</small></div></div>
        <div className="form-grid publish-file-select">
          <label>
            成品来源
            <select
              value={selectedPublishVideo}
              onChange={(event) => {
                setSelectedPublishVideo(event.target.value);
                setNativePublishToken("");
                setNativePublishName("");
                publishHydrateKey.current = "";
              }}
            >
              <option value="">{publishVideoReady && !selectedPublishVideo && !nativePublishToken ? `当前任务 · ${currentTaskPublishLabel}` : "当前任务成片（如已完成）"}</option>
              {nativePublishToken && <option value="__native__">本地选择 · {nativePublishName}</option>}
              {outputs?.tasks.flatMap((task) =>
                task.files
                  .filter((file) => file.kind === "成品视频")
                  .map((file) => (
                    <option key={file.id} value={file.id} title={`${outputTaskDisplayTitle(task, outputTaskTitles)} · ${task.name} · ${file.name}`}>
                      {formatPublishSourceLabel(task, file, outputTaskTitles)}
                    </option>
                  )),
              )}
            </select>
          </label>
        </div>
        <div className="inline-actions publish-file-actions">
          <button onClick={() => void chooseNativePublishVideo()}>选择本地成片</button>
          <button onClick={() => setView("subtitle")} disabled={!job?.result || Boolean(selectedPublishVideo)}>在字幕页预览当前成片</button>
        </div>
      </section>

      <section className="publish-section">
        <div className="publish-section-head">
          <b>02 · 投稿文案</b>
          <small>{publishPlatform === "bilibili" ? "标题、简介、标签会填入 B 站投稿页，仍需人工确认" : "标题、正文和话题会尝试填入抖音投稿页，仍需人工确认"}</small>
        </div>
        <label className="publish-template-select">
          {publishPlatform === "douyin" ? "抖音文案模板" : "B 站文案模板"}
          <select value={templateName} onChange={(event) => {
            if (templateDraftDirty && !window.confirm("「模板内容」中有未保存的草稿修改。切换模板会丢失这些修改，确定继续吗？")) return;
            const name = event.target.value;
            setTemplateName(name);
            localStorage.setItem(`ybl-publish-template-${publishPlatform}`, name);
            setTemplateBody(templates.find((item) => item.name === name)?.body ?? "");
          }}>{templates.map((item) => <option key={item.name}>{item.name}</option>)}</select>
        </label>
        <small className="drawer-note">
          {publishPlatform === "douyin"
            ? "抖音与 B 站模板分开保存；抖音默认模板不含原视频链接。"
            : "B 站模板可附原视频链接；自定义模板仅作用于当前平台。"}
        </small>
        <div className="inline-actions publish-copy-actions">
            <button onClick={() => void loadPublishCopy({ force: true })} disabled={!publishVideoReady}>载入成片文案</button>
            <button onClick={() => void loadPublishCopy({ generate: true, force: true })} disabled={!publishVideoReady}>重新生成标题标签</button>
            <button onClick={() => void generateDescription()}>更新简介</button>
        </div>
        <details className="publish-template-tools">
          <summary>模板编辑与管理 <span>展开 ›</span></summary>
          <div className="inline-actions">
            <button onClick={() => openTemplateEditor()}>新建/编辑模板</button>
            <button onClick={deleteTemplate}>删除自定义模板</button>
          </div>
          {templateDraftDirty && (
            <div className="template-draft-bar">
              <span className="dirty">草稿未保存：更新简介会优先使用当前模板草稿。</span>
              <button onClick={() => setTemplateBody(savedTemplateBody)}>恢复模板内容</button>
              <button onClick={() => openTemplateEditor()}>保存为模板</button>
            </div>
          )}
          <div className="publish-fields">
          <label>
            模板内容
            <textarea
              className="publish-template-body"
              value={templateBody}
              maxLength={5000}
              onChange={(event) => setTemplateBody(event.target.value)}
              placeholder={
                publishPlatform === "douyin"
                  ? "抖音模板可用 {custom_text}；一般不必写原视频链接"
                  : "模板支持 {source} 与 {custom_text}"
              }
            />
          </label>
          </div>
        </details>
        <div className="publish-fields">
          <label>
            标题
            <input
              value={options.title}
              maxLength={200}
              onChange={(event) => updateOptions("title", event.target.value)}
              placeholder="优先使用翻译阶段生成的中文标题"
            />
            {fieldErrors.title && <small className="field-error">{fieldErrors.title}</small>}
          </label>
          <label>
            简介
            <textarea
              className="publish-description-body"
              value={description}
              maxLength={5000}
              onChange={(event) => { setDescription(event.target.value); setFieldErrors((current) => ({ ...current, description: "" })); }}
              placeholder="生成后可继续人工修改"
            />
            {fieldErrors.description && <small className="field-error">{fieldErrors.description}</small>}
          </label>
          <div className="publish-meta-grid">
            <label>
              {publishPlatform === "douyin" ? "话题（逗号分隔）" : "标签（逗号分隔）"}
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
        </div>
      </section>

      <section className="publish-section">
        {publishPlatform === "bilibili" ? (
          <>
            <div className="publish-section-head"><b>03 · 连接 B 站并填表</b><small>自动化在人工提交前停止</small></div>
            <div className="publish-browser-row">
              <label>投稿浏览器<select value={publishBrowser} onChange={(event) => setPublishBrowser(event.target.value)}><option value="chromium">Chromium</option><option value="msedge">Microsoft Edge</option></select></label>
              <div className="inline-actions"><button onClick={publishCheck}>登录检查</button><button onClick={() => void loadPublishStatus()}>检查 Profile</button></div>
            </div>
            {profileMessage && <p className="publish-status" role="status">{profileMessage}</p>}
            <div className="publish-warning">
              <div className="publish-warning-checks">
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
              </div>
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
                {publishSession.error ? (
                  <p className="error">{publishSession.error}</p>
                ) : (
                  <p className="publish-session-message">{publishSession.message}</p>
                )}
                {publishSession.logs.length > 0 && (
                  <div className="publish-session-logs">
                    {publishSession.logs.slice(-12).map((line, index) => (
                      <p key={`${line}-${index}`}>{line}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="publish-section-head"><b>03 · 抖音上传与填表</b><small>使用独立登录资料；应用停在发布前</small></div>
            <div className="publish-browser-row">
              <label>投稿浏览器<select value={publishBrowser} onChange={(event) => setPublishBrowser(event.target.value)}><option value="chromium">Chromium</option><option value="msedge">Microsoft Edge</option></select></label>
              <div className="inline-actions"><button onClick={() => void loadPublishStatus()}>检查抖音 Profile</button><button onClick={() => void openDouyinPublisher()}>手动打开创作者中心</button></div>
            </div>
            {profileMessage && <p className="publish-status" role="status">{profileMessage}</p>}
            <div className="publish-douyin-actions">
              <CopyButton text={douyinPublishCopy} label="复制抖音文案（标题、正文、话题）" />
              <p>首次使用时请在打开的专用浏览器中登录。应用会上传所选成片并尝试填写标题、正文和话题，然后等待你检查封面、可见范围、声明及话题；不会自动点击发布。</p>
            </div>
            {(publishSession.active || publishSession.status !== "idle") && (
              <div className={`publish-session ${publishSession.status}`}>
                <div className="publish-session-head">
                  <b>
                    {publishSession.status === "waiting_review"
                      ? "已到人工发布前"
                      : publishSession.status === "failed"
                        ? "辅助失败"
                        : publishSession.status === "finished"
                          ? "辅助已结束"
                          : "抖音投稿辅助进行中"}
                  </b>
                  <span>{publishSession.active ? "运行中" : "已停止"}</span>
                </div>
                {publishSession.error ? <p className="error">{publishSession.error}</p> : <p className="publish-session-message">{publishSession.message}</p>}
                {publishSession.logs.length > 0 && (
                  <div className="publish-session-logs">
                    {publishSession.logs.slice(-12).map((line, index) => <p key={`${line}-${index}`}>{line}</p>)}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </section>

      <footer className="card-footer">
        <div className="publish-submit-hint">
          {!publishVideoReady && <span>请先选择一份可投稿的成片。</span>}
        </div>
        {publishPlatform === "bilibili" ? (
          <>
            <button className="danger" onClick={closePublishBrowser}>
              关闭后台 B 站浏览器
            </button>
            <button className="primary" onClick={publishAssist} disabled={!publishVideoReady} title={!publishVideoReady ? "请先选择成片" : "打开 B 站投稿辅助"}>
              打开 B 站投稿页面并填充 →
            </button>
          </>
        ) : (
          <>
            <button className="danger" onClick={closePublishBrowser}>关闭抖音辅助浏览器</button>
            <button className="primary" onClick={() => void publishDouyinAssist()} disabled={!publishVideoReady || publishSession.active} title={!publishVideoReady ? "请先选择成片" : "上传成片并辅助填写，停在最终发布前"}>
              上传成片并辅助填写 →
            </button>
          </>
        )}
      </footer>
      {templateEditorOpen && <div className="modal-backdrop" role="presentation">
        <section className="template-modal" role="dialog" aria-modal="true" aria-label="编辑文案模板">
          <header><div><p className="eyebrow">{publishPlatform === "douyin" ? "抖音文案模板" : "B 站文案模板"}</p><h2>新建或编辑模板</h2></div><button onClick={() => setTemplateEditorOpen(false)} aria-label="关闭模板编辑器">×</button></header>
          <label>模板名称<input value={templateDraftName} placeholder="例如：授权转载说明" onChange={(event) => setTemplateDraftName(event.target.value)} /></label>
          <label>模板内容<textarea value={templateBody} onChange={(event) => setTemplateBody(event.target.value)} /></label>
          <div className="inline-actions">
            {publishPlatform === "bilibili" && (
              <button onClick={() => insertTemplateVariable("{source}")}>{"{source}"}</button>
            )}
            <button onClick={() => insertTemplateVariable("{custom_text}")}>{"{custom_text}"}</button>
          </div>
          <div className="template-preview"><b>预览效果</b><p>{(templateBody || "填写模板内容后自动预览").replaceAll("{source}", publishPlatform === "douyin" ? "" : (sourceUrl || "https://example.com/video")).replaceAll("{custom_text}", "自定义内容")}</p></div>
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
          <h1>任务库</h1>
          <p>汇总待处理、处理中、可继续和已完成任务。同一时间只运行一个耗资源阶段。</p>
          <p className="library-cost-legend"><NavIcon name="text" className="app-icon" />金额为累计估算；0.2.52 之前的费用未记录。</p>
        </div>
        <div className="library-header-actions">
          <button onClick={() => void Promise.all([loadJobHistory(), loadRestorableJobs()])}>刷新</button>
          <button
            className={libraryBatchMode ? "danger" : ""}
            onClick={() => {
              setLibraryBatchMode((current) => !current);
              setSelectedOutputTaskIds([]);
            }}
          >
            {libraryBatchMode ? "退出多选" : "多选删除"}
          </button>
          <button className="primary" onClick={() => setView("material")}>添加视频</button>
        </div>
      </header>
      {batchQueue.length > 0 && (!batchSummaryDismissed || batchQueue.some((item) => item.status === "pending" || item.status === "running")) && (
        <div className="batch-panel" aria-label="批量处理" aria-live="polite">
          {(() => {
            const count = (status: BatchItem["status"]) => batchQueue.filter((item) => item.status === status).length;
            const done = count("done"), failed = count("failed"), cancelled = count("cancelled");
            const running = batchQueue.find((item) => item.status === "running");
            const active = running || batchQueue.some((item) => item.status === "pending");
            const total = batchQueue.length - cancelled;
            const pct = Math.round(((done + failed + (running?.progress ?? 0) / 100) / total) * 100);
            if (!active) return batchSummaryDismissed ? null : (
              <div className="batch-summary">
                <div><b>批量处理已结束：</b><b className="ok">完成 {done}</b> · <b className="bad">失败 {failed}</b> · 已停止 {cancelled}</div>
                <div className="library-header-actions">
                  {failed + cancelled > 0 && <button onClick={() => void retryBatch()}>重试失败项</button>}
                  <button onClick={() => setBatchSummaryDismissed(true)}>关闭</button>
                </div>
              </div>
            );
            return (
              <>
                <div className="batch-panel-head">
                  <div><h2>批量处理</h2><small>{batchQueue.length} 个视频 · 串行处理，不会自动投稿</small></div>
                  <div className="library-header-actions"><button className="danger" onClick={() => void cancelBatch()}>停止批量</button></div>
                </div>
                <div className="library-summary">
                  <div><b>{total}</b><span>总数</span></div>
                  <div className="done"><b>{done}</b><span>已完成</span></div>
                  <div><b>{running ? 1 : 0}</b><span>处理中</span></div>
                  <div className="failed"><b>{failed}</b><span>失败</span></div>
                </div>
                <div className="batch-current">
                  <div className="batch-current-line">
                    <div>{running ? <><ActivityBars /> 正在处理：<b>{running.title || running.url}</b> <span>· {running.stage ?? "准备中"} {running.progress ?? 0}%</span></> : <span>排队中（等待其他任务）</span>}</div>
                    <b className="pct">总进度 {pct}%</b>
                  </div>
                  <div className="batch-bar"><i style={{ width: `${pct}%` }} /></div>
                </div>
              </>
            );
          })()}
          {batchQueue.map((item) => {
            const category = item.status === "done" ? "completed" : item.status === "running" ? "active" : item.status === "failed" || item.status === "cancelled" ? "attention" : "";
            const statusText = { pending: "排队中", running: "处理中", done: "待投稿", failed: "失败", cancelled: "已停止" }[item.status];
            return (
              <article className={`video-library-row ${category}`} key={item.id}>
                <div className="video-library-main">
                  <div className="video-library-title"><b title={item.url}>{item.title || item.url}</b></div>
                  <div className="video-library-meta">
                    <span className={`library-status ${category}`}>{statusText}</span>
                    {item.status === "done" && <span>成片已完成，在下方任务列表载入后到发布页手动投稿</span>}
                  </div>
                  {item.error && <TaskErrorSummary message={item.error} />}
                </div>
                <div className="video-library-actions">
                  {(item.status === "failed" || item.status === "cancelled") && <button onClick={() => void retryBatch(item.id)}>续跑</button>}
                  {item.status === "running" && item.job_id && <button className="danger" onClick={() => void cancelBatchItem(item.job_id!)}>取消</button>}
                </div>
              </article>
            );
          })}
        </div>
      )}
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
      <div className={`library-tools ${libraryBatchMode ? "batch-mode" : ""}`}>
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
        {libraryBatchMode ? (
          <>
            <label className="library-select-all">
              <input
                type="checkbox"
                checked={allVisibleLibraryTasksSelected}
                disabled={!visibleLibraryTaskIds.length}
                onChange={(event) => {
                  if (!event.target.checked) {
                    setSelectedOutputTaskIds([]);
                    return;
                  }
                  if (visibleLibraryTaskIds.length > 200) setMessage("批量删除每次最多选择 200 个任务，已选择当前页前 200 个。 ");
                  setSelectedOutputTaskIds(visibleLibraryTaskIds.slice(0, 200));
                }}
              />
              已选 {selectedOutputTaskIds.length} / 200 项
            </label>
            <button
              className="history-clear"
              disabled={deleteBusy}
              onClick={() => void deleteSelectedHistoryTasks("record_only")}
              title="只删任务库记录，保留磁盘文件"
            >
              仅删记录
            </button>
            <button
              className="danger"
              disabled={deleteBusy}
              onClick={() => void deleteSelectedHistoryTasks("record_and_outputs")}
              title="删除任务库记录及其输出目录"
            >
              删除记录和文件
            </button>
            {historyDeleteError && <small className="library-batch-error" role="alert">{historyDeleteError}</small>}
          </>
        ) : (
          <small className="library-batch-hint">需要删除旧任务时，点右上角「多选删除」。</small>
        )}
      </div>
      <div className="video-library-list" aria-live="polite">
        {libraryJobs.length ? libraryJobs.map((item) => {
          const resumable = restorableJobs.some((candidate) => candidate.id === item.id);
          const category = videoLibraryCategory(item, resumable);
          const isCurrent = job?.id === item.id;
          const missingOutput = item.output_exists === false;
          const statusText = category === "active"
            ? item.status === "cancelling" ? "正在取消" : "处理中"
            : category === "completed"
              ? "已完成"
              : missingOutput && !resumable ? "输出已删除"
              : resumable ? "可继续" : item.status === "failed" ? "失败" : item.status === "cancelled" ? "已取消" : "待处理";
          return (
            <article className={`video-library-row ${category}${missingOutput ? " missing-output" : ""}`} key={item.id}>
              {libraryBatchMode && (
                <label className="library-select" title={category === "active" ? "运行中的任务不可删除" : "选择任务进行批量删除"}>
                  <input
                    type="checkbox"
                    checked={selectedOutputTaskIds.includes(item.id)}
                    disabled={category === "active" || (!selectedOutputTaskIds.includes(item.id) && selectedOutputTaskIds.length >= 200)}
                    onChange={(event) => setSelectedOutputTaskIds((current) => event.target.checked ? [...current, item.id] : current.filter((id) => id !== item.id))}
                  />
                </label>
              )}
              <div className="video-library-main">
                <div className="video-library-title">
                  <b title={item.title}>{item.title}</b>
                  {isCurrent && <span className="current-task">当前任务</span>}
                  {missingOutput && <span className="missing-output-tag">无输出目录</span>}
                </div>
                <div className="video-library-meta">
                  <span className={`library-status ${category}`}><NavIcon name={category === "completed" ? "check" : category === "active" ? "running" : category === "attention" ? "error" : "pause"} className="app-icon" />{statusText}</span>
                  <span>{item.stage || "等待配置"}</span>
                  <span>{item.created_at ? new Date(item.created_at * 1000).toLocaleString() : "时间未知"}</span>
                </div>
                {(category === "active" || (item.progress > 0 && item.progress < 100)) && (
                  <div className="library-progress" aria-label={`处理进度 ${item.progress}%`}><i style={{ width: `${item.progress}%` }} /></div>
                )}
                {item.error && <TaskErrorSummary message={item.error} />}
              </div>
              <div className="library-cost-column" aria-label="翻译费用">
                <span className="library-cost-label">翻译费用</span>
                <strong>{item.llm_usage_total?.cost ? costLabel(item.llm_usage_total) : "未记录"}</strong>
                <small>{item.llm_usage_total ? `累计估算 · ${item.llm_usage_total.http_requests} 次请求` : "旧任务未统计"}</small>
                <button className="library-cost-detail" onClick={() => setExpandedCostId((current) => current === item.id ? null : item.id)}>{item.llm_usage_total ? "查看费用" : "为什么未记录"}</button>
                {expandedCostId === item.id && (
                  <div className="library-cost-breakdown">
                    {item.llm_usage_total ? Object.entries(item.llm_usage_total.by_kind ?? {}).map(([kind, value]) => <span key={kind}><b>{({ source_correction: "源文校对", translation: "字幕翻译", review: "译文复审", formatting: "字幕精简", metadata: "标题标签", segmentation: "语义断句", single_translation: "单条重译" } as Record<string, string>)[kind] ?? kind}</b><em>{costLabel(value)}</em></span>) : <span>0.2.52 之前的任务没有费用用量记录，无法补算。</span>}
                  </div>
                )}
              </div>
              <div className="video-library-actions">
                {resumable || category === "active" || !missingOutput ? (
                  <button className="library-open" onClick={() => isCurrent && running ? setView("process") : void loadRestoredJob(item.id)}>
                    {isCurrent && running ? "查看进度" : resumable ? "继续处理" : category === "completed" ? "打开任务" : "检查任务"}
                  </button>
                ) : (
                  <button
                    className="history-clear small"
                    onClick={() => {
                      setLibraryBatchMode(true);
                      setSelectedOutputTaskIds([item.id]);
                    }}
                  >
                    清理记录
                  </button>
                )}
                <details className="library-row-more">
                  <summary aria-label={`更多操作：${item.title}`} title="更多操作">⋯</summary>
                  <div>
                    <button onClick={() => void openHistoryPath(item.rendered_video || "")} disabled={!item.rendered_exists} title={item.rendered_exists ? "打开最终成片" : "尚未生成成片"}>打开成片</button>
                    {item.dub_exists && <button onClick={() => void openHistoryPath(item.dub_video || "")}>打开配音版</button>}
                    <button onClick={() => void openHistoryPath(item.output_dir || "")} disabled={!item.output_exists} title={item.output_exists ? "打开该任务的输出目录" : "输出目录不存在"}>打开文件夹</button>
                  </div>
                </details>
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
        {historyHasMore && (
          <button className="history-load-more" onClick={() => void loadJobHistory(true)} disabled={historyLoadingMore}>
            {historyLoadingMore ? "正在加载…" : "加载更早任务"}
          </button>
        )}
      </div>
    </section>
  );
  const viewedTask = outputs?.tasks.find((task) => task.id === viewedTaskId) ?? null;
  const viewOutputTask = (taskId: string) => {
    setViewedTaskId(taskId);
    setSelectedOutputPaths((current) => current.filter((id) => outputs?.tasks.some((row) => row.id === id) || id.startsWith(`${taskId}/`)));
  };
  const storageKinds = ["下载/原视频", "成品视频", "字幕", "中间文件", "预览图", "其他"];
  const storageTotalBytes = storageKinds.reduce((sum, kind) => sum + (outputs?.category_totals[kind]?.bytes ?? 0), 0);
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
              onChange={(event) => { updateOptions("output_dir", event.target.value); setOutputs(null); setSelectedOutputPaths([]); setViewedTaskId(null); }}
            />
            <button onClick={() => void chooseOutputDir()}>选择文件夹</button>
          </div>
        </label>
        <button onClick={loadOutputs}>刷新</button>
        <button onClick={openOutput}>打开目录</button>
        <small>已选 {selectedOutputPaths.length} 项</small>
        <button
          className="danger"
          disabled={!selectedOutputPaths.length}
          onClick={deleteOutputs}
        >
          删除选中项
        </button>
      </div>
      <div className="storage-total"><strong>{outputs?.total_size ?? "—"}</strong><span>当前输出目录占用</span><small>{outputs?.task_count ?? 0} 个任务</small></div>
      <div className="storage-summary">
        {storageKinds.map((kind) => <span key={kind}><b>{outputs?.category_totals[kind]?.label ?? "0 B"}</b>{kind}</span>)}
      </div>
      <div className="storage-rail" role="img" aria-label="输出目录空间分类占比">
        {storageKinds.map((kind, index) => <i key={kind} className={`storage-segment segment-${index}`} style={{ width: `${storageTotalBytes ? ((outputs?.category_totals[kind]?.bytes ?? 0) / storageTotalBytes) * 100 : 0}%` }} title={`${kind}：${outputs?.category_totals[kind]?.label ?? "0 B"}`} />)}
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
                  <NavIcon name={item.status === "completed" ? "check" : item.status === "failed" ? "error" : item.status === "cancelled" ? "pause" : "running"} className="app-icon" />{item.status === "completed" ? "完成" : item.status === "failed" ? "失败" : item.status === "cancelled" ? "已取消" : item.status}
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
            {historyHasMore && (
              <button className="history-load-more" onClick={() => void loadJobHistory(true)} disabled={historyLoadingMore}>
                {historyLoadingMore ? "正在加载…" : "加载更早任务"}
              </button>
            )}
          </div>
        )}
      </section>
      <p className="file-manager-legend">
        说明：「查看」按钮用于查看某个任务的文件；「☐ 删除」勾选后点上方「删除选中项」。点击任务名称仅查看，不会删除任何内容。
      </p>
      <div className="file-manager">
        <section className="task-pane">
          <header><b>1. 任务列表</b><small>点「查看」显示该任务文件</small><button onClick={() => { const ids = (outputs?.tasks ?? []).slice(0, 200).map((task) => task.id); setSelectedOutputPaths(ids); if ((outputs?.tasks.length ?? 0) > 200) setMessage("本次已选前 200 个任务；其余任务可下一批处理。"); }} disabled={!outputs?.tasks.length}>选择当前任务{(outputs?.tasks.length ?? 0) > 200 ? "（前 200）" : ""}</button><button onClick={() => setSelectedOutputPaths([])} disabled={!selectedOutputPaths.length}>清空选择</button></header>
          {outputs?.tasks.length ? outputs.tasks.map((task) => (
            <article key={task.id} className={viewedTaskId === task.id ? "task-row selected" : "task-row"}>
              <label className="delete-check" title="勾选后点上方“删除选中项”"><input type="checkbox" checked={selectedOutputPaths.includes(task.id)} disabled={selectedOutputPaths.length >= 200 && !selectedOutputPaths.includes(task.id)} onChange={(event) => setSelectedOutputPaths((values) => event.target.checked ? [...values.filter((id) => !id.startsWith(`${task.id}/`)), task.id] : values.filter((value) => value !== task.id))} /><small>删除</small></label>
              <button className="task-row-main" onClick={() => viewOutputTask(task.id)} aria-pressed={viewedTaskId === task.id} title="查看该任务的文件"><b title={outputTaskDisplayTitle(task, outputTaskTitles)}>{truncateLabel(outputTaskDisplayTitle(task, outputTaskTitles), 48)}</b><small>{task.name} · {task.files.length} 个文件 · {new Date(task.modified_at * 1000).toLocaleString()}</small></button>
              <span>{task.size_label}</span>
              <button className="view-button" onClick={() => viewOutputTask(task.id)}>查看</button>
              <button onClick={(event) => { event.stopPropagation(); void openOutputPath(task.id); }}>打开</button>
            </article>
          )) : <div className="empty-editor"><b>还没有输出任务</b><span>完成一次处理后，文件会显示在这里。</span></div>}
        </section>
        <section className="task-pane">
          <header><b>2. 文件列表</b><small>{viewedTask ? viewedTask.name : "尚未选择任务"}</small></header>
          {viewedTask ? viewedTask.files.map((file) => (
            <article key={file.id} className="file-row">
              <label className="delete-check" title={selectedOutputPaths.includes(viewedTask.id) ? "已选择整个任务，无需重复选择内部文件" : "勾选后点上方“删除选中项”"}><input type="checkbox" disabled={selectedOutputPaths.includes(viewedTask.id) || (selectedOutputPaths.length >= 200 && !selectedOutputPaths.includes(file.id))} checked={selectedOutputPaths.includes(file.id) || selectedOutputPaths.includes(viewedTask.id)} onChange={(event) => setSelectedOutputPaths((values) => event.target.checked ? [...values, file.id] : values.filter((value) => value !== file.id))} /><small>删除</small></label>
              <span className="file-kind"><NavIcon name={fileIconName(file.kind, file.name)} className="app-icon" />{file.kind}</span>
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
            Localizer <small>v{appVersion || "—"}</small>
          </b>
        </a>
        <div className="project-name">
          <strong>{job?.material.name ?? selected?.name ?? "等待素材"}</strong>
          <span className={running ? "project-activity" : undefined}>
            {running && <ActivityBars />}
            {running
              ? `${job?.stage} ${job?.progress}% · ${formatElapsed(job?.started_at ? (now - job.started_at * 1000) / 1000 : 0)}`
              : selected?.duration_seconds
                ? `${selected.duration_seconds.toFixed(1)} 秒`
                : "本地工作台"}
          </span>
        </div>
        <div className="top-actions">
          <button ref={tutorialTriggerRef} className="tutorial-trigger" onClick={openTutorial} disabled={settingsOpen || dependenciesOpen || shortcutHelpOpen || templateEditorOpen} aria-label="打开新手教程">
            <NavIcon name="book" className="app-icon" />新手教程
          </button>
          <span className={online ? "status online" : "status"}>
            {online ? "本地已连接" : "未连接"}
          </span>
          <button className="text-button" onClick={() => setSettingsOpen(true)}>
            设置
          </button>
          <button className="text-button" onClick={() => setInspectorOpen((value) => !value)}>
            任务详情
          </button>
          <details className="top-more">
            <summary>更多</summary>
            <div className="top-more-menu">
              <button onClick={() => setDependenciesOpen(true)}>依赖中心</button>
              <button onClick={() => setShortcutHelpOpen(true)}>使用帮助</button>
            </div>
          </details>
          <button
            className="primary"
            onClick={topAction}
            disabled={preparingJob || (running ? !job : !job && !materialSourceReady)}
          >
            {topActionLabel}
          </button>
        </div>
      </header>
      {restorableJobs.length > 0 && !running && (
        <section className="restore-banner" aria-live="polite">
          <div>
            <b>↻ {restorableJobs.length} 个任务可继续</b>
            <span>选择任务恢复上次进度；载入不会自动开始处理。</span>
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
          {nav.map(([id, label, hint, icon]) => (
            <button
              key={id}
              className={view === id ? "active" : ""}
              onClick={() => setView(id)}
            >
              <b className="nav-label"><NavIcon name={icon} /><span>{label}</span></b>
              <small>{hint}</small>
            </button>
          ))}
          <button className="inspector-nav-button" onClick={() => setInspectorOpen(true)} disabled={!job} aria-label={job ? "打开详情与费用" : "请先载入任务"}>
            <b className="nav-label"><NavIcon name="check" /><span>详情与费用</span></b>
            <small>{job ? "进度、费用与日志" : "请先载入任务"}</small>
          </button>
          <div className="nav-status">
            <span>授权确认</span>
            <b>{authorized ? "已确认" : "待确认"}</b>
          </div>
        </nav>
        <section className={`main-workspace workspace-${view}`}>
          {view === "material" && weeklyHotCard}
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
                <p className="eyebrow">任务详情</p>
                <h2>{job?.stage ?? "等待任务"}</h2>
              </div>
              <button
                onClick={() => setInspectorOpen(false)}
                aria-label="收起任务详情"
              >
                ×
              </button>
            </div>
            <p className="inspector-subtitle">进度 · 费用 · 日志</p>
            <div className="inspector-body">
            {job && (
              <>
                <div className="progress-area">
                  <b>{job.progress}%</b>
                  {running && <ActivityBars />}
                  <small>{formatElapsed(job.finished_at && job.started_at ? job.finished_at - job.started_at : job.started_at ? (now - job.started_at * 1000) / 1000 : job.elapsed_seconds)}</small>
                  <div className={`progress ${running ? "is-active" : ""}`}>
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
                        <span aria-hidden="true"><NavIcon name={job.stages?.[id]?.status === "completed" ? "check" : job.stages?.[id]?.status === "running" || job.current_stage === id ? "running" : ["failed", "stale", "interrupted"].includes(job.stages?.[id]?.status ?? "") ? "error" : "pause"} className="app-icon" /></span>
                        <b>{label}</b>
                        <small>{stageStatusLabel[job.stages?.[id]?.status ?? "pending"]}</small>
                      </div>
                    ))}
                  </div>
                )}
                {(job.llm_usage || job.llm_usage_total || ["deepseek", "openai"].includes(job.options.translator)) && <LlmUsagePanel usage={job.llm_usage} total={job.llm_usage_total} />}
                <details className="job-options-details">
                  <summary>本次任务设置</summary>
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
                    <dt>CC 限制</dt>
                    <dd>{job.options.require_reuse_allowed ? "仅 reuse allowed" : "未启用"}</dd>
                  </div>
                  <div>
                    <dt>字幕渲染</dt>
                    <dd>{job.options.render_encoder}</dd>
                  </div>
                  <div>
                    <dt>字幕排版</dt>
                    <dd>
                      {job.options.subtitle_display_mode === "bilingual-source-first"
                        ? "原文在上 + 中文在下"
                        : job.options.subtitle_display_mode === "bilingual-translation-first"
                          ? "中文在上 + 原文在下"
                          : "中文单语"}
                    </dd>
                  </div>
                </dl>
                </details>
                <section className="compact-log">
                  <div className="log-head"><p className="eyebrow">{showFullLogs ? "完整日志" : "最近日志"}</p><div className="log-actions"><CopyButton text={buildDebugText()} label="复制调试信息" /><button onClick={() => void toggleFullLogs()}>{showFullLogs ? "收起" : `查看完整日志${job.log_total ? `（${job.log_total} 行）` : ""}`}</button></div></div>
                  {showFullLogs && fullLogsTruncated && <p>日志较长，仅显示最近 5000 行。</p>}
                  {(showFullLogs ? fullLogs : job.logs
                    .slice(-8)
                    .reverse()).map((line, index) => (
                      <p key={`${line}-${index}`}>{line}</p>
                    ))}
                </section>
                {(job.error || job.recovery) && (
                  <div className="error-block">
                    {job.error && <TaskErrorSummary message={job.error} />}
                    {job.error && job.suggested_action && <p className="error-suggestion">建议：{job.suggested_action}</p>}
                    {job.recovery && (
                      <section className="stage-recovery" aria-label="可选恢复操作">
                        <b>可选恢复操作</b>
                        <p>{job.recovery.impact}</p>
                        {job.recovery.id === "retry_extract_with_lower_resource" && (
                          <div className="recovery-buttons">
                            <button className="primary small" disabled={running || preparingJob} onClick={() => void runRecoveryAction(job.recovery!, { device: "cpu", compute_type: "int8" })}>
                              确认改用 CPU / int8 重试转写
                            </button>
                            {smallerInstalledWhisperModels.map((model) => (
                              <button className="small" key={model} disabled={running || preparingJob} onClick={() => void runRecoveryAction(job.recovery!, { options: { whisper_model_size: model } })}>
                                改用已安装的 {model} 模型
                              </button>
                            ))}
                            {!smallerInstalledWhisperModels.length && (
                              <button className="small" disabled={running || preparingJob} onClick={() => setDependenciesOpen(true)}>
                                查看或安装较小模型
                              </button>
                            )}
                          </div>
                        )}
                      </section>
                    )}
                    <div className="error-actions">
                      {job.error && <CopyButton text={job.error} label="复制错误" />}
                      {job.error && cookieHintMatches(job.error) && (
                        <button className="cookie-hint-button" onClick={() => setSettingsOpen(true)}>
                          去设置 → YouTube 访问设置
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
            </div>
          </aside>
        )}
      </section>
      {glossaryOpen && <div className="modal-backdrop" role="presentation">
        <section className="template-modal glossary-modal" role="dialog" aria-modal="true" aria-label="编辑术语表">
          <header><div><p className="eyebrow">术语表 · 全局</p><h2>编辑术语表</h2></div><button onClick={closeGlossary} aria-label="关闭术语表">×</button></header>
          <p className="settings-subfold-note">识别时用来纠正专有名词拼写，翻译时按「原文 = 译文」提示模型。保存后，已完成翻译的任务会提示重新翻译。原文为空的行保存时自动忽略；别名用逗号分隔。</p>
          <div className="glossary-tools">
            <input value={glossaryQuery} placeholder="搜索原文、译文或别名" onChange={(event) => setGlossaryQuery(event.target.value)} />
            <small>共 {glossaryRows.length} 条</small>
            <button onClick={() => { setGlossaryRows((rows) => [toGlossaryRow({}, true), ...rows]); setGlossaryQuery(""); }}>＋ 新增术语</button>
            <button onClick={() => glossaryFileRef.current?.click()}>导入 JSON / CSV</button>
            <button onClick={() => exportGlossary("json")}>导出 JSON</button>
            <button onClick={() => exportGlossary("csv")}>导出 CSV</button>
            <button className="danger" onClick={() => void clearGlossary()} disabled={!glossaryRows.length && !glossaryCount}>清空</button>
            <input ref={glossaryFileRef} type="file" accept=".json,.csv" hidden onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void importGlossary(file); }} />
          </div>
          {glossaryNote && <p className="glossary-note">{glossaryNote}</p>}
          <div className="llm-cost-table glossary-table"><table>
            <thead><tr><th>原文</th><th>译文</th><th>别名</th><th>备注</th><th /></tr></thead>
            <tbody>
              {glossaryRows.filter((row) => [row.source, row.target, row.aliases].some((value) => value.toLowerCase().includes(glossaryQuery.trim().toLowerCase()))).map((row) => {
                const duplicate = glossaryDuplicates.has(glossaryMatchKey(row.source));
                return (
                  <tr key={row.key} className={`${row.dirty ? "is-dirty" : ""} ${duplicate ? "is-duplicate" : ""}`}>
                    <td><input value={row.source} placeholder="原文（必填）" onChange={(event) => updateGlossaryRow(row.key, { source: event.target.value })} />{duplicate && <small className="field-error">原文重复（不区分大小写）</small>}</td>
                    <td><input value={row.target} placeholder="译文" onChange={(event) => updateGlossaryRow(row.key, { target: event.target.value })} /></td>
                    <td><input value={row.aliases} placeholder="逗号分隔" onChange={(event) => updateGlossaryRow(row.key, { aliases: event.target.value })} /></td>
                    <td><input value={row.note} placeholder="给翻译模型的说明" onChange={(event) => updateGlossaryRow(row.key, { note: event.target.value })} /></td>
                    <td><button className="danger" onClick={() => removeGlossaryRow(row)}>删除</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table></div>
          <footer>
            <span className={glossaryUnsaved ? "glossary-unsaved" : "glossary-saved"}>{glossaryUnsaved ? `● ${glossaryUnsaved} 处未保存改动` : "没有未保存的改动"}{glossaryDuplicates.size > 0 && "，有重复原文，改掉后才能保存"}</span>
            <button onClick={closeGlossary}>{glossaryUnsaved ? "放弃改动" : "关闭"}</button>
            <button className="primary" onClick={() => void saveGlossary()} disabled={!glossaryUnsaved || glossaryDuplicates.size > 0}>保存术语表</button>
          </footer>
        </section>
      </div>}
      {settingsOpen && (
        <div className="settings-backdrop" role="presentation" onClick={() => setSettingsOpen(false)}>
        <div className="settings-drawer" role="dialog" aria-modal="true" aria-label="接口与下载设置" onClick={(event) => event.stopPropagation()}>
          <div className="drawer-head">
            <div>
               <p className="eyebrow">本地设置</p>
               <h2>翻译服务与下载访问</h2>
            </div>
            <button onClick={() => setSettingsOpen(false)} aria-label="关闭接口与下载设置">×</button>
          </div>
           <p className="settings-intro">按翻译服务和下载访问分组。密钥只会写入本机 `.env`，不会通过接口重新读取或显示。</p>
           <div className="settings-section-label"><NavIcon name="translate" className="app-icon" /><b>翻译服务</b><span>在线 API / 本地模型</span></div>
          <TranslationPresetFields
            presets={translationPresets}
            presetId={translationPreset || "deepseek"}
            baseUrl={serviceBaseUrl}
            model={options.translate_model}
            apiKey={apiKey}
            disabled={savingSettings}
            onPresetChange={selectTranslationPreset}
            onBaseChange={(value) => { setServiceBaseUrl(value); setPresetDirty(true); }}
            onModelChange={(value) => { updateOptions("translate_model", value); setPresetDirty(true); }}
            onKeyChange={setApiKey}
            onOpenKeyPage={() => openKeyPage(translationPreset || "deepseek")}
          />
          <p className="drawer-note">保存后，下次打开仍使用这里选中的服务。在线密钥只写入本机，不会回显。换到另一个兼容厂商时，会覆盖上一次的兼容密钥。</p>
           <div className="settings-section-label"><NavIcon name="download" className="app-icon" /><b>下载访问</b><span>Cookies、验证和代理</span></div>
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
          <div className="settings-section-label"><NavIcon name="video" className="app-icon" /><b>关注频道</b><span>{weeklyTitle}</span></div>
          <label className="check settings-check">
            <input
              type="checkbox"
              checked={weekly?.enabled ?? false}
              disabled={channelSaving || !weekly}
              onChange={(event) => void setWeeklyEnabled(event.target.checked)}
            />
            启用周热
          </label>
          <p className="drawer-note">打开并添加频道后，素材页会把「关注频道」和「本周搜索」并排。频道页带齐播放量时按播放量排，标题是「本周热片」；播放量缺失时按上传时间排，标题是「本周新片」。列表缓存约 {weekly?.cache_ttl_hours ?? 12} 小时，点刷新会立刻重取。还没有频道时，热片卡片右侧有一个「关注频道」入口。热片列表不依赖这里的开关。</p>
          <div className="settings-weekly-fields">
            {weeklyNumber("window_days", "时间窗口（天）", 1, 30)}
            {weeklyNumber("top_k", "最多显示", 1, 50)}
            {weeklyNumber("fetch_count", "每频道拉取", 1, 50)}
            {weeklyNumber("cache_ttl_hours", "缓存（小时）", 1, 168)}
          </div>
          <label className="check settings-check">
            <input
              type="checkbox"
              checked={weekly?.show_thumbnails !== false}
              disabled={channelSaving || !weekly}
              onChange={(event) => void saveWeeklyPrefs({ show_thumbnails: event.target.checked })}
            />
            显示缩略图
          </label>
          <div className="settings-channels">
            {(weekly?.channels ?? []).map((channel) => (
              <div className="settings-channel-row" key={channel.id}>
                <span>
                  <b>{channel.title || "未命名频道"}</b>
                  <small title={channel.url}>{channel.url}</small>
                </span>
                <button type="button" onClick={() => void removeSubscribedChannel(channel.id)} disabled={channelSaving}>移除</button>
              </div>
            ))}
            {!weekly?.channels.length && <p className="drawer-note">还没有频道。粘贴频道主页或 /videos 链接。</p>}
            <div className="settings-channel-add">
              <input
                value={channelDraft}
                placeholder="https://www.youtube.com/@handle"
                onChange={(event) => setChannelDraft(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void addSubscribedChannel(); } }}
              />
              <button type="button" onClick={() => void addSubscribedChannel()} disabled={channelSaving || !channelDraft.trim()}>添加</button>
            </div>
          </div>
          <div className="settings-diagnostics">
            <div className="settings-section-label"><NavIcon name="book" className="app-icon" /><b>关于</b><span>v{appVersion || "—"}</span></div>
            <p className="drawer-note">诊断包包含最近日志、版本和已脱敏的本机设置，不含密钥、Cookies、登录状态或浏览器资料。</p>
            <button type="button" className="settings-link-button" onClick={() => void exportDiagnostics()} disabled={exportingDiagnostics}>
              {exportingDiagnostics ? "正在导出诊断包…" : "导出诊断包"}
            </button>
            {diagnosticsNote && <p className="drawer-note">{diagnosticsNote}</p>}
          </div>
          <div className="settings-drawer-footer"><button className="primary" onClick={saveSettings} disabled={savingSettings} aria-busy={savingSettings}>
            {savingSettings ? "正在保存本地设置…" : "保存本地设置并应用本次访问参数"}
          </button></div>
        </div>
        </div>
      )}
      {pendingDelete && (
        <div className="modal-backdrop" role="presentation">
          <section className="template-modal delete-preview-modal" role="dialog" aria-modal="true" aria-label="确认删除目标">
            <header>
              <div><p className="eyebrow">删除确认</p><h2>{deleteResults ? "删除结果" : "核对将要删除的内容"}</h2></div>
              <button onClick={() => { setPendingDelete(null); setDeleteResults(null); }} disabled={deleteBusy} aria-label="关闭">×</button>
            </header>
            <p className="delete-preview-summary">
              {pendingDelete.source === "history"
                ? pendingDelete.mode === "record_only"
                  ? `仅删除 ${pendingDelete.plan.task_count} 条任务库记录；磁盘文件保留。`
                  : `已验证 ${pendingDelete.plan.verified_files_count ?? 0} 条输出可删除（${pendingDelete.plan.file_count} 个文件，${formatBytes(pendingDelete.plan.bytes)}）；${pendingDelete.plan.record_only_required_count ?? 0} 条仅删除记录，磁盘文件保留。`
                : `删除 ${pendingDelete.plan.target_count} 个目标、${pendingDelete.plan.file_count} 个文件，预计 ${formatBytes(pendingDelete.plan.bytes)}；已登记的整任务会同步移除记录。`}
            </p>
            <div className="delete-preview-list">
              {(pendingDelete.source === "history" ? pendingDelete.plan.tasks : pendingDelete.plan.targets).map((item) => {
                const id = item.id;
                const result = deleteResults?.find((entry) => entry.id === id);
                const name = pendingDelete.source === "history" ? pendingDelete.plan.tasks.find((row) => row.id === id)?.title : pendingDelete.plan.targets.find((row) => row.id === id)?.name;
                const target = pendingDelete.source === "history"
                  ? pendingDelete.mode === "record_and_outputs" ? pendingDelete.plan.tasks.find((row) => row.id === id)?.output_dir : `任务记录：${id}`
                  : `${pendingDelete.plan.root.replace(/[\\/]$/, "")}\\${id.replace(/\//g, "\\")}`;
                return <div key={id} className="delete-preview-row">
                  <b>{name}</b><small>{target}</small>
                  <span>{result ? result.status === "deleted" ? "已删除" : result.status === "partial" ? "部分完成，请核对磁盘" : "未删除" : ("disposition" in item && item.disposition === "record_only_required") ? "仅删除记录 · 磁盘文件保留" : `${"files" in item ? item.files.length : item.file_count} 个文件 · ${formatBytes(item.bytes)}`}</span>
                  {pendingDelete.source === "history" && pendingDelete.mode === "record_and_outputs" && (item as HistoryDeletePlan["tasks"][number]).reason && <em>{(item as HistoryDeletePlan["tasks"][number]).reason}</em>}
                  {result?.error && <em>{result.error}</em>}
                </div>;
              })}
            </div>
            {!deleteResults && <p className="delete-preview-warning">请核对目标。确认后无法撤销；若预览期间文件发生变化，操作会被拒绝。</p>}
            <footer>
              <button onClick={() => { setPendingDelete(null); setDeleteResults(null); }} disabled={deleteBusy}>{deleteResults ? "关闭" : "取消"}</button>
              {!deleteResults && <button className="danger" onClick={() => void confirmPendingDelete()} disabled={deleteBusy}>{deleteBusy ? "正在删除…" : "确认删除"}</button>}
            </footer>
          </section>
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
            <p className="dependency-intro">默认流程只强制需要 FFmpeg。模型、OCR 与投稿浏览器按你实际使用的功能安装；状态会在应用运行期间自动刷新，安装完成后无需重启即可使用。</p>
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
                      {dependency.available ? <span>可用</span> : dependency.id === "tesseract" && dependency.installer_available ? (
                        <button className="primary small" disabled={tesseractInstallerBusy} onClick={() => void startTesseractInstaller()}>{tesseractInstallerBusy ? "等待安装完成…" : "启动安装器"}</button>
                      ) : dependency.installable ? (
                        <button className="primary small" disabled={Boolean(dependencyJob && ["queued", "running", "cancelling"].includes(dependencyJob.status))} onClick={() => void installDependency(dependency)}>
                          {active ? dependencyJob?.status === "cancelling" ? "取消中" : dependency.action_label === "下载模型" ? "下载中" : "安装中" : dependencyJob?.dependency_id === dependency.id && ["failed", "cancelled"].includes(dependencyJob.status) ? "重试" : dependency.action_label ?? "安装"}
                        </button>
                      ) : dependency.action_url ? (
                        <a href={dependency.action_url} target="_blank" rel="noreferrer">{dependency.action_label ?? "安装说明"}</a>
                      ) : dependency.id === "ollama-runtime" && dependency.state === "stopped" ? (
                        <button className="primary small" disabled={ollamaStartBusy} onClick={() => void startOllama()}>
                          {ollamaStartBusy ? "启动中…" : "启动 Ollama"}
                        </button>
                      ) : <span>{dependency.state === "unavailable" ? "请重新检测" : dependency.depends_on ? "等待运行环境" : "需手动安装"}</span>}
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
              <button onClick={() => void loadDependencies(true)}>重新检测</button>
              <button className="primary" onClick={() => setDependenciesOpen(false)}>完成</button>
            </footer>
          </section>
        </div>
      )}
      {setupOpen && (
        <SetupModal
          presets={translationPresets}
          presetId={setupPreset}
          baseUrl={setupBase}
          model={setupModel}
          ffmpegReady={ffmpegReady}
          modelReady={whisperReady}
          keyValue={setupKey}
          busy={setupBusy}
          active={setupActive}
          progress={setupProgress}
          error={setupError}
          onPresetChange={selectSetupPreset}
          onBaseChange={setSetupBase}
          onModelChange={setSetupModel}
          onKeyChange={setSetupKey}
          onOpenKeyPage={() => openKeyPage(setupPreset)}
          onInstall={() => void finishSetup()}
          onSkip={dismissSetup}
        />
      )}
      {tutorialOpen && <TutorialModal onClose={closeTutorial} onAction={handleTutorialAction} />}
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
              <dl className="shortcut-list"><div><dt>Ctrl + O</dt><dd>选择本地视频（先确认授权）</dd></div><div><dt>Ctrl + L</dt><dd>回到素材区并聚焦视频链接</dd></div><div><dt>Ctrl + R</dt><dd>执行顶部主操作（检查准备 / 继续下一阶段）</dd></div><div><dt>Alt + ↑ / ↓</dt><dd>字幕页切换上一条 / 下一条（输入框内不触发）</dd></div><div><dt>空格</dt><dd>字幕页播放或暂停当前字幕（输入框内不触发）</dd></div><div><dt>Ctrl + S</dt><dd>保存字幕草稿</dd></div><div><dt>Esc</dt><dd>只中断当前运行阶段，保留已完成检查点</dd></div></dl>
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
        <span>视频、字幕、浏览器 Profile 与密钥默认保存在本机；在线服务按确认内容处理请求。 <a href="https://github.com/wanghaofu124/youtube-bili-localizer/issues/new" target="_blank" rel="noreferrer">报告不当或错误的 AI 输出</a>（公开 Issue 请勿粘贴敏感信息）</span>
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

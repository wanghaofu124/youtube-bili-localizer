export type StageName = "acquire" | "extract" | "translate" | "render" | "dub" | "publish";

const order: StageName[] = ["acquire", "extract", "translate", "render", "dub", "publish"];

/** Treat API null and UI empty string as the same for optional text fields. */
export function sameTaskOption(_key: string, current: unknown, saved: unknown): boolean {
  const empty = (value: unknown) => value == null || value === "";
  if (empty(current) && empty(saved)) return true;
  // Public API maps ASS colors <-> Chinese names; both forms mean the same choice.
  const colorAliases: Record<string, string> = {
    白色: "&H00FFFFFF",
    黄色: "&H0000FFFF",
    青色: "&H00FFFF00",
    绿色: "&H0000FF00",
    黑色: "&H00000000",
    灰色: "&H00808080",
    蓝色: "&H00FF0000",
  };
  const normalizeColor = (value: unknown) => {
    if (typeof value !== "string") return value;
    const trimmed = value.trim();
    const upper = trimmed.toUpperCase();
    if (colorAliases[trimmed]) return colorAliases[trimmed];
    if (upper.startsWith("&H")) return upper;
    return trimmed;
  };
  if (_key === "subtitle_color" || _key === "subtitle_outline_color") {
    return normalizeColor(current) === normalizeColor(saved);
  }
  return JSON.stringify(current) === JSON.stringify(saved);
}

export function stageAvailability(
  stage: StageName,
  statuses: Partial<Record<StageName, string>>,
  checksReady: boolean,
  running: boolean,
  publishEnabled: boolean,
): { enabled: boolean; reason: string } {
  if (statuses[stage] === "completed") return { enabled: true, reason: "" };
  // "dub" only exists in statuses when the job has dubbing enabled.
  const previous = order.slice(0, order.indexOf(stage)).filter((name) => name !== "dub" || name in statuses);
  if (!checksReady) return { enabled: false, reason: "先通过准备检查" };
  if (!previous.every((name) => statuses[name] === "completed")) return { enabled: false, reason: "先完成上一个阶段" };
  if (stage === "publish" && !publishEnabled) return { enabled: false, reason: "处理方案中未启用投稿辅助" };
  if (running) return { enabled: false, reason: "另一个阶段正在运行" };
  return { enabled: true, reason: "" };
}

export function publishStageButton(input: {
  publishEnabled: boolean;
  renderStatus?: string;
  publishStatus?: string;
}): { enabled: boolean; label: string; action: "cancel" | "open" | "run"; reason: string } {
  const publishStatus = input.publishStatus ?? "pending";
  const renderDone = input.renderStatus === "completed";
  if (publishStatus === "running") {
    return { enabled: true, label: "取消当前阶段", action: "cancel", reason: "" };
  }
  if (publishStatus === "completed") {
    return { enabled: true, label: "打开投稿页", action: "open", reason: "" };
  }
  if (!input.publishEnabled && renderDone) {
    return { enabled: true, label: "打开投稿页", action: "open", reason: "" };
  }
  if (["failed", "cancelled", "stale", "interrupted"].includes(publishStatus)) {
    if (!input.publishEnabled) {
      return { enabled: false, label: "重试", action: "run", reason: "处理方案中未启用投稿辅助" };
    }
    return { enabled: renderDone, label: "重试", action: "run", reason: renderDone ? "" : "先完成上一个阶段" };
  }
  if (!input.publishEnabled) {
    return { enabled: false, label: "开始", action: "run", reason: "处理方案中未启用投稿辅助" };
  }
  return { enabled: renderDone, label: "开始", action: "run", reason: renderDone ? "" : "先完成上一个阶段" };
}

export function workflowPrimaryLabel(input: {
  running: boolean;
  hasJob: boolean;
  hasChecks: boolean;
  hasBlocking: boolean;
  canResume: boolean;
  nextStage: StageName | null;
  stageLabel: (stage: StageName) => string;
}): string {
  if (input.running) return "中断当前阶段";
  if (!input.hasJob) return "保存任务草稿";
  if (!input.hasChecks) return "检查准备";
  if (input.hasBlocking) return "重新检查准备";
  if (input.canResume) return "继续中断阶段";
  if (input.nextStage) return `继续${input.stageLabel(input.nextStage)}`;
  return "查看字幕成片";
}

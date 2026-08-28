export type StageName = "acquire" | "extract" | "translate" | "render" | "publish";

const order: StageName[] = ["acquire", "extract", "translate", "render", "publish"];

export function stageAvailability(
  stage: StageName,
  statuses: Partial<Record<StageName, string>>,
  checksReady: boolean,
  running: boolean,
  publishEnabled: boolean,
): { enabled: boolean; reason: string } {
  const previous = order.slice(0, order.indexOf(stage));
  if (!checksReady) return { enabled: false, reason: "先通过准备检查" };
  if (!previous.every((name) => statuses[name] === "completed")) return { enabled: false, reason: "先完成上一个阶段" };
  if (stage === "publish" && !publishEnabled) return { enabled: false, reason: "处理方案中未启用投稿辅助" };
  if (running) return { enabled: false, reason: "另一个阶段正在运行" };
  return { enabled: true, reason: "" };
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

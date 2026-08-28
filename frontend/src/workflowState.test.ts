import { describe, expect, it } from "vitest";
import { stageAvailability, workflowPrimaryLabel } from "./workflowState";

describe("staged workflow interactions", () => {
  it("explains why a stage is disabled instead of silently disabling it", () => {
    expect(stageAvailability("extract", { acquire: "pending" }, true, false, false)).toEqual({
      enabled: false,
      reason: "先完成上一个阶段",
    });
    expect(stageAvailability("acquire", {}, false, false, false).reason).toBe("先通过准备检查");
  });

  it("enables only the next stage after preparation and its upstream checkpoint", () => {
    expect(stageAvailability("extract", { acquire: "completed" }, true, false, false).enabled).toBe(true);
    expect(stageAvailability("publish", { acquire: "completed", extract: "completed", translate: "completed", render: "completed" }, true, false, false).reason).toContain("未启用");
  });

  it("changes the main action to match the current workflow state", () => {
    const label = workflowPrimaryLabel({
      running: false, hasJob: true, hasChecks: true, hasBlocking: false,
      canResume: false, nextStage: "translate", stageLabel: () => "翻译",
    });
    expect(label).toBe("继续翻译");
  });
});

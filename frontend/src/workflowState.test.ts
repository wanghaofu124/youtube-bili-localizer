import { describe, expect, it } from "vitest";
import { publishStageButton, sameTaskOption, stageAvailability, workflowPrimaryLabel } from "./workflowState";

describe("sameTaskOption", () => {
  it("treats null and empty string as equal for optional text", () => {
    expect(sameTaskOption("title", "", null)).toBe(true);
    expect(sameTaskOption("title", null, "")).toBe(true);
    expect(sameTaskOption("source_language", "", null)).toBe(true);
    expect(sameTaskOption("translate_model", "", undefined)).toBe(true);
  });

  it("equates ASS color codes with Chinese UI names", () => {
    expect(sameTaskOption("subtitle_color", "白色", "&H00FFFFFF")).toBe(true);
    expect(sameTaskOption("subtitle_outline_color", "黑色", "&H00000000")).toBe(true);
    expect(sameTaskOption("subtitle_color", "黄色", "&H00FFFFFF")).toBe(false);
  });
});

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

  it("keeps completed stage results accessible before a new preparation check", () => {
    expect(stageAvailability("acquire", { acquire: "completed" }, false, false, false).enabled).toBe(true);
  });

  it("changes the main action to match the current workflow state", () => {
    const label = workflowPrimaryLabel({
      running: false, hasJob: true, hasChecks: true, hasBlocking: false,
      canResume: false, nextStage: "translate", stageLabel: () => "翻译",
    });
    expect(label).toBe("继续翻译");
  });

  it("opens the publish page after render even when pipeline assist is off", () => {
    expect(publishStageButton({
      publishEnabled: false,
      renderStatus: "completed",
      publishStatus: "pending",
    })).toEqual({ enabled: true, label: "打开投稿页", action: "open", reason: "" });
    expect(publishStageButton({
      publishEnabled: true,
      renderStatus: "completed",
      publishStatus: "pending",
    })).toEqual({ enabled: true, label: "开始", action: "run", reason: "" });
  });
});

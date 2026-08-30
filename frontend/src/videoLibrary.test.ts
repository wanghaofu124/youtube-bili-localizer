import { describe, expect, it } from "vitest";
import { videoLibraryCategory, videoLibraryMatches } from "./videoLibrary";

const item = {
  id: "job-1",
  title: "授权演示视频",
  status: "failed",
  stage: "翻译",
  error: "API 暂时不可用",
};

describe("video library", () => {
  it("separates active, completed and recoverable work", () => {
    expect(videoLibraryCategory({ ...item, status: "running" }, false)).toBe("active");
    expect(videoLibraryCategory({ ...item, status: "completed" }, false)).toBe("completed");
    expect(videoLibraryCategory(item, true)).toBe("attention");
  });

  it("searches title, stage and error without changing the data", () => {
    expect(videoLibraryMatches(item, "授权演示")).toBe(true);
    expect(videoLibraryMatches(item, "翻译")).toBe(true);
    expect(videoLibraryMatches(item, "API 暂时")).toBe(true);
    expect(videoLibraryMatches(item, "不存在的内容")).toBe(false);
  });
});

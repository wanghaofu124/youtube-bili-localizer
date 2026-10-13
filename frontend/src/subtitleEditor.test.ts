import { describe, expect, it } from "vitest";
import { cueNeedsReview, replaceLiteral } from "./subtitleEditor";

describe("subtitle editor helpers", () => {
  it("includes automatic issues and saved manual marks in the review queue", () => {
    expect(cueNeedsReview({ issues: ["reading-speed"] })).toBe(true);
    expect(cueNeedsReview({ needs_review: true })).toBe(true);
    expect(cueNeedsReview({})).toBe(false);
    expect(cueNeedsReview({ needs_review: true, deleted: true })).toBe(false);
  });

  it("previews literal replacements without interpreting regex characters", () => {
    expect(replaceLiteral("Siftite siftite", "siftite", "晶石")).toEqual({
      value: "晶石 晶石",
      count: 2,
    });
    expect(replaceLiteral("a+b and a+b", "a+b", "fixed", true)).toEqual({
      value: "fixed and fixed",
      count: 2,
    });
    expect(replaceLiteral("unchanged", "", "x")).toEqual({ value: "unchanged", count: 0 });
  });
});

import { describe, expect, it, vi } from "vitest";
import { createSiteRequests } from "./siteRequests";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

describe("site requests", () => {
  it("saves the last chosen region after an older refresh and reads the saved choice", async () => {
    const requests = createSiteRequests();
    const oldRefresh = deferred<void>();
    let savedRegion = "US";
    const order: string[] = [];
    const refresh = requests.write(async () => {
      await oldRefresh.promise;
      savedRegion = "US";
      order.push("refresh US");
      return "US videos";
    });
    const first = requests.write(async () => { savedRegion = "JP"; order.push("JP"); return savedRegion; });
    const last = requests.write(async () => { savedRegion = "GB"; order.push("GB"); return savedRegion; });
    const read = requests.read(async () => savedRegion);
    expect(order).toEqual([]);
    oldRefresh.resolve();
    expect(await refresh).toBeNull();
    expect(await first).toBeNull();
    await last;
    expect(await read).toBe("GB");
    expect(order).toEqual(["refresh US", "JP", "GB"]);
    expect(savedRegion).toBe("GB");
  });

  it("ignores an older category response and an older error", async () => {
    const requests = createSiteRequests();
    const old = deferred<string>();
    const started = deferred<void>();
    const first = requests.read(async () => { started.resolve(); return old.promise; });
    await started.promise;
    expect(await requests.read(async () => "gaming")).toBe("gaming");
    old.reject(new Error("old category failed"));
    expect(await first).toBeNull();
  });

  it("does not send a read invalidated by a newer region selection", async () => {
    const requests = createSiteRequests();
    const save = deferred<string>();
    const first = requests.write(() => save.promise);
    const load = vi.fn(async () => "old region");
    const read = requests.read(load);
    const last = requests.write(async () => "JP");
    save.resolve("US");
    await first;
    expect(await read).toBeNull();
    expect(load).not.toHaveBeenCalled();
    expect(await last).toBe("JP");
  });

  it("reports a current save failure and still permits the next choice", async () => {
    const requests = createSiteRequests();
    await expect(requests.write(async () => { throw new Error("offline"); })).rejects.toThrow("offline");
    expect(await requests.write(async () => "JP")).toBe("JP");
  });
});

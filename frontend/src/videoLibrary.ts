export type VideoLibraryCategory = "active" | "attention" | "completed";

export type VideoLibraryItem = {
  id: string;
  title: string;
  status: string;
  stage: string;
  error: string | null;
};

const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);
const COMPLETED_STATUSES = new Set(["completed"]);

export function videoLibraryCategory(
  item: VideoLibraryItem,
  resumable: boolean,
): VideoLibraryCategory {
  if (ACTIVE_STATUSES.has(item.status)) return "active";
  if (COMPLETED_STATUSES.has(item.status)) return "completed";
  if (resumable || item.status !== "completed") return "attention";
  return "completed";
}

export function videoLibraryMatches(
  item: VideoLibraryItem,
  query: string,
): boolean {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return true;
  return [item.title, item.stage, item.status, item.error ?? ""]
    .join("\n")
    .toLocaleLowerCase()
    .includes(needle);
}

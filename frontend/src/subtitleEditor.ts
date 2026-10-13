export type ReviewCue = {
  issues?: string[];
  needs_review?: boolean;
  deleted?: boolean;
};

export function cueNeedsReview(cue: ReviewCue): boolean {
  return !cue.deleted && (Boolean(cue.needs_review) || Boolean(cue.issues?.length));
}

export function replaceLiteral(
  text: string,
  search: string,
  replacement: string,
  caseSensitive = false,
): { value: string; count: number } {
  if (!search) return { value: text, count: 0 };
  const escaped = search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matcher = new RegExp(escaped, caseSensitive ? "g" : "gi");
  let count = 0;
  const value = text.replace(matcher, () => {
    count += 1;
    return replacement;
  });
  return { value, count };
}

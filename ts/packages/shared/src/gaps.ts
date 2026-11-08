import type { QueryResponse } from "./schemas.ts";

export interface DataGap {
  id: "no_braking_data" | "no_lap_set" | "accuracy_limit";
  label: string;
}

const RULES: Array<{ gap: DataGap; pattern: RegExp }> = [
  {
    gap: { id: "no_braking_data", label: "No braking data was found" },
    pattern: /no braking data was found/i,
  },
  {
    gap: { id: "no_lap_set", label: "A driver set no lap" },
    pattern: /stopped on track|set no (?:lap )?time|did not make q3|no laps? recorded/i,
  },
  {
    gap: { id: "accuracy_limit", label: "Within position-accuracy limits" },
    pattern: /within the ~?15 ?m accuracy|section gaps? .{0,40}accurate/i,
  },
];

/**
 * Which honest-handling notes apply to this result. Read from the answer first, then from the evidence the
 * model was given, so a gap the context mentions is shown even if the answer words it differently.
 */
export function dataGaps(result: Pick<QueryResponse, "answer" | "sources">): DataGap[] {
  const evidence = [result.answer, ...result.sources.map((s) => s.text)].join("\n");
  return RULES.filter(({ pattern }) => pattern.test(evidence)).map(({ gap }) => gap);
}

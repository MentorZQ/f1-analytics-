import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  ApiErrorSchema,
  QueryRequestSchema,
  QueryResponseSchema,
  chartTitle,
  dataGaps,
  describeSection,
  lapGap,
  sectionBars,
  seconds,
  type Deltas,
} from "../src/index.ts";

// The real saved answers the Python server returns, used as a contract fixture.
const EXAMPLES_PATH = fileURLToPath(new URL("../../../../web/examples.json", import.meta.url));
const examples: Array<Record<string, unknown>> = JSON.parse(readFileSync(EXAMPLES_PATH, "utf8"));

describe("response contract against real saved answers", () => {
  it("has examples to test against", () => {
    expect(examples.length).toBeGreaterThan(0);
  });

  it.each(examples.map((e, i) => [i, e.question as string] as const))("example %i parses: %s", (i) => {
    const example = examples[i]!;
    const parsed = QueryResponseSchema.safeParse({ ...example, paid_by: "saved" });
    expect(parsed.success, parsed.success ? "" : JSON.stringify(parsed.error.issues)).toBe(true);
  });

  it("rejects a response missing a required field", () => {
    const { answer: _answer, ...rest } = examples[0]!;
    expect(QueryResponseSchema.safeParse({ ...rest, paid_by: "saved" }).success).toBe(false);
  });

  it("rejects an unknown paid_by value", () => {
    expect(QueryResponseSchema.safeParse({ ...examples[0], paid_by: "mystery" }).success).toBe(false);
  });

  it("accepts a routed answer: a router_fact source, no deltas, no free_remaining", () => {
    const routed = {
      answer: "RUS (Mercedes) qualified P1, before any grid penalties.",
      session: "barcelona",
      sources: [{ chunk_type: "router_fact", section: null, drivers: ["RUS"], text: "RUS (Mercedes), qualifying position P1" }],
      deltas: null,
      paid_by: "router",
      metrics: { retrieve_s: 0, llm_s: 0, total_s: 0.0021, input_tokens: 0, output_tokens: 0, model: null, routed: "position" },
    };
    const parsed = QueryResponseSchema.safeParse(routed);
    expect(parsed.success, parsed.success ? "" : JSON.stringify(parsed.error.issues)).toBe(true);
  });

  it("accepts the real refusal shape from the server, metrics included (model is null, filtered is true)", () => {
    const refusal = {
      answer: "I can only answer questions about the qualifying data.",
      session: "barcelona",
      sources: [],
      deltas: null,
      paid_by: "none",
      metrics: { retrieve_s: 0, deltas_s: 0, llm_s: 0, total_s: 0, input_tokens: 0, output_tokens: 0, prompt_chars: 0, model: null, filtered: true },
    };
    expect(QueryResponseSchema.safeParse(refusal).success).toBe(true);
  });

  it("accepts the refusal shape: no sources, no deltas", () => {
    const refusal = { answer: "x", session: null, sources: [], deltas: null, paid_by: "none" };
    expect(QueryResponseSchema.safeParse(refusal).success).toBe(true);
  });
});

describe("request and error shapes", () => {
  it("trims and rejects an empty question", () => {
    expect(QueryRequestSchema.safeParse({ question: "   " }).success).toBe(false);
    expect(QueryRequestSchema.parse({ question: "  hi  " }).question).toBe("hi");
  });

  it("rejects a question over the length limit", () => {
    expect(QueryRequestSchema.safeParse({ question: "x".repeat(501) }).success).toBe(false);
  });

  it("parses the server's error envelope", () => {
    const body = { detail: { code: "invalid_key", message: "no" } };
    expect(ApiErrorSchema.parse(body).detail.code).toBe("invalid_key");
  });
});

describe("chart helpers", () => {
  const deltas: Deltas = {
    driver_a: "RUS",
    driver_b: "HAM",
    lap_gap_s: 0.063,
    lap_gap_exact_s: 0.064,
    sections: [
      { section: "T1-T2", type: "corner_sequence", gain_s: -0.092, start_m: 634, end_m: 993, retrieved: true },
      { section: "T3", type: "corner", gain_s: 0.078, start_m: 993, end_m: 1304, retrieved: false },
      { section: "T4", type: "corner", gain_s: 0, start_m: null, end_m: null, retrieved: false },
      { section: "T5", type: "corner", gain_s: 0.01, start_m: 1, end_m: 2, retrieved: false },
    ],
  };

  it("keeps all three decimals and drops the sign", () => {
    expect(seconds(-0.064)).toBe("0.064s");
    expect(seconds(0.5)).toBe("0.500s");
  });

  it("prefers the exact lap gap over the section sum", () => {
    expect(lapGap(deltas)).toBe(0.064);
    expect(lapGap({ ...deltas, lap_gap_exact_s: null })).toBe(0.063);
    expect(lapGap({ ...deltas, lap_gap_exact_s: undefined })).toBe(0.063);
  });

  it("names the leader from the sign of the gap", () => {
    expect(chartTitle(deltas)).toBe("RUS 0.064s faster over the lap (RUS vs HAM)");
    expect(chartTitle({ ...deltas, lap_gap_exact_s: -0.2 })).toContain("HAM 0.200s faster");
  });

  it("describes sections, including level ones and ones with no range", () => {
    expect(describeSection(deltas.sections[0]!, deltas)).toBe("T1-T2 (634–993m): HAM faster by 0.092s");
    expect(describeSection(deltas.sections[2]!, deltas)).toBe("T4: level");
  });

  it("scales the largest swing to 38 and labels only the top three", () => {
    const bars = sectionBars(deltas);
    expect(bars[0]!.widthPct).toBeCloseTo(38);
    expect(bars.filter((b) => b.labelled)).toHaveLength(3);
    expect(bars[2]!.labelled).toBe(false); // the zero-gain section
    expect(bars[0]!.forA).toBe(false);
  });

  it("does not divide by zero when every gap is zero", () => {
    const flat = { ...deltas, sections: deltas.sections.map((s) => ({ ...s, gain_s: 0 })) };
    expect(sectionBars(flat).every((b) => b.widthPct === 0)).toBe(true);
  });
});

describe("data-gap badges", () => {
  const base = { answer: "RUS was faster.", sources: [] };

  it("shows nothing for a plain answer", () => {
    expect(dataGaps(base)).toEqual([]);
  });

  it("flags missing braking data from the answer", () => {
    const gaps = dataGaps({ ...base, answer: "No braking data was found for either driver." });
    expect(gaps.map((g) => g.id)).toEqual(["no_braking_data"]);
  });

  it("flags a stopped-on-track driver from the evidence", () => {
    const sources = [{ chunk_type: "driver_dnq_status", section: null, drivers: ["X"], text: "X stopped on track." }];
    expect(dataGaps({ ...base, sources }).map((g) => g.id)).toEqual(["no_lap_set"]);
  });

  it("works on the real saved examples without throwing", () => {
    for (const e of examples) {
      const parsed = QueryResponseSchema.parse({ ...e, paid_by: "saved" });
      expect(Array.isArray(dataGaps(parsed))).toBe(true);
    }
  });
});

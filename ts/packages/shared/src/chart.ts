import type { Deltas, Section } from "./schemas.ts";

/** Formats a signed gap as an unsigned figure with all three decimals, e.g. 0.064s. Never rounds further. */
export const seconds = (value: number): string => `${Math.abs(value).toFixed(3)}s`;

/** The lap gap to show: the exact timing figure when present, else the sum of interpolated sections. */
export function lapGap(deltas: Deltas): number {
  return deltas.lap_gap_exact_s ?? deltas.lap_gap_s;
}

export function leader(deltas: Deltas): string {
  return lapGap(deltas) >= 0 ? deltas.driver_a : deltas.driver_b;
}

export function chartTitle(deltas: Deltas): string {
  return `${leader(deltas)} ${seconds(lapGap(deltas))} faster over the lap (${deltas.driver_a} vs ${deltas.driver_b})`;
}

export function fasterDriver(section: Section, deltas: Deltas): string | null {
  if (section.gain_s === 0) return null;
  return section.gain_s > 0 ? deltas.driver_a : deltas.driver_b;
}

export function describeSection(section: Section, deltas: Deltas): string {
  const faster = fasterDriver(section, deltas);
  if (faster === null) return `${section.section}: level`;
  const range =
    section.start_m === null || section.end_m === null
      ? ""
      : ` (${Math.round(section.start_m)}–${Math.round(section.end_m)}m)`;
  return `${section.section}${range}: ${faster} faster by ${seconds(section.gain_s)}`;
}

export interface Bar {
  section: Section;
  description: string;
  /** Bar length as a percentage of the track half-width; the largest swing is capped at 38. */
  widthPct: number;
  forA: boolean;
  labelled: boolean;
}

/** Bar geometry for the section-gap chart. Only the three biggest swings carry a value label. */
export function sectionBars(deltas: Deltas): Bar[] {
  const largest = Math.max(...deltas.sections.map((s) => Math.abs(s.gain_s)), 0.001);
  const top3 = new Set(
    [...deltas.sections].sort((x, y) => Math.abs(y.gain_s) - Math.abs(x.gain_s)).slice(0, 3),
  );
  return deltas.sections.map((section) => ({
    section,
    description: describeSection(section, deltas),
    widthPct: (Math.abs(section.gain_s) / largest) * 38,
    forA: section.gain_s >= 0,
    labelled: top3.has(section),
  }));
}

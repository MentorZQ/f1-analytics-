import { useState } from "react";
import { chartTitle, fasterDriver, sectionBars, seconds, type Deltas } from "@f1rag/shared";

export function Chart({ deltas }: { deltas: Deltas }) {
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const { driver_a: a, driver_b: b } = deltas;
  const bars = sectionBars(deltas);

  return (
    <figure className="chart" style={{ marginInline: 0, marginBottom: 0 }}>
      <figcaption className="chart-title">{chartTitle(deltas)}</figcaption>
      <div className="chart-legend">
        <span className="legend-item"><span className="swatch swatch-a" />{a} faster</span>
        <span className="legend-item"><span className="swatch swatch-b" />{b} faster</span>
        <span>Bold sections were used in the answer</span>
      </div>

      <div className="bars">
        {bars.map(({ section, description, widthPct, forA, labelled }) => (
          <div
            className="bar-row"
            key={section.section}
            tabIndex={0}
            role="img"
            aria-label={description}
            onPointerMove={(e) => setTip({ text: description, x: e.clientX, y: e.clientY })}
            onPointerLeave={() => setTip(null)}
            onBlur={() => setTip(null)}
          >
            <span className={`bar-label${section.retrieved ? " is-cited" : ""}`}>{section.section}</span>
            <div className="bar-track">
              <div className={`bar ${forA ? "bar-a" : "bar-b"}`} style={{ width: `${widthPct}%` }} />
              {labelled && (
                <span
                  className="bar-value"
                  style={forA ? { right: `calc(50% + ${widthPct}% + 4px)` } : { left: `calc(50% + ${widthPct}% + 4px)` }}
                >
                  {seconds(section.gain_s)}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      <details>
        <summary>Show as a table</summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Section</th><th>Faster</th><th className="number">Gap</th></tr></thead>
            <tbody>
              {deltas.sections.map((s) => (
                <tr key={s.section}>
                  <td>{s.section}</td>
                  <td>{fasterDriver(s, deltas) ?? "Level"}</td>
                  <td className="number">{seconds(s.gain_s)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {tip && (
        <div className="tooltip" role="status" style={{ left: tip.x + 12, top: Math.max(8, tip.y - 40) }}>
          {tip.text}
        </div>
      )}
    </figure>
  );
}

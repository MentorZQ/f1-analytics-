import type { Source } from "@f1rag/shared";

export function Sources({ sources }: { sources: Source[] }) {
  return (
    <details>
      <summary>Telemetry behind this answer ({sources.length} sources)</summary>
      {sources.map((s, i) => (
        <div className="source" key={i}>
          <p className="source-title">
            {[s.section, s.drivers.join(" vs ")].filter(Boolean).join(" · ") || s.chunk_type}
          </p>
          <pre>{s.text}</pre>
        </div>
      ))}
    </details>
  );
}

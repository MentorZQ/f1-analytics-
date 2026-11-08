import type { ReactNode } from "react";

// Everything here is built as React elements from plain strings. Nothing is parsed as HTML, so markup inside an
// answer or a retrieved chunk can't run on a page that may hold an API key.

function inline(text: string, keyBase: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*)/)
    .filter(Boolean)
    .map((part, i) =>
      part.startsWith("**") && part.endsWith("**") && part.length > 4
        ? <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>
        : part,
    );
}

const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
const isTableRow = (l: string) => /^\s*\|/.test(l);
const isDivider = (l: string) => /^\s*\|?[\s:|-]+\|?\s*$/.test(l);
const isListItem = (l: string) => /^\s*([-*]|\d+\.)\s/.test(l);
const startsBlock = (l: string) => /^(#{1,6}\s|\s*\||\s*([-*]|\d+\.)\s)/.test(l);

export function parseMarkdown(source: string): ReactNode[] {
  const lines = source.split("\n");
  const out: ReactNode[] = [];
  let i = 0;
  let n = 0;

  while (i < lines.length) {
    const line = lines[i] ?? "";
    const key = `b${n++}`;

    if (!line.trim()) {
      i += 1;
    } else if (/^#{1,6}\s/.test(line)) {
      out.push(<h3 key={key}>{inline(line.replace(/^#+\s*/, ""), key)}</h3>);
      i += 1;
    } else if (isTableRow(line) && isDivider(lines[i + 1] ?? "")) {
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i] ?? "")) {
        rows.push(cells(lines[i] ?? ""));
        i += 1;
      }
      out.push(
        <div className="table-wrap" key={key}>
          <table>
            <thead><tr>{head.map((c, j) => <th key={j}>{inline(c, `${key}h${j}`)}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c, `${key}r${ri}c${j}`)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
    } else if (isListItem(line)) {
      const ordered = /^\s*\d/.test(line);
      const items: string[] = [];
      while (i < lines.length && isListItem(lines[i] ?? "")) {
        items.push((lines[i] ?? "").replace(/^\s*([-*]|\d+\.)\s+/, ""));
        i += 1;
      }
      const li = items.map((t, j) => <li key={j}>{inline(t, `${key}i${j}`)}</li>);
      out.push(ordered ? <ol key={key}>{li}</ol> : <ul key={key}>{li}</ul>);
    } else {
      const start = i;
      const parts: ReactNode[] = [];
      while (i < lines.length && (lines[i] ?? "").trim() && (i === start || !startsBlock(lines[i] ?? ""))) {
        parts.push(...inline((i > start ? " " : "") + (lines[i] ?? "").trim(), `${key}p${i}`));
        i += 1;
      }
      out.push(<p key={key}>{parts}</p>);
    }
  }
  return out;
}

export function Markdown({ source }: { source: string }) {
  return <div className="answer">{parseMarkdown(source)}</div>;
}

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ask } from "../src/api.ts";
import { Chart } from "../src/Chart.tsx";
import { Markdown } from "../src/Markdown.tsx";
import { Sources } from "../src/Sources.tsx";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("Markdown never turns text into markup", () => {
  it("escapes HTML in an answer", () => {
    const html = renderToStaticMarkup(<Markdown source={'<img src=x onerror="alert(1)"> and <script>x</script>'} />);
    expect(html).not.toContain("<img");
    expect(html).not.toContain("<script");
    expect(html).toContain("&lt;img");
  });

  it("renders bold, lists, headings and tables", () => {
    const html = renderToStaticMarkup(
      <Markdown source={"## Head\n\n**RUS** was faster.\n\n- one\n- two\n\n| A | B |\n|---|---|\n| 1 | 2 |"} />,
    );
    expect(html).toContain("<h3>Head</h3>");
    expect(html).toContain("<strong>RUS</strong>");
    expect(html).toContain("<ul><li>one</li><li>two</li></ul>");
    expect(html).toContain("<th>A</th>");
    expect(html).toContain("<td>2</td>");
  });

  it("joins wrapped lines into one paragraph", () => {
    expect(renderToStaticMarkup(<Markdown source={"line one\nline two"} />)).toContain("<p>line one line two</p>");
  });
});

describe("Chart", () => {
  const deltas = {
    driver_a: "RUS",
    driver_b: "HAM",
    lap_gap_s: 0.063,
    lap_gap_exact_s: 0.064,
    sections: [
      { section: "T1-T2", type: "corner_sequence", gain_s: -0.092, start_m: 634, end_m: 993, retrieved: true },
      { section: "T3", type: "corner", gain_s: 0.078, start_m: 993, end_m: 1304, retrieved: false },
    ],
  };

  it("titles the chart with the exact gap, not the section sum", () => {
    const html = renderToStaticMarkup(<Chart deltas={deltas} />);
    expect(html).toContain("RUS 0.064s faster over the lap (RUS vs HAM)");
  });

  it("marks sections used in the answer and lists every value in the table", () => {
    const html = renderToStaticMarkup(<Chart deltas={deltas} />);
    expect(html).toContain("is-cited");
    expect(html).toContain("0.092s");
    expect(html).toContain("0.078s");
  });
});

describe("Sources", () => {
  it("shows chunk text as text", () => {
    const html = renderToStaticMarkup(
      <Sources sources={[{ chunk_type: "x", section: "T1", drivers: ["A", "B"], text: "<b>hi</b>" }]} />,
    );
    expect(html).toContain("&lt;b&gt;hi&lt;/b&gt;");
    expect(html).toContain("T1 · A vs B");
  });
});

describe("ask()", () => {
  const good = { answer: "a", session: "monza", sources: [], deltas: null, paid_by: "free", free_remaining: 4 };

  it("returns validated data", async () => {
    const r = await ask("q", "monza", null, vi.fn(async () => json(200, good)));
    expect(r).toMatchObject({ ok: true });
  });

  it("sends the key header only when there is a key", async () => {
    const fake = vi.fn<typeof fetch>(async () => json(200, good));
    await ask("q", null, "sk-ant-x", fake);
    await ask("q", null, null, fake);
    const first = (fake.mock.calls[0]![1] as RequestInit).headers as Record<string, string>;
    const second = (fake.mock.calls[1]![1] as RequestInit).headers as Record<string, string>;
    expect(first["X-Anthropic-Key"]).toBe("sk-ant-x");
    expect(second).not.toHaveProperty("X-Anthropic-Key");
  });

  it("surfaces a typed server error", async () => {
    const r = await ask("q", null, null, vi.fn(async () => json(401, { detail: { code: "invalid_key", message: "no" } })));
    expect(r).toEqual({ ok: false, code: "invalid_key", message: "no" });
  });

  it("rejects an answer in the wrong format", async () => {
    const r = await ask("q", null, null, vi.fn(async () => json(200, { answer: 1 })));
    expect(r).toMatchObject({ ok: false, code: "bad_response" });
  });

  it("does not throw when the network fails", async () => {
    const r = await ask("q", null, null, vi.fn(async () => { throw new TypeError("offline"); }));
    expect(r).toMatchObject({ ok: false, code: "network" });
  });

  it("falls back to a generic message for an unreadable error body", async () => {
    const r = await ask("q", null, null, vi.fn(async () => new Response("<html>oops</html>", { status: 500 })));
    expect(r).toMatchObject({ ok: false, code: "unknown" });
  });
});

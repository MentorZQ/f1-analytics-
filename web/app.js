"use strict";

const KEY_STORAGE = "f1rag.anthropicKey";
const KEY_ERRORS = new Set([
  "free_limit_reached",
  "daily_budget_reached",
  "no_server_key",
  "invalid_key",
]);

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const askForm = $("ask-form");
const askButton = $("ask-button");
const questionInput = $("question-input");
const sessionSelect = $("session-select");
const keyPanel = $("key-panel");
const keyInput = $("key-input");
const keyStatus = $("key-status");
const tooltip = $("tooltip");

let sessionLabels = {};
let lastRequest = null;
let busy = false;

// All page content is built with textContent, never parsed as HTML, so nothing
// in an answer or a chunk can inject markup into a page that may hold an API key.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ── Own-key storage ──────────────────────────────────────────────────────────
// Browser storage can throw in private windows; the key then lasts for the page.

let memoryKey = null;

function getKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || memoryKey;
  } catch {
    return memoryKey;
  }
}

function setKey(value) {
  memoryKey = value;
  try {
    if (value) sessionStorage.setItem(KEY_STORAGE, value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    // Kept in memoryKey only.
  }
  keyStatus.hidden = !value;
}

// ── Answer text ──────────────────────────────────────────────────────────────

function appendInline(parent, text) {
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      parent.append(el("strong", null, part.slice(2, -2)));
    } else if (part) {
      parent.append(part);
    }
  });
}

function tableCells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function renderMarkdown(source) {
  const root = el("div", "answer");
  const lines = source.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
    } else if (/^#{1,6}\s/.test(line)) {
      const heading = el("h3");
      appendInline(heading, line.replace(/^#+\s*/, ""));
      root.append(heading);
      i += 1;
    } else if (/^\s*\|/.test(line) && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1] || "")) {
      const table = el("table");
      const headRow = el("tr");
      tableCells(line).forEach((cell) => {
        const th = el("th");
        appendInline(th, cell);
        headRow.append(th);
      });
      const thead = el("thead");
      thead.append(headRow);
      const tbody = el("tbody");
      i += 2;
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        const row = el("tr");
        tableCells(lines[i]).forEach((cell) => {
          const td = el("td");
          appendInline(td, cell);
          row.append(td);
        });
        tbody.append(row);
        i += 1;
      }
      table.append(thead, tbody);
      const wrap = el("div", "table-wrap");
      wrap.append(table);
      root.append(wrap);
    } else if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const list = el(/^\s*\d/.test(line) ? "ol" : "ul");
      while (i < lines.length && /^\s*([-*]|\d+\.)\s/.test(lines[i])) {
        const item = el("li");
        appendInline(item, lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        list.append(item);
        i += 1;
      }
      root.append(list);
    } else {
      const paragraph = el("p");
      const start = i;
      while (
        i < lines.length && lines[i].trim() &&
        (i === start || !/^(#{1,6}\s|\s*\||\s*([-*]|\d+\.)\s)/.test(lines[i]))
      ) {
        appendInline(paragraph, (i > start ? " " : "") + lines[i].trim());
        i += 1;
      }
      root.append(paragraph);
    }
  }
  return root;
}

// ── Lap comparison chart ─────────────────────────────────────────────────────

const seconds = (value) => `${Math.abs(value).toFixed(3)}s`;

function describeSection(section, deltas) {
  if (section.gain_s === 0) return `${section.section}: level`;
  const faster = section.gain_s > 0 ? deltas.driver_a : deltas.driver_b;
  const range = section.start_m === null
    ? ""
    : ` (${Math.round(section.start_m)}–${Math.round(section.end_m)}m)`;
  return `${section.section}${range}: ${faster} faster by ${seconds(section.gain_s)}`;
}

function showTooltip(text, x, y) {
  tooltip.textContent = text;
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  tooltip.style.left = `${Math.max(8, Math.min(x + 12, window.innerWidth - box.width - 8))}px`;
  tooltip.style.top = `${Math.max(8, y - box.height - 8)}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function renderChart(deltas) {
  const chart = el("figure", "chart");
  chart.style.marginInline = "0";
  chart.style.marginBottom = "0";

  const { driver_a: a, driver_b: b, lap_gap_s: gap, sections } = deltas;
  const leader = gap >= 0 ? a : b;
  chart.append(
    el("figcaption", "chart-title", `${leader} ${seconds(gap)} faster over the lap (${a} vs ${b})`)
  );

  const legend = el("div", "chart-legend");
  [["swatch-a", `${a} faster`], ["swatch-b", `${b} faster`]].forEach(([swatch, label]) => {
    const item = el("span", "legend-item");
    item.append(el("span", `swatch ${swatch}`), label);
    legend.append(item);
  });
  legend.append(el("span", null, "Bold sections were used in the answer"));
  chart.append(legend);

  const largest = Math.max(...sections.map((s) => Math.abs(s.gain_s)), 0.001);
  // Label only the biggest swings. The table below has every value.
  const labelled = new Set(
    [...sections].sort((x, y) => Math.abs(y.gain_s) - Math.abs(x.gain_s)).slice(0, 3)
  );

  const bars = el("div", "bars");
  sections.forEach((section) => {
    const description = describeSection(section, deltas);
    const row = el("div", "bar-row");
    row.tabIndex = 0;
    row.setAttribute("role", "img");
    row.setAttribute("aria-label", description);
    row.append(el("span", `bar-label${section.retrieved ? " is-cited" : ""}`, section.section));

    const track = el("div", "bar-track");
    const share = (Math.abs(section.gain_s) / largest) * 38; // leaves room for a label
    const forA = section.gain_s >= 0;
    const bar = el("div", `bar ${forA ? "bar-a" : "bar-b"}`);
    bar.style.width = `${share}%`;
    track.append(bar);

    if (labelled.has(section)) {
      const value = el("span", "bar-value", seconds(section.gain_s));
      if (forA) value.style.right = `calc(50% + ${share}% + 4px)`;
      else value.style.left = `calc(50% + ${share}% + 4px)`;
      track.append(value);
    }
    row.append(track);

    row.addEventListener("pointermove", (e) => showTooltip(description, e.clientX, e.clientY));
    row.addEventListener("pointerleave", hideTooltip);
    row.addEventListener("focus", () => {
      const box = row.getBoundingClientRect();
      showTooltip(description, box.left, box.top);
    });
    row.addEventListener("blur", hideTooltip);
    bars.append(row);
  });
  chart.append(bars);

  const details = el("details");
  details.append(el("summary", null, "Show as a table"));
  const table = el("table");
  const head = el("tr");
  head.append(el("th", null, "Section"), el("th", null, "Faster"), el("th", "number", "Gap"));
  const thead = el("thead");
  thead.append(head);
  const tbody = el("tbody");
  sections.forEach((section) => {
    const row = el("tr");
    const faster = section.gain_s === 0 ? "Level" : section.gain_s > 0 ? a : b;
    row.append(
      el("td", null, section.section),
      el("td", null, faster),
      el("td", "number", seconds(section.gain_s))
    );
    tbody.append(row);
  });
  table.append(thead, tbody);
  const wrap = el("div", "table-wrap");
  wrap.append(table);
  details.append(wrap);
  chart.append(details);
  return chart;
}

// ── Sources ──────────────────────────────────────────────────────────────────

function renderSources(sources) {
  const details = el("details");
  details.append(el("summary", null, `Telemetry behind this answer (${sources.length} sources)`));
  sources.forEach((source) => {
    const block = el("div", "source");
    const title = [source.section, source.drivers.join(" vs ")].filter(Boolean).join(" · ");
    block.append(el("p", "source-title", title || source.chunk_type));
    block.append(el("pre", null, source.text));
    details.append(block);
  });
  return details;
}

// ── Asking ───────────────────────────────────────────────────────────────────

function paidByText(data) {
  if (data.paid_by === "saved") return "Saved answer";
  if (data.paid_by === "visitor") return "Answered with your key";
  const left = data.free_remaining;
  return `${left} free question${left === 1 ? "" : "s"} left today`;
}

function renderResult(card, question, data) {
  card.replaceChildren(
    el("p", "card-question", question),
    el("p", "card-meta", `${sessionLabels[data.session] || data.session} · ${paidByText(data)}`),
    renderMarkdown(data.answer)
  );
  if (data.deltas) card.append(renderChart(data.deltas));
  if (data.sources.length) card.append(renderSources(data.sources));
}

function renderError(card, question, message) {
  card.replaceChildren(el("p", "card-question", question), el("p", "card-error", message));
}

function setBusy(value) {
  busy = value;
  askButton.disabled = value;
  askButton.textContent = value ? "Working…" : "Ask";
  document.querySelectorAll(".example, #key-form button").forEach((button) => {
    button.disabled = value;
  });
}

async function ask(question, session) {
  if (busy) return;
  setBusy(true);
  lastRequest = { question, session };

  const card = el("article", "card");
  card.append(
    el("p", "card-question", question),
    el("p", "loading", "Searching the telemetry and writing an answer…")
  );
  thread.append(card);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });

  const headers = { "Content-Type": "application/json" };
  const key = getKey();
  if (key) headers["X-Anthropic-Key"] = key;

  try {
    const response = await fetch("query", {
      method: "POST",
      headers,
      body: JSON.stringify({ question, session: session || null }),
    });
    const data = await response.json().catch(() => null);

    if (response.ok) {
      renderResult(card, question, data);
      keyPanel.hidden = true;
    } else {
      const detail = data && data.detail;
      const code = detail && detail.code;
      const message = (detail && detail.message) || "Something went wrong. Please try again.";
      renderError(card, question, message);
      if (KEY_ERRORS.has(code)) {
        if (code === "invalid_key") setKey(null);
        keyPanel.hidden = false;
        keyInput.focus();
      }
    }
  } catch {
    renderError(card, question, "Couldn't reach the server. Check your connection and try again.");
  } finally {
    setBusy(false);
  }
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  ask(question, sessionSelect.value);
});

$("key-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const value = keyInput.value.trim();
  if (!value) return;
  setKey(value);
  keyInput.value = "";
  keyPanel.hidden = true;
  if (lastRequest) ask(lastRequest.question, lastRequest.session);
});

$("forget-key").addEventListener("click", () => setKey(null));

// ── Start-up ─────────────────────────────────────────────────────────────────

async function init() {
  keyStatus.hidden = !getKey();
  try {
    const response = await fetch("api/sessions");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    data.sessions.forEach(({ key, label }) => {
      sessionLabels[key] = label;
      const option = el("option", null, label);
      option.value = key;
      sessionSelect.append(option);
    });
    $("session-names").textContent = data.sessions.map((s) => s.label).join(", ");

    const list = $("example-list");
    data.examples.forEach(({ session, question }) => {
      const button = el("button", "example", question);
      button.type = "button";
      button.append(el("small", null, sessionLabels[session] || session));
      button.addEventListener("click", () => ask(question, session));
      list.append(button);
    });
  } catch {
    $("example-list").append(el("p", "card-error", "Couldn't load the sessions. Please refresh."));
  }
}

init();

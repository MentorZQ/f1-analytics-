import { useEffect, useRef, useState, type FormEvent } from "react";
import { KEY_ERRORS, dataGaps, type QueryResponse, type SessionsResponse } from "@f1rag/shared";
import { ask, loadSessions } from "./api.ts";
import { Chart } from "./Chart.tsx";
import { getKey, setKey } from "./keyStore.ts";
import { Markdown } from "./Markdown.tsx";
import { Sources } from "./Sources.tsx";

type Card =
  | { id: number; question: string; status: "loading" }
  | { id: number; question: string; status: "done"; data: QueryResponse }
  | { id: number; question: string; status: "error"; message: string };

function paidByText(data: QueryResponse): string {
  if (data.paid_by === "saved") return "Saved answer";
  if (data.paid_by === "visitor") return "Answered with your key";
  if (data.paid_by === "router") return "Answered from the data, no model call";
  if (data.paid_by === "none") return "No model call";
  const left = data.free_remaining ?? 0;
  return `${left} free question${left === 1 ? "" : "s"} left today`;
}

function Result({ card, labels }: { card: Extract<Card, { status: "done" }>; labels: Record<string, string> }) {
  const { data } = card;
  const gaps = dataGaps(data);
  const session = data.session ? labels[data.session] ?? data.session : "Session not detected";
  return (
    <>
      <p className="card-question">{card.question}</p>
      <p className="card-meta">{session} · {paidByText(data)}</p>
      {gaps.length > 0 && (
        <ul className="gaps" aria-label="Limits of the data for this answer">
          {gaps.map((g) => <li className="gap-badge" key={g.id}>{g.label}</li>)}
        </ul>
      )}
      <Markdown source={data.answer} />
      {data.deltas && <Chart deltas={data.deltas} />}
      {data.sources.length > 0 && <Sources sources={data.sources} />}
    </>
  );
}

export function App() {
  const [info, setInfo] = useState<SessionsResponse | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [cards, setCards] = useState<Card[]>([]);
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState("");
  const [question, setQuestion] = useState("");
  const [needKey, setNeedKey] = useState(false);
  const [hasKey, setHasKey] = useState(() => getKey() !== null);
  const [keyDraft, setKeyDraft] = useState("");
  const last = useRef<{ question: string; session: string | null } | null>(null);
  const nextId = useRef(0);

  useEffect(() => {
    loadSessions().then(setInfo).catch(() => setLoadFailed(true));
  }, []);

  const labels = Object.fromEntries((info?.sessions ?? []).map((s) => [s.key, s.label]));

  async function run(q: string, s: string | null) {
    if (busy) return;
    setBusy(true);
    last.current = { question: q, session: s };
    const id = nextId.current++;
    const update = (card: Card) => setCards((prev) => prev.map((c) => (c.id === id ? card : c)));
    setCards((prev) => [...prev, { id, question: q, status: "loading" }]);

    const result = await ask(q, s, getKey());
    if (result.ok) {
      update({ id, question: q, status: "done", data: result.data });
      setNeedKey(false);
    } else {
      update({ id, question: q, status: "error", message: result.message });
      if (KEY_ERRORS.has(result.code)) {
        if (result.code === "invalid_key") {
          setKey(null);
          setHasKey(false);
        }
        setNeedKey(true);
      }
    }
    setBusy(false);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q) return;
    setQuestion("");
    void run(q, session || null);
  }

  function saveKey(e: FormEvent) {
    e.preventDefault();
    const value = keyDraft.trim();
    if (!value) return;
    setKey(value);
    setHasKey(true);
    setKeyDraft("");
    setNeedKey(false);
    if (last.current) void run(last.current.question, last.current.session);
  }

  return (
    <main className="page">
      <header className="masthead">
        <h1>F1 Qualifying RAG</h1>
        <p>Ask where a qualifying lap was won or lost. Answers come from telemetry, with the evidence shown underneath.</p>
      </header>

      <section className="examples" aria-label="Example questions">
        <h2>Try an example <span className="muted">(instant and free)</span></h2>
        <div className="example-list">
          {loadFailed && <p className="card-error">Couldn't load the sessions. Please refresh.</p>}
          {info?.examples.map((ex) => (
            <button className="example" type="button" key={ex.question} disabled={busy} onClick={() => void run(ex.question, ex.session)}>
              {ex.question}
              <small>{labels[ex.session] ?? ex.session}</small>
            </button>
          ))}
        </div>
      </section>

      <div className="thread" aria-live="polite">
        {cards.map((card) => (
          <article className="card" key={card.id}>
            {card.status === "loading" && (
              <>
                <p className="card-question">{card.question}</p>
                <p className="loading">Searching the telemetry and writing an answer…</p>
              </>
            )}
            {card.status === "error" && (
              <>
                <p className="card-question">{card.question}</p>
                <p className="card-error">{card.message}</p>
              </>
            )}
            {card.status === "done" && <Result card={card} labels={labels} />}
          </article>
        ))}
      </div>

      {needKey && (
        <section className="key-panel" aria-labelledby="key-title">
          <h2 id="key-title">Use your own Anthropic key</h2>
          <p>Your key goes to this site's server with each question, is used for that call only, and is never stored or logged there. Your browser keeps it until you close this tab.</p>
          <form className="key-form" autoComplete="off" onSubmit={saveKey}>
            <label htmlFor="key-input" className="visually-hidden">Anthropic API key</label>
            <input id="key-input" type="password" placeholder="sk-ant-..." autoComplete="off" spellCheck={false} autoFocus value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)} />
            <button type="submit" disabled={busy}>Save and retry</button>
          </form>
        </section>
      )}

      <form className="ask-form" onSubmit={submit}>
        <label htmlFor="session-select" className="visually-hidden">Session</label>
        <select id="session-select" value={session} onChange={(e) => setSession(e.target.value)}>
          <option value="">Detect session</option>
          {info?.sessions.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
        </select>
        <label htmlFor="question-input" className="visually-hidden">Your question</label>
        <input id="question-input" type="text" maxLength={info?.max_question_chars ?? 500} required placeholder="Where did Norris lose time to Russell at Barcelona?" autoComplete="off" value={question} onChange={(e) => setQuestion(e.target.value)} />
        <button type="submit" disabled={busy}>{busy ? "Working…" : "Ask"}</button>
      </form>
      {hasKey && (
        <p className="key-status">
          Using your own key.{" "}
          <button type="button" className="link-button" onClick={() => { setKey(null); setHasKey(false); }}>Forget key</button>
        </p>
      )}

      <footer className="footer">
        Sessions: {info?.sessions.map((s) => s.label).join(", ")}. Answers are written by Claude from retrieved telemetry and can still contain mistakes, so check them against the sources.{" "}
        <a href="https://github.com/MentorZQ/f1-analytics-">Source code</a>
      </footer>
    </main>
  );
}

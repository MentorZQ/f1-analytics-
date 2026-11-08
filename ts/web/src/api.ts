import {
  ApiErrorSchema,
  QueryResponseSchema,
  SessionsResponseSchema,
  type QueryResponse,
  type SessionsResponse,
} from "@f1rag/shared";

export type AskResult =
  | { ok: true; data: QueryResponse }
  | { ok: false; code: string; message: string };

const GENERIC = "Something went wrong. Please try again.";

/** Asks one question. Never throws: every failure comes back as a typed {ok: false, code, message}. */
export async function ask(
  question: string,
  session: string | null,
  key: string | null,
  fetchImpl: typeof fetch = fetch,
): Promise<AskResult> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (key) headers["X-Anthropic-Key"] = key;

  let response: Response;
  try {
    response = await fetchImpl("/query", { method: "POST", headers, body: JSON.stringify({ question, session }) });
  } catch {
    return { ok: false, code: "network", message: "Couldn't reach the server. Check your connection and try again." };
  }

  const body: unknown = await response.json().catch(() => null);
  if (response.ok) {
    const parsed = QueryResponseSchema.safeParse(body);
    return parsed.success
      ? { ok: true, data: parsed.data }
      : { ok: false, code: "bad_response", message: "The server sent an answer in an unexpected format." };
  }
  const error = ApiErrorSchema.safeParse(body);
  return error.success
    ? { ok: false, code: error.data.detail.code, message: error.data.detail.message }
    : { ok: false, code: "unknown", message: GENERIC };
}

export async function loadSessions(fetchImpl: typeof fetch = fetch): Promise<SessionsResponse> {
  const response = await fetchImpl("/api/sessions");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return SessionsResponseSchema.parse(await response.json());
}

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildGateway } from "../src/gateway.ts";

const examples = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../web/examples.json", import.meta.url)), "utf8"),
) as Array<Record<string, unknown>>;
const goodResponse = { ...examples[0], paid_by: "saved" };
const VALID_KEY = "sk-ant-" + "a".repeat(40);

type FakeFetch = ReturnType<typeof vi.fn<typeof fetch>>;

function reply(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function gatewayWith(fetchImpl: FakeFetch, extra: Record<string, unknown> = {}) {
  return buildGateway({ upstreamUrl: "http://py.test", fetchImpl, ...extra });
}

const ask = (app: ReturnType<typeof gatewayWith>, payload: unknown, headers: Record<string, string> = {}) =>
  app.inject({ method: "POST", url: "/query", payload: payload as object, headers });

let app: ReturnType<typeof gatewayWith> | undefined;
afterEach(async () => {
  await app?.close();
  app = undefined;
});

describe("POST /query", () => {
  it("returns a validated answer from the upstream", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake);
    const res = await ask(app, { question: "Why was RUS ahead of HAM?", session: "monza" });
    expect(res.statusCode).toBe(200);
    expect(res.json().paid_by).toBe("saved");
    expect(res.json().sources.length).toBeGreaterThan(0);
  });

  it("rejects an empty question before calling Python", async () => {
    const fake: FakeFetch = vi.fn();
    app = gatewayWith(fake);
    const res = await ask(app, { question: "   " });
    expect(res.statusCode).toBe(400);
    expect(res.json().detail.code).toBe("invalid_request");
    expect(fake).not.toHaveBeenCalled();
  });

  it("rejects a question over 500 characters", async () => {
    const fake: FakeFetch = vi.fn();
    app = gatewayWith(fake);
    const res = await ask(app, { question: "x".repeat(501) });
    expect(res.statusCode).toBe(400);
    expect(fake).not.toHaveBeenCalled();
  });

  it("rejects a malformed key without calling Python", async () => {
    const fake: FakeFetch = vi.fn();
    app = gatewayWith(fake);
    const res = await ask(app, { question: "hi" }, { "x-anthropic-key": "not a key!!" });
    expect(res.statusCode).toBe(401);
    expect(res.json().detail.code).toBe("invalid_key");
    expect(fake).not.toHaveBeenCalled();
  });

  it("forwards a trimmed key, and the visitor's address instead of any supplied one", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake);
    await ask(app, { question: "hi" }, { "x-anthropic-key": `  ${VALID_KEY}  `, "x-forwarded-for": "6.6.6.6" });
    const init = fake.mock.calls[0]![1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers["x-anthropic-key"]).toBe(VALID_KEY);
    expect(headers["x-forwarded-for"]).not.toContain("6.6.6.6");
  });

  it("sends no key header when the visitor gave none", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake);
    await ask(app, { question: "hi" });
    const headers = (fake.mock.calls[0]![1] as RequestInit).headers as Record<string, string>;
    expect(headers).not.toHaveProperty("x-anthropic-key");
  });

  it("passes Python's typed errors through with their status", async () => {
    const fake: FakeFetch = vi.fn(async () =>
      reply(503, { detail: { code: "no_server_key", message: "Add your own key." } }),
    );
    app = gatewayWith(fake);
    const res = await ask(app, { question: "hi" });
    expect(res.statusCode).toBe(503);
    expect(res.json().detail.code).toBe("no_server_key");
  });

  it("hides an upstream error body that is not the expected shape", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(500, { trace: "secret internals" }));
    app = gatewayWith(fake);
    const res = await ask(app, { question: "hi" });
    expect(res.statusCode).toBe(502);
    expect(res.body).not.toContain("secret internals");
    expect(res.json().detail.code).toBe("upstream_error");
  });

  it("does not echo a response that fails the schema", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, { answer: 42, leaked: "internal" }));
    app = gatewayWith(fake);
    const res = await ask(app, { question: "hi" });
    expect(res.statusCode).toBe(502);
    expect(res.json().detail.code).toBe("bad_upstream_response");
    expect(res.body).not.toContain("internal");
  });

  it("returns a typed error when Python is down", async () => {
    const fake: FakeFetch = vi.fn(async () => {
      throw new TypeError("fetch failed");
    });
    app = gatewayWith(fake);
    const res = await ask(app, { question: "hi" });
    expect(res.statusCode).toBe(502);
    expect(res.json().detail.code).toBe("upstream_unreachable");
  });

  it("gives up on a hung upstream", async () => {
    const fake: FakeFetch = vi.fn(
      (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          (init?.signal as AbortSignal).addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );
    app = gatewayWith(fake, { upstreamTimeoutMs: 20 });
    const res = await ask(app, { question: "hi" });
    expect(res.json().detail.code).toBe("upstream_unreachable");
  });

  it("rate limits a visitor at the gateway", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake, { questionsPerMinute: 2 });
    const statuses: number[] = [];
    for (let i = 0; i < 3; i++) statuses.push((await ask(app, { question: "hi" })).statusCode);
    expect(statuses).toEqual([200, 200, 429]);
    expect(fake).toHaveBeenCalledTimes(2);
  });
});

describe("GET /api/sessions and /health", () => {
  const sessions = {
    sessions: [{ key: "monza", label: "Monza 2024" }],
    examples: [{ session: "monza", question: "Q?" }],
    max_question_chars: 500,
  };

  it("validates and returns the session list", async () => {
    app = gatewayWith(vi.fn(async () => reply(200, sessions)));
    const res = await app.inject({ method: "GET", url: "/api/sessions" });
    expect(res.statusCode).toBe(200);
    expect(res.json().sessions[0].key).toBe("monza");
  });

  it("rejects a malformed session list", async () => {
    app = gatewayWith(vi.fn(async () => reply(200, { sessions: "nope" })));
    const res = await app.inject({ method: "GET", url: "/api/sessions" });
    expect(res.statusCode).toBe(502);
  });

  it("reports degraded when Python's health check fails", async () => {
    app = gatewayWith(vi.fn(async () => reply(503, { status: "degraded" })));
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.statusCode).toBe(503);
  });

  it("reports ok when Python is healthy", async () => {
    app = gatewayWith(vi.fn(async () => reply(200, { status: "ok" })));
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.statusCode).toBe(200);
    expect(res.json().gateway).toBe("ok");
  });

  it("sets security headers", async () => {
    app = gatewayWith(vi.fn(async () => reply(200, { status: "ok" })));
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.headers["x-content-type-options"]).toBe("nosniff");
    expect(res.headers["referrer-policy"]).toBe("no-referrer");
  });
});

describe("running behind another service", () => {
  const headersOf = (fake: FakeFetch) => (fake.mock.calls[0]![1] as RequestInit).headers as Record<string, string>;

  it("sends Python the shared secret and the visitor's address so its quotas stay per visitor", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake, { gatewaySecret: "shared-secret-value" });
    await app.inject({ method: "POST", url: "/query", payload: { question: "hi" }, remoteAddress: "7.7.7.7" });
    expect(headersOf(fake)["x-gateway-secret"]).toBe("shared-secret-value");
    expect(headersOf(fake)["x-visitor-ip"]).toBe("7.7.7.7");
  });

  it("sends neither header when no secret is configured", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake);
    await ask(app, { question: "hi" });
    expect(headersOf(fake)).not.toHaveProperty("x-gateway-secret");
    expect(headersOf(fake)).not.toHaveProperty("x-visitor-ip");
  });

  it("with one proxy hop trusted, the visitor is the address the proxy appended, not one the visitor wrote", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake, { trustProxy: 1, gatewaySecret: "s" });
    await app.inject({
      method: "POST",
      url: "/query",
      payload: { question: "hi" },
      remoteAddress: "10.0.0.1",
      headers: { "x-forwarded-for": "6.6.6.6, 9.9.9.9" },
    });
    expect(headersOf(fake)["x-visitor-ip"]).toBe("9.9.9.9");
  });

  it("without a trusted proxy, a forwarded address the visitor wrote is ignored", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake, { gatewaySecret: "s" });
    await app.inject({
      method: "POST",
      url: "/query",
      payload: { question: "hi" },
      remoteAddress: "8.8.4.4",
      headers: { "x-forwarded-for": "6.6.6.6" },
    });
    expect(headersOf(fake)["x-visitor-ip"]).toBe("8.8.4.4");
  });

  it("never puts the shared secret in a response", async () => {
    const fake: FakeFetch = vi.fn(async () => reply(200, goodResponse));
    app = gatewayWith(fake, { gatewaySecret: "shared-secret-value" });
    const res = await ask(app, { question: "hi" });
    expect(res.body).not.toContain("shared-secret-value");
    expect(JSON.stringify(res.headers)).not.toContain("shared-secret-value");
  });

  it("sends the same content security policy as the Python server", async () => {
    app = gatewayWith(vi.fn(async () => reply(200, { status: "ok" })));
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.headers["content-security-policy"]).toBe(
      "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    );
  });
});

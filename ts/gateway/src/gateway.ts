import { existsSync } from "node:fs";
import Fastify, { type FastifyInstance, type FastifyReply } from "fastify";
import rateLimit from "@fastify/rate-limit";
import fastifyStatic from "@fastify/static";
import {
  ApiErrorSchema,
  QueryRequestSchema,
  QueryResponseSchema,
  SessionsResponseSchema,
  type ApiError,
} from "@f1rag/shared";

export interface GatewayOptions {
  /** Base URL of the Python server, e.g. http://localhost:8080 */
  upstreamUrl: string;
  /** Injected in tests; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
  /** Built frontend to serve. Skipped when the folder does not exist. */
  webDir?: string;
  /** Questions per visitor per minute at the gateway, before anything reaches Python. */
  questionsPerMinute?: number;
  /** An answer takes 10-20 s, so the default leaves generous room. */
  upstreamTimeoutMs?: number;
  /** Set when another proxy (such as Render's) sits in front of the gateway. */
  trustProxy?: boolean;
  logger?: boolean;
}

// What a visitor's key is allowed to look like. Python checks this too; failing here saves a hop.
const KEY_FORMAT = /^[A-Za-z0-9_-]{20,300}$/;

function fail(reply: FastifyReply, status: number, code: string, message: string) {
  const body: ApiError = { detail: { code, message } };
  return reply.code(status).send(body);
}

export function buildGateway(options: GatewayOptions): FastifyInstance {
  const {
    upstreamUrl,
    fetchImpl = fetch,
    webDir,
    questionsPerMinute = 20,
    upstreamTimeoutMs = 120_000,
    trustProxy = false,
    logger = false,
  } = options;

  const app = Fastify({
    trustProxy,
    // A visitor's key must never reach a log line, even by accident.
    logger: logger ? { redact: ["req.headers.x-anthropic-key", "req.headers.authorization"] } : false,
  });

  app.register(rateLimit, {
    global: false,
    errorResponseBuilder: () => ({
      statusCode: 429,
      detail: { code: "gateway_rate_limited", message: "Too many questions. Try again in a minute." },
    }),
  });

  app.addHook("onSend", async (_req, reply) => {
    reply.header("Referrer-Policy", "no-referrer");
    reply.header("X-Content-Type-Options", "nosniff");
  });

  /** Calls Python. Returns the parsed JSON and status, or null when it could not be reached. */
  async function upstream(
    path: string,
    init: { method: string; headers?: Record<string, string>; body?: string },
  ): Promise<{ status: number; json: unknown } | null> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), upstreamTimeoutMs);
    try {
      const response = await fetchImpl(new URL(path, upstreamUrl), { ...init, signal: controller.signal });
      const json: unknown = await response.json().catch(() => null);
      return { status: response.status, json };
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  // Registered as a plugin so the rate-limit plugin above has loaded before these routes are defined.
  app.register(async (api) => {
    api.get("/health", async (_req, reply) => {
      const result = await upstream("/health", { method: "GET" });
      if (!result) return fail(reply, 502, "upstream_unreachable", "The analysis server is not reachable.");
      return reply.code(result.status === 200 ? 200 : 503).send({ gateway: "ok", upstream: result.json });
    });

    api.get("/api/sessions", async (_req, reply) => {
      const result = await upstream("/api/sessions", { method: "GET" });
      if (!result) return fail(reply, 502, "upstream_unreachable", "The analysis server is not reachable.");
      const parsed = SessionsResponseSchema.safeParse(result.json);
      if (result.status !== 200 || !parsed.success) {
        return fail(reply, 502, "bad_upstream_response", "The analysis server sent an unexpected reply.");
      }
      return parsed.data;
    });

    api.post(
      "/query",
      { config: { rateLimit: { max: questionsPerMinute, timeWindow: "1 minute" } } },
      async (request, reply) => {
        const body = QueryRequestSchema.safeParse(request.body);
        if (!body.success) {
          const message = body.error.issues[0]?.message ?? "That request wasn't valid.";
          return fail(reply, 400, "invalid_request", message);
        }

        const rawKey = request.headers["x-anthropic-key"];
        const key = (Array.isArray(rawKey) ? rawKey[0] : rawKey)?.trim() || undefined;
        if (key && !KEY_FORMAT.test(key)) {
          return fail(reply, 401, "invalid_key", "That doesn't look like an Anthropic API key.");
        }

        const headers: Record<string, string> = {
          "content-type": "application/json",
          // Overwritten, never appended: Python takes the last entry, so a value the visitor supplied can't be used
          // to dodge its per-visitor quota.
          "x-forwarded-for": request.ip,
        };
        if (key) headers["x-anthropic-key"] = key;

        const result = await upstream("/query", {
          method: "POST",
          headers,
          body: JSON.stringify({ question: body.data.question, session: body.data.session ?? null }),
        });
        if (!result) return fail(reply, 502, "upstream_unreachable", "The analysis server is not reachable.");

        if (result.status !== 200) {
          // Rebuilt from the validated fields, so nothing unexpected is echoed to the visitor.
          const error = ApiErrorSchema.safeParse(result.json);
          if (error.success) return reply.code(result.status).send({ detail: error.data.detail });
          return fail(reply, 502, "upstream_error", "Something went wrong. Please try again.");
        }

        const answer = QueryResponseSchema.safeParse(result.json);
        if (!answer.success) {
          // Field paths only: the values could contain the model's answer or a visitor's text.
          request.log.warn({ paths: answer.error.issues.map((i) => i.path.join(".")) }, "bad upstream response");
          return fail(reply, 502, "bad_upstream_response", "The analysis server sent an unexpected reply.");
        }
        return answer.data;
      },
    );
  });

  if (webDir && existsSync(webDir)) {
    app.register(fastifyStatic, { root: webDir });
  }

  return app;
}

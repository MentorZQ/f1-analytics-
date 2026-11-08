# TypeScript layer

A typed gateway and frontend in front of the Python RAG server. The Python pipeline (retrieval, comparisons, the
model call, the checks) is unchanged; this layer owns the contract and the presentation.

```
browser (React + TypeScript)  ->  Node gateway (Fastify)  ->  Python server (FastAPI, ../server.py)
          packages/shared: Zod schemas and types used by both sides
```

| Folder | What it does |
|---|---|
| `packages/shared` | Zod schemas for every request, response and error code; the chart maths (bars, titles, exact lap gap); the data-gap rules behind the badges. |
| `gateway` | Validates the request, rate limits per visitor, forwards the visitor's key without logging it, validates what Python returns, and serves the built frontend. |
| `web` | Vite + React. Answer text is rendered as React elements, never parsed as HTML. |

## What the types buy

- **The response is validated at the gateway.** A renamed or missing field from Python becomes a typed
  `bad_upstream_response` error and nothing from the bad payload is echoed to the visitor.
- **The saved answers are the contract fixture.** `packages/shared/test` parses every answer in `../web/examples.json`.
- **The UI branches on error codes, not message text** (`KEY_ERRORS` decides when to ask for a key).
- **Honest handling is visible.** Badges show when an answer rests on missing data ("No braking data was found",
  "A driver set no lap"), read from the answer and from the evidence the model was given.

## Run it

```bash
cd ts && npm install
npm run typecheck && npm test && npm run build

# in one shell: the Python server (from the repo root)
uvicorn server:app --port 8080
# in another: the gateway, which also serves ts/web/dist
UPSTREAM_URL=http://localhost:8080 npm run dev:gateway    # http://localhost:3000
# hot reload for the frontend (proxies to the gateway)
npm run dev:web
```

Settings: `UPSTREAM_URL`, `PORT` (3000), `QUESTIONS_PER_MINUTE` (20), `WEB_DIR`, `TRUST_PROXY=1` behind a host's proxy.

## Visitor quotas

Python keeps its per-visitor quotas by IP. The gateway sends the real visitor address in `X-Forwarded-For`
(overwritten, never appended) and Python must run with `TRUST_PROXY=1` to read it. Without that, every visitor
looks like the gateway and shares one quota.

## Not in this layer yet

Token streaming (needs a streaming endpoint in `server.py`) and deployment as a second Render service.

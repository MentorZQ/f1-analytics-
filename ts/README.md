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

Python keeps its per-visitor quotas by address. Behind the gateway every request would look like it came from the gateway, so
the gateway sends the visitor's address in `X-Visitor-IP` together with `X-Gateway-Secret`. Python believes that header only when
the secret matches its own `GATEWAY_SECRET` (compared in constant time); otherwise it is ignored, so a visitor cannot name an
address to dodge the limit. The gateway itself reads the visitor from the address Render's proxy appended (`TRUST_PROXY=1`
trusts exactly one hop), not from a value the visitor can write.

## Deploying on Render

`render.yaml` defines both services:

```
visitor -> f1-rag-gateway (Node) -> f1-qualifying-rag (Python, over Render's private network)
```

- **`f1-rag-gateway`** builds with `npm ci --include=dev && npm run build` (the build needs vite and typescript even when
  `NODE_ENV=production`), starts with `npm run start --workspace gateway`, and serves the built page itself. `tsx`, which
  runs the gateway, is a production dependency.
- **`GATEWAY_SECRET`** is generated once on the gateway and read by the Python service (`fromService`), so no secret is in the repo.
- **`UPSTREAM_URL`** is `http://f1-qualifying-rag:8080`; the Python service sets `PORT=8080` to match the Dockerfile.
- The Python service is still public. Making it a private service later would stop anyone calling it without the gateway.

Steps: push, then in Render sync the Blueprint (New > Blueprint, or the existing one's "Sync"). Render creates the gateway service,
adds the three variables to the Python service and redeploys it. Open the gateway's URL, not the Python one, to use the TypeScript layer.

`tests/test_deploy_files.py` checks the blueprint against the Dockerfile and `ts/package.json` (port, secret wiring, start script,
production dependency). The deploy was also rehearsed on a clean copy: install from the lockfile, build with `NODE_ENV=production`,
start with production dependencies only, in front of the real Python server.

## Not in this layer yet

Token streaming (needs a streaming endpoint in `server.py`).

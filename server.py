"""
server.py
---------
Public web app around query.py's RAG pipeline: serves the UI in web/ and the
JSON API it calls.

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /               -> web UI
    GET  /health         -> liveness check
    GET  /api/sessions   -> sessions and example questions
    POST /query          -> {"question": "...", "session": "..."}
                            Optional header X-Anthropic-Key: the visitor's own key.

Who pays for an answer:
    1. A saved example answer costs nothing and needs no key.
    2. Otherwise the visitor's key is used if they sent one. It is used for that
       single call and is never stored or logged.
    3. Otherwise the server's key is used, within the free quotas below.

Settings (environment variables):
    F1_MODEL              Claude model for answers (default: claude-sonnet-4-6)
    FREE_PER_IP_PER_DAY   free questions per visitor per day (default: 5)
    FREE_TOTAL_PER_DAY    free questions per day across all visitors (default: 100)
    OWN_KEY_PER_IP_PER_HOUR  own-key questions per visitor per hour (default: 30)
    TRUST_PROXY           set to 1 behind a host's proxy so visitors are told apart
                          using X-Forwarded-For (default: off)

Quotas live in memory, so a restart resets them. Keep a hard spend limit in the
Anthropic console as well.
"""

import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import query as query_module

log = logging.getLogger("f1_rag")

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
EXAMPLES_PATH = WEB_DIR / "examples.json"

MODEL = os.environ.get("F1_MODEL", "claude-sonnet-4-6")
N_CHUNKS = 6
MAX_QUESTION_CHARS = 500

SESSION_LABELS = {
    "melbourne": "Melbourne 2026",
    "barcelona": "Barcelona 2026",
    "monza":     "Monza 2024",
}

_API_KEY_FORMAT = re.compile(r"^[A-Za-z0-9_-]{20,300}$")


# ── Quotas ────────────────────────────────────────────────────────────────────

class WindowCounter:
    """Thread-safe count of uses per key, reset every `window_s` seconds."""

    def __init__(self, limit: int, window_s: int) -> None:
        self.limit = limit
        self.window_s = window_s
        self._uses: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def _current(self, key: str, now: float) -> tuple[float, int]:
        started, count = self._uses.get(key, (now, 0))
        return (now, 0) if now - started >= self.window_s else (started, count)

    def take(self, key: str) -> bool:
        """Use one slot. Returns False, using nothing, if the key is at its limit."""
        now = time.monotonic()
        with self._lock:
            if len(self._uses) > 10_000:
                self._uses = {
                    k: v for k, v in self._uses.items() if now - v[0] < self.window_s
                }
            started, count = self._current(key, now)
            if count >= self.limit:
                return False
            self._uses[key] = (started, count + 1)
            return True

    def give_back(self, key: str) -> None:
        """Return a slot, for a request that failed before producing an answer."""
        with self._lock:
            started, count = self._current(key, time.monotonic())
            self._uses[key] = (started, max(count - 1, 0))

    def remaining(self, key: str) -> int:
        with self._lock:
            _, count = self._current(key, time.monotonic())
            return max(self.limit - count, 0)


class Quotas:
    def __init__(self, free_per_ip: int, free_total: int, own_key_per_ip: int) -> None:
        self.free_per_ip = WindowCounter(free_per_ip, 86_400)
        self.free_total = WindowCounter(free_total, 86_400)
        self.own_key = WindowCounter(own_key_per_ip, 3_600)


quotas = Quotas(
    free_per_ip=int(os.environ.get("FREE_PER_IP_PER_DAY", "5")),
    free_total=int(os.environ.get("FREE_TOTAL_PER_DAY", "100")),
    own_key_per_ip=int(os.environ.get("OWN_KEY_PER_IP_PER_HOUR", "30")),
)


def _client_ip(request: Request) -> str:
    if os.environ.get("TRUST_PROXY") == "1":
        # The nearest proxy appends the address it saw. Entries further left are
        # supplied by the visitor and could be forged to dodge the quota.
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _reserve_free_question(ip: str) -> None:
    if not quotas.free_per_ip.take(ip):
        raise _api_error(
            429, "free_limit_reached",
            "You've used today's free questions. Add your own Anthropic key to keep going.",
        )
    if not quotas.free_total.take("all"):
        quotas.free_per_ip.give_back(ip)
        raise _api_error(
            429, "daily_budget_reached",
            "Today's free questions have run out. Add your own Anthropic key to keep going.",
        )


def _release_free_question(ip: str) -> None:
    quotas.free_per_ip.give_back(ip)
    quotas.free_total.give_back("all")


# ── Saved example answers ─────────────────────────────────────────────────────

def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def _load_examples() -> list[dict]:
    if not EXAMPLES_PATH.exists():
        return []
    return json.loads(EXAMPLES_PATH.read_text())


def _find_saved_example(question: str, session: str | None) -> dict | None:
    wanted = _normalise(question)
    for example in EXAMPLES:
        if (
            example.get("answer")
            and _normalise(example["question"]) == wanted
            and session in (None, example["session"])
        ):
            return example
    return None


EXAMPLES = _load_examples()


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    query_module._load_dotenv()
    query_module.get_embed_fn()  # so the first visitor doesn't wait for the model
    yield


app = FastAPI(title="F1 RAG API", lifespan=_lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    # Visitors may type an API key into this page, so it loads only its own files.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/sessions")
def sessions() -> dict:
    return {
        "sessions": [
            {"key": key, "label": SESSION_LABELS.get(key, key)}
            for key in query_module.SESSIONS
        ],
        "examples": [
            {"session": e["session"], "question": e["question"]} for e in EXAMPLES
        ],
        "max_question_chars": MAX_QUESTION_CHARS,
    }


@app.post("/query")
def run_query(
    req: QueryRequest,
    request: Request,
    x_anthropic_key: str | None = Header(default=None),
) -> dict:
    question = req.question.strip()
    if not question:
        raise _api_error(400, "empty_question", "Please enter a question.")
    if req.session is not None and req.session not in query_module.SESSIONS:
        raise _api_error(400, "unknown_session", "That session isn't available.")

    saved = _find_saved_example(question, req.session)
    if saved:
        return {
            "answer":  saved["answer"],
            "session": saved["session"],
            "sources": saved["sources"],
            "deltas":  saved["deltas"],
            "paid_by": "saved",
        }

    # Checked by hand: a declared validation rule would echo the key in its error.
    own_key = (x_anthropic_key or "").strip() or None
    if own_key and not _API_KEY_FORMAT.match(own_key):
        raise _api_error(401, "invalid_key", "That doesn't look like an Anthropic API key.")

    ip = _client_ip(request)
    if own_key:
        if not quotas.own_key.take(ip):
            raise _api_error(429, "too_many_requests", "Too many questions. Try again in a while.")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise _api_error(
                503, "no_server_key",
                "Free questions aren't available right now. Add your own Anthropic key to ask one.",
            )
        _reserve_free_question(ip)

    try:
        result = query_module.answer_with_sources(
            question,
            session_key=req.session,
            n_chunks=N_CHUNKS,
            model=MODEL,
            api_key=own_key,
        )
        if not result["answer"].strip():
            raise _api_error(502, "empty_answer", "No answer came back. Please try again.")
    except Exception as exc:
        if not own_key:
            _release_free_question(ip)
        if isinstance(exc, HTTPException):
            raise
        # Only the exception type is logged: messages can quote request details.
        log.warning("Query failed: %s", type(exc).__name__)
        if own_key and isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            raise _api_error(401, "invalid_key", "Anthropic didn't accept that key.") from None
        if isinstance(exc, anthropic.RateLimitError):
            raise _api_error(503, "busy", "The service is busy. Please try again shortly.") from None
        if own_key and isinstance(exc, anthropic.APIStatusError):
            raise _api_error(
                502, "upstream_error",
                "Anthropic returned an error for your key. Check its credit balance and try again.",
            ) from None
        raise _api_error(502, "upstream_error", "Something went wrong. Please try again.") from None

    result["paid_by"] = "visitor" if own_key else "free"
    if not own_key:
        result["free_remaining"] = min(
            quotas.free_per_ip.remaining(ip), quotas.free_total.remaining("all")
        )
    return result


# Mounted last so it doesn't shadow the API routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

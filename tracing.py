"""
tracing.py
----------
Optional Phoenix / OpenTelemetry tracing. Off unless PHOENIX_COLLECTOR_ENDPOINT is set
(for example http://localhost:6006), so the deployed site never records visitor questions.

    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 python query.py "..."

Each question becomes one trace:
    f1.query        (CHAIN)      question in, answer out, plus the evidence as JSON attributes
      f1.retrieve   (RETRIEVER)  the chunks with rank, distance and metadata
      Anthropic call             captured by the OpenInference Anthropic instrumentor

The API key is never put on a span: it is only passed to the Anthropic client, and the
instrumentor records messages and model parameters, not credentials.
"""

from __future__ import annotations

import contextlib
import json
import os

_tracer = None
_initialised = False
PROJECT = os.environ.get("PHOENIX_PROJECT_NAME", "f1-qualifying-rag")


def init_tracing() -> bool:
    """Start tracing if an endpoint is configured. Safe to call more than once."""
    global _tracer, _initialised
    if _initialised:
        return _tracer is not None
    _initialised = True
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return False
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from phoenix.otel import register
        provider = register(project_name=PROJECT, endpoint=endpoint.rstrip("/") + "/v1/traces",
                            batch=False, auto_instrument=False, verbose=False)
        AnthropicInstrumentor().instrument(tracer_provider=provider)
        _tracer = provider.get_tracer("f1-rag")
    except Exception as exc:                       # tracing must never break answering
        print(f"[tracing disabled: {type(exc).__name__}]")
        _tracer = None
    return _tracer is not None


def enabled() -> bool:
    return _tracer is not None


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", input_value: str | None = None):
    """A span when tracing is on, otherwise a no-op that yields None."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as sp:
        sp.set_attribute("openinference.span.kind", kind)
        if input_value is not None:
            sp.set_attribute("input.value", input_value)
        yield sp


def set_attrs(sp, **attrs) -> None:
    if sp is None:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        sp.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str))


def record_documents(sp, chunks: list[dict]) -> None:
    """Retrieved chunks in the OpenInference retrieval format, so Phoenix shows them as documents."""
    if sp is None:
        return
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        pre = f"retrieval.documents.{i}.document"
        sp.set_attribute(f"{pre}.id", str(meta.get("event_id") or c.get("id") or i))
        sp.set_attribute(f"{pre}.content", c.get("text", ""))
        if c.get("distance") is not None:
            sp.set_attribute(f"{pre}.score", float(c["distance"]))
        sp.set_attribute(f"{pre}.metadata", json.dumps(
            {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}, default=str))

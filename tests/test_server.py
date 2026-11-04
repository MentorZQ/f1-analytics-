"""
Tests for the public web API. No API key is needed and no paid call is made: the
LLM call is replaced, while retrieval runs for real against a copy of the store.

Run from the project root:
    pytest tests/test_server.py
"""

import logging
import re
import shutil
import sys
from pathlib import Path

import anthropic
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import query as query_module  # noqa: E402
import server  # noqa: E402

VISITOR_KEY = "sk-ant-visitor-key-abcdefghijklmnop"
QUESTION = "Where did Norris lose time to Russell at Barcelona?"


@pytest.fixture(scope="module", autouse=True)
def store_copy(tmp_path_factory):
    # Newer chromadb versions rewrite the store on open, so tests use a copy.
    copy = tmp_path_factory.mktemp("chroma") / "store"
    shutil.copytree(ROOT / "RAG_data_layers/chroma_store", copy)
    patcher = pytest.MonkeyPatch()
    patcher.setenv("CHROMA_PATH", str(copy))
    patcher.delenv("CHROMA_HOST", raising=False)
    query_module._get_chroma_client.cache_clear()
    query_module._get_session_context.cache_clear()
    yield
    patcher.undo()
    query_module._get_chroma_client.cache_clear()
    query_module._get_session_context.cache_clear()


@pytest.fixture
def llm_calls(monkeypatch):
    """Replace the paid LLM call and record what it was given."""
    calls = []

    def fake_call_llm(prompt, model="", api_key=None):
        calls.append({"prompt": prompt, "model": model, "api_key": api_key})
        return "A grounded answer."

    monkeypatch.setattr(query_module, "call_llm", fake_call_llm)
    return calls


@pytest.fixture
def client(monkeypatch, llm_calls):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-key")
    monkeypatch.setattr(server, "quotas", server.Quotas(2, 3, 2))
    monkeypatch.setattr(server, "EXAMPLES", [])
    return TestClient(server.app)


def post(client, question=QUESTION, session="barcelona", key=None, ip=None):
    headers = {}
    if key:
        headers["X-Anthropic-Key"] = key
    if ip:
        headers["X-Forwarded-For"] = ip
    return client.post("/query", json={"question": question, "session": session}, headers=headers)


def code_of(response):
    return response.json()["detail"]["code"]


def failing_llm(exception_type):
    def call(prompt, model="", api_key=None):
        # The constructor needs a live HTTP response; only the type matters here.
        raise exception_type.__new__(exception_type)

    return call


# ── Answers and evidence ─────────────────────────────────────────────────────

def test_answer_includes_sources_and_chart_data(client):
    data = post(client).json()
    assert data["answer"] == "A grounded answer."
    assert data["session"] == "barcelona"
    assert data["paid_by"] == "free"
    assert data["free_remaining"] == 1
    retrieved = [x for x in data["sources"] if x["chunk_type"] != "lap_comparison"]
    assert 1 <= len(retrieved) <= server.N_CHUNKS                 # at most N retrieved chunks, plus the fixed comparison block
    assert len(data["sources"]) - len(retrieved) <= 1
    assert {data["deltas"]["driver_a"], data["deltas"]["driver_b"]} == {"RUS", "NOR"}


def test_lap_comparison_matches_known_pole_gap(client):
    deltas = post(client, "Why did Russell take pole?").json()["deltas"]
    by_section = {s["section"]: s for s in deltas["sections"]}

    assert (deltas["driver_a"], deltas["driver_b"]) == ("RUS", "HAM")
    assert deltas["lap_gap_s"] == pytest.approx(0.064, abs=0.002)
    # From the stored text (OpenF1 data, official corners): "RUS faster by 0.125s" (T12), "HAM faster by 0.037s".
    assert by_section["T12"]["gain_s"] == pytest.approx(0.125)
    assert by_section["Main straight (start, before T1)"]["gain_s"] == pytest.approx(-0.037)


def test_sections_are_in_lap_order(client):
    sections = post(client, "Why did Russell take pole?").json()["deltas"]["sections"]
    starts = [s["start_m"] for s in sections]
    assert starts == sorted(starts)
    assert starts[0] == 0
    assert any(s["retrieved"] for s in sections)


def test_client_cannot_choose_model_or_chunk_count(client, llm_calls):
    response = client.post("/query", json={
        "question": QUESTION, "session": "barcelona", "model": "claude-fable-5-1", "chunks": 500,
    })
    assert response.status_code == 200
    assert llm_calls[0]["model"] == server.MODEL
    sources = response.json()["sources"]
    assert len([x for x in sources if x["chunk_type"] != "lap_comparison"]) <= server.N_CHUNKS
    assert len(sources) <= server.N_CHUNKS + 1


@pytest.mark.parametrize("question", ["", "x" * (server.MAX_QUESTION_CHARS + 1)])
def test_rejects_question_of_invalid_length(client, llm_calls, question):
    assert post(client, question).status_code == 422
    assert llm_calls == []


def test_rejects_blank_question_and_unknown_session(client, llm_calls):
    assert code_of(post(client, "   ")) == "empty_question"
    assert code_of(post(client, session="silverstone")) == "unknown_session"
    assert llm_calls == []


# ── Saved examples ───────────────────────────────────────────────────────────

SAVED = {
    "session": "barcelona",
    "question": "Why did Russell take pole?",
    "answer": "Saved text.",
    "sources": [],
    "deltas": None,
}


def test_saved_example_makes_no_call_and_uses_no_quota(client, monkeypatch, llm_calls):
    monkeypatch.setattr(server, "EXAMPLES", [SAVED])
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    for _ in range(5):
        data = post(client, "  why did RUSSELL take pole? ", session=None).json()
        assert (data["answer"], data["paid_by"]) == ("Saved text.", "saved")
    assert llm_calls == []
    assert server.quotas.free_per_ip.remaining("testclient") == 2


def test_saved_example_is_not_served_for_a_different_session(client, monkeypatch):
    monkeypatch.setattr(server, "EXAMPLES", [SAVED])
    data = post(client, "Why did Russell take pole?", session="monza").json()
    assert data["paid_by"] == "free"


def test_example_without_saved_answer_falls_back_to_live(client, monkeypatch):
    monkeypatch.setattr(server, "EXAMPLES", [{"session": "barcelona", "question": QUESTION}])
    assert post(client).json()["paid_by"] == "free"


def test_shipped_examples_all_have_a_driver_comparison(client):
    for example in server._load_examples():
        data = post(client, example["question"], example["session"], key=VISITOR_KEY).json()
        assert data["deltas"], example["question"]
        server.quotas.own_key.give_back("testclient")


# ── Quotas ───────────────────────────────────────────────────────────────────

def test_visitor_limit_then_own_key_still_works(client, llm_calls):
    assert post(client).status_code == 200
    assert post(client).status_code == 200

    blocked = post(client)
    assert blocked.status_code == 429
    assert code_of(blocked) == "free_limit_reached"
    assert len(llm_calls) == 2

    data = post(client, key=VISITOR_KEY).json()
    assert data["paid_by"] == "visitor"
    assert "free_remaining" not in data


def test_total_daily_budget_applies_across_visitors(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        assert post(client, ip=ip).status_code == 200

    blocked = post(client, ip="4.4.4.4")
    assert code_of(blocked) == "daily_budget_reached"
    # The refused visitor keeps their own allowance.
    assert server.quotas.free_per_ip.remaining("4.4.4.4") == 2


def test_failed_call_does_not_use_up_quota(client, monkeypatch):
    monkeypatch.setattr(query_module, "call_llm", failing_llm(anthropic.APIConnectionError))
    for _ in range(4):
        assert post(client).status_code == 502
    assert server.quotas.free_per_ip.remaining("testclient") == 2
    assert server.quotas.free_total.remaining("all") == 3


def test_empty_answer_is_an_error_and_uses_no_quota(client, monkeypatch):
    monkeypatch.setattr(query_module, "call_llm", lambda *a, **k: "  ")
    assert code_of(post(client)) == "empty_answer"
    assert server.quotas.free_per_ip.remaining("testclient") == 2


def test_own_key_requests_are_limited_too(client):
    assert post(client, key=VISITOR_KEY).status_code == 200
    assert post(client, key=VISITOR_KEY).status_code == 200
    assert code_of(post(client, key=VISITOR_KEY)) == "too_many_requests"


def test_forged_forwarded_address_cannot_dodge_the_limit(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert post(client, ip="9.9.9.1, 7.7.7.7").status_code == 200
    assert post(client, ip="9.9.9.2, 7.7.7.7").status_code == 200
    assert code_of(post(client, ip="9.9.9.3, 7.7.7.7")) == "free_limit_reached"


def test_forwarded_header_is_ignored_unless_trusted(client, monkeypatch):
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert post(client, ip="1.1.1.1").status_code == 200
    assert post(client, ip="2.2.2.2").status_code == 200
    assert code_of(post(client, ip="3.3.3.3")) == "free_limit_reached"


# ── Keys ─────────────────────────────────────────────────────────────────────

def test_own_key_is_passed_for_that_call_only(client, llm_calls):
    post(client, key=VISITOR_KEY)
    post(client)
    assert llm_calls[0]["api_key"] == VISITOR_KEY
    assert llm_calls[1]["api_key"] is None


def test_no_server_key_offers_own_key(client, monkeypatch, llm_calls):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert code_of(post(client)) == "no_server_key"
    assert llm_calls == []
    assert post(client, key=VISITOR_KEY).status_code == 200


def test_malformed_key_is_rejected_without_being_echoed(client, llm_calls):
    response = post(client, key="not a real key!")
    assert code_of(response) == "invalid_key"
    assert "not a real key" not in response.text
    assert llm_calls == []


def test_rejected_own_key_reports_invalid_key(client, monkeypatch):
    monkeypatch.setattr(query_module, "call_llm", failing_llm(anthropic.AuthenticationError))
    response = post(client, key=VISITOR_KEY)
    assert response.status_code == 401
    assert code_of(response) == "invalid_key"


def test_rejected_server_key_is_not_blamed_on_the_visitor(client, monkeypatch):
    monkeypatch.setattr(query_module, "call_llm", failing_llm(anthropic.AuthenticationError))
    response = post(client)
    assert response.status_code == 502
    assert code_of(response) == "upstream_error"


def test_keys_never_reach_logs_or_error_text(client, monkeypatch, caplog):
    def leaky(prompt, model="", api_key=None):
        raise ValueError(f"failed with {api_key}")

    monkeypatch.setattr(query_module, "call_llm", leaky)
    with caplog.at_level(logging.DEBUG):
        response = post(client, key=VISITOR_KEY)
    assert response.status_code == 502
    assert VISITOR_KEY not in response.text
    assert VISITOR_KEY not in caplog.text


# ── Page ─────────────────────────────────────────────────────────────────────

def test_page_and_assets_are_served_with_a_strict_policy(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "default-src 'self'" in page.headers["content-security-policy"]
    # The policy blocks inline code, so the page must not rely on any.
    assert not re.search(r"<script(?![^>]*\bsrc=)|\sstyle=|\son\w+=", page.text)
    for asset in re.findall(r'(?:src|href)="((?!https?:)[^"]+)"', page.text):
        assert client.get(f"/{asset}").status_code == 200, asset


def test_sessions_endpoint_lists_every_session_with_a_label(client):
    data = client.get("/api/sessions").json()
    assert {s["key"] for s in data["sessions"]} == set(query_module.SESSIONS)
    assert all(s["key"] in server.SESSION_LABELS for s in data["sessions"])


def test_api_docs_are_not_exposed(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


# ── CLI compatibility ────────────────────────────────────────────────────────

def test_cli_answer_still_returns_plain_text(llm_calls):
    assert query_module.answer(QUESTION, "barcelona") == "A grounded answer."
    assert query_module.answer(QUESTION, "silverstone").startswith("Unknown session")


def test_haiku_is_called_with_thinking_disabled(monkeypatch):
    """Haiku 5.5 thinks by default and the thinking consumes the whole token budget, truncating answers."""
    import types
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="ok")],
                                         usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))

    class FakeClient:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    query_module.call_llm("prompt", model="claude-haiku-5-5")
    assert captured["thinking"] == {"type": "disabled"}
    captured.clear()
    query_module.call_llm("prompt", model="claude-sonnet-4-6")
    assert "thinking" not in captured

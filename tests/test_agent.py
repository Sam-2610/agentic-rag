"""Fast tests: no API key, no internet. The LLM is replaced by a scripted fake."""
import pytest
from fastapi.testclient import TestClient

import agent
import api
import llm
import sandbox

GOOD = """```python
rows = [l for l in CONTEXT.splitlines() if l.strip().startswith('| North')]
print('North Q1 =', [c.strip() for c in rows[0].strip('|').split('|')][1])
```"""
BROKEN = "```python\nprint(undefined_variable)\n```"
CHART = """```python
import matplotlib.pyplot as plt
plt.bar(['Q1', 'Q2'], [1, 2])
plt.savefig('chart.png')
print('chart saved')
```"""


class FakeLLM:
    def __init__(self, intent="retrieve", code_replies=()):
        self.intent, self.code_replies, self.calls = intent, list(code_replies), []

    def __call__(self, messages, task="general", temperature=0.1):
        self.calls.append((task, messages))
        if task == "router":
            return '{"intent": "%s", "reason": "test"}' % self.intent
        if task in ("write_code", "fix_code"):
            return self.code_replies.pop(0)
        if task == "summarize":
            return "North Q1 revenue is 420 thousand dollars [1]."
        if task == "answer":
            return "Maria Alvarez is the CEO [1]."
        return "Hello! I can answer questions about your documents."


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        f = FakeLLM(**kw)
        monkeypatch.setattr(llm, "chat", f)
        return f
    return install


def test_sandbox_runs_valid_code():
    res = sandbox.run("print(sum([1, 2, 3]))", "ctx")
    assert res["ok"] and res["stdout"].strip() == "6"


@pytest.mark.parametrize("code", [
    "import os\nprint(os.listdir('.'))",
    "print(open('context.txt').read())",
    "import pandas as pd\nprint(pd.read_csv('context.txt'))",
    "print(().__class__.__bases__)",
    "eval('1+1')",
])
def test_sandbox_blocks_unsafe_code(code):
    res = sandbox.run(code, "ctx")
    assert not res["ok"] and res["error"]


def test_sandbox_timeout():
    res = sandbox.run("while True:\n    pass", "ctx", timeout=2)
    assert not res["ok"] and "Timed out" in res["error"]


def test_sandbox_chart_artifact():
    res = sandbox.run(CHART.strip("`").replace("python\n", "", 1), "ctx")
    assert res["ok"] and any(a.endswith(".png") for a in res["artifacts"])


def test_greeting_skips_router_llm(fake):
    f = fake()
    out = agent.ask("hello", "t-greeting")
    assert out["intent"] == "chat" and [c[0] for c in f.calls] == ["chat"]


def test_router_fallback_rules(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "not json")
    assert agent.classify("plot revenue as a chart", [])[0] == "analyze"
    assert agent.classify("who is the CEO", [])[0] == "retrieve"


def test_retrieve_path_cites_sources(fake):
    fake(intent="retrieve")
    out = agent.ask("Who is the CEO of Northwind Traders?", "t-retrieve")
    assert "Alvarez" in out["answer"] and out["sources"]


def test_analyze_retries_after_error(fake):
    f = fake(intent="analyze", code_replies=[BROKEN, GOOD])
    out = agent.ask("What was North revenue in Q1?", "t-retry")
    assert [c[0] for c in f.calls].count("fix_code") == 1 and "420" in out["answer"]
    assert any("attempt 2): ok" in t for t in out["trace"])


def test_analyze_falls_back_to_text(fake):
    fake(intent="analyze", code_replies=[BROKEN] * 3)
    out = agent.ask("What was North revenue in Q1?", "t-fallback")
    assert any("falling back" in t for t in out["trace"]) and out["answer"]


def test_analyze_chart_returns_artifact(monkeypatch, fake):
    fake(intent="analyze", code_replies=[CHART])
    hit = {"text": "| North | 420 | 450 |", "source": "s.md", "page": 1, "score": 0.9}
    monkeypatch.setattr(agent.store, "search", lambda q, **kw: [hit])
    out = agent.ask("Plot revenue as a bar chart", "t-chart")
    assert any(a.endswith(".png") for a in out["artifacts"])


def test_out_of_scope_refused(monkeypatch, fake):
    fake(intent="retrieve")
    monkeypatch.setattr(agent.store, "search", lambda q, **kw: [])
    assert agent.ask("Who won the 2023 cricket world cup?", "t-refuse")["answer"] == agent.NOT_FOUND_TEXT


def test_memory_per_session(fake):
    f = fake(intent="chat")
    agent.ask("My favorite region is East.", "mem-a")
    agent.ask("Which region did I say I like?", "mem-a")
    assert any("favorite region is East" in m["content"] for m in f.calls[-1][1])
    agent.ask("Which region did I say I like?", "mem-b")
    assert not any("favorite region is East" in m["content"] for m in f.calls[-1][1])


def test_api(fake):
    fake(intent="retrieve")
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "hi", "session_id": "../x"}).status_code == 422
    body = client.post("/chat", json={"question": "Who is the CEO of Northwind Traders?", "session_id": "a1"}).json()
    assert body["intent"] == "retrieve" and body["sources"]
    assert client.get("/artifacts/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_analyze_keeps_low_score_chunks(monkeypatch, fake):
    seen = {}
    fake(intent="analyze", code_replies=[GOOD])
    hit = {"text": "| North | 420 | 450 |", "source": "s.md", "page": 1, "score": 0.1}

    def search(q, **kw):
        seen.update(query=q, min_score=kw.get("min_score"))
        return [hit]

    monkeypatch.setattr(agent.store, "search", search)
    out = agent.ask("Plot revenue by quarter as a bar chart", "t-lowscore")
    assert seen["min_score"] == 0.0 and "plot" not in seen["query"].lower()
    assert any("write_code" in t for t in out["trace"])


def test_chart_request_overrides_router(monkeypatch, fake):
    fake(intent="retrieve", code_replies=[CHART])  # the LLM wrongly says "retrieve"
    hit = {"text": "| North | 420 |", "source": "s.md", "page": 1, "score": 0.2}
    monkeypatch.setattr(agent.store, "search", lambda q, **kw: [hit])
    out = agent.ask("Plot revenue as a chart", "t-override")
    assert out["intent"] == "analyze" and any(a.endswith(".png") for a in out["artifacts"])
"""AgentStore 单测（离线）：runs 快照 + session_turns 记忆 + 孤儿检测，全用临时 SQLite 文件。"""
import json

import pytest

from note_assistant.agent.store import AgentStore
from note_assistant.config import settings


@pytest.fixture
def store(tmp_path):
    s = AgentStore(tmp_path / "agent.sqlite")
    s.reset()
    yield s
    s.reset()


def test_create_and_get_run(store):
    rid = store.create_run("什么是 RAG？")
    assert isinstance(rid, str) and len(rid) == 32  # uuid4.hex
    run = store.get_run(rid)
    assert run["run_id"] == rid
    assert run["question"] == "什么是 RAG？"
    assert run["status"] == "running"
    assert run["answer"] == ""
    assert run["trajectory"] == []


def test_append_event_and_finish(store):
    rid = store.create_run("q")
    store.append_event(rid, {"type": "thought", "content": "x"}, 0)
    store.append_event(rid, {"type": "answer", "content": "a"}, 1)
    store.set_answer(rid, "a")
    store.finish_run(rid)
    run = store.get_run(rid)
    assert run["status"] == "finished"
    assert run["answer"] == "a"
    assert [e["type"] for e in run["trajectory"]] == ["thought", "answer"]


def test_ensure_run_idempotent(store):
    rid = "fixed-id-123"
    store.ensure_run(rid, "q1")
    store.ensure_run(rid, "q2")  # 不应覆盖已有 question
    run = store.get_run(rid)
    assert run is not None
    assert run["question"] == "q1"


def test_interrupted_when_orphan_ttl_passed(monkeypatch, store):
    # 把 TTL 设为负数 → 任何 running 的 run 都判为 interrupted
    monkeypatch.setattr(settings, "agent_run_orphan_ttl", -1)
    rid = store.create_run("q")
    run = store.get_run(rid)
    assert run["status"] == "interrupted"


def test_session_turns_roundtrip(store):
    store.append_turn("s1", "user", "你好")
    store.append_turn("s1", "assistant", "我是助手")
    store.append_turn("s1", "user", "再来一次")
    hist = store.get_history("s1")
    assert [h["role"] for h in hist] == ["user", "assistant", "user"]
    assert hist[0]["content"] == "你好"
    assert hist[-1]["content"] == "再来一次"
    # 另一个 session 互不影响
    assert store.get_history("s2") == []


def test_get_run_missing_returns_none(store):
    assert store.get_run("nope") is None


def test_sources_json_roundtrip(store):
    rid = store.create_run("q")
    src = [{"filepath": "a.md", "title": "A"}]
    store.set_sources(rid, src)
    store.finish_run(rid)
    run = store.get_run(rid)
    assert run["sources"] == src
    # 确保是合法 JSON 往返
    assert json.loads(json.dumps(run["sources"])) == src

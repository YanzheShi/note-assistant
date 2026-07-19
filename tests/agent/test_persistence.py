"""Agent 持久化端到端测试（离线）：注入临时 SQLite store + fake LLM。

覆盖：
- ainvoke 产出 run_id，且 run 快照可回查（finished + 完整轨迹 + 来源）
- 跨会话记忆：同 session_id 两次调用，历史持久化且服务端持有
- 断流续传：用已 finished 的 run_id 重放 astream，回放完整轨迹
- 缓存命中也会登记 run（cached=True）
- agent_session_enabled=False → 退化为无状态（run_id 为空、不写历史）
"""
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent import agent as agent_mod
from note_assistant.agent.runner import (
    ainvoke, astream, get_store, reset_cache, reset_store, set_store_for_test,
)
from note_assistant.agent.store import AgentStore
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


class FakeAgentLLM(BaseChatModel):
    @property
    def _llm_type(self):
        return "fake-agent"

    @staticmethod
    def _role(messages):
        for m in messages:
            if isinstance(m, SystemMessage):
                c = m.content
                if "意图分类器" in c:
                    return "router"
                if "反思评判器" in c:
                    return "reflect"
                if "闲聊" in c:
                    return "chat"
                if "基于个人知识库的问答助手" in c:
                    return "generate"
                if "可用工具" in c:
                    return "agent"
        return "unknown"

    def _respond(self, role, messages):
        if role == "router":
            q = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
            needs = "你好" not in q
            return AIMessage(content=json.dumps({"needs_search": needs, "reason": "x"}))
        if role == "reflect":
            return AIMessage(content=json.dumps(
                {"verdict": "sufficient", "reason": "已有相关片段", "rewritten_query": ""}))
        if role == "chat":
            return AIMessage(content="我是你的知识库助手。")
        if role == "generate":
            return AIMessage(content="根据笔记，FlashAttention 通过分块减少显存占用。")
        if role == "agent":
            has_tool = any(isinstance(m, ToolMessage) for m in messages)
            if not has_tool:
                return AIMessage(content="", tool_calls=[{
                    "name": "hybrid_search", "args": {"query": "FlashAttention", "top_k": 2}, "id": "c1"}])
            return AIMessage(content="基于检索结果：FlashAttention 改进了显存。")
        return AIMessage(content="fallback")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_tool(name, args):
    results = [
        RetrievalResult(score=0.9, page_content="FA 背景", metadata={
            "title": "FlashAttention", "filepath": "fa.md", "heading_path": "一、背景"}),
        RetrievalResult(score=0.8, page_content="FA 改进", metadata={
            "title": "FlashAttention", "filepath": "fa2.md", "heading_path": "二、改进"}),
    ]
    return ("obs: 命中 2 段", results)


@pytest.fixture
def patched(tmp_path, monkeypatch):
    store = AgentStore(tmp_path / "agent.sqlite")
    store.reset()
    set_store_for_test(store)
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: FakeAgentLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call", _fake_tool)
    reset_cache()
    yield store
    reset_store()
    reset_cache()


@pytest.mark.asyncio
async def test_ainvoke_records_run(patched):
    result = await ainvoke("FlashAttention 是什么？", session_id="s1")
    assert result.run_id
    run = patched.get_run(result.run_id)
    assert run["status"] == "finished"
    assert "FlashAttention" in run["answer"]
    assert run["sources"]  # 来源已落盘
    types = [e["type"] for e in run["trajectory"]]
    assert "thought" in types and "tool_call" in types and "answer" in types


@pytest.mark.asyncio
async def test_cross_session_memory(patched):
    await ainvoke("FlashAttention 是什么？", session_id="s1")
    # 第二次带相同 session_id，且显式传空 history（证明服务端从 store 取历史）
    await ainvoke("FlashAttention 的改进点？", session_id="s1", history=[])
    hist = patched.get_history("s1")
    # 两轮 = 2 user + 2 assistant
    assert [h["role"] for h in hist] == ["user", "assistant", "user", "assistant"]
    assert hist[0]["content"] == "FlashAttention 是什么？"
    # 最终答案是 generate 节点产出（agent 节点的「基于检索结果」分支不会被走到）
    assert hist[-1]["content"] == "根据笔记，FlashAttention 通过分块减少显存占用。"


@pytest.mark.asyncio
async def test_resume_finished_run_via_astream(patched):
    result = await ainvoke("FlashAttention 是什么？", session_id="s1")
    # 断流后用同一 run_id 重放
    events = [e async for e in astream("FlashAttention 是什么？", run_id=result.run_id)]
    assert events[0]["type"] == "run"
    assert events[0]["resumable"] is True
    types = [e["type"] for e in events]
    assert "sources" in types
    assert "answer" in types
    # 重放不应再次触发新的工具检索（轨迹来自已落盘的快照）
    assert types.count("tool_call") >= 1


@pytest.mark.asyncio
async def test_cache_hit_still_records_run(patched):
    await ainvoke("FlashAttention 是什么？", session_id="s1")
    r2 = await ainvoke("FlashAttention 是什么？", session_id="s1")
    assert r2.cached is True
    assert r2.run_id  # 缓存命中也登记了 run
    run = patched.get_run(r2.run_id)
    assert run["status"] == "finished"


@pytest.mark.asyncio
async def test_session_disabled_falls_back_to_stateless(monkeypatch, tmp_path):
    import note_assistant.agent.agent as agent_mod
    # 关掉持久化但仍需用 fake LLM，避免打到真实 API
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: FakeAgentLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call", _fake_tool)
    monkeypatch.setattr(settings, "agent_session_enabled", False)
    reset_store()  # 清全局，强制 get_store() 走 settings 判定
    # 无临时 store 注入（disabled）→ get_store 返回 None
    assert get_store() is None
    result = await ainvoke("FlashAttention 是什么？")
    assert result.run_id == ""  # 退化为无状态
    reset_store()
    monkeypatch.setattr(settings, "agent_session_enabled", True)

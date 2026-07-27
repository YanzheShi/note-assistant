"""Agent runner 端到端测试（离线）：用 fake LLM + 假工具跑通完整 StateGraph。

不依赖 Ollama / DeepSeek / ChromaDB，覆盖：
- 检索路径：Router → agent → tools → reflect(Judge) → generate，轨迹/来源/答案齐全
- 闲聊路径：Router → direct_chat，不检索
- 语义缓存：第二次命中返回 cached=True
- 流式：astream 逐项产出事件
"""
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent.context import set_context_manager_for_test
from note_assistant.agent.runner import ainvoke, astream, reset_cache
from note_assistant.retrieval.types import RetrievalResult


class FakeAgentLLM(BaseChatModel):
    """根据 SystemMessage 内容识别节点角色，返回脚本化响应。"""

    @property
    def _llm_type(self):
        return "fake-agent"

    @staticmethod
    def _role(messages):
        for m in messages:
            if isinstance(m, (SystemMessage, HumanMessage)):
                c = m.content
                if "意图分类器" in c:
                    return "router"
                if "检索质量" in c:
                    return "reflect"
                if "闲聊" in c:
                    return "chat"
                if "基于个人知识库的问答助手" in c:
                    return "generate"
                if "可用工具" in c:
                    return "agent"
                if "对话改写器" in c:
                    return "condense"
        return "unknown"

    def _respond(self, role, messages):
        if role == "router":
            q = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
            needs = "你好" not in q  # 含"你好"视为闲聊
            return AIMessage(content=json.dumps({"needs_search": needs, "reason": "x"}))
        if role == "reflect":
            return AIMessage(content=json.dumps({
                "verdict": "sufficient", "reason": "已有相关片段", "rewritten_query": ""
            }))
        if role == "chat":
            return AIMessage(content="我是你的知识库助手。")
        if role == "generate":
            return AIMessage(content="根据笔记，FlashAttention 通过分块减少显存占用。")
        if role == "agent":
            has_tool = any(isinstance(m, ToolMessage) for m in messages)
            if not has_tool:
                return AIMessage(content="", tool_calls=[{
                    "name": "hybrid_search",
                    "args": {"query": "FlashAttention", "top_k": 2},
                    "id": "call_1",
                }])
            return AIMessage(content="基于检索结果：FlashAttention 改进了显存。")
        if role == "condense":
            # 返回空 → context.py 降级用原问题，保持 e2e 行为稳定
            return AIMessage(content="")
        return AIMessage(content="fallback")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_tool(name, args):
    results = [
        RetrievalResult(score=0.9, page_content="FA 背景内容", metadata={
            "title": "FlashAttention", "filepath": "fa.md", "heading_path": "一、背景"}),
        RetrievalResult(score=0.8, page_content="FA 改进内容", metadata={
            "title": "FlashAttention", "filepath": "fa2.md", "heading_path": "二、改进"}),
    ]
    return ("obs: 命中 2 段", results)


@pytest.fixture
def patched(monkeypatch):
    import note_assistant.agent.agent as agent_mod
    from note_assistant.agent.runner import reset_store
    from note_assistant.config import settings
    from unittest.mock import MagicMock
    # 关掉持久化，避免 e2e 测试触碰真实 SQLite 文件（保持无副作用、快速）
    monkeypatch.setattr(settings, "agent_session_enabled", False)
    # 关掉相关性裁剪，避免 e2e 触碰 Ollama embedder（离线、确定性）
    monkeypatch.setattr(settings, "agent_history_relevance_enabled", False)
    # 关掉自动图扩展：避免 e2e 触碰真实图谱数据 / ChromaDB
    monkeypatch.setattr(settings, "agent_graph_expand_enabled", False)
    # mock reranker 工厂：graph 中仍保留 rerank 节点（双层精排结构不变），
    # 但用 mock 替代 1.1GB 模型加载，保证离线、快速、确定
    _fake_reranker = MagicMock()
    _fake_reranker.rerank.side_effect = (
        lambda q, results, top_k=None: results if top_k is None else results[:top_k]
    )
    monkeypatch.setattr(agent_mod, "get_reranker", lambda *a, **k: _fake_reranker)
    reset_store()
    # 当前 settings（含上述开关）已定稿，清空 build_graph 缓存以便用最新开关重建 graph
    agent_mod.build_graph.cache_clear()
    set_context_manager_for_test(None)  # 每次按当前 settings 懒重建 ContextManager
    # context.py 的凝练走 note_assistant.llm.client.get_llm，需一并 mock 才离线
    monkeypatch.setattr("note_assistant.llm.client.get_llm", lambda *a, **k: FakeAgentLLM())
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: FakeAgentLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call", _fake_tool)
    reset_cache()
    yield
    monkeypatch.setattr(settings, "agent_session_enabled", True)


@pytest.mark.asyncio
async def test_ainvoke_search_path(patched):
    result = await ainvoke("FlashAttention 是什么？")
    assert "FlashAttention" in result.answer
    assert len(result.sources) == 2
    types = [t["type"] for t in result.trajectory]
    assert types[0] == "thought"            # 路由判定
    assert "tool_call" in types
    assert "observation" in types
    assert "judge" in types
    assert "answer" in types
    judges = [t for t in result.trajectory if t["type"] == "judge"]
    assert judges[0]["verdict"] == "sufficient"
    assert result.cached is False


@pytest.mark.asyncio
async def test_ainvoke_chat_path(patched):
    result = await ainvoke("你好")
    assert "助手" in result.answer
    types = [t["type"] for t in result.trajectory]
    assert "thought" in types
    assert "answer" in types
    # 闲聊不应进入检索
    assert not any(t["type"] == "tool_call" for t in result.trajectory)
    assert result.sources == []


@pytest.mark.asyncio
async def test_cache_hit_second_time(patched):
    r1 = await ainvoke("FlashAttention 是什么？")
    r2 = await ainvoke("FlashAttention 是什么？")  # 相同问题 → 命中缓存
    assert r2.cached is True
    assert r2.answer == r1.answer
    assert r2.trajectory == r1.trajectory


@pytest.mark.asyncio
async def test_astream_search_path(patched):
    events = [e async for e in astream("FlashAttention 是什么？")]
    types = [e["type"] for e in events]
    assert "thought" in types
    assert "tool_call" in types
    assert "observation" in types
    assert "judge" in types
    assert "answer" in types
    assert "sources" in types
    answers = [e for e in events if e["type"] == "answer"]
    assert answers and "FlashAttention" in answers[-1]["content"]
    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert len(sources_events[0]["sources"]) == 2

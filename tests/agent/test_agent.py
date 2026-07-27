"""Agentic RAG 测试（不依赖外部服务 Ollama/DeepSeek/ChromaDB 的纯逻辑）。"""
import asyncio
import logging

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent import agent as agent_mod
from note_assistant.agent.agent import (
    _agent_branch,
    _extract_json,
    _norm_verdict,
    _reflect_branch,
    _route_branch,
    _top_k_context,
    agent_node,
    tools_node,
)
from note_assistant.agent.runner import _initial_state, _log_task_exception
from note_assistant.agent.tools import _format_results
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


def _mk(score: float, filepath: str = "a.md", heading: str = "h") -> RetrievalResult:
    return RetrievalResult(
        score=score,
        page_content=f"content-{score}",
        metadata={"title": "T", "filepath": filepath, "heading_path": heading},
    )


def _ai_with_tools(*calls):
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[
        {"name": c["name"], "args": c.get("args", {}), "id": c.get("id", f"c{i}")}
        for i, c in enumerate(calls)
    ])


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def test_extract_json_strips_code_fence():
    assert _extract_json('```json\n{"needs_search": true}\n```') == '{"needs_search": true}'


def test_extract_json_plain():
    assert _extract_json('{"needs_search": false}') == '{"needs_search": false}'


@pytest.mark.parametrize("v,expect", [
    ("sufficient", "sufficient"),
    ("need_rewrite", "need_rewrite"),
    ("need_more", "need_more"),
    ("give_up", "give_up"),
    ("垃圾", "sufficient"),
    ("", "sufficient"),
    ("NEED_MORE", "need_more"),
])
def test_norm_verdict(v, expect):
    assert _norm_verdict(v) == expect


def test_top_k_context_sorts_and_truncates():
    results = [_mk(0.2), _mk(0.9), _mk(0.5), _mk(0.1)]
    out = _top_k_context(results)
    assert [r.score for r in out] == [0.9, 0.5, 0.2, 0.1]
    # 超过 top_k_rerank 会截断
    assert len(_top_k_context(results[:settings.top_k_rerank + 3])) <= settings.top_k_rerank


def test_format_results_includes_source_path():
    text = _format_results([_mk(0.8)])
    assert "来源路径: a.md" in text
    assert "content-0.8" in text


def test_format_results_empty():
    assert "未检索到" in _format_results([])


def test_build_initial_state_maps_history():
    state = _initial_state(
        "问题?", history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    assert state["question"] == "问题?"
    # 当前问题作为首条消息；历史单独存放在 state["history"]，生成阶段注入
    assert len(state["messages"]) == 1
    assert state["history"] == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}
    ]
    assert state["accumulated"] == []
    assert state["iteration"] == 0


# ──────────────────────────────────────────────
# 分支逻辑（纯函数）
# ──────────────────────────────────────────────

def test_route_branch():
    assert _route_branch({"route": "search"}) == "search"
    assert _route_branch({"route": "chat"}) == "chat"


def test_agent_branch():
    assert _agent_branch({"messages": [_ai_with_tools({"name": "hybrid_search"})]}) == "tools"
    from langchain_core.messages import AIMessage
    assert _agent_branch({"messages": [AIMessage(content="直接回答")]}) == "generate"


def test_reflect_branch_under_max():
    # 未达上限：sufficient / give_up → 生成；need_* → 改写
    s = {"iteration": 1, "judge_verdict": "sufficient"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": 1, "judge_verdict": "give_up"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": 1, "judge_verdict": "need_rewrite"}
    assert _reflect_branch(s) == "rewrite"
    s = {"iteration": 1, "judge_verdict": "need_more"}
    assert _reflect_branch(s) == "rewrite"


def test_reflect_branch_at_max_forces_generate():
    # 达到 max_iter 无论如何强制生成（硬性降级）
    s = {"iteration": settings.agent_max_iter, "judge_verdict": "need_more"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": settings.agent_max_iter, "judge_verdict": "need_rewrite"}
    assert _reflect_branch(s) == "generate"


# ──────────────────────────────────────────────
# Context Accumulator 确定性去重（直接测 tools_node）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tools_node_dedup_and_iteration(monkeypatch):
    import note_assistant.agent.agent as agent_mod

    calls = [
        # 第一轮：a/b
        (_format_results([_mk(0.9, "a.md", "h1"), _mk(0.8, "b.md", "h2")]),
         [_mk(0.9, "a.md", "h1"), _mk(0.8, "b.md", "h2")]),
        # 第二轮：b（重复）/ c（新）
        (_format_results([_mk(0.7, "b.md", "h2"), _mk(0.6, "c.md", "h3")]),
         [_mk(0.7, "b.md", "h2"), _mk(0.6, "c.md", "h3")]),
    ]
    monkeypatch.setattr(agent_mod, "run_tool_call", lambda name, args: calls.pop(0))

    state = {
        "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "q"}, "id": "c1"})],
        "accumulated": [], "iteration": 0,
    }
    out1 = await tools_node(state)
    assert len(out1["accumulated"]) == 2
    assert out1["iteration"] == 1

    state2 = {
        "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "q2"}, "id": "c2"})],
        "accumulated": out1["accumulated"], "iteration": 1,
    }
    out2 = await tools_node(state2)
    # b 被去重，结果应为 a,b,c
    assert len(out2["accumulated"]) == 3
    fps = {(r.filepath, r.metadata.get("heading_path")) for r in out2["accumulated"]}
    assert fps == {("a.md", "h1"), ("b.md", "h2"), ("c.md", "h3")}
    assert out2["iteration"] == 2


@pytest.mark.asyncio
async def test_tools_node_no_tool_calls_returns_empty(monkeypatch):
    import note_assistant.agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_tool_call", lambda name, args: ("x", []))
    from langchain_core.messages import AIMessage
    out = await tools_node({"messages": [AIMessage(content="no tools")], "accumulated": [], "iteration": 0})
    assert out == {}


# ──────────────────────────────────────────────
# agent_node：当前问题使用凝练版（消指代），与 router/reflect/generate 同源
# ──────────────────────────────────────────────

class _CaptureLLM(BaseChatModel):
    """记录最后一次 ainvoke 收到的 messages，返回无工具调用的空 AIMessage。"""
    captured: list = []

    @property
    def _llm_type(self):
        return "capture"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _CaptureLLM.captured = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        _CaptureLLM.captured = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_agent_node_uses_condensed_question(monkeypatch):
    """agent_node 应向 LLM 发送凝练后的独立问题，而非原始未消解问题。"""
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    state = {
        "messages": [HumanMessage("它有什么缺点？")],
        "history_messages": [],
        "question": "它有什么缺点？",
        "condensed_question": "FlashAttention 有什么缺点？",
    }
    await agent_node(state)
    msgs = _CaptureLLM.captured
    user_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    assert user_msgs, "agent_node 必须向 LLM 发送用户问题"
    # 用户问题应为凝练版，而非原始未消解的「它有什么缺点？」
    assert user_msgs[0].content == "FlashAttention 有什么缺点？"
    assert HumanMessage("它有什么缺点？") not in msgs


@pytest.mark.asyncio
async def test_agent_node_keeps_in_run_messages(monkeypatch):
    """改写循环后再进 agent_node：保留本轮工具调用 / 观察，仅替换首条原始问题。"""
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    state = {
        "messages": [
            HumanMessage("它有什么缺点？"),
            _ai_with_tools({"name": "hybrid_search", "args": {"query": "x"}, "id": "c1"}),
            ToolMessage(content="obs", tool_call_id="c1"),
            AIMessage(content="（反思改写）重新检索"),
        ],
        "history_messages": [],
        "question": "它有什么缺点？",
        "condensed_question": "FlashAttention 有什么缺点？",
    }
    await agent_node(state)
    msgs = _CaptureLLM.captured
    # 凝练问题注入，且本轮内的工具调用 / 观察顺序保留
    assert HumanMessage("FlashAttention 有什么缺点？") in msgs
    assert any(isinstance(m, ToolMessage) and m.content == "obs" for m in msgs)
    # 原始未消解问题不应再出现
    assert HumanMessage("它有什么缺点？") not in msgs


@pytest.mark.asyncio
async def test_router_uses_original_question_not_condensed(monkeypatch):
    """回归：路由必须基于用户原始问题判意图，不能因凝练抹掉「笔记」信号而误判闲聊。

    当凝练版把「你看看我之前的笔记，…」改写成不含「笔记」的独立问题时，router 仍应收到
    原始问题（含「笔记」），而非被改写后的纯常识问法；凝练版仅作为消歧参考附上。
    """
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    state = {
        "question": "你看看我之前的笔记，上下文工程的五大策略有哪些？",
        "condensed_question": "上下文工程的五大策略有哪些？",  # 凝练抹掉了「笔记」
    }
    await agent_mod.router(state)
    user_msgs = [m for m in _CaptureLLM.captured if isinstance(m, HumanMessage)]
    assert user_msgs, "router 必须向 LLM 发送用户问题"
    assert "笔记" in user_msgs[0].content                       # 原始「笔记」信号保留
    assert "上下文工程的五大策略有哪些？" in user_msgs[0].content  # 凝练版作为消歧参考


# ──────────────────────────────────────────────
# 后台任务异常兜底
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_task_exception_logs_and_does_not_propagate(caplog):
    """后台任务抛异常时，_log_task_exception 兜底记日志而不向外抛。"""
    caplog.set_level(logging.WARNING)

    async def boom():
        raise RuntimeError("kaboom")

    task = asyncio.create_task(boom())
    task.add_done_callback(_log_task_exception)
    await asyncio.gather(task, return_exceptions=True)
    assert any("kaboom" in r.message for r in caplog.records)

# ──────────────────────────────────────────────
# Reranker 路由测试（出口开关控制 _reflect_branch 行为）
# ──────────────────────────────────────────────

def test_reflect_branch_routes_to_rerank_exit_when_enabled(monkeypatch):
    """rerank_exit 开启时，sufficient/give_up/达上限 应走 rerank_exit 而非 generate。"""
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", True)
    s = {"iteration": 1, "judge_verdict": "sufficient"}
    assert _reflect_branch(s) == "rerank_exit"
    s = {"iteration": 1, "judge_verdict": "give_up"}
    assert _reflect_branch(s) == "rerank_exit"
    # 达上限也应走 rerank_exit
    s = {"iteration": settings.agent_max_iter, "judge_verdict": "need_more"}
    assert _reflect_branch(s) == "rerank_exit"


def test_reflect_branch_routes_to_generate_when_disabled():
    """rerank_exit 关闭时，行为不变，仍走 generate。"""
    s = {"iteration": 1, "judge_verdict": "sufficient"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": 1, "judge_verdict": "give_up"}
    assert _reflect_branch(s) == "generate"


def test_rerank_loop_disabled_returns_empty():
    """rerank_loop 关闭时，直接透传，不做事。"""
    from note_assistant.agent.agent import rerank_loop
    import asyncio
    result = asyncio.run(rerank_loop({"accumulated": [], "condensed_question": "q"}))
    assert result == {}


def test_rerank_exit_disabled_returns_empty():
    """rerank_exit 关闭时，直接透传，不做事。"""
    from note_assistant.agent.agent import rerank_exit
    import asyncio
    result = asyncio.run(rerank_exit({"accumulated": [], "condensed_question": "q"}))
    assert result == {}

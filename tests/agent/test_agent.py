"""Agentic RAG 测试（不依赖外部服务 Ollama/DeepSeek/ChromaDB 的纯逻辑）。"""
import pytest

from note_assistant.agent.agent import (
    _agent_branch,
    _extract_json,
    _norm_verdict,
    _reflect_branch,
    _route_branch,
    _top_k_context,
    tools_node,
)
from note_assistant.agent.runner import _initial_state
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

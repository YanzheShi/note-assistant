"""force_search 兜底强制检索测试（2026-08-31 修「agentic 多轮后全部答当前不包含」）。

背景：多轮会话里 agent_node 的 LLM 看到 history 中自己上一轮的长答案后，常误以为
「资料已经有了」而直接输出纯文本（无 tool_calls）。旧行为直落 generate，导致本轮
零检索——生成只能吃上一轮 seed ×0.9 衰减后的陈旧片段。本测试锁定：

    - _agent_branch 硬闸门：search 路由 + 本轮未检索过 + 兜底未用尽 → force_search；
    - force_search_node：代码层跑 hybrid_search、identity_key 去重、标记置位、
      allowed_files 白名单、L2 注入计数、工具失败不崩溃、不 +1 iteration；
    - 图级：router(search) → agent(纯文本) → force_search → reflect → generate
      完整路径（公共下游保证检索结果过 rerank + Judge，不裸奔进生成）。
"""
import asyncio

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import note_assistant.agent.agent as agent_mod
from note_assistant.agent.agent import _agent_branch, build_graph, force_search_node
from note_assistant.agent.runner import _initial_state
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


def _mk(score: float, filepath: str = "a.md", heading: str = "h") -> RetrievalResult:
    return RetrievalResult(
        score=score,
        page_content=f"content-{score}",
        metadata={"title": "T", "filepath": filepath, "heading_path": heading},
    )


def _plain_ai(content: str = "直接回答") -> AIMessage:
    return AIMessage(content=content)


def _ai_with_tools(*calls):
    return AIMessage(content="", tool_calls=[
        {"name": c["name"], "args": c.get("args", {}), "id": c.get("id", f"c{i}")}
        for i, c in enumerate(calls)
    ])


# ──────────────────────────────────────────────
# _agent_branch 硬闸门
# ──────────────────────────────────────────────

def test_branch_with_tool_calls_still_tools():
    """有 tool_calls：正常路径不变，仍走 tools。"""
    s = {"messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "q"}})],
         "route": "search", "searched_once": False, "force_search_tries": 0}
    assert _agent_branch(s) == "tools"


def test_branch_unsearched_search_route_forces_search():
    """无 tool_calls + search 路由 + 本轮未检索过 → 强制兜底检索。"""
    s = {"messages": [_plain_ai()], "route": "search",
         "searched_once": False, "force_search_tries": 0}
    assert _agent_branch(s) == "force_search"


def test_branch_missing_flags_also_forces_search():
    """旧 state / 字段未初始化：searched_once 缺失视为未检索过 → 同样兜底（防漏网）。"""
    s = {"messages": [_plain_ai()], "route": "search"}
    assert _agent_branch(s) == "force_search"


def test_branch_chat_route_no_force():
    """chat 路由：闲聊不走检索，也不走兜底。"""
    s = {"messages": [_plain_ai()], "route": "chat",
         "searched_once": False, "force_search_tries": 0}
    assert _agent_branch(s) == "generate"


def test_branch_searched_once_no_force():
    """本轮已检索过（tools_node 置位）→ 不重复兜底，直接生成。"""
    s = {"messages": [_plain_ai()], "route": "search",
         "searched_once": True, "force_search_tries": 0}
    assert _agent_branch(s) == "generate"


def test_branch_tries_exhausted_no_force():
    """兜底已用尽（rewrite 回 agent 后 LLM 仍短路）→ 防循环，直接生成。"""
    s = {"messages": [_plain_ai()], "route": "search",
         "searched_once": False, "force_search_tries": 1}
    assert _agent_branch(s) == "generate"


# ──────────────────────────────────────────────
# force_search_node
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_force_search_node_runs_hybrid_and_dedups(monkeypatch):
    """强制检索：跑 hybrid_search，结果按 identity_key 去重并入 accumulated，标记置位。"""
    called = {}

    def fake_run(name, args):
        called["name"], called["args"] = name, args
        return ("obs", [_mk(0.9, "a.md", "h1"), _mk(0.8, "b.md", "h2")])

    monkeypatch.setattr(agent_mod, "run_tool_call", fake_run)
    state = {
        "condensed_question": "什么是 RAG？",
        "question": "什么是 RAG？",
        "accumulated": [_mk(0.7, "a.md", "h1")],  # a.md/h1 已在 → 去重
        "allowed_files": set(),
        "injection_hits": 0,
        "force_search_tries": 0,
    }
    out = await force_search_node(state)
    assert called["name"] == "hybrid_search"
    assert called["args"]["query"] == "什么是 RAG？"
    assert called["args"]["top_k"] == settings.top_k_retrieve
    # 去重：a.md/h1 已有，只新增 b.md/h2
    fps = {(r.filepath, r.metadata.get("heading_path")) for r in out["accumulated"]}
    assert fps == {("a.md", "h1"), ("b.md", "h2")}
    assert out["searched_once"] is True
    assert out["force_search_tries"] == 1
    assert out["allowed_files"] == {"a.md", "b.md"}
    # 不 +1 iteration：返回 dict 不含 iteration 字段（避免提前撞 MAX_ITER）
    assert "iteration" not in out


@pytest.mark.asyncio
async def test_force_search_node_fallback_to_original_question(monkeypatch):
    """condensed_question 缺失时回退用原始 question 作检索 query。"""
    called = {}
    monkeypatch.setattr(agent_mod, "run_tool_call",
                        lambda name, args: called.update(args) or ("obs", []))
    await force_search_node({"question": "原始问题", "accumulated": [],
                             "allowed_files": set(), "injection_hits": 0,
                             "force_search_tries": 0})
    assert called["query"] == "原始问题"


@pytest.mark.asyncio
async def test_force_search_node_tool_failure_no_crash(monkeypatch):
    """检索工具全失败：不崩溃，标记照常置位，accumulated 保持原样。"""
    def boom(name, args):
        raise RuntimeError("search down")
    monkeypatch.setattr(agent_mod, "run_tool_call", boom)
    state = {
        "condensed_question": "q", "question": "q",
        "accumulated": [_mk(0.5, "a.md", "h1")],
        "allowed_files": set(), "injection_hits": 0, "force_search_tries": 0,
    }
    out = await force_search_node(state)
    assert len(out["accumulated"]) == 1          # 原有片段不动
    assert out["searched_once"] is True          # 兜底已执行（结果为空也要标记）
    assert out["force_search_tries"] == 1


@pytest.mark.asyncio
async def test_force_search_node_injection_hits_accumulate(monkeypatch):
    """L2 注入扫描命中计数并入返回（与 tools_node 同规，升级护栏用）。"""
    monkeypatch.setattr(agent_mod, "run_tool_call", lambda name, args: ("obs", [_mk(0.9)]))
    monkeypatch.setattr(agent_mod, "sanitize_text", lambda content, source=None: (content, 2))
    out = await force_search_node({"condensed_question": "q", "question": "q",
                                   "accumulated": [], "allowed_files": set(),
                                   "injection_hits": 3, "force_search_tries": 0})
    assert out["injection_hits"] == 3 + 2


@pytest.mark.asyncio
async def test_tools_node_marks_searched_once(monkeypatch):
    """正常检索路径 tools_node 也要置 searched_once=True（与兜底路径对称）。"""
    monkeypatch.setattr(agent_mod, "run_tool_call",
                        lambda name, args: ("obs", [_mk(0.9, "a.md", "h1")]))
    state = {
        "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "q"}})],
        "accumulated": [], "iteration": 0,
    }
    out = await agent_mod.tools_node(state)
    assert out["searched_once"] is True


# ──────────────────────────────────────────────
# 图级端到端：router → agent(短路) → force_search → reflect → generate
# ──────────────────────────────────────────────

class _SeqObjLLM(BaseChatModel):
    """按调用次数返回预设对象（str 或 AIMessage）的假 LLM。

    用于区分两条路径的 agent 行为：输出 2 可以是纯文本（短路 → force_search）
    或带 tool_calls 的 AIMessage（正常 → tools）。
    """
    outputs: list = []
    calls: int = 0

    @property
    def _llm_type(self):
        return "seqobj"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        out = _SeqObjLLM.outputs[_SeqObjLLM.calls % len(_SeqObjLLM.outputs)]
        _SeqObjLLM.calls += 1
        msg = out if isinstance(out, AIMessage) else AIMessage(content=out)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_full_graph_force_search_path(monkeypatch):
    """端到端：agent 短路（纯文本无 tool_calls）→ force_search 兜底检索，
    结果走公共下游（reflect）→ generate，最终答案有据可依（不再「当前不包含」）。"""
    # 简化下游：关闭 graph_expand / rerank_loop / rerank_exit，force_search → reflect → generate
    monkeypatch.setattr(settings, "agent_graph_expand_enabled", False)
    monkeypatch.setattr(settings, "agent_reranker_loop_enabled", False)
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)

    _SeqObjLLM.outputs = [
        '{"needs_search": true, "reason": "涉及知识库", "confidence": 0.9}',   # 1 router
        "根据历史回答即可，无需检索",                                           # 2 agent（短路！）
        '{"verdict": "sufficient", "relevance_score": 1.5, "reason": "x", '
        '"rewritten_query": "", "clarify_question": ""}',                      # 3 reflect judge
        "这是基于检索片段的最终答案。",                                          # 4 generate
    ]
    _SeqObjLLM.calls = 0
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _SeqObjLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call",
                        lambda name, args: ("obs", [_mk(0.9, "a.md", "h1"), _mk(0.8, "b.md", "h2")]))

    # build_graph 是 lru_cache：monkeypatch settings 后必须清缓存重编译
    agent_mod.build_graph.cache_clear()
    try:
        graph = build_graph()
        final = await graph.ainvoke(_initial_state("什么是 RAG？", history=[], condensed="什么是 RAG？"))
    finally:
        agent_mod.build_graph.cache_clear()  # 还原，避免污染其它测试的缓存图

    # 兜底检索确已执行
    assert final["searched_once"] is True
    assert final["force_search_tries"] == 1
    # 检索结果进了 accumulated（有据可依）
    assert len(final["accumulated"]) >= 1
    # 公共下游：走过了 reflect（Judge 判定）
    assert final["judge_log"], "兜底检索结果必须经过 reflect（公共下游），不得裸奔进生成"
    assert final["judge_verdict"] == "sufficient"
    # 生成答案正常
    assert final["answer"] == "这是基于检索片段的最终答案。"
    # 走的是 force_search 而非 tools：messages 里不应有 ToolMessage
    assert not any(isinstance(m, ToolMessage) for m in final["messages"])


@pytest.mark.asyncio
async def test_full_graph_agent_with_tools_still_normal(monkeypatch):
    """对照：agent 正常输出 tool_calls 时走 tools（而非 force_search），行为零回归。"""
    monkeypatch.setattr(settings, "agent_graph_expand_enabled", False)
    monkeypatch.setattr(settings, "agent_reranker_loop_enabled", False)
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)

    _SeqObjLLM.outputs = [
        '{"needs_search": true, "reason": "涉及知识库", "confidence": 0.9}',   # 1 router
        AIMessage(content="", tool_calls=[                                    # 2 agent（正常：带 tool_calls）
            {"name": "hybrid_search", "args": {"query": "什么是 RAG？",
                                               "top_k": settings.top_k_retrieve}, "id": "c1"},
        ]),
        '{"verdict": "sufficient", "relevance_score": 1.5, "reason": "x", '
        '"rewritten_query": "", "clarify_question": ""}',                      # 3 reflect judge
        "正常检索路径的答案。",                                                  # 4 generate
    ]
    _SeqObjLLM.calls = 0
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _SeqObjLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call",
                        lambda name, args: ("obs", [_mk(0.9, "a.md", "h1")]))

    agent_mod.build_graph.cache_clear()
    try:
        graph = build_graph()
        final = await graph.ainvoke(_initial_state("什么是 RAG？", history=[], condensed="什么是 RAG？"))
    finally:
        agent_mod.build_graph.cache_clear()

    # 正常路径：走了 tools（有 ToolMessage），未触发兜底
    assert final["searched_once"] is True
    assert final["force_search_tries"] == 0
    assert any(isinstance(m, ToolMessage) for m in final["messages"])
    assert len(final["accumulated"]) >= 1

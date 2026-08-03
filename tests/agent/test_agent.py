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
    _coverage_has_extra_topics,
    _coverage_view,
    _extract_json,
    _format_judge_evidence,
    _norm_verdict,
    _reflect_branch,
    _route_branch,
    _top_k_context,
    agent_node,
    generate_node,
    reflect,
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


def test_reflect_branch_under_max(monkeypatch):
    # rerank_exit 关闭时：未达上限 sufficient / give_up → 生成；need_* → 改写
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
    s = {"iteration": 1, "judge_verdict": "sufficient"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": 1, "judge_verdict": "give_up"}
    assert _reflect_branch(s) == "generate"
    s = {"iteration": 1, "judge_verdict": "need_rewrite"}
    assert _reflect_branch(s) == "rewrite"
    s = {"iteration": 1, "judge_verdict": "need_more"}
    assert _reflect_branch(s) == "rewrite"


def test_reflect_branch_at_max_forces_generate(monkeypatch):
    # 达到 max_iter 无论如何强制生成（硬性降级），即便 rerank_exit 关闭
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
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


def test_reflect_branch_routes_to_generate_when_disabled(monkeypatch):
    """rerank_exit 关闭时，行为不变，仍走 generate。"""
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
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


# ──────────────────────────────────────────────
# ① 覆盖概览（治本修复 Judge 盲判）
# ──────────────────────────────────────────────

def test_coverage_view_lists_titles_and_headings():
    """覆盖概览应去重列出文档标题 + 各级 heading，供 Judge 做广度判断。"""
    results = [
        _mk(0.9, "a.md", "一、背景 > 检索方法"),
        _mk(0.8, "a.md", "一、背景 > 检索方法"),  # 同 heading 去重
        _mk(0.7, "a.md", "二、实现 > 接口设计"),
        _mk(0.6, "b.md", "SFT 原理"),
    ]
    view = _coverage_view(results)
    assert "《T》" in view              # 标题（_mk 默认 title="T"）
    assert "一、背景 > 检索方法" in view
    assert "二、实现 > 接口设计" in view
    assert "SFT 原理" in view
    # 重复的 heading 只出现一次
    assert view.count("一、背景 > 检索方法") == 1


def test_coverage_has_extra_topics_true_and_false():
    """top-N 正文未覆盖的 heading 存在 → True；全覆盖 → False。"""
    results = [
        _mk(0.9, "a.md", "A1"),   # 高分
        _mk(0.8, "a.md", "A2"),   # 高分
        _mk(0.1, "a.md", "A-low"),  # 低分、heading 不在 top-N 正文里
    ]
    # top_n=2 时 top 正文只覆盖 A1/A2，A-low 在概览里 → True
    assert _coverage_has_extra_topics(results, top_n=2) is True
    # top_n 覆盖全部 heading → False
    assert _coverage_has_extra_topics(results, top_n=3) is False
    # 空结果 → False
    assert _coverage_has_extra_topics([], top_n=2) is False


def test_format_judge_evidence_appends_coverage(monkeypatch):
    """_format_judge_evidence 末尾应附『知识库覆盖概览』。"""
    monkeypatch.setattr(settings, "agent_judge_evidence_top_n", 2)
    results = [_mk(0.9, "a.md", "A1"), _mk(0.8, "b.md", "B1")]
    text = _format_judge_evidence(results)
    assert "【知识库覆盖概览】" in text
    assert "A1" in text and "B1" in text


# ──────────────────────────────────────────────
# ④ 生成窗口反向放宽（top_k 覆盖）
# ──────────────────────────────────────────────

def test_top_k_context_override_top_k():
    results = [_mk(s) for s in (0.2, 0.9, 0.5, 0.1, 0.8, 0.7, 0.3)]
    # 默认截断到 top_k_rerank（默认 5）
    assert len(_top_k_context(results)) == settings.top_k_rerank
    # 显式覆盖到 10
    assert len(_top_k_context(results, top_k=10)) == 7


# ──────────────────────────────────────────────
# ③④ 收敛闸门 + 反向放宽 + 诚实声明（reflect / generate_node）
# ──────────────────────────────────────────────

class _JsonLLM(BaseChatModel):
    """返回固定 JSON 内容的假 LLM（供 reflect 的 Judge 调用）。"""
    content: str = ""

    def __init__(self, content: str = "", **kwargs):
        super().__init__()
        self.content = content

    @property
    def _llm_type(self):
        return "json"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.content))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self


def _judge_json(verdict: str, rewritten: str = "q") -> str:
    return (
        f'{{"verdict": "{verdict}", "relevance_score": 0.5, '
        f'"reason": "x", "rewritten_query": "{rewritten}", "clarify_question": ""}}'
    )


@pytest.mark.asyncio
async def test_reflect_convergence_gate_forces_sufficient_after_streak(monkeypatch):
    """连续 agent_convergence_streak 轮改写后『新增独特文档=0』→ 强制 sufficient（gate_overrode）。"""
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _JsonLLM(_judge_json("need_rewrite")))
    monkeypatch.setattr(settings, "agent_convergence_streak", 2)

    # 单文档，多 chunk（模拟「同文档换同义词空转」）
    def _state(iter_val, prev_count, prev_streak):
        return {
            "iteration": iter_val,
            "accumulated": [
                _mk(0.9, "a.md", f"h{i}") for i in range(8)
            ],
            "condensed_question": "大模型训练有哪些方法？",
            "question": "大模型训练有哪些方法？",
            "doc_count_at_last_reflect": prev_count,
            "no_new_doc_streak": prev_streak,
        }

    # iter1: 1 文档 > 基线 0 → streak 归零
    out1 = await reflect(_state(1, 0, 0))
    assert out1["judge_verdict"] == "need_rewrite"
    assert out1["no_new_doc_streak"] == 0
    assert out1["gate_overrode"] is False

    # iter2: 仍 1 文档，streak=1，未达阈值 → 继续
    out2 = await reflect(_state(2, out1["doc_count_at_last_reflect"], out1["no_new_doc_streak"]))
    assert out2["judge_verdict"] == "need_rewrite"
    assert out2["no_new_doc_streak"] == 1
    assert out2["gate_overrode"] is False

    # iter3: 仍 1 文档，streak=2 达阈值 → 强制 sufficient + gate_overrode
    out3 = await reflect(_state(3, out2["doc_count_at_last_reflect"], out2["no_new_doc_streak"]))
    assert out3["judge_verdict"] == "sufficient"
    assert out3["gate_overrode"] is True
    assert out3["widen_context"] is True


@pytest.mark.asyncio
async def test_reflect_convergence_gate_new_doc_resets_streak(monkeypatch):
    """改写轮次若检索到新文档，streak 应重置（不误杀有价值的重检）。"""
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _JsonLLM(_judge_json("need_rewrite")))
    monkeypatch.setattr(settings, "agent_convergence_streak", 2)

    # iter1: 单文档 → streak 0, doc_count=1
    s1 = {"iteration": 1, "accumulated": [_mk(0.9, "a.md", "h1")],
          "condensed_question": "q", "question": "q",
          "doc_count_at_last_reflect": 0, "no_new_doc_streak": 0}
    o1 = await reflect(s1)
    assert o1["no_new_doc_streak"] == 0 and o1["doc_count_at_last_reflect"] == 1

    # iter2: 出现新文档 b.md（2 文档 > 1）→ streak 重置 0
    s2 = {"iteration": 2, "accumulated": [_mk(0.9, "a.md", "h1"), _mk(0.8, "b.md", "h2")],
          "condensed_question": "q", "question": "q",
          "doc_count_at_last_reflect": o1["doc_count_at_last_reflect"], "no_new_doc_streak": o1["no_new_doc_streak"]}
    o2 = await reflect(s2)
    assert o2["no_new_doc_streak"] == 0
    assert o2["judge_verdict"] == "need_rewrite"  # 未强制


@pytest.mark.asyncio
async def test_reflect_widen_context_set_when_coverage_extra(monkeypatch):
    """覆盖概览存在 top-N 正文未覆盖的章节 → widen_context=True（④ 触发条件之一）。"""
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _JsonLLM(_judge_json("sufficient")))
    # 缩小 top_n 到 2：top 正文只覆盖 hiA/hiB，低分 chunk 的 low-heading 不在其中 → widen=True
    monkeypatch.setattr(settings, "agent_judge_evidence_top_n", 2)
    results = [_mk(0.9, "a.md", "hiA"), _mk(0.8, "a.md", "hiB"), _mk(0.1, "a.md", "low-heading")]
    s = {"iteration": 1, "accumulated": results,
         "condensed_question": "q", "question": "q",
         "doc_count_at_last_reflect": 0, "no_new_doc_streak": 0}
    out = await reflect(s)
    assert out["judge_verdict"] == "sufficient"
    assert out["widen_context"] is True


@pytest.mark.asyncio
async def test_generate_node_widen_includes_extra_chunks(monkeypatch):
    """widen_context=True 时生成上下文应放宽到 agent_generate_widen_top_k，纳入低分但相关的 chunk。"""
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    # 8 条 chunk，分数从 0.8 到 0.1 递减，page_content 各不相同
    results = [_mk(round(0.8 - i * 0.1, 2), "a.md", f"h{i}") for i in range(8)]

    # 不放宽：top-5，仅前 5 条进上下文
    await generate_node({"accumulated": results, "iteration": 1, "judge_verdict": "sufficient",
                          "widen_context": False, "gate_overrode": False,
                          "question": "q", "condensed_question": "q",
                          "history": [], "history_messages": []})
    ctx_narrow = "".join(m.content for m in _CaptureLLM.captured if isinstance(m, HumanMessage))
    narrow_count = sum(1 for i in range(8) if f"content-{round(0.8 - i*0.1,2)}" in ctx_narrow)

    # 放宽：top-10，全部 8 条进上下文
    await generate_node({"accumulated": results, "iteration": 1, "judge_verdict": "sufficient",
                          "widen_context": True, "gate_overrode": False,
                          "question": "q", "condensed_question": "q",
                          "history": [], "history_messages": []})
    ctx_wide = "".join(m.content for m in _CaptureLLM.captured if isinstance(m, HumanMessage))
    wide_count = sum(1 for i in range(8) if f"content-{round(0.8 - i*0.1,2)}" in ctx_wide)

    assert narrow_count == settings.top_k_rerank
    assert wide_count == 8


@pytest.mark.asyncio
async def test_generate_node_honest_disclosure_on_gate_overrode(monkeypatch):
    """③ 诚实声明：gate_overrode 时，生成 prompt 末尾应附『知识库已穷尽』提示，且不缩小上下文。"""
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    await generate_node({"accumulated": [_mk(0.9, "a.md", "h1")], "iteration": 2,
                          "judge_verdict": "need_rewrite", "widen_context": True, "gate_overrode": True,
                          "question": "q", "condensed_question": "q",
                          "history": [], "history_messages": []})
    ctx = "".join(m.content for m in _CaptureLLM.captured if isinstance(m, HumanMessage))
    assert "可能未涵盖" in ctx
    # 上下文仍包含已检索片段（未被缩小）
    assert "content-0.9" in ctx

# ──────────────────────────────────────────────
# Graph Expand 节点测试
# ──────────────────────────────────────────────

def test_graph_expand_node_disabled_returns_empty():
    """graph_expand_node 关闭时，直接透传，不做事。"""
    from note_assistant.agent.agent import graph_expand_node
    import asyncio
    result = asyncio.run(graph_expand_node({"accumulated": []}))
    assert result == {}


def test_graph_expand_node_skips_stub_filepaths(monkeypatch):
    """stub 节点（[[xxx]]）跳过，不调 graph_expand_impl。"""
    from note_assistant.agent.agent import graph_expand_node
    import asyncio
    from note_assistant.retrieval.types import RetrievalResult

    called = []
    monkeypatch.setattr(settings, "agent_graph_expand_enabled", True)

    # 只有 stub 节点，graph_expand_impl 不应被调用
    def fake_impl(fp, hop=1):
        called.append(fp)
        return []
    monkeypatch.setattr("note_assistant.agent.agent.graph_expand_impl", fake_impl)

    state = {
        "accumulated": [
            RetrievalResult(score=0.5, page_content="x", metadata={"filepath": "[[stub]]", "heading_path": ""}),
        ]
    }
    result = asyncio.run(graph_expand_node(state))
    assert called == []  # 不应调 graph_expand_impl
    assert result == {}


def test_graph_expand_node_dedup(monkeypatch):
    """图扩展返回的 chunk 与已有 accumulated 按 (filepath, heading) 去重。"""
    from note_assistant.agent.agent import graph_expand_node
    import asyncio
    from note_assistant.retrieval.types import RetrievalResult

    monkeypatch.setattr(settings, "agent_graph_expand_enabled", True)

    existing = RetrievalResult(score=0.9, page_content="已有内容", metadata={"filepath": "a.md", "heading_path": "h1"})
    new_unique = RetrievalResult(score=0.5, page_content="新内容", metadata={"filepath": "b.md", "heading_path": "h2"})
    new_dup = RetrievalResult(score=0.4, page_content="重复内容", metadata={"filepath": "a.md", "heading_path": "h1"})

    def fake_impl(fp, hop=1):
        return [new_unique, new_dup]

    monkeypatch.setattr("note_assistant.agent.agent.graph_expand_impl", fake_impl)

    state = {"accumulated": [existing]}
    result = asyncio.run(graph_expand_node(state))
    assert len(result["accumulated"]) == 2  # existing + new_unique，new_dup 被去重
    fps = {(r.filepath, r.metadata.get("heading_path", "")) for r in result["accumulated"]}
    assert fps == {("a.md", "h1"), ("b.md", "h2")}


# ──────────────────────────────────────────────
# 去重：图片与同节文本/父块共存（identity_key）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tools_node_image_and_parent_coexist(monkeypatch):
    """同 (filepath, heading) 的 image summary chunk 与父块必须都进 accumulated。

    旧去重键 (filepath, heading) 让二者按分数竞速二选一——图意图 query 图赢、
    普通 query 长文赢，这是「有的 case 图进 sources、有的不进」的直接原因。
    """
    from note_assistant.agent.context import ContextManager, set_context_manager_for_test

    set_context_manager_for_test(ContextManager(embed_fn=None))
    try:
        parent = RetrievalResult(score=0.9, page_content="整节正文", metadata={
            "filepath": "n.md", "heading_path": "H1", "kind": "parent", "title": "T"})
        img = RetrievalResult(score=0.5, page_content="图片理解：三层架构", metadata={
            "filepath": "n.md", "heading_path": "H1", "kind": "image",
            "placeholder": "[IMAGE_UID_aaaaaaaa]", "title": "T"})

        # 父块在前（分数高）——旧实现在这里就把图片丢了
        monkeypatch.setattr(agent_mod, "run_tool_call",
                            lambda name, args: ("obs", [parent, img]))
        state = {
            "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "架构图"}})],
            "accumulated": [],
            "iteration": 0,
        }
        out = await tools_node(state)
        kinds = {r.metadata.get("kind") for r in out["accumulated"]}
        assert kinds == {"parent", "image"}
    finally:
        set_context_manager_for_test(None)


@pytest.mark.asyncio
async def test_tools_node_image_first_parent_still_kept(monkeypatch):
    """图片在前、父块在后：两者同样共存（顺序无关）。"""
    from note_assistant.agent.context import ContextManager, set_context_manager_for_test

    set_context_manager_for_test(ContextManager(embed_fn=None))
    try:
        img = RetrievalResult(score=0.9, page_content="图片理解：三层架构", metadata={
            "filepath": "n.md", "heading_path": "H1", "kind": "image",
            "placeholder": "[IMAGE_UID_aaaaaaaa]", "title": "T"})
        parent = RetrievalResult(score=0.5, page_content="整节正文", metadata={
            "filepath": "n.md", "heading_path": "H1", "kind": "parent", "title": "T"})
        monkeypatch.setattr(agent_mod, "run_tool_call",
                            lambda name, args: ("obs", [img, parent]))
        state = {
            "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "架构图"}})],
            "accumulated": [],
            "iteration": 0,
        }
        out = await tools_node(state)
        kinds = {r.metadata.get("kind") for r in out["accumulated"]}
        assert kinds == {"parent", "image"}
    finally:
        set_context_manager_for_test(None)


@pytest.mark.asyncio
async def test_tools_node_text_same_heading_still_deduped(monkeypatch):
    """普通正文 chunk 之间仍按 (filepath, heading) 去重——旧行为零回归。"""
    from note_assistant.agent.context import ContextManager, set_context_manager_for_test

    set_context_manager_for_test(ContextManager(embed_fn=None))
    try:
        a = RetrievalResult(score=0.9, page_content="正文A", metadata={
            "filepath": "n.md", "heading_path": "H1", "title": "T"})
        b = RetrievalResult(score=0.8, page_content="正文B", metadata={
            "filepath": "n.md", "heading_path": "H1", "title": "T"})
        monkeypatch.setattr(agent_mod, "run_tool_call",
                            lambda name, args: ("obs", [a, b]))
        state = {
            "messages": [_ai_with_tools({"name": "hybrid_search", "args": {"query": "x"}})],
            "accumulated": [],
            "iteration": 0,
        }
        out = await tools_node(state)
        assert len(out["accumulated"]) == 1
    finally:
        set_context_manager_for_test(None)


# ──────────────────────────────────────────────
# rerank 图片保位（ensure_image_selected 接入点）
# ──────────────────────────────────────────────

class _FakeCutReranker:
    """模拟交叉编码器：文本分 > 图片分，top_k 截断时图片被挤出。"""

    def rerank(self, q, results, top_k=None):
        scored = []
        for r in results:
            s = 0.1 if r.metadata.get("kind") == "image" else 0.9
            scored.append(RetrievalResult(score=s, page_content=r.page_content, metadata=r.metadata))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k] if top_k is not None else scored


def _img_result(heading="h-img"):
    return RetrievalResult(score=0.8, page_content="图片理解：三层架构", metadata={
        "filepath": "a.md", "heading_path": heading, "kind": "image",
        "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
        "title": "T"})


def test_rerank_exit_pins_image_on_image_intent(monkeypatch):
    """图意图 query：rerank top-k 裁掉图片时，从全量精排补回最高分图片。"""
    import asyncio
    from note_assistant.agent.agent import rerank_exit

    monkeypatch.setattr(agent_mod, "get_reranker", lambda *a, **k: _FakeCutReranker())
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", True)

    acc = [_mk(0.9, heading=f"h{i}") for i in range(settings.top_k_rerank)]
    acc.append(_img_result())
    state = {"accumulated": acc, "question": "架构图长什么样",
             "condensed_question": "", "widen_context": False}
    out = asyncio.run(rerank_exit(state))
    kinds = [r.metadata.get("kind") for r in out["accumulated"]]
    assert "image" in kinds
    assert len(out["accumulated"]) == settings.top_k_rerank  # 预算不变，替换末位


def test_rerank_exit_no_pin_without_intent(monkeypatch):
    """非图意图 query：图片被精排裁掉就不强塞（避免无关图干扰）。"""
    import asyncio
    from note_assistant.agent.agent import rerank_exit

    monkeypatch.setattr(agent_mod, "get_reranker", lambda *a, **k: _FakeCutReranker())
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", True)

    acc = [_mk(0.9, heading=f"h{i}") for i in range(settings.top_k_rerank)]
    acc.append(_img_result())
    state = {"accumulated": acc, "question": "RAG 的检索流程是什么",
             "condensed_question": "", "widen_context": False}
    out = asyncio.run(rerank_exit(state))
    assert all(r.metadata.get("kind") != "image" for r in out["accumulated"])


def test_rerank_loop_pins_image_on_image_intent(monkeypatch):
    """循环内闸门同样保位：否则图片在 reflect 前就丢了，Judge 证据也看不到图。"""
    import asyncio
    from note_assistant.agent.agent import rerank_loop

    monkeypatch.setattr(agent_mod, "get_reranker", lambda *a, **k: _FakeCutReranker())
    monkeypatch.setattr(settings, "agent_reranker_loop_enabled", True)

    acc = [_mk(0.9, heading=f"h{i}") for i in range(settings.agent_reranker_loop_top_k)]
    acc.append(_img_result())
    state = {"accumulated": acc, "question": "架构图长什么样", "condensed_question": ""}
    out = asyncio.run(rerank_loop(state))
    kinds = [r.metadata.get("kind") for r in out["accumulated"]]
    assert "image" in kinds
    assert len(out["accumulated"]) == settings.agent_reranker_loop_top_k

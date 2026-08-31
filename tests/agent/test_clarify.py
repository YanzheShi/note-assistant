"""澄清 / 反问（clarify-as-terminal，方案 B）测试。

覆盖设计文档 `docs/clarification-design.md` 15.5 节的 12 条规格：
消解置信信号、级联守卫五道闸门、clarify 节点终止语义、reflect 证据注入（P0 回归），
以及**最重要的**回归保护 —— `agent_clarify_enabled=False` 时行为与改造前完全一致。

全离线：不依赖 Ollama / DeepSeek / ChromaDB。
"""
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent import agent as agent_mod
from note_assistant.agent.agent import (
    _format_judge_evidence,
    _norm_verdict,
    _reflect_branch,
    _should_clarify,
    clarify_node,
    reflect,
)
from note_assistant.agent.context import (
    CondenseSignal,
    ContextManager,
    set_context_manager_for_test,
)
from note_assistant.agent.runner import ainvoke, reset_cache, reset_store
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


# ──────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────

class FakeLLM:
    """最简异步 LLM 桩（用于 ContextManager.condense_llm）。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)


class _CaptureLLM(BaseChatModel):
    """记录最后一次 ainvoke 收到的 messages，返回可配置 JSON。"""

    captured: list = []
    reply: str = '{"verdict": "sufficient", "relevance_score": 1.5, "reason": "ok", "rewritten_query": "", "clarify_question": ""}'

    @property
    def _llm_type(self):
        return "capture"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _CaptureLLM.captured = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        _CaptureLLM.captured = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def bind_tools(self, tools, **kwargs):
        return self


def _rr(score: float, content: str, title: str, heading: str = "h") -> RetrievalResult:
    return RetrievalResult(
        score=score,
        page_content=content,
        metadata={"filepath": f"{title}.md", "title": title, "heading_path": heading},
    )


def _state(**kw) -> dict:
    """构造 _should_clarify 需要的最小 state。"""
    base = {
        "judge_verdict": "need_clarify",
        "clarify_question": "你问的是 FlashAttention-2 还是 Paged Attention？",
        "condense_confidence": 0.4,
        "condense_candidates": ["FlashAttention-2", "Paged Attention"],
        "just_clarified": False,
        "iteration": 1,
    }
    base.update(kw)
    return base


@pytest.fixture
def clarify_on(monkeypatch):
    monkeypatch.setattr(settings, "agent_clarify_enabled", True)
    monkeypatch.setattr(settings, "agent_clarify_confidence_threshold", 0.6)
    return settings


# ──────────────────────────────────────────────
# 用例 1：need_clarify 不再被静默降级
# ──────────────────────────────────────────────

def test_norm_verdict_keeps_need_clarify():
    # 改造前 need_clarify 不在白名单里，会被静默归一成 sufficient
    assert _norm_verdict("need_clarify") == "need_clarify"
    assert _norm_verdict("NEED_CLARIFY") == "need_clarify"
    # 非法值仍然兜底为 sufficient（原行为不变）
    assert _norm_verdict("胡言乱语") == "sufficient"


# ──────────────────────────────────────────────
# 用例 2~5：_should_clarify 级联守卫（五道闸门）
# ──────────────────────────────────────────────

def test_should_clarify_false_when_disabled(monkeypatch):
    """回归保护：总开关关闭 → 恒 False，无论其他条件多么符合。"""
    monkeypatch.setattr(settings, "agent_clarify_enabled", False)
    assert _should_clarify(_state()) is False


def test_should_clarify_false_when_confidence_high(clarify_on):
    """级联终点的核心语义：消解已经成功（高置信）→ 不反问，照常回答。"""
    assert _should_clarify(_state(condense_confidence=0.9)) is False
    # 恰好等于阈值也不反问（阈值取严）
    assert _should_clarify(_state(condense_confidence=0.6)) is False


def test_should_clarify_true_on_low_confidence(clarify_on):
    assert _should_clarify(_state(condense_confidence=0.4)) is True


def test_should_clarify_false_when_just_clarified(clarify_on):
    """防连续追问：上一轮刚反问过，本轮无论如何都要给答案。"""
    assert _should_clarify(_state(just_clarified=True)) is False


def test_should_clarify_false_without_verdict_or_question(clarify_on):
    # Judge 没判 need_clarify → 不反问
    assert _should_clarify(_state(judge_verdict="need_rewrite")) is False
    # 判了但没给出问句 → 不反问（防吐空问句）
    assert _should_clarify(_state(clarify_question="")) is False
    assert _should_clarify(_state(clarify_question="   ")) is False


# ──────────────────────────────────────────────
# 用例 6：_reflect_branch 路由 + 开关关闭时的回归
# ──────────────────────────────────────────────

def test_reflect_branch_routes_to_clarify(clarify_on, monkeypatch):
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
    assert _reflect_branch(_state()) == "clarify"


def test_reflect_branch_need_clarify_degrades_when_disabled(monkeypatch):
    """开关关闭时 need_clarify 退化为 sufficient 路径 —— 与改造前逐字节等价。

    改造前 `_norm_verdict` 白名单不含 need_clarify，Judge 就算判了也会被归一成
    sufficient 直接进生成。现在白名单放行了它，必须在分支里补回这个降级，
    否则关掉开关的用户会莫名走到未知分支。
    """
    monkeypatch.setattr(settings, "agent_clarify_enabled", False)
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
    assert _reflect_branch(_state()) == "generate"
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", True)
    assert _reflect_branch(_state()) == "rerank_exit"


def test_reflect_branch_max_iter_beats_clarify(clarify_on, monkeypatch):
    """守卫不通过时上限硬性降级仍然生效（未破坏原有兜底）。"""
    monkeypatch.setattr(settings, "agent_reranker_exit_enabled", False)
    s = _state(iteration=settings.agent_max_iter, condense_confidence=0.9)
    assert _reflect_branch(s) == "generate"


# ──────────────────────────────────────────────
# 用例 7：clarify_node 终止语义
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clarify_node_returns_question_as_answer():
    q = "你问的是 FlashAttention-2 还是 Paged Attention？"
    out = await clarify_node(_state(clarify_question=q))
    assert out["answer"] == q
    assert isinstance(out["messages"][-1], AIMessage)
    assert str(out["messages"][-1].content) == q
    # clarified 标记供 runner 跳过缓存收录（防反问死循环）
    assert out["clarified"] is True


@pytest.mark.asyncio
async def test_clarify_node_falls_back_on_empty_question():
    out = await clarify_node(_state(clarify_question=""))
    assert out["answer"]  # 不会吐空串
    assert "补充" in out["answer"] or "哪一方面" in out["answer"]


# ──────────────────────────────────────────────
# 用例 8：短问题补漏（方案0 前置拦截漏洞）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_short_followup_enters_condense():
    """「性能呢？」不含代词但零主语，改造前会被前置拦截直接透传给检索。"""
    llm = FakeLLM("FlashAttention 的性能如何")
    cm = ContextManager(condense_llm=llm)
    hist = [{"role": "user", "content": "FlashAttention 是什么"},
            {"role": "assistant", "content": "一种高效注意力机制"}]
    out = await cm.condense_question("性能呢？", hist, session_id="s1")
    assert out == "FlashAttention 的性能如何"
    assert llm.calls == 1


# ──────────────────────────────────────────────
# 用例 9~10、12：CondenseSignal 置信度规则
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_signal_entity_disagreement_lowers_confidence():
    """方案2（LLM 改写）与方案3（实体槽位）指向不同实体 → 最强歧义信号。"""
    cm = ContextManager(condense_llm=FakeLLM("Paged Attention 的改进是什么"))
    cm._last_entity["s1"] = "FlashAttention"      # 上一轮主导实体
    cm._last_topics["s1"] = ["FlashAttention", "Paged Attention"]
    cm._last_topic_margin["s1"] = 0.5             # 主题不竞争，仅实体分歧
    await cm.condense_question("那个的改进呢", [{"role": "user", "content": "x"}], session_id="s1")
    sig = cm.get_condense_signal("s1")
    assert sig.entity_agreement is False
    assert sig.confidence < settings.agent_clarify_confidence_threshold
    assert sig.candidates == ["FlashAttention", "Paged Attention"]


@pytest.mark.asyncio
async def test_signal_topic_competition_lowers_confidence(monkeypatch):
    """实体一致但上一轮召回本就存在竞争主题 → 置信降档（0.9 → 0.5）。"""
    monkeypatch.setattr(settings, "agent_clarify_topic_margin", 0.15)
    cm = ContextManager(condense_llm=FakeLLM("FlashAttention 的改进是什么"))
    cm._last_entity["s1"] = "FlashAttention"
    cm._last_topics["s1"] = ["FlashAttention", "Paged Attention"]
    cm._last_topic_margin["s1"] = 0.02            # top1/top2 几乎打平
    await cm.condense_question("那个的改进呢", [{"role": "user", "content": "x"}], session_id="s1")
    sig = cm.get_condense_signal("s1")
    assert sig.entity_agreement is True
    assert sig.confidence == 0.5
    assert sig.confidence < settings.agent_clarify_confidence_threshold


@pytest.mark.asyncio
async def test_signal_agreement_high_confidence():
    """两条独立路径互相印证 + 无竞争主题 → 高置信，不该触发反问。"""
    cm = ContextManager(condense_llm=FakeLLM("FlashAttention 的改进是什么"))
    cm._last_entity["s1"] = "FlashAttention"
    cm._last_topic_margin["s1"] = 0.8
    await cm.condense_question("那个的改进呢", [{"role": "user", "content": "x"}], session_id="s1")
    sig = cm.get_condense_signal("s1")
    assert sig.confidence == 0.9
    assert sig.confidence >= settings.agent_clarify_confidence_threshold


@pytest.mark.asyncio
async def test_signal_fallback_on_llm_error():
    """LLM 异常 → 只剩零模型兜底，置信最低（0.3）。"""

    class BoomLLM:
        async def ainvoke(self, messages):
            raise RuntimeError("boom")

    cm = ContextManager(condense_llm=BoomLLM())
    out = await cm.condense_question("它是什么", [{"role": "user", "content": "RAG 是什么"}], session_id="s1")
    assert out  # 不抛异常，正常降级
    sig = cm.get_condense_signal("s1")
    assert sig.method == "fallback"
    assert sig.confidence == 0.3


def test_signal_default_is_fully_trusted():
    """未记录信号时默认「完全可信」，绝不会平白触发反问。"""
    cm = ContextManager()
    sig = cm.get_condense_signal("never-seen")
    assert sig == CondenseSignal()
    assert sig.confidence == 1.0


def test_clarified_flag_is_one_shot():
    cm = ContextManager()
    assert cm.pop_clarified("s1") is False
    cm.mark_clarified("s1")
    assert cm.pop_clarified("s1") is True
    assert cm.pop_clarified("s1") is False   # 一次性，读完即清
    # 无 session_id 时是 no-op，不污染全局
    cm.mark_clarified("")
    assert cm.pop_clarified("") is False


# ──────────────────────────────────────────────
# 用例 11：reflect 证据注入（P0 盲判修复回归）
# ──────────────────────────────────────────────

def test_format_judge_evidence_sorts_and_truncates(monkeypatch):
    monkeypatch.setattr(settings, "agent_judge_evidence_top_n", 2)
    monkeypatch.setattr(settings, "agent_judge_evidence_chars", 10)
    results = [
        _rr(0.2, "低分内容", "C"),
        _rr(0.9, "高分内容" * 20, "A", "一、背景"),
        _rr(0.5, "中分内容", "B"),
    ]
    text = _format_judge_evidence(results)
    assert "A" in text and "B" in text
    # 正文按 top_n 截断：最低分 C 的正文不应出现在证据正文里
    assert "低分内容" not in text
    assert text.index("A") < text.index("B")   # 按分降序
    assert "一、背景" in text                   # heading_path 参与判定
    assert len(text) < 200                     # 正文按 chars 截断
    # 末尾附『知识库覆盖概览』：列出全部已命中笔记标题（含低分 C）→ 支撑 Judge 广度判定
    assert "【知识库覆盖概览】" in text
    assert "《C》" in text


def test_format_judge_evidence_empty():
    assert "未检索" in _format_judge_evidence([]) or _format_judge_evidence([]).strip()


@pytest.mark.asyncio
async def test_reflect_injects_evidence_into_prompt(monkeypatch):
    """P0 回归：改造前只把 len(accumulated) 传给 Judge，Judge 在盲判。"""
    _CaptureLLM.captured = []
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _CaptureLLM())
    state = {
        "question": "那个的改进",
        "condensed_question": "FlashAttention 的改进",
        "accumulated": [_rr(0.9, "FA2 通过减少非矩阵乘法运算提速", "FlashAttention", "二、FA2")],
        "iteration": 1,
    }
    await reflect(state)
    human = str(_CaptureLLM.captured[-1].content)
    assert "FlashAttention" in human            # 标题进了 prompt
    assert "FA2 通过减少非矩阵乘法运算提速" in human  # 正文进了 prompt
    assert "二、FA2" in human                    # heading_path 进了 prompt


@pytest.mark.asyncio
async def test_reflect_passes_through_clarify_question(monkeypatch):
    _CaptureLLM.captured = []
    llm = _CaptureLLM()
    llm.reply = ('{"verdict": "need_clarify", "relevance_score": 0.5, "reason": "两个主题",'
                 ' "rewritten_query": "", "clarify_question": "你问的是 A 还是 B？"}')
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: llm)
    out = await reflect({
        "question": "那个", "condensed_question": "那个",
        "accumulated": [_rr(0.9, "内容", "A"), _rr(0.88, "内容", "B")], "iteration": 1,
    })
    assert out["judge_verdict"] == "need_clarify"
    assert out["clarify_question"] == "你问的是 A 还是 B？"


# ──────────────────────────────────────────────
# 端到端：clarify-as-terminal 在真实图里跑通
#
# 重点验证文档点名的「唯一真陷阱」——澄清问句若进了语义缓存，
# 同一个模糊问题再问会命中缓存直接吐问句，ctx_key 又相同，永远出不来答案。
# ──────────────────────────────────────────────

class _ClarifyAgentLLM(BaseChatModel):
    """脚本化多角色 LLM：Judge 恒判 need_clarify，其余节点正常作答。"""

    @property
    def _llm_type(self):
        return "fake-clarify"

    @staticmethod
    def _role(messages):
        for m in messages:
            if isinstance(m, (SystemMessage, HumanMessage)):
                c = str(m.content)
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
            return AIMessage(content=json.dumps({"needs_search": True, "reason": "x"}))
        if role == "reflect":
            return AIMessage(content=json.dumps({
                "verdict": "need_clarify", "relevance_score": 0.5,
                "reason": "片段命中两个互不相干主题",
                "rewritten_query": "",
                "clarify_question": "你问的是 FlashAttention 的改进，还是 Paged Attention 的改进？",
            }))
        if role == "generate":
            return AIMessage(content="根据笔记，FlashAttention 通过分块减少显存占用。")
        if role == "agent":
            if any(isinstance(m, ToolMessage) for m in messages):
                return AIMessage(content="基于检索结果作答。")
            return AIMessage(content="", tool_calls=[{
                "name": "hybrid_search", "args": {"query": "改进", "top_k": 2}, "id": "call_1",
            }])
        if role == "condense":
            return AIMessage(content="")  # 空 → 走 fallback 降级，置信度 0.3
        return AIMessage(content="fallback")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._respond(self._role(messages), messages))])

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_tool(name, args):
    return ("obs: 命中 2 段", [
        _rr(0.90, "FA2 改进内容", "FlashAttention", "二、改进"),
        _rr(0.89, "PA 改进内容", "PagedAttention", "二、改进"),
    ])


@pytest.fixture
def e2e(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(settings, "agent_session_enabled", False)
    monkeypatch.setattr(settings, "agent_history_relevance_enabled", False)
    monkeypatch.setattr(settings, "agent_graph_expand_enabled", False)
    _rk = MagicMock()
    _rk.rerank.side_effect = lambda q, results, top_k=None: results if top_k is None else results[:top_k]
    monkeypatch.setattr(agent_mod, "get_reranker", lambda *a, **k: _rk)
    reset_store()
    agent_mod.build_graph.cache_clear()
    set_context_manager_for_test(None)
    monkeypatch.setattr("note_assistant.llm.client.get_llm", lambda *a, **k: _ClarifyAgentLLM())
    # 5df66f9 新增凝练专属入口，不拦则 condense 走真实 LLM 消解成功（置信度 1.0），
    # clarify 前提（低置信度）永远不成立 → 反问测试恒失败
    monkeypatch.setattr("note_assistant.llm.client.get_condense_llm", lambda *a, **k: _ClarifyAgentLLM())
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ClarifyAgentLLM())
    monkeypatch.setattr(agent_mod, "run_tool_call", _fake_tool)
    reset_cache()
    yield
    monkeypatch.setattr(settings, "agent_session_enabled", True)
    set_context_manager_for_test(None)


_HIST = [
    {"role": "user", "content": "FlashAttention 和 PagedAttention 有什么区别"},
    {"role": "assistant", "content": "一个优化算子，一个优化显存管理。"},
]


@pytest.mark.asyncio
async def test_e2e_clarify_terminates_with_question(e2e, clarify_on):
    r = await ainvoke("那个的改进呢", history=_HIST, session_id="s-clarify")
    assert "还是" in r.answer                       # 返回的是澄清问句而非答案
    assert r.cached is False
    types = [t["type"] for t in r.trajectory]
    assert "judge" in types and "answer" in types   # 终止语义与 generate 同构
    judges = [t for t in r.trajectory if t["type"] == "judge"]
    assert judges[-1]["verdict"] == "need_clarify"


@pytest.mark.asyncio
async def test_e2e_clarify_answer_never_cached(e2e, clarify_on):
    """陷阱回归：澄清问句进缓存 → 同问题再问永远命中问句，永远出不来答案。"""
    r1 = await ainvoke("那个的改进呢", history=_HIST, session_id="s-cache")
    assert "还是" in r1.answer
    r2 = await ainvoke("那个的改进呢", history=_HIST, session_id="s-cache")
    assert r2.cached is False                       # 没被缓存收录
    # 且第二轮因 just_clarified 守卫不再反问，正常给答案
    assert "还是" not in r2.answer
    assert "FlashAttention" in r2.answer


@pytest.mark.asyncio
async def test_e2e_disabled_switch_never_clarifies(e2e, monkeypatch):
    """回归保护：开关关闭时，Judge 就算判 need_clarify 也照常出答案。"""
    monkeypatch.setattr(settings, "agent_clarify_enabled", False)
    r = await ainvoke("那个的改进呢", history=_HIST, session_id="s-off")
    assert "还是" not in r.answer
    assert "FlashAttention" in r.answer

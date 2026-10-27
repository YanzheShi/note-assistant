# -*- coding: utf-8 -*-
"""决策模型 × router/reflect 开关矩阵测试。

覆盖四种组合：默认关（原路径不变）/ 直接切换（dm 主判）/ dm 失败回退 LLM /
影子模式（LLM 主判 + dm_shadow 记录）。不依赖外部服务（dm/LLM 全部 fake）。
"""
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from note_assistant.agent import agent as agent_mod
from note_assistant.config import settings


# ──────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────

class _FakeDM:
    """决策模型替身：可配置返回值与是否抛异常。"""

    def __init__(self, route=None, judge=None, route_error=False, judge_error=False):
        self._route = route or {
            "needs_search": True, "p_search": 0.87, "p_reference": 0.1, "server_ms": 50,
        }
        self._judge = judge or {
            "verdict": "sufficient",
            "probs": {"sufficient": 0.9, "need_rewrite": 0.1},
            "p_sufficient": 0.9, "confidence": 0.9, "server_ms": 50,
        }
        self._route_error = route_error
        self._judge_error = judge_error
        self.route_calls: list = []
        self.judge_calls: list = []

    def route_intent(self, question, condensed=""):
        self.route_calls.append((question, condensed))
        if self._route_error:
            raise RuntimeError("dm down")
        return dict(self._route)

    def judge_verdict(self, question, evidence_text, iteration, max_iter):
        self.judge_calls.append((question, iteration, max_iter))
        if self._judge_error:
            raise RuntimeError("dm down")
        return dict(self._judge)


class _ScriptLLM(BaseChatModel):
    """按序返回预置文本的 LLM 替身（记录收到的 messages）。"""

    script: list = []
    captured: list = []

    @property
    def _llm_type(self):
        return "script-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._take(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._take(messages)

    def _take(self, messages):
        _ScriptLLM.captured.append(list(messages))
        content = _ScriptLLM.script.pop(0) if _ScriptLLM.script else "{}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """每用例恢复默认关 + 清 LLM 脚本，避免跨用例污染。"""
    monkeypatch.setattr(settings, "decision_model_enabled", False)
    monkeypatch.setattr(settings, "decision_model_scope", "all")
    monkeypatch.setattr(settings, "decision_model_shadow", False)
    monkeypatch.setattr(settings, "decision_model_judge_rewrite_llm", True)
    _ScriptLLM.script = []
    _ScriptLLM.captured = []
    yield


def _router_state():
    return {"question": "笔记里 v2 切分策略是什么？", "condensed_question": ""}


def _reflect_state():
    return {
        "question": "笔记里 v2 切分策略是什么？",
        "condensed_question": "v2 切分策略是什么？",
        "accumulated": [],
        "iteration": 0,
        "messages": [],
    }


# ──────────────────────────────────────────────
# router
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_dm_active_skips_llm(monkeypatch):
    """直接切换：决策模型主判，LLM 完全不被调用。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_scope", "all")
    fake = _FakeDM(route={"needs_search": False, "p_search": 0.05, "p_reference": 0.0, "server_ms": 45})
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)

    def _boom(*a, **k):
        raise AssertionError("决策模型 active 模式下不应调用 LLM")
    monkeypatch.setattr(agent_mod, "get_llm", _boom)

    result = await agent_mod.router(_router_state())
    assert result["route"] == "chat"          # p_search=0.05 → chat
    assert result["dm_shadow"] == []          # 非影子模式无影子记录
    assert fake.route_calls and fake.route_calls[0][0].startswith("笔记里")


@pytest.mark.asyncio
async def test_router_dm_error_falls_back_to_llm(monkeypatch):
    """决策模型抛异常 → 回退 LLM 原路径（JSON 解析照常）。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    fake = _FakeDM(route_error=True)
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = [json.dumps({"needs_search": False, "reason": "llm 说不用查"})]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.router(_router_state())
    assert result["route"] == "chat"
    assert len(_ScriptLLM.captured) == 1      # LLM 被调用了一次


@pytest.mark.asyncio
async def test_router_shadow_llm_decides_and_records(monkeypatch):
    """影子模式：LLM 照常主判，决策模型并行记录 dm_shadow。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_shadow", True)
    fake = _FakeDM(route={"needs_search": True, "p_search": 0.93, "p_reference": 0.2, "server_ms": 47})
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = [json.dumps({"needs_search": False, "reason": "llm"})]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.router(_router_state())
    assert result["route"] == "chat"          # 主判来自 LLM
    rec = result["dm_shadow"]
    assert len(rec) == 1 and rec[0]["node"] == "router"
    assert rec[0]["dm_needs_search"] is True and rec[0]["dm_p_search"] == 0.93


@pytest.mark.asyncio
async def test_router_disabled_unchanged(monkeypatch):
    """默认关：工厂返回 None（真实实现按 ENABLED 判定），行为与改造前一致。"""
    def _dm():
        return None if not settings.decision_model_enabled else _FakeDM()
    monkeypatch.setattr(agent_mod, "get_decision_client", _dm)
    _ScriptLLM.script = [json.dumps({"needs_search": True, "reason": ""})]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.router(_router_state())
    assert result["route"] == "search"
    assert len(_ScriptLLM.captured) == 1
    assert result["dm_shadow"] == []


# ──────────────────────────────────────────────
# reflect
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reflect_dm_active_sufficient_skips_judge_llm(monkeypatch):
    """直接切换：verdict 来自决策模型（sufficient），LLM 不被调用，judge_log 带 dm 概率。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    fake = _FakeDM()
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)

    def _boom(*a, **k):
        raise AssertionError("dm active 且判定成功时不应调用 LLM")
    monkeypatch.setattr(agent_mod, "get_llm", _boom)

    result = await agent_mod.reflect(_reflect_state())
    assert result["judge_verdict"] == "sufficient"
    entry = result["judge_log"][0]
    assert entry["dm_probs"]["sufficient"] == pytest.approx(0.9)
    assert result["dm_shadow"][0]["source"] == "dm"


@pytest.mark.asyncio
async def test_reflect_dm_need_rewrite_generates_query_via_llm(monkeypatch):
    """need_rewrite：决策模型判类，LLM 只补改写问句（生成字段拆分）。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    fake = _FakeDM(judge={
        "verdict": "need_rewrite",
        "probs": {"sufficient": 0.2, "need_rewrite": 0.7},
        "p_sufficient": 0.2, "confidence": 0.8, "server_ms": 45,
    })
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = ["v2 chunking 800 150 切分策略对比"]   # LLM 产出改写
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.reflect(_reflect_state())
    assert result["judge_verdict"] == "need_rewrite"
    assert result["rewritten_query"] == "v2 chunking 800 150 切分策略对比"
    assert result["clarify_question"] == ""
    assert len(_ScriptLLM.captured) == 1      # 只有一次轻量改写调用


@pytest.mark.asyncio
async def test_reflect_dm_error_falls_back_to_llm_judge(monkeypatch):
    """决策模型异常 → 回退 LLM 全功能判定（含 rewritten_query）。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_scope", "judge")
    fake = _FakeDM(judge_error=True)
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = [json.dumps({
        "verdict": "need_more", "rewritten_query": "", "clarify_question": "", "reason": "llm",
    })]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.reflect(_reflect_state())
    assert result["judge_verdict"] == "need_more"
    assert result["judge_log"][0]["reason"] == "llm"
    assert "dm_probs" not in result["judge_log"][0]   # 回退后无 dm 字段


@pytest.mark.asyncio
async def test_reflect_shadow_records_both_verdicts(monkeypatch):
    """影子模式：LLM 主判 + dm 并行判定，dm_shadow 同时带 llm_verdict 与 dm_verdict。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_shadow", True)
    fake = _FakeDM(judge={"verdict": "need_more", "probs": {"need_more": 0.6},
                          "p_sufficient": 0.3, "confidence": 0.7, "server_ms": 46})
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = [json.dumps({"verdict": "sufficient", "reason": "llm"})]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.reflect(_reflect_state())
    assert result["judge_verdict"] == "sufficient"          # 主判来自 LLM
    rec = result["dm_shadow"][0]
    assert rec["source"] == "dm_shadow"
    assert rec["llm_verdict"] == "sufficient"
    assert rec["dm_verdict"] == "need_more"


@pytest.mark.asyncio
async def test_reflect_scope_router_disables_judge_dm(monkeypatch):
    """SCOPE=router：judge 不使用决策模型（灰度粒度隔离）。"""
    monkeypatch.setattr(settings, "decision_model_enabled", True)
    monkeypatch.setattr(settings, "decision_model_scope", "router")
    fake = _FakeDM()
    monkeypatch.setattr(agent_mod, "get_decision_client", lambda: fake)
    _ScriptLLM.script = [json.dumps({"verdict": "sufficient", "reason": ""})]
    monkeypatch.setattr(agent_mod, "get_llm", lambda *a, **k: _ScriptLLM())

    result = await agent_mod.reflect(_reflect_state())
    assert fake.judge_calls == []             # judge 未触碰决策模型
    assert result["judge_verdict"] == "sufficient"

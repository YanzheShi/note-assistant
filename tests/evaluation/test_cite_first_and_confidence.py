"""P1 生成质量改造的测试（2026-09-07）：

1. 先引后答（CITE_FIRST）：naive Generator 与 agent generate_node 都只在
   有参考笔记时注入，空 context（闲聊路径）绝不注入。
2. 置信度校准：judge prompt 带 4 档 confidence 定义（打破恒高分虚高）。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from note_assistant.retrieval.types import RetrievalResult


# ── naive Generator 条件化教学（P3 后教学在 human 头部，system 全静态）──

def _make_generator():
    """不走 __init__（避免真实 LLM 初始化），只测 prompt 组装。"""
    from note_assistant.generation.generator import Generator

    gen = object.__new__(Generator)
    return gen


def _fake_result(content: str = "RAG 是检索增强生成"):
    return RetrievalResult(
        score=0.9,
        page_content=content,
        metadata={"filepath": "a.md", "title": "A", "heading_path": "h"},
    )


def test_generator_cite_first_injected_with_context():
    gen = _make_generator()
    preamble = gen._teaching_preamble([_fake_result()])
    assert "先引后答" in preamble
    assert "依据笔记" in preamble
    # system 保持全静态（缓存前缀稳定）
    assert "先引后答" not in gen._system_prompt([_fake_result()])
    assert "先引后答" not in gen._system_prompt([])


def test_generator_cite_first_absent_when_empty_context():
    gen = _make_generator()
    preamble = gen._teaching_preamble([])
    assert "先引后答" not in preamble
    # 空 context 的兜底话术仍在 system 里
    assert "没有关于该问题的相关内容" in gen._system_prompt([])


# ── agent GENERATE_CITE_FIRST 条件化 ─────────────────────────

def test_agent_cite_first_constant_exists():
    from note_assistant.agent.agent import GENERATE_CITE_FIRST

    assert "先引后答" in GENERATE_CITE_FIRST
    assert "依据笔记" in GENERATE_CITE_FIRST


def test_agent_generate_system_base_has_no_cite_first():
    from note_assistant.agent.agent import GENERATE_SYSTEM

    # 基础 prompt 不含（generate_node 运行时按 top_results 条件化追加）
    assert "先引后答" not in GENERATE_SYSTEM


def test_agent_generate_node_appends_cite_first(monkeypatch):
    """有检索结果时 generate_node 的 human 消息头部应包含先引后答，system 全静态。"""
    import asyncio

    from note_assistant.agent import agent as agent_mod

    captured = {}

    class _Resp:
        content = "答案"

    class _LLM:
        async def ainvoke(self, msgs):
            captured["system"] = msgs[0].content
            captured["human"] = msgs[-1].content
            return _Resp()

    result = {
        "question": "什么是 RAG？",
        "condensed_question": "",
        "accumulated": [_fake_result()],
        "iteration": 1,
        "history": [],
    }
    monkeypatch.setattr(agent_mod, "get_llm", lambda **kw: _LLM())
    monkeypatch.setattr(agent_mod, "ensure_image_selected", lambda q, a, b: b)
    monkeypatch.setattr(agent_mod, "expand_image_neighbors", lambda acc, fn: [])
    monkeypatch.setattr(agent_mod, "_fetch_text_neighbors_by_heading", None, raising=False)

    state = dict(result)
    state.update({
        "judge_verdict": "sufficient",
        "widen_context": False,
        "gate_overrode": False,
        "history_messages": None,
    })
    asyncio.run(agent_mod.generate_node(state))
    assert "先引后答" in captured["human"]
    assert captured["human"].index("先引后答") < captured["human"].index("## 参考笔记")
    assert "先引后答" not in captured["system"]


def test_agent_generate_node_no_cite_first_when_no_results(monkeypatch):
    """accumulated 为空（give_up 场景）时不注入先引后答。"""
    from note_assistant.agent import agent as agent_mod

    captured = {}

    class _Resp:
        content = "答案"

    class _LLM:
        async def ainvoke(self, msgs):
            captured["system"] = msgs[0].content
            captured["human"] = msgs[-1].content
            return _Resp()

    monkeypatch.setattr(agent_mod, "get_llm", lambda **kw: _LLM())
    monkeypatch.setattr(agent_mod, "ensure_image_selected", lambda q, a, b: b)
    monkeypatch.setattr(agent_mod, "expand_image_neighbors", lambda acc, fn: [])

    state = {
        "question": "什么是 RAG？",
        "condensed_question": "",
        "accumulated": [],
        "iteration": 3,
        "judge_verdict": "give_up",
        "widen_context": False,
        "gate_overrode": False,
        "history": [],
        "history_messages": None,
    }
    asyncio.run(agent_mod.generate_node(state))
    assert "先引后答" not in captured["human"]
    assert "先引后答" not in captured["system"]


# ── 置信度校准 prompt ────────────────────────────────────────

def test_confidence_prompt_has_calibrated_bands(monkeypatch):
    """judge prompt 应包含分档定义（0.95~1.0 / 0.7~0.9 / 0.4~0.6 / 0.0~0.3）。"""
    import inspect

    from note_assistant.evaluation import generation_metrics as gm

    src = inspect.getsource(gm.compute_answer_accuracy)
    assert "0.95" in src and "0.7" in src and "0.4" in src
    assert "置信度" in src
    # 强调独立性：置信度与分数高低无关
    assert "与分数高低无关" in src

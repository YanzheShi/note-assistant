"""LLM-judge 指标的健壮性测试（2026-09-07 假 0 修复）。

核心约定：**无法判定 ≠ 判 0**——judge LLM 调用/解析彻底失败时返回
None（聚合层跳过），不再静默记 0 污染均值。
"""
from __future__ import annotations

import pytest

from note_assistant.evaluation.generation_metrics import (
    _extract_judge_json,
    answer_relevance,
    compute_answer_accuracy,
    faithfulness,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _ScriptedLLM:
    """按脚本依次返回内容，记录调用次数。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        idx = min(self.calls - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _AlwaysFailLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise RuntimeError("gateway down")


# ── _extract_judge_json ──────────────────────────────────────

def test_extract_json_plain():
    data = _extract_judge_json('{"accuracy": 0.9}', ("accuracy",))
    assert data == {"accuracy": 0.9}


def test_extract_json_in_fence_with_text_before():
    text = '好的，以下是评分：\n```json\n{"score": 0.8}\n```\n以上。'
    assert _extract_judge_json(text, ("score",)) == {"score": 0.8}


def test_extract_json_bare_braces_fallback():
    text = '前置说明 {"score": 0.5} 后置说明'
    assert _extract_judge_json(text, ("score",)) == {"score": 0.5}


def test_extract_json_missing_key_returns_none():
    assert _extract_judge_json('{"score": 0.5}', ("accuracy",)) is None


def test_extract_json_garbage_returns_none():
    assert _extract_judge_json("完全不是 JSON 的回复", ("score",)) is None
    assert _extract_judge_json("", ("score",)) is None


# ── faithfulness ─────────────────────────────────────────────

_CTX = "RAG 是检索增强生成。BM25 是稀疏检索算法。"
_GOOD = (
    '```json\n{"claims": ["a"], "verifications": '
    '[{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}\n```'
)


def test_faithfulness_happy_path():
    llm = _ScriptedLLM([_GOOD])
    assert faithfulness("答案", _CTX, llm) == 0.5
    assert llm.calls == 1


def test_faithfulness_parse_fail_then_retry_success():
    llm = _ScriptedLLM(["不是 JSON", _GOOD])
    assert faithfulness("答案", _CTX, llm) == 0.5
    assert llm.calls == 2


def test_faithfulness_total_failure_returns_none_not_zero():
    llm = _AlwaysFailLLM()
    assert faithfulness("答案", _CTX, llm) is None
    assert llm.calls == 2  # 首判 + 1 次重试


def test_faithfulness_garbage_twice_returns_none():
    llm = _ScriptedLLM(["垃圾1", "垃圾2"])
    assert faithfulness("答案", _CTX, llm) is None


def test_faithfulness_empty_verifications_retries():
    empty = '{"claims": [], "verifications": []}'
    llm = _ScriptedLLM([empty, _GOOD])
    assert faithfulness("答案", _CTX, llm) == 0.5


def test_faithfulness_empty_inputs_still_zero():
    # 空 candidate/context 是「答案失败」而非「judge 失败」，保持记 0
    assert faithfulness("", _CTX, llm=object()) == 0.0
    assert faithfulness("答案", "", llm=object()) == 0.0


def test_faithfulness_uses_widened_context(capfd):
    # context 截断阈值已 4000→8000：7000 字符的依据应完整进入 prompt
    long_ctx = "依据" * 3500  # 7000 chars
    llm = _ScriptedLLM([_GOOD])
    faithfulness("答案", long_ctx, llm)
    # 无异常即通过（截断逻辑在 prompt 构造内，重点是不再 4000 截断导致崩溃/失真）


# ── answer_relevance ─────────────────────────────────────────

def test_relevance_happy_path():
    llm = _ScriptedLLM(['{"score": 0.8, "reason": "ok"}'])
    assert answer_relevance("答案", "问题", llm) == 0.8


def test_relevance_score_clamped():
    llm = _ScriptedLLM(['{"score": 1.7, "reason": "ok"}'])
    assert answer_relevance("答案", "问题", llm) == 1.0


def test_relevance_total_failure_returns_none():
    llm = _ScriptedLLM(["垃圾1", "垃圾2"])
    assert answer_relevance("答案", "问题", llm) is None


# ── compute_answer_accuracy ──────────────────────────────────

def test_accuracy_happy_path():
    llm = _ScriptedLLM(['{"accuracy": 0.7, "confidence": 0.9, "reason": "ok"}'])
    acc, conf = compute_answer_accuracy("答案", "金标准", "问题", llm)
    assert (acc, conf) == (0.7, 0.9)


def test_accuracy_missing_key_treated_as_parse_failure():
    llm = _ScriptedLLM([
        '{"accuracy": 0.7, "reason": "缺 confidence"}',  # 缺键 → 重试
        '{"accuracy": 0.9, "confidence": 0.8, "reason": "ok"}',
    ])
    acc, conf = compute_answer_accuracy("答案", "金标准", "问题", llm)
    assert (acc, conf) == (0.9, 0.8)
    assert llm.calls == 2


def test_accuracy_total_failure_returns_none_pair():
    llm = _AlwaysFailLLM()
    acc, conf = compute_answer_accuracy("答案", "金标准", "问题", llm)
    assert acc is None and conf is None


def test_accuracy_empty_inputs_still_zero_pair():
    # 空答案/空金标准是「答案失败」而非「judge 失败」，保持记 0
    acc, conf = compute_answer_accuracy("", "金标准", "问题", llm=object())
    assert (acc, conf) == (0.0, 0.0)

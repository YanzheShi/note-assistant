"""多轮评测 + Token 统计 + 语义缓存命中（鸭子类型，不依赖真实 LLM）。

覆盖：
    - naive target：history 逐轮累积 + 全局 handler 旁路累加 token
    - agent target：session_id 多轮串联 + semantic_cache_stats 字段落报告
    - EvalQuestion.turns 序列化 / 反序列化往返
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path, Path as _P
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.evaluation.eval_dataset import EvalQuestion, EvalTurn, EvalDataset
from note_assistant.evaluation.evaluator import Evaluator
from note_assistant.llm.usage import get_token_handler


def _src(filepath):
    s = MagicMock()
    s.filepath = filepath
    s.preview = "preview"
    return s


    def test_naive_multiturn_accumulates_history_and_tokens():
        ask_target = MagicMock()
        seen_histories = []  # 每次调用时的 history 快照（避开 MagicMock 引用陷阱）

        def fake_ask(question, history=None):
            # 快照当前调用时的 history（list 引用会被后续 append 污染，必须 copy）
            seen_histories.append(list(history or []))
            # 模拟一次真实 LLM 调用触发全局 handler
            # （Evaluator.run 开头已 set_meter，此处 on_llm_end 会累加）
            h = get_token_handler()
            resp = MagicMock()
            resp.generations = [[MagicMock(message=MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 20}))]]
            h.on_llm_end(resp)
            return MagicMock(answer=f"ans:{question}", sources=[_src(f"f_{question}.md")])

        ask_target.ask.side_effect = fake_ask

    q = EvalQuestion(question="首问", turns=[
        EvalTurn(question="t1", golden_answer="g1", relevant_files=["a.md"]),
        EvalTurn(question="t2", golden_answer="g2", relevant_files=["b.md"]),
    ])
    dataset = EvalDataset(name="mt", questions=[q])

    # patch agent.ainvoke 以隔离（naive 路径不会真正调用）
    with patch("note_assistant.agent.runner.ainvoke") as mock_ainvoke:
        evaluator = Evaluator(ask_target, target_kind="naive")
        report = evaluator.run(dataset)

    # 两轮都被问到
    assert ask_target.ask.call_count == 2
    # 第二轮调用时 history 应仅含第一轮的 user + assistant（2 条）
    assert len(seen_histories[1]) == 2
    assert seen_histories[1][0]["role"] == "user"
    assert seen_histories[1][1]["role"] == "assistant"

    # token 被回调累加（两轮各 100 prompt / 20 completion）
    assert report.token_usage_total["prompt_tokens"] == 200
    assert report.token_usage_total["completion_tokens"] == 40
    assert report.token_usage_total["llm_calls"] == 2

    # 多轮明细
    assert len(report.per_conversation) == 1
    assert len(report.per_conversation[0]["turns"]) == 2
    # 每轮 token_usage 差值正确
    assert report.per_conversation[0]["turns"][0]["token_usage"]["prompt_tokens"] == 100

    # naive 无语义缓存统计
    assert report.semantic_cache_stats is None
    # handler 用完恢复 None（不影响线上）
    assert get_token_handler().meter is None


def test_agent_multiturn_session_and_cache_stats():
    with patch("note_assistant.agent.runner.ainvoke") as mock_ainvoke:
        seen = {}

        async def fake_ainvoke(question, history=None, session_id="", return_contexts=False):
            seen.setdefault(session_id, 0)
            seen[session_id] += 1
            result = MagicMock()
            result.answer = f"a:{question}"
            result.sources = [{"filepath": "x.md", "title": "X"}]
            result.contexts = ["ctx chunk"]
            return result

        mock_ainvoke.side_effect = fake_ainvoke

        q = EvalQuestion(question="首", turns=[
            EvalTurn(question="t1"), EvalTurn(question="t2"),
        ])
        dataset = EvalDataset(name="agent_mt", questions=[q])
        evaluator = Evaluator(None, target_kind="agent")
        report = evaluator.run(dataset)

        # 同一 conv 两轮共享 session_id（仅 1 个 session key）
        assert len(seen) == 1
        assert list(seen.values())[0] == 2
        # agent 报告含语义缓存字段
        assert report.semantic_cache_stats is not None
        assert "hit_rate" in report.semantic_cache_stats
        assert "enabled" in report.semantic_cache_stats
        # handler 恢复
        assert get_token_handler().meter is None


def test_dataset_turns_roundtrip():
    """EvalQuestion.turns 序列化 / 反序列化往返。"""
    q = EvalQuestion(
        question="首", golden_answer="g0",
        turns=[
            EvalTurn(question="t1", golden_answer="g1", relevant_files=["a.md"]),
            EvalTurn(question="t2"),
        ],
    )
    ds = EvalDataset(name="rt", questions=[q])
    with tempfile.TemporaryDirectory() as tmp:
        p = _P(tmp) / "rt.json"
        ds.save(p)
        ds2 = EvalDataset.load(p)
    assert ds2.questions[0].is_multiturn()
    assert len(ds2.questions[0].turns) == 2
    assert ds2.questions[0].turns[0].question == "t1"
    assert ds2.questions[0].turns[0].relevant_files == ["a.md"]


def test_eval_report_new_fields_serializable():
    """EvalReport 新字段能正常序列化（to_dict / save）。"""
    from note_assistant.evaluation.evaluator import EvalReport

    report = EvalReport(
        dataset_name="x", total_questions=1,
        token_usage_total={"prompt_tokens": 1, "llm_calls": 1, "cache_hit_rate": 0.0},
        llm_cache_hit_rate=0.0,
        semantic_cache_stats={"enabled": True, "hits": 0, "misses": 0, "hit_rate": 0.0},
        per_conversation=[{"question_index": 0, "turns": []}],
    )
    d = report.to_dict()
    assert d["token_usage_total"]["prompt_tokens"] == 1
    assert d["semantic_cache_stats"]["enabled"] is True
    assert d["per_conversation"][0]["question_index"] == 0
    # 能写 JSON
    import json
    json.dumps(d, ensure_ascii=False)

"""评测闭环测试（P7c）：指标聚合纯函数 + 离线 run_evaluation（注入 fake runner）。"""
import json

from note_assistant.agent import evaluation
from note_assistant.agent.evaluation import aggregate, extract_metrics
from note_assistant.agent.runner import _contexts_from_results
from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion
from note_assistant.retrieval.types import RetrievalResult


class _FakeResult:
    def __init__(self, answer, sources, trajectory):
        self.answer = answer
        self.sources = sources
        self.trajectory = trajectory


def _rec(question, answer, tools, verdicts, sources=2):
    traj = [{"type": "thought", "content": "路由判定：检索"}]
    for t in tools:
        traj.append({"type": "tool_call", "tool": t, "args": {}})
    for v in verdicts:
        traj.append({"type": "judge", "verdict": v})
    traj.append({"type": "answer", "content": answer})
    return extract_metrics(question, _FakeResult(answer, [{}] * sources, traj))


def test_aggregate_basic():
    recs = [
        _rec("q1", "a1", ["hybrid_search"], ["sufficient"]),
        _rec("q2", "a2", ["hybrid_search", "graph_expand"], ["need_more", "sufficient"]),
        _rec("q3", "a3", [], []),  # 闲聊，无工具
    ]
    agg = aggregate(recs)
    assert agg["count"] == 3
    assert agg["success"] == 3
    assert agg["route_distribution"] == {"search": 2, "chat": 1}
    assert agg["tool_distribution"] == {"hybrid_search": 2, "graph_expand": 1}
    assert agg["judge_distribution"] == {"sufficient": 2, "need_more": 1}
    assert agg["avg_tool_calls"] == (1 + 2 + 0) / 3
    assert agg["avg_iterations"] == (1 + 2 + 0) / 3


def test_extract_metrics_error():
    rec = extract_metrics("q", None, error="boom")
    assert rec.error == "boom"
    assert rec.route == ""


def test_run_evaluation_offline(monkeypatch, tmp_path):
    """注入 fake async runner，离线跑完评测并写出 JSON 报告。"""
    monkeypatch.setattr(evaluation, "EVAL_OUTPUT", tmp_path / "eval_agent.json")

    async def fake_run(q):
        traj = [
            {"type": "thought", "content": "路由判定：检索"},
            {"type": "tool_call", "tool": "hybrid_search", "args": {"query": q}},
            {"type": "observation", "content": "片段"},
            {"type": "judge", "verdict": "sufficient"},
            {"type": "answer", "content": f"答:{q}"},
        ]
        return _FakeResult(f"答:{q}", [{"filepath": "a.md"}], traj)

    import asyncio
    report = asyncio.run(evaluation.run_evaluation(
        questions=["问题A", "问题B"], run_fn=fake_run, with_ragas=False
    ))

    assert report["aggregate"]["count"] == 2
    assert report["aggregate"]["success"] == 2
    assert report["aggregate"]["route_distribution"] == {"search": 2}
    out = tmp_path / "eval_agent.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["aggregate"]["count"] == 2
    assert len(data["records"]) == 2


def _mk_result(question, answer, sources, trajectory, contexts=None):
    class _R:
        def __init__(self, a, s, t, c):
            self.answer = a
            self.sources = s
            self.trajectory = t
            self.contexts = c or []
    return _R(answer, sources, trajectory, contexts)


def test_contexts_from_results_picks_topk():
    """_contexts_from_results 取 top_k_rerank 条正文，按 score 降序，不截断。"""
    results = [
        RetrievalResult(score=0.1, page_content="低分正文", metadata={"filepath": "low.md"}),
        RetrievalResult(score=0.9, page_content="高分正文", metadata={"filepath": "high.md"}),
        RetrievalResult(score=0.5, page_content="中分正文", metadata={"filepath": "mid.md"}),
    ]
    # 默认 top_k_rerank=5 → 全部返回，按 score 降序
    ctx = _contexts_from_results(results)
    assert ctx == ["高分正文", "中分正文", "低分正文"]


def test_run_evaluation_with_dataset_retrieval_metrics(monkeypatch, tmp_path):
    """注入带 relevant_files 的黄金集 + fake runner，应算出检索指标（Recall/MRR/nDCG）。"""
    monkeypatch.setattr(evaluation, "EVAL_OUTPUT", tmp_path / "eval_agent.json")

    dataset = EvalDataset(
        name="mini",
        questions=[
            EvalQuestion(
                question="什么是 RAG？",
                golden_answer="RAG 是检索增强生成。",
                relevant_files=["08-RAG基础概念.md"],
            )
        ],
    )

    async def fake_run(q):
        traj = [
            {"type": "thought", "content": "路由判定：检索"},
            {"type": "tool_call", "tool": "hybrid_search", "args": {"query": q}},
            {"type": "observation", "content": "片段"},
            {"type": "judge", "verdict": "sufficient"},
            {"type": "answer", "content": f"答:{q}"},
        ]
        # 检索命中了黄金集标注的文件 → 检索指标应为满分
        return _mk_result(
            question=q,
            answer=f"答:{q}",
            sources=[{"filepath": "08-RAG基础概念.md"}],
            trajectory=traj,
            contexts=["chunk 正文"],
        )

    import asyncio
    report = asyncio.run(evaluation.run_evaluation(dataset=dataset, run_fn=fake_run, with_ragas=False))

    rec = report["records"][0]
    assert rec["retrieval_metrics"]["mrr"] == 1.0
    assert rec["retrieval_metrics"]["recall@3"] == 1.0
    assert rec["retrieval_metrics"]["ndcg@3"] == 1.0
    # 未开 ragas → 生成质量指标留空
    assert rec["faithfulness"] is None
    # 汇总里应有检索指标均值
    assert report["aggregate"]["avg_mrr"] == 1.0


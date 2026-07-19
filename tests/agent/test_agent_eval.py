"""评测闭环测试（P7c）：指标聚合纯函数 + 离线 run_evaluation（注入 fake runner）。"""
import json

from note_assistant.agent import evaluation
from note_assistant.agent.evaluation import aggregate, extract_metrics


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

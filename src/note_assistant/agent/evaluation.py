"""Agent 评测闭环骨架（P7c）。

提供离线的「轨迹级」评测：对一组黄金问题跑通 Agentic RAG，收集可观测指标
（路由决策、检索轮次、工具调用分布、Judge 判定、延迟、答案质量），
输出 JSON 报告到 data/eval_agent.json。

可选择性接入 ragas 做 faithfulness / answer_relevance：
    - 若已安装 ragas 且配置了 LLM，自动启用；
    - 否则跳过，不影响主流程（纯轨迹级指标仍然产出）。

设计上 ``run_evaluation`` 的 ``run_fn`` 可注入，因此**完全离线**测试时
可用 fake runner 验证指标聚合逻辑，无需 Ollama / DeepSeek / ChromaDB。

用法：
    uv run python -m note_assistant.agent.evaluation
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from note_assistant.config import PROJECT_ROOT

EVAL_OUTPUT: Path = PROJECT_ROOT / "data" / "eval_agent.json"

# 一组公开的示例问题（覆盖：需检索 / 闲聊 / 多子话题 / 对比）
GOLDEN_QUESTIONS = [
    "FlashAttention 是什么？它相比标准注意力有什么改进？",
    "我的知识库里关于检索方法都记了哪些笔记？",
    "你好，你是谁？",
    "RAG 系统怎么处理长文档的切分？",
    "对比一下 dense 向量检索和 BM25 关键词检索的优劣。",
]


@dataclass
class EvalRecord:
    question: str
    route: str = ""
    answer: str = ""
    latency_ms: float = 0.0
    num_iterations: int = 0
    num_tool_calls: int = 0
    tool_calls: List[dict] = field(default_factory=list)
    judge_verdicts: List[str] = field(default_factory=list)
    num_sources: int = 0
    error: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None


# ──────────────────────────────────────────────
# 指标聚合（纯函数，可离线测试）
# ──────────────────────────────────────────────

def _count(items: List[str]) -> dict:
    d: dict[str, int] = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return d


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(records: List[EvalRecord]) -> dict:
    """聚合一批评测记录为汇总指标（纯函数）。"""
    n = len(records) or 1
    ok = [r for r in records if not r.error]
    return {
        "count": len(records),
        "success": len(ok),
        "avg_latency_ms": round(sum(r.latency_ms for r in records) / n, 2),
        "avg_iterations": round(sum(r.num_iterations for r in records) / n, 2),
        "avg_tool_calls": round(sum(r.num_tool_calls for r in records) / n, 2),
        "avg_sources": round(sum(r.num_sources for r in records) / n, 2),
        "route_distribution": _count([r.route for r in records]),
        "tool_distribution": _count(
            [t["tool"] for r in records for t in r.tool_calls if t.get("tool")]
        ),
        "judge_distribution": _count(
            [v for r in records for v in r.judge_verdicts if v]
        ),
        "faithfulness_avg": _avg([r.faithfulness for r in ok]),
        "answer_relevance_avg": _avg([r.answer_relevance for r in ok]),
    }


def extract_metrics(question: str, result, error: Optional[str] = None) -> EvalRecord:
    """从一次运行结果抽取轨迹级指标（纯函数）。"""
    rec = EvalRecord(question=question)
    if error:
        rec.error = error
        return rec
    rec.answer = getattr(result, "answer", "") or ""
    rec.num_sources = len(getattr(result, "sources", []) or [])
    traj = getattr(result, "trajectory", []) or []
    rec.num_tool_calls = sum(1 for t in traj if t.get("type") == "tool_call")
    rec.tool_calls = [
        {"tool": t.get("tool"), "args": t.get("args")}
        for t in traj
        if t.get("type") == "tool_call"
    ]
    rec.judge_verdicts = [
        t.get("verdict")
        for t in traj
        if t.get("type") == "judge" and t.get("verdict")
    ]
    # 路由：有工具调用说明走检索，否则走直接对话
    rec.route = "search" if rec.num_tool_calls > 0 else "chat"
    # 检索轮次用 Judge 判定次数近似（每轮检索后都过一次 reflect）
    rec.num_iterations = len(rec.judge_verdicts)
    return rec


# ──────────────────────────────────────────────
# 可选 ragas 评测（守卫式导入）
# ──────────────────────────────────────────────

def evaluate_with_ragas(question: str, answer: str, context: str):
    """用 ragas 计算 faithfulness / answer_relevance。无 ragas 或异常时返回 (None, None)。"""
    try:
        from datasets import Dataset  # noqa: F401
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception:
        return None, None
    try:
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[context]],
        }
        dataset = EvaluationDataset.from_dict(data)
        score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        res = score.to_dict()
        return (
            res.get("faithfulness"),
            res.get("answer_relevancy"),
        )
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

async def run_evaluation(
    questions: Optional[List[str]] = None,
    run_fn: Optional[Callable[[str], Awaitable]] = None,
    with_ragas: bool = False,
) -> dict:
    """运行评测，返回报告 dict 并写入 EVAL_OUTPUT。

    Args:
        questions: 黄金问题列表；默认用 GOLDEN_QUESTIONS。
        run_fn:    异步运行函数，签名 ``async def run_fn(q) -> AgentRunResult``。
                   默认接 agent_runner.ainvoke。
        with_ragas: 是否启用 ragas 答案质量评测（需要 ragas + LLM）。
    """
    from note_assistant.agent import runner as agent_runner

    questions = questions or GOLDEN_QUESTIONS
    run_fn = run_fn or (lambda q: agent_runner.ainvoke(q))

    records: List[EvalRecord] = []
    for q in questions:
        t0 = time.time()
        try:
            result = await run_fn(q)
            rec = extract_metrics(q, result)
            if with_ragas:
                ctx = "\n".join(
                    str(t.get("content", ""))
                    for t in (getattr(result, "trajectory", []) or [])
                    if t.get("type") == "observation"
                )
                f, ar = evaluate_with_ragas(q, rec.answer, ctx)
                rec.faithfulness = f
                rec.answer_relevance = ar
        except Exception as e:  # noqa: BLE001
            rec = extract_metrics(q, None, error=str(e))
        rec.latency_ms = round((time.time() - t0) * 1000, 2)
        records.append(rec)

    report = {
        "aggregate": aggregate(records),
        "records": [asdict(r) for r in records],
    }
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    rep = asyncio.run(run_evaluation())
    print(json.dumps(rep["aggregate"], ensure_ascii=False, indent=2))

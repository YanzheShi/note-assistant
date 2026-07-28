"""Agent 评测闭环骨架（P7c）。

提供离线的「轨迹级」评测：对一组黄金问题跑通 Agentic RAG，收集可观测指标
（路由决策、检索轮次、工具调用分布、Judge 判定、延迟、检索质量、答案质量），
输出 JSON 报告到 data/eval_agent.json。

答案质量指标复用 ``evaluation.ragas_metrics.batch_compute_ragas``（faithfulness /
context_precision / context_recall + 手写 ROUGE-L / BLEU / 语义相似度）；
检索质量指标复用 ``evaluation.retrieval_metrics.compute_retrieval_metrics``
（Recall@K / Precision@K / MRR / nDCG），需要黄金集标注 ``relevant_files``。

设计上 ``run_evaluation`` 的 ``run_fn`` 可注入，因此**完全离线**测试时
可用 fake runner 验证指标聚合逻辑，无需 Ollama / DeepSeek / ChromaDB。

用法：
    uv run python -m note_assistant.agent.evaluation          # 用内置黄金集 + 全指标
    uv run python -m note_assistant.agent.evaluation --ragas  # 额外算 ragas 生成指标
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from note_assistant.config import PROJECT_ROOT
from note_assistant.evaluation.eval_dataset import EvalQuestion, get_builtin_dataset
from note_assistant.evaluation.retrieval_metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)

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
    # 生成质量指标（由 batch_compute_ragas 统一产出：ragas + 手写）
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    rouge_l: Optional[float] = None
    bleu_1: Optional[float] = None
    bleu_4: Optional[float] = None
    semantic_similarity: Optional[float] = None
    # 检索质量指标（Recall@K / Precision@K / MRR / nDCG，来自黄金集 relevant_files）
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)


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


def _flatten_retrieval_metrics(rm) -> Dict[str, float]:
    """将 RetrievalMetrics 展平为 flat dict（与 naive evaluator 同口径）。"""
    d: Dict[str, float] = {"mrr": rm.mrr}
    for k, v in rm.recall_at_k.items():
        d[f"recall@{k}"] = v
    for k, v in rm.precision_at_k.items():
        d[f"precision@{k}"] = v
    for k, v in rm.ndcg_at_k.items():
        d[f"ndcg@{k}"] = v
    return d


def aggregate(records: List[EvalRecord]) -> dict:
    """聚合一批评测记录为汇总指标（纯函数）。"""
    n = len(records) or 1
    ok = [r for r in records if not r.error]
    gen_keys = [
        "faithfulness", "context_precision", "context_recall",
        "rouge_l", "bleu_1", "bleu_4", "semantic_similarity",
    ]
    # 检索指标 key 动态汇总（可能含 recall@3 / mrr / ndcg@5 等）
    retr_keys: List[str] = []
    for r in ok:
        for k in r.retrieval_metrics:
            if k not in retr_keys:
                retr_keys.append(k)
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
        **{f"avg_{k}": _avg([getattr(r, k) for r in ok]) for k in gen_keys},
        **{f"avg_{k}": _avg([r.retrieval_metrics.get(k) for r in ok]) for k in retr_keys},
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
# 主入口
# ──────────────────────────────────────────────

async def run_evaluation(
    questions: Optional[List[str]] = None,
    dataset=None,
    run_fn: Optional[Callable[[str], Awaitable]] = None,
    with_ragas: bool = False,
) -> dict:
    """运行评测，返回报告 dict 并写入 EVAL_OUTPUT。

    Args:
        questions:   黄金问题列表（纯字符串）。与 ``dataset`` 二选一；
                     传 questions 时仅有轨迹级 + 检索指标（无金标准答案）。
        dataset:     EvalDataset（含 golden_answer + relevant_files）。
                     默认用 get_builtin_dataset()，可算全指标（检索 + 生成 + ROUGE/BLEU）。
        run_fn:      异步运行函数，签名 ``async def run_fn(q) -> AgentRunResult``。
                     默认接 agent_runner.ainvoke（with_ragas 时自动 return_contexts）。
        with_ragas:  是否启用 ragas 生成质量评测（需要 ragas + 配置好的 LLM）。
    """
    from note_assistant.agent import runner as agent_runner

    # 解析评测项（统一为 EvalQuestion）
    if dataset is not None:
        items = list(dataset.questions)
    elif questions:
        items = [EvalQuestion(question=q, golden_answer="", relevant_files=[]) for q in questions]
    else:
        items = list(get_builtin_dataset().questions)

    # 默认 runner：直接接 ainvoke；with_ragas 时自动回传完整上下文给评测
    run_fn = run_fn or (lambda q: agent_runner.ainvoke(q, return_contexts=with_ragas))

    records: List[EvalRecord] = []
    # RAGAS 模式：先收集，循环结束后批量 evaluate（框架内部并行，效率更高）
    if with_ragas:
        rq: List[str] = []
        ra: List[str] = []
        rc: List[List[str]] = []
        rg: List[str] = []

    for item in items:
        q = item.question
        t0 = time.time()
        try:
            result = await run_fn(q)
            rec = extract_metrics(q, result)
            # 检索质量指标（需黄金集标注 relevant_files）
            if item.relevant_files:
                retrieved_files = [
                    s.get("filepath") for s in (getattr(result, "sources", []) or [])
                ]
                rec.retrieval_metrics = _flatten_retrieval_metrics(
                    compute_retrieval_metrics(retrieved_files, set(item.relevant_files))
                )
            if with_ragas:
                rq.append(q)
                ra.append(rec.answer)
                rc.append(getattr(result, "contexts", []) or [])
                rg.append(item.golden_answer)
        except Exception as e:  # noqa: BLE001
            rec = extract_metrics(q, None, error=str(e))
        rec.latency_ms = round((time.time() - t0) * 1000, 2)
        records.append(rec)

    # RAGAS 批量评测后回写生成指标
    if with_ragas and rq:
        from note_assistant.evaluation.ragas_metrics import batch_compute_ragas
        try:
            scores = batch_compute_ragas(questions=rq, answers=ra, contexts=rc, ground_truths=rg)
            for i, r in enumerate(records):
                if i < len(scores):
                    s = scores[i]
                    r.faithfulness = s.get("faithfulness")
                    r.context_precision = s.get("context_precision")
                    r.context_recall = s.get("context_recall")
                    r.rouge_l = s.get("rouge_l")
                    r.bleu_1 = s.get("bleu_1")
                    r.bleu_4 = s.get("bleu_4")
                    r.semantic_similarity = s.get("semantic_similarity")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RAGAS 批量评测失败，生成指标留空: {e}")

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
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m note_assistant.agent.evaluation",
        description="Agentic RAG 评测：轨迹级 + 检索质量指标，可选 ragas 生成质量指标。",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="额外计算 ragas 生成质量指标（faithfulness/context_precision/"
        "context_recall + ROUGE/BLEU/语义相似度），需已配置 ragas LLM "
        "（settings.ragas_base_url / ragas_api_key / ragas_llm_model）。",
    )
    args = parser.parse_args()

    rep = asyncio.run(run_evaluation(with_ragas=args.ragas))

    print(json.dumps(rep["aggregate"], ensure_ascii=False, indent=2))
    if args.ragas:
        print(
            "\n[ragas] 已纳入生成质量指标；若报告中 faithfulness/context_precision/"
            "context_recall 等为 null，请检查 ragas LLM 配置是否就绪。",
            file=sys.stderr,
        )

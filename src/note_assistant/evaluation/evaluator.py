"""
评测编排器：将检索 + 生成管线接入评测流程，批量跑评测集，输出指标报告。

架构：
    EvalDataset (评测集)
            ↓
    Evaluator.run() → 对每条问题：
        1. rag_chain.ask(question) → AskResponse
        2. 提取 retrieved_files + context（sources 中的 preview）
        3. 计算检索指标：compute_retrieval_metrics()
        4. 计算生成指标：compute_generation_metrics()
        ↓
    汇总所有问题的指标 → EvalReport
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion
from note_assistant.evaluation.retrieval_metrics import compute_retrieval_metrics, RetrievalMetrics
from note_assistant.evaluation.generation_metrics import compute_generation_metrics
from note_assistant.pipeline.rag_chain import AskResponse, SourceInfo

logger = logging.getLogger(__name__)


def _flatten_retrieval_metrics(rm: RetrievalMetrics) -> Dict[str, float]:
    """
    将 RetrievalMetrics dataclass 展平为 flat dict，方便 _aggregate_metrics 遍历。

    输入：
        RetrievalMetrics(mrr=0.5, recall_at_k={3: 1.0}, precision_at_k={3: 0.67}, ndcg_at_k={3: 0.75})
    输出：
        {"mrr": 0.5, "recall@3": 1.0, "precision@3": 0.67, "ndcg@3": 0.75}
    """
    d: Dict[str, float] = {"mrr": rm.mrr}
    for k, v in rm.recall_at_k.items():
        d[f"recall@{k}"] = v
    for k, v in rm.precision_at_k.items():
        d[f"precision@{k}"] = v
    for k, v in rm.ndcg_at_k.items():
        d[f"ndcg@{k}"] = v
    return d


@dataclass
class SingleEvalResult:
    """单条问题的评测结果。"""
    question: str
    retrieved_files: List[str] = field(default_factory=list)
    generated_answer: str = ""
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """完整评测报告。"""
    dataset_name: str
    total_questions: int
    avg_elapsed_ms: float = 0.0
    retrieval_metrics_avg: Dict[str, float] = field(default_factory=dict)
    generation_metrics_avg: Dict[str, float] = field(default_factory=dict)
    per_question: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为字典，用于 JSON 序列化。"""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """保存为 JSON 文件。"""
        p = Path(path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class Evaluator:
    """
    RAG 管线评测器。

    用法：
        evaluator = Evaluator(rag_chain, llm=my_llm)
        report = evaluator.run(dataset)
        report.save("eval_report.json")
        print(report.retrieval_metrics_avg)

    rag_chain 只需要有 ask(question) → AskResponse 方法即可（鸭子类型）。
    llm 单独传是因为 RAGChain 内部可能没有暴露 LLM 实例。
    """

    def __init__(self, rag_chain, llm=None):
        """
        Args:
            rag_chain: RAGChain 实例（或任何有 ask 方法的对象）
            llm: 可选的 LLM 实例，用于 faithfulness + answer_relevance
                 需要有 invoke(messages) 方法
        """
        self.rag_chain = rag_chain
        self.llm = llm

    def run(self, dataset: EvalDataset, k_values: List[int] | None = None) -> EvalReport:
        """
        在评测集上运行完整评测。

        流程：
        1. 遍历 dataset.questions
        2. 对每条问题调用 rag_chain.ask()，计算检索指标和生成指标
        3. 汇总所有结果 → 求平均值 → 返回 EvalReport

        异常处理：rag_chain.ask() 抛出异常时，记录为空结果继续下一条（容错模式）。

        Args:
            dataset: 评测数据集
            k_values: 检索指标截断点，默认 [3, 5, 10]

        Returns:
            EvalReport
        """

        eval_results = []
        for question in dataset.questions:

            start = time.time()
            try:
                ans = self.rag_chain.ask(question)
            except Exception as e:
                logger.error(f"评测失败: {e}")
                ans = AskResponse(answer="", sources=[], graph_expansion=0, retrieved=0)

            elapsed = (time.time() - start) * 1000
            
            retrieved_files = [source.filepath for source in ans.sources]
            context = " ".join(s.preview for s in ans.sources)

            retrieval_metrics = compute_retrieval_metrics(retrieved_files, question.relevant_files, k_values)
            generation_metrics = compute_generation_metrics(ans.answer, question.golden_answer, llm=self.llm, context=context, question=question.question)

            eval_results.append(
                SingleEvalResult(
                    question=question.question,
                    retrieved_files=retrieved_files,
                    generated_answer=ans.answer,
                    retrieval_metrics=_flatten_retrieval_metrics(retrieval_metrics),
                    generation_metrics=generation_metrics.to_dict(),
                    elapsed_ms=elapsed,
                )
            )

        avg_retrieval, avg_generation = self._aggregate_metrics(eval_results)
        avg_elapsed = sum(result.elapsed_ms  for result in eval_results) / len(eval_results)
        return EvalReport(
            dataset_name=dataset.name,
            total_questions=len(dataset.questions),
            avg_elapsed_ms=avg_elapsed,
            retrieval_metrics_avg=avg_retrieval,
            generation_metrics_avg=avg_generation,
            per_question=[asdict(r) for r in eval_results],
        )

    def _aggregate_metrics(self, results: List[SingleEvalResult]) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        将多条评测结果的指标汇总为平均值。

        按指标名分组收集所有值，对每组求平均。值为 None 时跳过（LLM 调用失败时）。
        """
        agg_retrieval = defaultdict(list)
        agg_generation = defaultdict(list)

        for r in results:
            for k, v in r.retrieval_metrics.items():
                if v is not None:
                    agg_retrieval[k].append(v)
            for k, v in r.generation_metrics.items():
                if v is not None:
                    agg_generation[k].append(v)

        avg_retrieval = {k: sum(vals) / len(vals) for k, vals in agg_retrieval.items()}
        avg_generation = {
            k: sum(vals) / len(vals) for k, vals in agg_generation.items()
        }

        return avg_retrieval, avg_generation

    def run_single(self, question: str, golden_answer: str, relevant_files: List[str]) -> SingleEvalResult:
        """
        评测单条问题（调试用）。

        和 run() 的逻辑相同，只是不聚合，直接返回 SingleEvalResult。
        """
        start = time.time()
        try:
            ans = self.rag_chain.ask(question)
        except Exception as e:
            logger.error(f"评测失败: {e}")
            ans = AskResponse(answer="", sources=[], graph_expansion=0, retrieved=0)

        elapsed = (time.time() - start) * 1000

        retrieved_files = [source.filepath for source in ans.sources]
        context = " ".join(s.preview for s in ans.sources)

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_files, relevant_files
        )
        generation_metrics = compute_generation_metrics(
            ans.answer,
            golden_answer,
            llm=self.llm,
            context=context,
            question=question,
        )

        return SingleEvalResult(
                question=question,
                retrieved_files=retrieved_files,
                generated_answer=ans.answer,
                retrieval_metrics=_flatten_retrieval_metrics(retrieval_metrics),
                generation_metrics=generation_metrics.to_dict(),
                elapsed_ms=elapsed,
            )
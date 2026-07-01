"""
评测模块：检索指标 + 生成指标 + 评测编排器。

模块：
    eval_dataset     — 评测数据集（问题 + 金标准）
    retrieval_metrics — 检索指标（recall@k, precision@k, MRR, nDCG）
    generation_metrics — 生成指标（ROUGE-L, BLEU, 语义相似度）
    evaluator         — 评测编排器（批量跑评测集，输出报告）
"""

from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion, get_builtin_dataset
from note_assistant.evaluation.retrieval_metrics import compute_retrieval_metrics, recall_at_k, precision_at_k, mrr, ndcg_at_k, RetrievalMetrics
from note_assistant.evaluation.generation_metrics import (
    compute_generation_metrics, rouge_l, bleu_1, bleu_4,
    semantic_similarity, faithfulness, answer_relevance,
    GenerationMetrics,
)
from note_assistant.evaluation.evaluator import Evaluator, EvalReport, SingleEvalResult

__all__ = [
    "EvalDataset",
    "EvalQuestion",
    "get_builtin_dataset",
    "compute_retrieval_metrics",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "RetrievalMetrics",
    "GenerationMetrics",
    "compute_generation_metrics",
    "rouge_l",
    "bleu_1",
    "bleu_4",
    "semantic_similarity",
    "faithfulness",
    "answer_relevance",
    "Evaluator",
    "EvalReport",
    "SingleEvalResult",
]

"""
RAGAS 评估指标包装器。

在保留手写指标（ROUGE-L / BLEU / 语义相似度）的基础上，新增 RAGAS 指标：
  - faithfulness        — 答案是否忠于检索到的上下文
  - context_precision   — 检索到的上下文中，有多少是真正需要的（新增，手写版没有）
  - context_recall      — 金标准答案所需的信息是否都被检索到了（新增，手写版没有）

用法（由 Evaluator 自动调用，通常不直接使用）：
    from note_assistant.evaluation.ragas_metrics import batch_compute_ragas
    scores = batch_compute_ragas(
        questions=[...], answers=[...],
        contexts=[[...], ...], ground_truths=[...],
    )

模型由 settings.ragas_llm_model 配置（在 config.py 或 .env 中设置）。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from note_assistant.evaluation.generation_metrics import (
    rouge_l,
    bleu_1,
    bleu_4,
    semantic_similarity,
)
from note_assistant.config import settings

logger = logging.getLogger(__name__)

# RAGAS 指标中，需要 LLM 而不需要 embedding 的
# answer_relevancy 需要 embedding，暂时跳过（手写版已有）
_RAGAS_LLM_ONLY_METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
]


def _clean_float(value: float) -> float:
    """将 NaN/Inf 转为 0.0，避免报告里出现 NaN。"""
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _make_ragas_llm(ollama_model: str | None = None):
    """
    通过 OpenAI 兼容接口连接 LLM，构造 RAGAS 所需的 LLM 实例。

    默认使用 settings 中的配置（base_url / api_key / model），
    可以在 .env 中覆盖：
        RAGAS_BASE_URL=https://api.deepseek.com/v1
        RAGAS_API_KEY=sk-xxx
        RAGAS_LLM_MODEL=deepseek-chat

    也可以传 ollama_model 覆盖模型名。

    Args:
        ollama_model: Ollama 上的 chat 模型名。不传则用配置文件的值。
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    model = ollama_model or settings.ragas_llm_model
    base_url = settings.ragas_base_url
    api_key = settings.ragas_api_key

    client = OpenAI(base_url=base_url, api_key=api_key)
    llm = llm_factory(model, client=client)
    # judge LLM 默认 max_tokens 偏小，遇到较长 answer/context 会被截断，
    # 触发 ragas 的 IncompleteOutputException，进而污染 faithfulness /
    # context_precision（指标变 nan → 被 _clean_float 记为 0.0）。
    # ragas 的 RagasLLM 把真实模型挂在 .llm 上，直接调大即可。
    _RAGAS_JUDGE_MAX_TOKENS = 4096
    try:
        if hasattr(llm, "llm") and hasattr(llm.llm, "max_tokens"):
            llm.llm.max_tokens = _RAGAS_JUDGE_MAX_TOKENS
    except Exception as e:
        logger.warning(f"设置 RAGAS judge max_tokens 失败（忽略）: {e}")
    return llm


def batch_compute_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    ollama_model: str | None = None,
) -> List[Dict[str, float]]:
    """
    批量计算 RAGAS 指标 + 保留手写指标（ROUGE-L / BLEU / 语义相似度）。

    RAGAS 在批量模式下效率更高——所有问题一次性传入 evaluate()，框架内部
    自动并行化 LLM 调用。

    Args:
        questions: 所有用户问题
        answers: 所有模型生成的答案
        contexts: 每条问题检索到的上下文片段列表（list of list of str）
        ground_truths: 金标准答案
        ollama_model: Ollama 上的 chat 模型名。不传则用 settings.ragas_llm_model。

    Returns:
        每条结果一个 dict，包含：
            faithfulness, context_precision, context_recall,      (RAGAS)
            rouge_l, bleu_1, bleu_4, semantic_similarity          (手写)
    """
    n = len(questions)
    if n == 0:
        return []

    # 1. 构造 RAGAS LLM
    try:
        llm = _make_ragas_llm(ollama_model)
    except Exception as e:
        logger.error(f"RAGAS LLM 初始化失败: {e}")
        # 回退：只返回手写指标，RAGAS 指标填 0.0
        return _fallback_metrics(questions, answers, contexts, ground_truths)

    # 2. 运行 RAGAS evaluate
    try:
        ragas_scores = _run_ragas_evaluate(questions, answers, contexts, ground_truths, llm)
    except Exception as e:
        logger.warning(f"RAGAS evaluate 失败: {e}，回退到手写指标")
        ragas_scores = [{"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0} for _ in range(n)]

    # 3. 合并手写指标
    results = []
    for i in range(n):
        candidate = answers[i]
        reference = ground_truths[i]

        scores = {
            "faithfulness": _clean_float(ragas_scores[i].get("faithfulness", 0.0)),
            "context_precision": _clean_float(ragas_scores[i].get("context_precision", 0.0)),
            "context_recall": _clean_float(ragas_scores[i].get("context_recall", 0.0)),
            "rouge_l": rouge_l(candidate, reference),
            "bleu_1": bleu_1(candidate, reference),
            "bleu_4": bleu_4(candidate, reference),
            "semantic_similarity": semantic_similarity(candidate, reference),
        }
        results.append(scores)

    return results


def _run_ragas_evaluate(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    llm,
) -> List[Dict[str, float]]:
    """
    调用 RAGAS evaluate() 批量计算。

    Args:
        questions: 用户问题列表
        answers: 模型答案列表
        contexts: 每条的检索上下文列表
        ground_truths: 金标准答案列表
        llm: RAGAS LLM 实例（来自 _make_ragas_llm）

    Returns:
        每行一个 dict，key 为指标名
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate

    # ragas 0.4.3 的 ``ragas.metrics.collections`` 仍是占位包（import 到的是
    # 子模块而非 metric 实例），官方 deprecation 提示的
    # ``from ragas.metrics.collections import faithfulness`` 在 0.4.3 不可用。
    # 0.4.3 真正可运行的是公共 ``ragas.metrics.*``（已实例化的单例，仅产生
    # deprecation warning，v1.0 会移除）。升级到 ragas>=1.0 后，将此处改为
    # ``from ragas.metrics.collections import faithfulness, context_precision,
    # context_recall`` 即可（届时 collections 导出的是实例）。
    # ``_as_metric`` 兼容「类」与「实例」两种导出形态，避免再次踩坑。
    from ragas.metrics import faithfulness, context_precision, context_recall

    def _as_metric(m):
        return m() if isinstance(m, type) else m

    f = _as_metric(faithfulness)
    cp = _as_metric(context_precision)
    cr = _as_metric(context_recall)
    f.llm = llm
    cp.llm = llm
    cr.llm = llm
    metrics = [f, cp, cr]

    # 构造 Dataset
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    # 运行
    score = ragas_evaluate(
        dataset,
        metrics=metrics,
        raise_exceptions=False,  # 单条失败不中断整体
        batch_size=1,  # 逐条调用，避免 API 限流
    )

    df = score.to_pandas()

    # 提取指标列
    result_cols = [m.name for m in metrics]
    results = []
    for _, row in df.iterrows():
        results.append({col: float(row[col]) for col in result_cols if col in row})

    return results


def _fallback_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> List[Dict[str, float]]:
    """RAGAS 不可用时的回退：只计算手写指标，RAGAS 指标填 0.0。"""
    results = []
    for i in range(len(questions)):
        candidate = answers[i]
        reference = ground_truths[i]
        results.append({
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "rouge_l": rouge_l(candidate, reference),
            "bleu_1": bleu_1(candidate, reference),
            "bleu_4": bleu_4(candidate, reference),
            "semantic_similarity": semantic_similarity(candidate, reference),
        })
    return results
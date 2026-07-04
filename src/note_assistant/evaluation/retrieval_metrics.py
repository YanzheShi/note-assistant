"""
检索评测指标：Recall@K、Precision@K、MRR、nDCG。

所有指标基于"检索结果 vs 金标准 relevant_files"计算：

    retrieved_files (List[str])  +  relevant_files (Set[str])
            ↓
    recall_at_k() / precision_at_k() / mrr() / ndcg_at_k()
            ↓
    compute_retrieval_metrics() → RetrievalMetrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set


@dataclass(frozen=True)
class RetrievalMetrics:
    """检索指标结果，不可变。"""
    mrr: float
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


def recall_at_k(retrieved_files: List[str], relevant_files: Set[str], k: int | None = None) -> float:
    """
    Recall@K：前 K 个检索结果中，有多少比例是相关文档。

    公式：Recall = |Retrieved@K ∩ Relevant| / |Relevant|

    边界情况：
    - relevant_files 为空且 retrieved_files 也为空 → 返回 1.0（没有检索也没有遗漏）
    - relevant_files 为空但有检索结果 → 返回 0.0（全错）

    Args:
        retrieved_files: 检索返回的文件路径列表（不去重）
        relevant_files: 金标准相关文件集合
        k: 截断位置（None 表示全部）

    Returns:
        0.0 ~ 1.0
    """
    if not retrieved_files and not relevant_files:
        return 1.0

    if not relevant_files:
        return 0.0

    intersection = set(retrieved_files[:k]).intersection(relevant_files)

    return len(intersection) / len(relevant_files)


def precision_at_k(retrieved_files: List[str], relevant_files: Set[str], k: int | None = None) -> float:
    """
    Precision@K：前 K 个检索结果中，有多少比例是相关文档。

    公式：Precision@K = |Retrieved@K ∩ Relevant| / K

    分母是 K（IR 标准定义），不是实际检索数量。K 为 None 时分母为 len(retrieved_files)。

    Args:
        retrieved_files: 检索返回的文件路径列表
        relevant_files: 金标准相关文件集合
        k: 截断位置（None 表示全部）

    Returns:
        0.0 ~ 1.0
    """
    if not retrieved_files:
        return 0.0

    top_k = retrieved_files[:k]
    if not top_k:
        return 0.0

    intersection = set(top_k).intersection(relevant_files)
    denominator = k if k is not None else len(retrieved_files)
    return len(intersection) / denominator


def mrr(retrieved_files: List[str], relevant_files: Set[str]) -> float:
    """
    MRR (Mean Reciprocal Rank)：第一个相关文档的排名倒数。

    公式：MRR = 1 / rank_of_first_relevant

    - 只看"第一个命中"的位置，越靠前越好
    - 如果完全没有命中 → 返回 0.0

    Args:
        retrieved_files: 检索返回的文件路径列表（有序，排名决定顺序）
        relevant_files: 金标准相关文件集合

    Returns:
        0.0 ~ 1.0
    """

    for i, retrieved_file in enumerate(retrieved_files):
        if retrieved_file in relevant_files:
            return 1 / (i + 1)

    return 0.0


def ndcg_at_k(retrieved_files: List[str], relevant_files: Set[str], k: int | None = None) -> float:
    """
    nDCG@K (Normalized Discounted Cumulative Gain)：考虑排名的加权相关度。

    公式：
        DCG@K = Σ_{i=1}^{min(K, |R|)} rel_i / log2(i + 1)
        IDCG@K = Σ_{i=1}^{min(K, |Rel|)} 1 / log2(i + 1)
        nDCG@K = DCG@K / IDCG@K

    其中 rel_i ∈ {0, 1} 表示第 i 个检索结果是否相关。
    注意：重复文档只计一次相关（后续重复视为不相关），防止 nDCG > 1。

    边界情况：
    - relevant_files 为空 → 返回 0.0
    - retrieved_files 为空 → 返回 0.0

    Args:
        retrieved_files: 检索返回的文件路径列表（有序）
        relevant_files: 金标准相关文件集合
        k: 截断位置（None 表示全部）

    Returns:
        0.0 ~ 1.0
    """
    if not retrieved_files or not relevant_files:
        return 0.0

    top_k = retrieved_files[:k] if k else retrieved_files

    # 去重：每个文件第一次出现算相关，后续重复算不相关
    seen: Set[str] = set()
    rels = []
    for f in top_k:
        if f in relevant_files and f not in seen:
            rels.append(1)
            seen.add(f)
        else:
            rels.append(0)

    dcg = sum(rels[i] / math.log2(i + 2) for i in range(len(top_k)))
    ideal_len = min(len(top_k), len(relevant_files)) if k is None else min(k, len(relevant_files))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, 1 + ideal_len))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_files: List[str],
    relevant_files: Set[str],
    k_values: List[int] | None = None,
) -> RetrievalMetrics:
    """
    一站式计算所有检索指标。

    路径归一化：如果 relevant_files 是短文件名（不含路径分隔符），
    则自动从 retrieved_files 中提取 basename 进行比较。
    支持两种匹配方式：
        - 短文件名匹配："BM25.md" 匹配 ".../BM25.md" 或 "...\\BM25.md"
        - 全路径匹配：直接字符串相等

    Args:
        retrieved_files: 检索返回的文件路径列表
        relevant_files: 金标准相关文件集合
        k_values: 要计算的 K 值列表，默认 [3, 5, 10]

    Returns:
        RetrievalMetrics
    """
    if k_values is None:
        k_values = [3, 5, 10]

    # 路径归一化：如果 relevant_files 是短文件名，提取 basename 匹配
    if relevant_files:
        sample = next(iter(relevant_files))
        if not any(sep in sample for sep in ("/", "\\")):
            retrieved_files = [Path(f).name for f in retrieved_files]

    mrr_score = mrr(retrieved_files, relevant_files)
    recall = {k: recall_at_k(retrieved_files, relevant_files, k) for k in k_values}
    precision = {k: precision_at_k(retrieved_files, relevant_files, k) for k in k_values}
    ndcg = {k: ndcg_at_k(retrieved_files, relevant_files, k) for k in k_values}

    return RetrievalMetrics(
        mrr=mrr_score,
        recall_at_k=recall,
        precision_at_k=precision,
        ndcg_at_k=ndcg,
    )
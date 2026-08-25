"""
生成评测指标：ROUGE-L、BLEU-1/4、语义相似度、忠诚度、答案相关性。

架构：
    词面匹配指标（无需 LLM）：
        candidate + reference → rouge_l() / bleu_n() / semantic_similarity()
    LLM 辅助指标（需要 LLM）：
        candidate + context → faithfulness()      # 答案是否忠于上下文
        candidate + question → answer_relevance()  # 答案是否直接回答问题
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable

import numpy as np


@dataclass(frozen=True)
class GenerationMetrics:
    """生成指标结果，不可变。

    字段：
        rouge_l: ROUGE-L F1 分数（基于 LCS）
        bleu_1: BLEU-1 unigram 精确度
        bleu_4: BLEU-4 4-gram 精确度
        semantic_similarity: 语义相似度（embedding 余弦或 Jaccard）
        faithfulness: 忠诚度（答案是否忠于上下文，可选）
        answer_relevance: 答案相关性（答案是否直接回答问题，可选）
        accuracy: 准确率（LLM-as-judge 对比金标准答案，0~1，可选）
        confidence: 评分置信度（Judge 对自身 grading 的把握，0~1，可选）
    """
    rouge_l: float
    bleu_1: float
    bleu_4: float
    semantic_similarity: float
    faithfulness: Optional[float] = field(default=None)
    answer_relevance: Optional[float] = field(default=None)
    accuracy: Optional[float] = field(default=None)
    confidence: Optional[float] = field(default=None)

    def to_dict(self) -> dict:
        """转为字典，用于 JSON 序列化。"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ──────────────────────────────────────────────────────────────
# 词面匹配指标（无需 LLM）
# ──────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """
    标准化文本：小写、去标点、保留中文、合并空格。

    为什么需要标准化？
    - ROUGE/BLEU 对大小写和标点敏感
    - 中英文混排时需要同时保留 Unicode 汉字和 ASCII 字符

    Args:
        text: 原始文本

    Returns:
        标准化后的文本
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def rouge_l(candidate: str, reference: str) -> float:
    """
    ROUGE-L：基于最长公共子序列（LCS）的 F1 分数。

    公式：
        L = LCS(candidate, reference)
        P = L / len(candidate)
        R = L / len(reference)
        F1 = 2 * P * R / (P + R)，调和平均

    边界情况：
    - candidate 和 reference 都为空 → 返回 1.0（完全一致）
    - candidate 为空但 reference 非空，或反之 → 返回 0.0

    Args:
        candidate: 模型生成的答案
        reference: 金标准答案

    Returns:
        0.0 ~ 1.0
    """
    if not candidate and not reference:
        return 1

    if not candidate or not reference:
        return 0

    l = _longest_common_subsequence(candidate, reference)
    p = l / len(candidate)
    r = l / len(reference)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _longest_common_subsequence(s1: str, s2: str) -> int:
    """返回两个字符串的最长公共子序列长度。标准 DP 解法，O(mn) 时间 / O(mn) 空间。"""
    m, n = len(s1), len(s2)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def bleu_n(candidate: str, reference: str, n: int = 4) -> float:
    """
    BLEU-N：N-gram 精确度（带 brevity penalty）。

    核心思想：
    - 截断计数（clipped count）：防止模型通过重复刷分
    - Brevity Penalty：短答案受惩罚

    Args:
        candidate: 模型生成的答案
        reference: 金标准答案
        n: N-gram 阶数（1, 2, 3, 4）

    Returns:
        0.0 ~ 1.0
    """
    if not candidate:
        return 0.0

    cand_tokens = _normalize_text(candidate).split()
    ref_tokens = _normalize_text(reference).split()

    if not cand_tokens:
        return 0.0

    from collections import Counter
    cand_ngrams = Counter(tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))

    if not cand_ngrams:
        return 0.0

    clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
    precision = clipped / len(cand_ngrams)

    # Brevity Penalty
    if len(cand_tokens) > len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))

    return bp * precision


def bleu_1(candidate: str, reference: str) -> float:
    """BLEU-1：unigram 精确度。"""
    return bleu_n(candidate, reference, n=1)


def bleu_4(candidate: str, reference: str) -> float:
    """BLEU-4：4-gram 精确度（最常用）。"""
    return bleu_n(candidate, reference, n=4)


def semantic_similarity(candidate: str, reference: str, embedder=None) -> float:
    """
    语义相似度：用 sentence embedding 的余弦相似度。

    - 有 embedder 时：用 embedding 余弦
    - 无 embedder 时：回退到 Jaccard 词重叠

    Args:
        candidate: 模型生成的答案
        reference: 金标准答案
        embedder: 可选的 embedding 函数，接受 str → List[float]

    Returns:
        0.0 ~ 1.0
    """
    if not candidate and not reference:
        return 1.0

    if embedder is not None:
        emb_c = np.array(embedder(candidate), dtype=np.float64)
        emb_r = np.array(embedder(reference), dtype=np.float64)
        norm_c = np.linalg.norm(emb_c)
        norm_r = np.linalg.norm(emb_r)
        if norm_c == 0 or norm_r == 0:
            return 0.0
        return float(np.dot(emb_c, emb_r) / (norm_c * norm_r))

    # 回退：Jaccard 词重叠
    cand_words = set(_normalize_text(candidate).split())
    ref_words = set(_normalize_text(reference).split())
    if not cand_words or not ref_words:
        return 0.0
    return len(cand_words & ref_words) / len(cand_words | ref_words)


# ──────────────────────────────────────────────────────────────
# LLM 辅助指标（需要 LLM）
# ──────────────────────────────────────────────────────────────


def faithfulness(candidate: str, context: str, llm=None) -> float:
    """
    忠诚度：答案中的每个主张是否有上下文依据？

    实现思路：将答案拆分为若干事实主张（claims），逐一检查每个主张能否从上下文中找到依据。
    最终得分为：Faithfulness = 有依据的主张数 / 总主张数

    Args:
        candidate: 模型生成的答案
        context: 检索到的上下文（所有 chunk 拼接）
        llm: 可选的 LLM 实例，需要有 invoke(messages) 方法

    Returns:
        0.0 ~ 1.0，越高越忠于上下文
    """
    if not llm or not candidate or not context:
        return 0.0

    prompt = (
        "你是一个事实核查员。请检查以下答案中的每一条事实主张，"
        "判断它们是否能从给定的上下文中找到依据。\n\n"
        "上下文：\n{context}\n\n"
        "答案：\n{candidate}\n\n"
        "请按以下 JSON 格式输出：\n"
        "{{\n"
        '  "claims": ["主张1", "主张2", ...],\n'
        '  "verifications": [\n'
        '    {{"claim": "主张1", "supported": true/false, "reason": "..."}},\n'
        '    ...\n'
        "  ]\n"
        "}}\n\n"
        "注意：\n"
        "- 每个主张必须是独立的、可验证的事实陈述\n"
        '- "supported" 为 true 表示上下文中有直接依据\n'
        '- 如果上下文没有提到该主张，则 supported 为 false'
    ).format(context=context[:4000], candidate=candidate[:2000])

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import json as json_mod
        json_str = content.strip()
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json_mod.loads(json_str)
        verifications = result.get("verifications", [])
        if not verifications:
            return 0.0
        supported_count = sum(1 for v in verifications if v.get("supported"))
        return supported_count / len(verifications)
    except Exception:
        return 0.0


def answer_relevance(candidate: str, question: str, llm=None) -> float:
    """
    答案相关性：答案是否直接回答了问题？

    实现思路：让 LLM 对 candidate 和 question 的相关性在 0~1 范围内打分，
    配合评分标准提高稳定性。

    Args:
        candidate: 模型生成的答案
        question: 用户原始问题
        llm: 可选的 LLM 实例

    Returns:
        0.0 ~ 1.0，越高越相关
    """
    if not llm or not candidate or not question:
        return 0.0

    prompt = (
        "你是一个评分员。请判断以下答案是否直接回答了用户的问题。\n\n"
        "问题：{question}\n\n"
        "答案：{candidate}\n\n"
        "请按以下 JSON 格式输出评分：\n"
        "{{\n"
        '  "score": 0.0~1.0 之间的浮点数,\n'
        '  "reason": "简短理由"\n'
        "}}\n\n"
        "评分标准：\n"
        "- 1.0: 答案直接、完整地回答了问题\n"
        "- 0.8: 答案回答了问题但不够完整\n"
        "- 0.5: 答案部分相关但未直接回答\n"
        "- 0.0: 答案完全答非所问\n\n"
        "注意：只输出 JSON，不要输出其他内容。"
    ).format(question=question[:1000], candidate=candidate[:2000])

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import json as json_mod
        json_str = content.strip()
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json_mod.loads(json_str)
        score = float(result.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def compute_answer_accuracy(
    candidate: str,
    reference: str,
    question: str = "",
    llm=None,
) -> tuple[float, float]:
    """
    准确率 + 评分置信度（LLM-as-judge，一次调用同时产出两个值）。

    - accuracy（0~1）：待评测答案相对「金标准答案」在**事实正确性 + 内容覆盖度**
      上的一致程度（1.0 = 事实全对且覆盖全部关键信息点；0.0 = 严重不符/答非所问）。
    - confidence（0~1）：Judge 对自己这条评分的把握（信息充分、判断明确时高；
      答案含糊、难以判断时低）。即评测框架所需的「置信度」维度。

    合并为一次 LLM 调用，避免 accuracy / confidence 各占一次请求、拖慢评测。

    Args:
        candidate: 模型生成的答案
        reference: 金标准答案（用于对比）
        question: 用户原始问题（仅作参考，不计入评分）
        llm: 可选的 LLM 实例（需有 invoke(messages)）

    Returns:
        (accuracy, confidence)，失败时返回 (0.0, 0.0)
    """
    if not llm or not candidate or not reference:
        return 0.0, 0.0

    prompt = (
        "你是一个严格的评测员。我会给你一个「标准答案」和一个「待评测答案」，"
        "请评估待评测答案相对于标准答案的「准确率」。\n\n"
        "定义（accuracy）：衡量待评测答案在**事实正确性与内容覆盖度**上，"
        "与标准答案的一致程度：\n"
        "- 1.0：事实完全正确，且覆盖了标准答案的所有关键信息点\n"
        "- 0.7：大部分关键信息正确，但有少量遗漏或轻微不精确\n"
        "- 0.4：仅部分正确，有明显事实偏差或重大遗漏\n"
        "- 0.0：与标准答案严重不符或答非所问\n\n"
        "同时，请给出你对自己评分的「置信度」（confidence，0~1）："
        "信息充分、判断明确时为高置信度；答案含糊、难以判断时为低置信度。\n\n"
        "标准答案：\n{reference}\n\n"
        "待评测答案：\n{candidate}\n\n"
        "用户问题（仅供参考，不计入评分）：\n{question}\n\n"
        "只输出 JSON，不要输出其他内容：\n"
        "{{\n"
        '  "accuracy": 0.0~1.0 的浮点数,\n'
        '  "confidence": 0.0~1.0 的浮点数,\n'
        '  "reason": "简短理由"\n'
        "}}"
    ).format(
        reference=reference[:2500],
        candidate=candidate[:2500],
        question=question[:1000],
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import json as json_mod
        json_str = content.strip()
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()

        result = json_mod.loads(json_str)
        accuracy = max(0.0, min(1.0, float(result.get("accuracy", 0.0))))
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        return accuracy, confidence
    except Exception:
        return 0.0, 0.0


# ──────────────────────────────────────────────────────────────
# 一站式入口
# ──────────────────────────────────────────────────────────────


def compute_generation_metrics(
    candidate: str,
    reference: str,
    embedder=None,
    llm=None,
    context: str = "",
    question: str = "",
) -> GenerationMetrics:
    """
    一站式计算所有生成指标。

    - 始终计算：    rouge_l / bleu_1 / bleu_4 / semantic_similarity
- 有条件计算（需 llm + context）：faithfulness
- 有条件计算（需 llm + question）：answer_relevance
- 有条件计算（需 llm + reference）：accuracy + confidence（一次 LLM 调用）
- 缺少条件时，对应指标设为 None

Args:
    candidate: 模型生成的答案
    reference: 金标准答案（用于 ROUGE/BLEU/语义相似度/accuracy）
    embedder: 可选的 embedding 函数
    llm: 可选的 LLM 实例（用于 faithfulness + answer_relevance + accuracy/confidence）
    context: 检索到的上下文（用于 faithfulness）
    question: 用户问题（用于 answer_relevance）

Returns:
    GenerationMetrics
"""
    rouge_l_score = rouge_l(candidate, reference)
    bleu_1_score = bleu_1(candidate, reference)
    bleu_4_score = bleu_4(candidate, reference)
    sim_score = semantic_similarity(candidate, reference, embedder)

    faith = None
    if llm and context:
        faith = faithfulness(candidate, context, llm)

    rel = None
    if llm and question:
        rel = answer_relevance(candidate, question, llm)

    acc = None
    conf = None
    if llm and reference:
        acc, conf = compute_answer_accuracy(candidate, reference, question, llm)

    return GenerationMetrics(
        rouge_l=rouge_l_score,
        bleu_1=bleu_1_score,
        bleu_4=bleu_4_score,
        semantic_similarity=sim_score,
        faithfulness=faith,
        answer_relevance=rel,
        accuracy=acc,
        confidence=conf,
    )
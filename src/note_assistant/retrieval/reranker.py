"""
Reranker 重排模块 —— 对混合检索的 top-k 候选做交叉编码器重排
"""

import logging
import time
from typing import List
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult
from transformers import PreTrainedTokenizerBase
from FlagEmbedding import FlagReranker
from functools import lru_cache

logger = logging.getLogger(__name__)

# ─── 兼容层：FlagEmbedding 1.4.0 依赖 prepare_for_model（transformers < 4.34）───
# transformers >= 4.34 移除了此方法，加回去让 FlagEmbedding 正常工作
if not hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
    def _prepare_for_model(self, text, text_pair=None, **kwargs):
        """
        兼容旧 API：
        - 如果 text 是 dict（已编码），直接返回
        - 如果 text 是 list[int]（token IDs），解码后重新编码
        - 如果 text 是 str，委托给 __call__
        """
        if isinstance(text, dict):
            return text
        if isinstance(text, list) and text and isinstance(text[0], int):
            # 已编码的 token IDs → 解码为文本再编码
            text = self.decode(text, skip_special_tokens=True)
            if text_pair and isinstance(text_pair, list) and isinstance(text_pair[0], int):
                text_pair = self.decode(text_pair, skip_special_tokens=True)
        return self(text, text_pair=text_pair, **kwargs)
    PreTrainedTokenizerBase.prepare_for_model = _prepare_for_model


@lru_cache(maxsize=None)
def get_reranker(model_path: str | None = None, use_fp16: bool = True) -> "LocalReranker":
    """LocalReranker 懒加载工厂：@lru_cache 保证全局单例，防重复加载 1.1GB 模型。

    Args:
        model_path: 模型路径，默认读 settings.reranker_model
        use_fp16: 是否半精度推理

    Returns:
        LocalReranker 单例
    """
    return LocalReranker(model_path=model_path, use_fp16=use_fp16)


class LocalReranker:
    """
    本地 Reranker：使用 FlagReranker 对候选文档做精排

    架构：
        混合检索返回 top-20
            ↓
        Reranker 交叉编码（query + doc 同时输入 Transformer）
            ↓
        按重排分数取 top-5
            ↓
        送入 LLM 生成回答
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_fp16: bool = True,
    ):
        """
        Args:
            model_path: reranker 模型路径，默认读 config.reranker_model
            use_fp16: 是否用半精度推理（省显存，速度快）
        """
        self.model_path = model_path or settings.reranker_model
        self.use_fp16 = use_fp16
        self.model = FlagReranker(
            model_name_or_path=self.model_path,
            use_fp16=self.use_fp16
        )

    # ──────────────────────────────────────────────
    # 推理
    # ──────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """
        对混合检索返回的 RetrievalResult 做重排。

        Args:
            query: 用户查询
            results: HybridRetriever.search() 的返回值（保留 metadata 和 breakdown）
            top_k: 返回多少个结果，默认读 config.top_k_rerank

        Returns:
            [RetrievalResult, ...]，按 rerank score 降序排列
            每个 result 的 score 被更新为 rerank 分数，metadata/breakdown 保留原值
        """
        top_k = top_k if top_k is not None else settings.top_k_rerank
        _t0 = time.perf_counter()
        logger.info("rerank.start", extra={"candidates": len(results), "top_k": top_k})

        # 1. 构造 pairs（取 page_content 做交叉编码）
        documents = [r.page_content for r in results]
        pairs = [[query, doc] for doc in documents]
        # 2. batch 推理
        scores = self.model.compute_score(pairs)
        # 3. 排序取 top_k，更新 score 并保留原始 metadata
        top_k_idx = sorted(range(len(scores)),
                           key=lambda k: scores[k],
                           reverse=True)[:top_k]
        batch_ms = (time.perf_counter() - _t0) * 1000

        reranked = []
        for idx in top_k_idx:
            r = results[idx]
            reranked.append(RetrievalResult(
                score=float(scores[idx]),
                page_content=r.page_content,
                metadata=r.metadata,
                dense_score=r.dense_score,
                sparse_score=r.sparse_score,
            ))
        elapsed = (time.perf_counter() - _t0) * 1000
        logger.info(
            "rerank.done", extra={"results": len(reranked), "top_k": top_k, "elapsed_ms": round(elapsed)}
        )
        return reranked

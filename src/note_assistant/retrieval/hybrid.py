# src/note_assistant/retrieval/hybrid.py
"""
混合检索器：dense（ChromaDB 向量）+ sparse（BM25 稀疏）加权融合
"""

import logging
import time
from typing import List

from note_assistant.config import settings
from note_assistant.indexing.embedder import OllamaEmbedder
from note_assistant.indexing.ingestor import Ingestor
from note_assistant.retrieval.sparse_retriever import BM25Retriever
from note_assistant.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器：两路并行检索 → 归一化 → 加权融合

    架构：
        用户 query
            ├─→ embedding → ChromaDB（dense） → cosine 距离
            ├─→ 分词 → BM25Okapi（sparse）    → BM25 分数
            ↓
        归一化到 [0, 1]
            ↓
        加权融合：final = α × dense_sim + (1-α) × sparse_norm
            ↓
        排序 → top_k → 返回
    """

    def __init__(
        self,
        alpha: float | None = None,
        top_k: int | None = None,
    ):
        """
        Args:
            alpha: dense 权重，默认读 config.bm25_weight（注意：config 里 bm25_weight 是 sparse 权重）
                   alpha = 1 - bm25_weight，即 dense 权重
            top_k: 返回多少个结果
        """
        self.alpha = alpha if alpha is not None else settings.dense_weight
        self.top_k = top_k if top_k is not None else settings.top_k_retrieve

        self.embedder = OllamaEmbedder()
        self.ingestor = Ingestor()
        self.bm25 = BM25Retriever()

    # ──────────────────────────────────────────────
    # Dense 检索（ChromaDB）
    # ──────────────────────────────────────────────

    def _dense_search(self, query_embedding: List[float], n_results: int = 50) -> List[RetrievalResult]:
        """
        向量检索：ChromaDB query → cosine 距离 → 转 similarity

        ChromaDB 返回的是 cosine 距离（越小越近），需要转成 similarity（越大越近）。
        """
        query_results = self.ingestor.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return self._results_from_chroma(query_results)

    def _results_from_chroma(self, query_results: dict) -> List[RetrievalResult]:
        """把 ChromaDB query 输出（含 distances）转为归一化 similarity 的 RetrievalResult 列表。"""
        if not query_results.get("distances") or not query_results["distances"][0]:
            return []
        distances = query_results["distances"][0]
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0]
        min_d, max_d = min(distances), max(distances)
        rng = max_d - min_d if max_d != min_d else 1.0
        results = []
        for i in range(len(distances)):
            sim = 1 - (distances[i] - min_d) / rng
            results.append(RetrievalResult(
                score=sim,
                page_content=docs[i],
                metadata=metas[i] or {},
                dense_score=sim,
            ))
        return results

    # ──────────────────────────────────────────────
    # 公开子检索方法（供 agent 工具集单独调用）
    # ──────────────────────────────────────────────

    def _safe_n(self, top_k: int) -> int:
        """clamp n_results 不超过集合规模，避免 ChromaDB 报错。"""
        try:
            count = self.ingestor.collection.count()
        except Exception:
            count = 1
        return max(1, min(top_k, max(1, count)))

    @staticmethod
    def _build_where(
        filepath: str | None = None,
        heading: str | None = None,
        tag: str | None = None,
    ) -> dict | None:
        """构造 ChromaDB `where` 过滤条件（按元数据过滤）。"""
        conds = []
        if filepath:
            conds.append({"filepath": filepath})
        if heading:
            conds.append({"heading_path": {"$contains": heading}})
        if tag:
            conds.append({"tags": {"$contains": tag}})
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {"$and": conds}

    def vector_search(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """仅语义向量检索（ChromaDB dense）。"""
        top_k = top_k or self.top_k
        qe = self.embedder.embed_one(query)
        qr = self.ingestor.collection.query(
            query_embeddings=[qe],
            n_results=self._safe_n(top_k),
            include=["documents", "metadatas", "distances"],
        )
        return self._results_from_chroma(qr)

    def bm25_search(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """仅关键词（BM25 稀疏）检索。"""
        return self.bm25.search(query, top_k=top_k or self.top_k)

    def filtered_search(
        self,
        query: str,
        filepath: str | None = None,
        heading: str | None = None,
        tag: str | None = None,
        top_k: int | None = None,
    ) -> List[RetrievalResult]:
        """按元数据过滤（filepath / heading / tag）后再做向量检索。"""
        top_k = top_k or self.top_k
        where = self._build_where(filepath, heading, tag)
        qe = self.embedder.embed_one(query)
        qr = self.ingestor.collection.query(
            query_embeddings=[qe],
            n_results=self._safe_n(top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._results_from_chroma(qr)

    # ──────────────────────────────────────────────
    # Sparse 检索（BM25）
    # ──────────────────────────────────────────────

    def _sparse_search(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """
        稀疏检索：BM25Okapi 算分数
        """
        return self.bm25.search(query, top_k=top_k)

    # ──────────────────────────────────────────────
    # 融合：归一化 + 加权
    # ──────────────────────────────────────────────

    def _merge_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        两路结果融合：sparse 归一化 → 加权求和

        dense 已在 [0,1]（_dense_search 归一化过），sparse 需要 min-max 归一化。
        用 page_content 做 key 匹配（两路同源，文本相同）。
        并集融合：保留所有候选。
        """
        dense_by_content = {r.page_content: r for r in dense_results}
        sparse_by_content = {r.page_content: r for r in sparse_results}

        # 归一化 sparse 分数
        sparse_scores = [r.score for r in sparse_results]
        if sparse_scores:
            min_s, max_s = min(sparse_scores), max(sparse_scores)
            s_rng = max_s - min_s if max_s != min_s else 1.0
        else:
            min_s, s_rng = 0.0, 1.0

        # 并集融合
        all_contents = dense_by_content.keys() | sparse_by_content.keys()
        merged = []
        for content in all_contents:
            dense_score = dense_by_content[content].score if content in dense_by_content else 0.0
            raw_sparse = sparse_by_content[content].score if content in sparse_by_content else 0.0
            sparse_norm = (raw_sparse - min_s) / s_rng

            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_norm

            metadata = (
                dense_by_content[content].metadata
                if content in dense_by_content
                else sparse_by_content[content].metadata
            )
            merged.append(RetrievalResult(
                score=final_score,
                page_content=content,
                metadata=metadata,
                dense_score=dense_score,
                sparse_score=sparse_norm,
            ))

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """
        混合检索主入口。

        Args:
            query: 用户查询（可能已被 QueryRewriter 改写）
            top_k: 返回多少个结果
        Returns:
            [RetrievalResult, ...]，按 final_score 降序排列
        """
        _t0 = time.perf_counter()
        top_k = top_k if top_k is not None else self.top_k
        logger.info("hybrid_search.start", extra={"query_preview": query[:50], "top_k": top_k})

        # 1. embedding
        qe_t0 = time.perf_counter()
        query_embedding = self.embedder.embed_one(query)
        embed_ms = (time.perf_counter() - qe_t0) * 1000

        # 2. 两路并行检索（各取 top-50，融合后截断）
        dense_t0 = time.perf_counter()
        dense_results = self._dense_search(query_embedding, n_results=50)
        dense_ms = (time.perf_counter() - dense_t0) * 1000

        sparse_t0 = time.perf_counter()
        sparse_results = self._sparse_search(query, top_k=50)
        sparse_ms = (time.perf_counter() - sparse_t0) * 1000

        # 3. 融合
        merge_t0 = time.perf_counter()
        merged = self._merge_results(dense_results, sparse_results)
        merge_ms = (time.perf_counter() - merge_t0) * 1000

        out = merged[:top_k]
        elapsed = (time.perf_counter() - _t0) * 1000
        logger.info(
            "hybrid_search.done",
            extra={
                "query_preview": query[:50],
                "top_k": top_k,
                "results": len(out),
                "alpha": self.alpha,
                "embed_ms": round(embed_ms),
                "dense_ms": round(dense_ms),
                "sparse_ms": round(sparse_ms),
                "merge_ms": round(merge_ms),
                "elapsed_ms": round(elapsed),
            },
        )
        return out

    # ──────────────────────────────────────────────
    # 跟踪检索（逐步骤输出）
    # ──────────────────────────────────────────────

    def search_with_trace(self, query: str, top_k: int | None = None) -> tuple[list[RetrievalResult], list[dict]]:
        """
        混合检索 + 逐步骤跟踪信息。

        Returns:
            (merged_results, trace_steps)
            trace_steps: [{"step": "embedding", "ms": 150},
                          {"step": "dense_retrieval", "results": 50, "ms": 200}, ...]
        """
        top_k = top_k if top_k is not None else self.top_k
        trace = []

        # 1. Embedding
        t0 = time.time()
        query_embedding = self.embedder.embed_one(query)
        trace.append({"step": "embedding", "ms": int((time.time() - t0) * 1000)})

        # 2. Dense 检索
        t0 = time.time()
        dense_results = self._dense_search(query_embedding, n_results=50)
        trace.append({"step": "dense_retrieval", "results": len(dense_results), "ms": int((time.time() - t0) * 1000)})

        # 3. Sparse 检索
        t0 = time.time()
        sparse_results = self._sparse_search(query, top_k=50)
        trace.append({"step": "sparse_retrieval", "results": len(sparse_results), "ms": int((time.time() - t0) * 1000)})

        # 4. 融合
        t0 = time.time()
        merged = self._merge_results(dense_results, sparse_results)
        trace.append({"step": "hybrid_fusion", "results": len(merged), "ms": int((time.time() - t0) * 1000)})

        return merged[:top_k], trace

    # ──────────────────────────────────────────────
    # 便捷方法：从 ChromaDB 建 BM25 索引
    # ──────────────────────────────────────────────

    def build_bm25_from_chroma(self) -> None:
        """
        从 ChromaDB 全量拉取 chunks 建 BM25 索引。

        这个方法把 indexing 和 retrieval 串起来：
        Ingestor 写 ChromaDB → 这里从 ChromaDB 读 → 建 BM25 → 两路检索可用
        """
        self.bm25 = BM25Retriever.from_chroma()
        self.bm25.save()

# src/note_assistant/retrieval/hybrid.py
"""
混合检索器：dense（ChromaDB 向量）+ sparse（BM25 稀疏）加权融合
"""

from typing import List

from note_assistant.config import settings
from note_assistant.indexing.embedder import OllamaEmbedder
from note_assistant.indexing.ingestor import Ingestor
from note_assistant.retrieval.sparse_retriever import BM25Retriever
from note_assistant.retrieval.types import RetrievalResult


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

        # cosine distance → similarity（越大越好）
        distances = query_results["distances"][0]
        min_d, max_d = min(distances), max(distances)
        rng = max_d - min_d if max_d != min_d else 1.0

        results = []
        for i in range(len(distances)):
            sim = 1 - (distances[i] - min_d) / rng
            results.append(RetrievalResult(
                score=sim,
                page_content=query_results["documents"][0][i],
                metadata=query_results["metadatas"][0][i],
                dense_score=sim,
            ))

        return results

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
        top_k = top_k if top_k is not None else self.top_k

        # 1. embedding
        query_embedding = self.embedder.embed_one(query)

        # 2. 两路并行检索（各取 top-50，融合后截断）
        dense_results = self._dense_search(query_embedding, n_results=50)
        sparse_results = self._sparse_search(query, top_k=50)

        # 3. 融合
        merged = self._merge_results(dense_results, sparse_results)

        # 4. 截断
        return merged[:top_k]

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

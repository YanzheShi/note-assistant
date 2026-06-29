"""
BM25 稀疏检索器

基于 rank_bm25 实现关键词检索，与 ChromaDB 向量检索互补。
中文分词使用 jieba。
"""

import pickle
from pathlib import Path
from typing import List

import jieba
from rank_bm25 import BM25Okapi

from note_assistant.config import settings
from note_assistant.indexing.types import Chunk
from note_assistant.retrieval.types import RetrievalResult


class BM25Retriever:
    """
    BM25 稀疏检索器

    使用 pickle 持久化索引，避免内存常驻。
    中文分词用 jieba（比逐字符更精准，"评测指标" → ['评测', '指标']）。
    """

    def __init__(self, index_path: str | Path | None = None):
        """
        Args:
            index_path: pickle 文件路径，默认读 config.bm25_index_path
        """
        self.index_path = Path(index_path) if index_path else settings.bm25_index_path
        self.corpus: List[Chunk] = []            # 原始 chunk 列表
        self.tokenized: List[List[str]] = []     # 分词后的 token 列表
        self.bm25: BM25Okapi | None = None       # BM25 索引对象

    # ──────────────────────────────────────────────
    # 分词（jieba）
    # ──────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        中文分词：将文本切分为 token 列表。
        jieba 分词比逐字符更精准，"评测指标" → ['评测', '指标']。
        """
        return jieba.lcut(text)

    # ──────────────────────────────────────────────
    # 建索引
    # ──────────────────────────────────────────────

    def build_index(self, corpus: List[Chunk]) -> None:
        """
        从 chunk 列表建 BM25 索引。

        Args:
            corpus: Chunk 对象列表（来自 Ingestor 的 chunks）
        """
        self.corpus = corpus
        self.tokenized = [self.tokenize(chunk.page_content) for chunk in corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    # ──────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 50) -> List[RetrievalResult]:
        """
        检索 query，返回 top-k 个最相关的 chunk。

        Args:
            query: 用户查询（可能已被 QueryRewriter 改写）
            top_k: 返回多少个结果

        Returns:
            [RetrievalResult, ...]，按 score 降序排列
        """
        if self.bm25 is None:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        result = []
        for idx in top_k_indices:
            chunk = self.corpus[idx]
            result.append(RetrievalResult(
                score=float(scores[idx]),
                page_content=chunk.page_content,
                metadata=chunk.metadata,
                index=idx,
            ))

        return result
    # ──────────────────────────────────────────────
    # 持久化（pickle）
    # ──────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        """
        将索引保存到 pickle 文件。

        保存内容：corpus + tokenized（BM25Okapi 可从 tokenized 重建）
        为什么不存 self.bm25？因为 BM25Okapi 只需 tokenized 就能重建，
        存 corpus + tokenized 更紧凑，也方便调试时查看原始数据。
        """
        save_path = Path(path) if path else self.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": self.corpus,
            "tokenized": self.tokenized,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path | None = None) -> None:
        """
        从 pickle 文件加载索引。

        加载后重建 self.bm25（因为只存了 tokenized，没存 BM25Okapi 对象）。
        """
        load_path = Path(path) if path else self.index_path

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.corpus = data["corpus"]
        self.tokenized = data["tokenized"]
        self.bm25 = BM25Okapi(self.tokenized)

    # ──────────────────────────────────────────────
    # 便捷方法：从 ChromaDB 全量加载建索引
    # ──────────────────────────────────────────────

    @classmethod
    def from_chroma(cls, collection_name: str | None = None) -> "BM25Retriever":
        """
        从 ChromaDB collection 全量拉取 chunks 建 BM25 索引。

        这个方法把 indexing 和 retrieval 串起来：
        Ingestor 写 ChromaDB → 这里从 ChromaDB 读 → 建 BM25 → 两路检索
        """
        from note_assistant.indexing.ingestor import Ingestor

        ingestor = Ingestor()
        collection = ingestor.collection

        # 从 ChromaDB 拿所有文档
        result = collection.get(
            include=["documents", "metadatas"]
        )

        corpus = [
            Chunk(page_content=doc, metadata=meta)
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

        retriever = cls()
        retriever.build_index(corpus)
        return retriever

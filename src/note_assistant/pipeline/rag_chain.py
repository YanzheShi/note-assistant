# src/note_assistant/pipeline/rag_chain.py
"""
RAG 完整管线：检索 -> 图扩展 -> Rerank -> 生成。

架构：
    用户问题
        ->
    HybridRetriever.search(top_k=20) -> 候选
        ->
    LocalReranker.rerank(top_k=5) -> 精选
        ->
    WikiGraph.expand(hit_files) -> 关联笔记
        ->
    组装 context（精选 + 关联）
        ->
    Generator.generate() -> 最终回答
"""
import logging
from dataclasses import dataclass, asdict
from typing import List

from note_assistant.retrieval.types import RetrievalResult


@dataclass
class SourceInfo:
    """单个来源片段的信息。"""
    type: str                       # "direct" | "graph"
    filepath: str
    heading: str
    preview: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AskResponse:
    """RAGChain.ask() 的返回值。"""
    answer: str
    sources: List[SourceInfo]
    graph_expansion: int = 0
    retrieved: int = 0

    def to_dict(self) -> dict:
        """转为字典，用于 JSON 序列化。"""
        d = asdict(self)
        d["sources"] = [s.to_dict() for s in self.sources]
        return d


class RAGChain:
    """RAG 管线编排器。"""

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        graph=None,
        generator=None,
    ):
        """
        Args:
            hybrid_retriever: HybridRetriever 实例
            reranker: LocalReranker 实例
            graph: WikiGraph 实例（可选，不传则跳过图扩展）
            generator: Generator 实例（可选，不传则只返回 context）
        """
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.graph = graph
        self.generator = generator

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        """
        【核心逻辑待实现】完整管线：检索 -> rerank -> 图扩展 -> 生成。

        流程：
        1. 混合检索 top-20（dense + sparse 融合）
        2. Rerank -> top-5（交叉编码精排）
        3. 图扩展：从命中文档出发找关联（BFS 1-hop）
        4. 组装 context：直接命中 + 关联扩展
        5. LLM 生成最终回答

        Args:
            question: 用户问题
            top_k: 最终返回几个结果（默认 config.top_k_rerank）

        Returns:
            AskResponse(answer=..., sources=[SourceInfo(...), ...], ...)
        """
        hybrid_result = self.retriever.search(question, top_k=top_k)
        rerank_result = self.reranker.rerank(question, hybrid_result, top_k=top_k)

        hit_files_paths = [doc.metadata["filepath"] for doc in rerank_result]

        graph_expand_chunks = []
        if self.graph and hit_files_paths:
            expand_file_paths = self.graph.expand(hit_files_paths)
            graph_expand_chunks = self._fetch_neighbor_chunks(expand_file_paths)

        merge_chunks = rerank_result + graph_expand_chunks

        answer = ""
        if self.generator:
            answer = self.generator.generate(question, merge_chunks)

        sources = []
        for r in rerank_result:
            sources.append(SourceInfo(
                type="direct",
                filepath=r.metadata.get("filepath", ""),
                heading=r.metadata.get("heading_path", ""),
                preview=r.page_content[:200],
                score=r.score,
            ))

        return AskResponse(
            answer=answer,
            sources=sources,
            graph_expansion=len(graph_expand_chunks),
            retrieved=len(hybrid_result),
        )

    # ------------------------------------------------------------------
    # 图扩展辅助
    # ------------------------------------------------------------------

    def _fetch_neighbor_chunks(
        self,
        neighbors: List[tuple[str, float]]
    ) -> List[RetrievalResult]:
        """
        【核心逻辑待实现】从邻居文件中获取 chunks 内容。

        策略：
        - stub 节点（"[[xxx]]"）跳过
        - 每个邻居最多取 2 个 chunk（取最前面的）
        - 返回 chunk 文本列表

        Args:
            neighbors: [(filepath, decay_score), ...]

        Returns:
            [RetrievalResult, ...]
        """
        chunks = []
        for filepath, score in neighbors:
            if filepath.startswith("[["):
                # stub 节点，跳过
                continue

            # 从 chromadb 获取文件的 chunks（用 get 不需要 embedding）
            try:
                results = self.retriever.ingestor.collection.get(
                    where={"filepath": filepath},
                    include=["documents", "metadatas"],
                )
                if results and results.get("documents"):
                    for doc, meta in zip(results["documents"], results["metadatas"]):
                        chunks.append(
                            RetrievalResult(
                                score=0.0,  # 图扩展没有 rerank score
                                page_content=doc,
                                metadata=meta,
                            )
                        )

            except Exception as e:
                logging.error(e)

        return chunks

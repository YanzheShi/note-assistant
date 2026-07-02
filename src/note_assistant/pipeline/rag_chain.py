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
import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import AsyncIterator, List

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
    # 流式入口
    # ------------------------------------------------------------------

    async def ask_stream(self, question: str, top_k: int | None = None) -> AsyncIterator[dict]:
        """
        真流式问答：检索（同步）→ 流式生成（逐 token）→ 推送 sources。

        流程：
        1. 同步检索 + rerank + 图扩展（和 ask() 一样）
        2. yield meta 事件（含检索耗时+图扩展数）
        3. 流式生成，逐 token yield
        4. 最后 yield sources 事件

        Yields:
            {"type": "meta", "retrieve_ms": int, "graph_expansion": int}
            {"type": "char", "content": str}
            {"type": "sources", "content": list[SourceInfo], "graph_expansion": int}

        面试考点：
        - 检索是同步的（~200ms），用 run_in_executor 包进 async context 也没必要
        - 真正的"流"在 LLM 生成阶段（几秒到十几秒）
        - sources 在检索完成后就已确定，但放最后推送避免干扰阅读
        """
        t0 = time.time()

        # ── 同步检索（放到线程池，避免阻塞 async generator 的事件循环） ──
        def _do_retrieval():
            hybrid_result = self.retriever.search(question, top_k=top_k)
            rerank_result = self.reranker.rerank(question, hybrid_result, top_k=top_k)
            hit_files_paths = [doc.metadata["filepath"] for doc in rerank_result]

            graph_expand_chunks = []
            if self.graph and hit_files_paths:
                expand_file_paths = self.graph.expand(hit_files_paths)
                graph_expand_chunks = self._fetch_neighbor_chunks(expand_file_paths)

            merge_chunks = rerank_result + graph_expand_chunks

            # 组装 sources
            sources = []
            for r in rerank_result:
                sources.append(SourceInfo(
                    type="direct",
                    filepath=r.metadata.get("filepath", ""),
                    heading=r.metadata.get("heading_path", ""),
                    preview=r.page_content[:200],
                    score=r.score,
                ))

            return merge_chunks, sources, len(graph_expand_chunks)

        merge_chunks, sources, graph_exp_count = await asyncio.to_thread(_do_retrieval)
        t1 = time.time()

        # ── yield meta ──
        yield {
            "type": "meta",
            "retrieve_ms": int((t1 - t0) * 1000),
            "graph_expansion": graph_exp_count,
        }

        # ── 流式生成 ──
        answer_text = ""
        if self.generator:
            try:
                async for token in self.generator.generate_stream(question, merge_chunks):
                    answer_text += token
                    yield {"type": "char", "content": token}
            except Exception as e:
                logging.error(f"流式生成失败: {e}")
                yield {"type": "char", "content": f"\n\n[生成中断: {e}]"}

        # ── 最后 yield sources ──
        yield {
            "type": "sources",
            "content": [s.to_dict() for s in sources],
            "graph_expansion": graph_exp_count,
        }

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

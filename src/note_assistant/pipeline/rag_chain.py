# src/note_assistant/pipeline/rag_chain.py
"""
RAG 完整管线：检索路由 -> 检索 -> 图扩展 -> Rerank -> 生成。

架构：
    用户问题
        ->
    检索路由（_needs_retrieval）判断是否需要查知识库
        -> 不需要 -> 直接生成（空 context，闲聊/通用回答）
        -> 需要 -> HybridRetriever.search(top_k=20) -> 候选
             -> LocalReranker.rerank(top_k=5) -> 精选
             -> WikiGraph.expand(hit_files) -> 关联笔记
             -> 组装 context（精选 + 关联）
             -> Generator.generate() -> 最终回答
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import AsyncIterator, List

from langchain_core.messages import HumanMessage
from note_assistant.llm.client import get_llm
from note_assistant.pipeline.source_kind import classify_source
from note_assistant.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


# ─── 路由结果解析工具函数 ───────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """清理 LLM 返回的可能包裹在 markdown 代码块中的 JSON 文本。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json 或 ``` 开头的代码块
        rest = text[3:].lstrip()
        # 如果是 ```json，去掉 json 标签
        if rest.lower().startswith("json"):
            rest = rest[4:].lstrip()
        # 取第一行到最后一个 ``` 之前的内容
        lines = rest.split("\n")
        cleaned_parts = []
        for line in lines:
            if line.strip().startswith("```"):
                break
            cleaned_parts.append(line)
        text = "\n".join(cleaned_parts).strip()
    # 兜底：去掉首尾多余的反引号
    text = text.strip("`").strip()
    return text


def _parse_routing_result(result) -> bool:
    """将 LLM 返回的路由结果转换为 bool。支持多种格式。"""
    if isinstance(result, dict):
        # 支持多种可能的键名变体
        return result.get("need_retrieval",
               result.get("needs_retrieval",
               result.get("needRetrieval",
               result.get("needsRetrieval", True))))
    if isinstance(result, bool):
        return result
    if isinstance(result, str):
        return result.lower() not in ("false", "no", "不需要", "否")
    return True  # 保守：解析不了就检索


@dataclass
class SourceInfo:
    """
    单个来源片段的信息。

    `origin` 与 `kind` 是两套正交语义，历史上被挤在同一个 `type` 字段里，
    导致 API 层把「来源渠道」当「内容类型」透传给前端，前端的 table/mermaid/image
    渲染分支永远进不去（详见 pipeline/source_kind.py 模块注释）：

        origin — 来源渠道：direct（检索直接命中） | graph（双链扩展带出）
        kind   — 内容类型：text | table | mermaid | image
    """
    origin: str                     # "direct" | "graph"
    filepath: str
    heading: str
    preview: str
    score: float
    kind: str = "text"              # "text" | "table" | "mermaid" | "image"
    # 渲染载荷：能抽就抽，不受 kind 约束（一个 chunk 可能同时含表格与图片）
    img_path: str = ""
    raw_table: str = ""
    raw_mermaid: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_result(cls, r: RetrievalResult, origin: str = "direct") -> "SourceInfo":
        """从检索结果组装来源信息，内容类型与渲染载荷交给 classify_source 判定。"""
        meta = getattr(r, "metadata", None) or {}
        rich = classify_source(r.page_content, meta)
        return cls(
            origin=origin,
            filepath=meta.get("filepath", ""),
            heading=meta.get("heading_path", ""),
            preview=r.page_content[:200],
            score=r.score,
            kind=rich["kind"],
            img_path=rich["img_path"],
            raw_table=rich["raw_table"],
            raw_mermaid=rich["raw_mermaid"],
        )


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

    # ─── 检索路由 prompt ───
    NEEDS_RETRIEVAL_PROMPT = (
        "你是一个助手，判断用户的问题是否需要查询个人知识库。\n\n"
        "判断规则：\n"
        "- 需要检索：问题涉及具体知识、概念解释、操作方法、项目细节、笔记内容等\n"
        "- 不需要检索：打招呼（你好/hi/在吗）、情感表达（谢谢/再见/辛苦了）、纯闲聊\n\n"
        "示例：\n"
        "- \"你好\" → false\n"
        "- \"hi\" → false\n"
        "- \"谢谢\" → false\n"
        "- \"如何进行上下文工程的评估？\" → true\n"
        "- \"RAG 是什么？\" → true\n\n"
        "请严格只返回如下 JSON 格式，不要包含 markdown 代码块标记，不要输出任何其他内容：\n"
        "{{\"need_retrieval\": false}}\n\n"
        "问题：{question}"
    )

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
    # 检索路由
    # ------------------------------------------------------------------

    def _needs_retrieval_sync(self, question: str) -> bool:
        """同步调用 LLM 判断是否需要检索（用于 ask 主入口）。"""
        try:
            prompt = self.NEEDS_RETRIEVAL_PROMPT.format(question=question)
            llm = get_llm(temperature=0, max_tokens=50)
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            text = text.strip()
            # 去掉可能的 markdown 代码块包裹（如 ```json ... ```）
            text = _clean_json_text(text)
            logging.debug(f"路由返回(sync): {text}")
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # 不是合法 JSON，直接当字符串处理（如 "不需要"/"否"）
                result = text
            result = _parse_routing_result(result)
            logger.info("routing.decision", extra={"needs_retrieval": result, "sync": True})
            return result
        except Exception as e:
            logging.warning(f"路由判断失败，默认检索: {e}")
            logging.error(f"路由 LLM 调用失败，已降级为默认检索，请检查路由服务: {e}")
            logger.warning("routing.failed", extra={"error": str(e)[:120]})
            return True

    async def _needs_retrieval_async(self, question: str) -> bool:
        """异步调用 LLM 判断是否需要检索（用于 ask_stream / ask_with_trace）。"""
        try:
            prompt = self.NEEDS_RETRIEVAL_PROMPT.format(question=question)
            llm = get_llm(temperature=0, max_tokens=50)
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            text = text.strip()
            # 去掉可能的 markdown 代码块包裹
            text = _clean_json_text(text)
            logging.debug(f"路由返回(async): {text}")
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # 不是合法 JSON，直接当字符串处理（如 "不需要"/"否"）
                result = text
            result = _parse_routing_result(result)
            logger.info("routing.decision", extra={"needs_retrieval": result, "sync": False})
            return result
        except Exception as e:
            logging.warning(f"路由判断失败，默认检索: {e}")
            logging.error(f"路由 LLM 调用失败，已降级为默认检索，请检查路由服务: {e}")
            logger.warning("routing.failed", extra={"error": str(e)[:120]})
            return True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None, history: list[dict] | None = None) -> AskResponse:
        """
        完整管线：检索路由 -> 检索 -> rerank -> 图扩展 -> 生成，支持多轮历史。

        流程：
        1. 检索路由：判断是否需要查知识库
        2. 需要检索：混合检索 top-20（dense + sparse 融合）
        3. Rerank -> top-5（交叉编码精排）
        4. 图扩展：从命中文档出发找关联（BFS 1-hop）
        5. 组装 context：直接命中 + 关联扩展
        6. LLM 生成最终回答（含历史对话）

        Args:
            question: 用户问题
            top_k: 最终返回几个结果（默认 config.top_k_rerank）
            history: 历史对话列表

        Returns:
            AskResponse(answer=..., sources=[SourceInfo(...), ...], ...)
        """
        # ── 检索路由 ──
        needs_retrieval = self._needs_retrieval_sync(question)

        if not needs_retrieval:
            # 不需要检索：直接返回友好回复，不走 LLM
            return AskResponse(answer="你好，我是知识库问答助手，可以帮你检索个人知识库内容，回答相关的问题", sources=[], graph_expansion=0, retrieved=0)

        # ── 需要检索：正常流程 ──
        hybrid_result = self.retriever.search(question, top_k=top_k)
        rerank_result = self.reranker.rerank(question, hybrid_result, top_k=top_k)

        hit_files_paths = [doc.metadata.get("filepath", "") for doc in rerank_result if doc.metadata.get("filepath")]

        graph_expand_chunks = []
        if self.graph and hit_files_paths:
            expand_file_paths = self.graph.expand(hit_files_paths)
            graph_expand_chunks = self._fetch_neighbor_chunks(expand_file_paths)

        merge_chunks = rerank_result + graph_expand_chunks

        answer = ""
        if self.generator:
            answer = self.generator.generate(question, merge_chunks, history=history)
            logger.info("rag_chain.generate", extra={"answer_len": len(answer)})

        sources = [SourceInfo.from_result(r, origin="direct") for r in rerank_result]

        return AskResponse(
            answer=answer,
            sources=sources,
            graph_expansion=len(graph_expand_chunks),
            retrieved=len(hybrid_result),
        )

    # ------------------------------------------------------------------
    # 流式入口
    # ------------------------------------------------------------------

    async def ask_stream(self, question: str, top_k: int | None = None, history: list[dict] | None = None) -> AsyncIterator[dict]:
        """
        真流式问答：检索（同步）→ 流式生成（逐 token）→ 推送 sources，支持多轮历史。

        流程：
        1. 同步检索 + rerank + 图扩展（和 ask() 一样）
        2. yield meta 事件（含检索耗时+图扩展数）
        3. 流式生成（含历史对话），逐 token yield
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
        # ── 检索路由 ──
        needs_retrieval = await self._needs_retrieval_async(question)

        if not needs_retrieval:
            yield {"type": "meta", "retrieve_ms": 0, "graph_expansion": 0, "routed": False}
            yield {"type": "char", "content": "你好！有什么笔记相关的问题可以帮你？"}
            yield {"type": "sources", "content": [], "graph_expansion": 0}
            return

        t0 = time.time()

        # ── 同步检索（放到线程池，避免阻塞 async generator 的事件循环） ──
        def _do_retrieval():
            hybrid_result = self.retriever.search(question, top_k=top_k)
            rerank_result = self.reranker.rerank(question, hybrid_result, top_k=top_k)
            hit_files_paths = [doc.metadata.get("filepath", "") for doc in rerank_result if doc.metadata.get("filepath")]

            graph_expand_chunks = []
            if self.graph and hit_files_paths:
                expand_file_paths = self.graph.expand(hit_files_paths)
                graph_expand_chunks = self._fetch_neighbor_chunks(expand_file_paths)

            merge_chunks = rerank_result + graph_expand_chunks

            # 组装 sources
            sources = [SourceInfo.from_result(r, origin="direct") for r in rerank_result]

            return merge_chunks, sources, len(graph_expand_chunks)

        merge_chunks, sources, graph_exp_count = await asyncio.to_thread(_do_retrieval)
        t1 = time.time()

        # ── yield meta ──
        yield {
            "type": "meta",
            "retrieve_ms": int((t1 - t0) * 1000),
            "graph_expansion": graph_exp_count,
            "routed": True,
        }

        # ── 流式生成（含历史） ──
        answer_text = ""
        if self.generator:
            try:
                async for token in self.generator.generate_stream(question, merge_chunks, history=history):
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
    # 跟踪检索入口（逐步骤输出）
    # ------------------------------------------------------------------

    async def ask_with_trace(self, question: str, top_k: int | None = None, history: list[dict] | None = None) -> AsyncIterator[dict]:
        """
        真·逐步骤检索：每完成一步立即 yield，前端实时展示，支持多轮历史。

        流程：
            1. Embedding       → yield trace + 得到向量
            2. 稠密检索         → yield trace + 得到结果
            3. 稀疏检索         → yield trace + 得到结果
            4. 融合排序         → yield trace + 得到结果
            5. Rerank          → yield trace + 得到结果
            6. 图扩展           → yield trace + 得到结果
            7. 流式生成（含历史） → yield char
            8. 推送 sources

        Yields:
            {"type": "trace", "step": "embedding", "ms": 150, "status": "done"}
            {"type": "trace", "step": "dense_retrieval", "ms": 200, "results": 50, "status": "done"}
            {"type": "trace", "step": "sparse_retrieval", ...}
            {"type": "trace", "step": "hybrid_fusion", ...}
            {"type": "trace", "step": "rerank", ...}
            {"type": "trace", "step": "graph_expansion", ...}
            {"type": "meta", ...}
            {"type": "char", ...}
            {"type": "sources", ...}

        面试考点：
        - 每个检索步骤单独 await asyncio.to_thread，确保事件循环不被阻塞
        - 前端可在收到 trace 事件后立即展示，实现"边检索边显示"
        """
        top_k = top_k or self.retriever.top_k
        t_global = time.time()

        # ── 检索路由 ──
        needs_retrieval = await self._needs_retrieval_async(question)

        if not needs_retrieval:
            yield {"type": "trace", "step": "routing", "ms": 0, "status": "skipped"}
            yield {"type": "meta", "retrieve_ms": 0, "graph_expansion": 0, "routed": False}
            yield {"type": "char", "content": "你好！有什么笔记相关的问题可以帮你？"}
            yield {"type": "sources", "content": [], "graph_expansion": 0}
            return

        # ── 1. Embedding ──
        t0 = time.time()
        query_embedding = await asyncio.to_thread(self.retriever.embedder.embed_one, question)
        yield {"type": "trace", "step": "embedding", "ms": int((time.time() - t0) * 1000), "status": "done"}

        # ── 2. 稠密检索 ──
        t0 = time.time()
        dense_results = await asyncio.to_thread(self.retriever._dense_search, query_embedding, 50)
        yield {
            "type": "trace", "step": "dense_retrieval", "ms": int((time.time() - t0) * 1000),
            "results": len(dense_results), "status": "done",
            "preview": dense_results[0].page_content[:100] if dense_results else "",
        }

        # ── 3. 稀疏检索 ──
        t0 = time.time()
        sparse_results = await asyncio.to_thread(self.retriever._sparse_search, question, 50)
        yield {
            "type": "trace", "step": "sparse_retrieval", "ms": int((time.time() - t0) * 1000),
            "results": len(sparse_results), "status": "done",
            "preview": sparse_results[0].page_content[:100] if sparse_results else "",
        }

        # ── 4. 融合排序 ──
        t0 = time.time()
        merged = await asyncio.to_thread(self.retriever._merge_results, dense_results, sparse_results)
        merged = merged[:top_k]
        yield {
            "type": "trace", "step": "hybrid_fusion", "ms": int((time.time() - t0) * 1000),
            "results": len(merged), "status": "done",
        }

        # ── 5. Rerank ──
        t0 = time.time()
        rerank_result = await asyncio.to_thread(self.reranker.rerank, question, merged, top_k)
        yield {
            "type": "trace", "step": "rerank", "ms": int((time.time() - t0) * 1000),
            "results": len(rerank_result), "status": "done",
            "preview": rerank_result[0].page_content[:100] if rerank_result else "",
        }

        # ── 6. 图扩展 ──
        hit_files_paths = [doc.metadata.get("filepath", "") for doc in rerank_result if doc.metadata.get("filepath")]
        graph_expand_chunks = []
        graph_exp_count = 0
        if self.graph and hit_files_paths:
            t0 = time.time()
            expand_file_paths = await asyncio.to_thread(self.graph.expand, hit_files_paths)
            graph_expand_chunks = await asyncio.to_thread(self._fetch_neighbor_chunks, expand_file_paths)
            graph_exp_count = len(graph_expand_chunks)
            yield {
                "type": "trace", "step": "graph_expansion", "ms": int((time.time() - t0) * 1000),
                "results": graph_exp_count, "status": "done",
            }

        merge_chunks = rerank_result + graph_expand_chunks
        t_retrieval = time.time()

        # ── 组装 sources ──
        sources = [SourceInfo.from_result(r, origin="direct") for r in rerank_result]

        # ── yield meta ──
        yield {
            "type": "meta",
            "retrieve_ms": int((t_retrieval - t_global) * 1000),
            "graph_expansion": graph_exp_count,
            "routed": True,
        }

        # ── 流式生成（含历史） ──
        if self.generator:
            try:
                async for token in self.generator.generate_stream(question, merge_chunks, history=history):
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
                                score=score * 0.1,  # 图扩展 decay_score 压缩 0.1 倍
                                page_content=doc,
                                metadata=meta,
                            )
                        )

            except Exception as e:
                logging.error(e)

        return chunks

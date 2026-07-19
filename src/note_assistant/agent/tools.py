"""Agent 工具集（完整集，P6 扩展 + P7a 重试/降级）。

第一批（最小集）：hybrid_search、graph_expand。
第二批（扩展）：vector_search、bm25_search、filtered_search、query_rewrite、get_note。

设计要点（配合自写 StateGraph 的 Context Accumulator）：
    - 暴露 *_impl 函数，返回结构化 ``List[RetrievalResult]``，供 tools 节点
      累积 / 去重 / 生成前 Top-K 裁剪使用。
    - 暴露 ``@tool`` 包装（给 LLM 看的自然语言描述 + 格式化文本）。
    - ``run_tool_call(name, args)`` 是 tools 节点的统一入口：
        · 带重试（settings.agent_max_tool_retry）
        · hybrid_search 失败自动降级为 vector_search
        · 任何异常都优雅兜底（返回空结果 + 提示，不让整条链路崩）
    - 工具返回文本里带「来源路径: <filepath>」，便于 LLM 引用，也便于解析来源。
"""
import logging
from functools import lru_cache
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from note_assistant.config import settings
from note_assistant.retrieval.graph import WikiGraph
from note_assistant.retrieval.hybrid import HybridRetriever
from note_assistant.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(alpha=settings.dense_weight)


@lru_cache(maxsize=None)
def _wiki_graph() -> WikiGraph:
    g = WikiGraph()
    try:
        g.load()
    except Exception:
        # 图谱可能尚未构建，忽略即可（graph_expand 会返回空）
        pass
    return g


@lru_cache(maxsize=None)
def _embedder():
    from note_assistant.indexing.embedder import OllamaEmbedder
    return OllamaEmbedder()


def _format_results(results: List[RetrievalResult]) -> str:
    """把检索结果格式化为含来源信息的文本（给 LLM 阅读）。"""
    if not results:
        return "（未检索到相关内容）"
    parts = []
    for i, r in enumerate(results, 1):
        title = r.metadata.get("title", "未知笔记")
        filepath = r.metadata.get("filepath", "")
        heading = r.metadata.get("heading_path", "")
        loc = f"{filepath} › {heading}" if heading else filepath
        parts.append(f"[{i}] 《{title}》\n来源路径: {loc}\n{r.page_content}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# 结构化实现（返回 RetrievalResult，供 Accumulator 使用）
# ──────────────────────────────────────────────

def hybrid_search_impl(query: str, top_k: int = 5) -> List[RetrievalResult]:
    return _hybrid_retriever().search(query, top_k=top_k)


def vector_search_impl(query: str, top_k: int = 5) -> List[RetrievalResult]:
    return _hybrid_retriever().vector_search(query, top_k=top_k)


def bm25_search_impl(query: str, top_k: int = 5) -> List[RetrievalResult]:
    return _hybrid_retriever().bm25_search(query, top_k=top_k)


def filtered_search_impl(
    query: str,
    filepath: str | None = None,
    heading: str | None = None,
    tag: str | None = None,
    top_k: int = 5,
) -> List[RetrievalResult]:
    return _hybrid_retriever().filtered_search(
        query, filepath=filepath, heading=heading, tag=tag, top_k=top_k
    )


def graph_expand_impl(filepaths: List[str], hop: int = 1) -> List[RetrievalResult]:
    g = _wiki_graph()
    neighbors = g.expand(set(filepaths), hop=hop)
    if not neighbors:
        return []
    chunks: List[RetrievalResult] = []
    collection = _hybrid_retriever().ingestor.collection
    for filepath, _score in neighbors:
        if filepath.startswith("[["):
            continue
        try:
            res = collection.get(
                where={"filepath": filepath},
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(
                res.get("documents", []), res.get("metadatas", [])
            ):
                chunks.append(
                    RetrievalResult(
                        score=0.0,
                        page_content=doc,
                        metadata=meta or {},
                    )
                )
        except Exception:
            continue
    return chunks


def get_note_impl(filepath: str) -> List[RetrievalResult]:
    """读取整篇笔记的全部片段（按 filepath 拉全量）。"""
    if not filepath:
        return []
    collection = _hybrid_retriever().ingestor.collection
    try:
        res = collection.get(
            where={"filepath": filepath},
            include=["documents", "metadatas"],
        )
        chunks = [
            RetrievalResult(score=0.0, page_content=doc, metadata=meta or {})
            for doc, meta in zip(res.get("documents", []), res.get("metadatas", []))
        ]
        return chunks
    except Exception:
        return []


def query_rewrite_impl(original_query: str, missing_aspect: str = "") -> str:
    """用 LLM 把原查询改写为更利于知识库检索的查询（只返回改写后的字符串）。"""
    from note_assistant.llm.client import get_llm

    aspect = f"\n缺失角度：{missing_aspect}" if missing_aspect else ""
    prompt = (
        f"原问题：{original_query}{aspect}\n"
        "请改写为更利于个人知识库（Obsidian 笔记）检索的查询。"
        "要求：陈述句、保留核心实体、去掉口语冗余。只输出改写后的查询，不要解释。"
    )
    llm = get_llm(temperature=0.3)
    resp = llm.invoke([
        SystemMessage("你是检索查询改写助手，擅长把口语问题改写为精准检索词。"),
        HumanMessage(prompt),
    ])
    return str(resp.content).strip()


# ──────────────────────────────────────────────
# @tool 包装（给 LLM 看的描述 + 格式化文本）
# ──────────────────────────────────────────────

@tool("hybrid_search")
def hybrid_search(query: str, top_k: int = 5) -> str:
    """在个人知识库中做混合检索（语义向量 + BM25 关键词融合），通用首选入口。

    适用场景：用户提问、或需要查找某个概念/操作/知识点时。
    返回最相关的若干笔记片段，每段都带标题与来源路径。

    Args:
        query: 检索问题（陈述句效果优于口语问句）
        top_k: 返回的片段数量，默认 5
    """
    return _format_results(hybrid_search_impl(query, top_k))


@tool("graph_expand")
def graph_expand(filepaths: List[str], hop: int = 1) -> str:
    """基于 Obsidian [[双链]] 从已命中的笔记出发，扩展其关联笔记。

    适用场景：已拿到若干相关笔记（hybrid_search 结果里的来源路径），
    想进一步查看它们链接/被链接的笔记，补全上下文。

    Args:
        filepaths: 已命中笔记的来源路径列表（来自 hybrid_search 的「来源路径」）
        hop: 扩展跳数，默认 1（只扩展直接相邻的一跳）
    """
    return _format_results(graph_expand_impl(filepaths, hop))


@tool("vector_search")
def vector_search(query: str, top_k: int = 5) -> str:
    """仅做语义向量检索（不依赖关键词）。

    适用场景：hybrid_search 关键词噪声大、或问题偏语义/概念时。

    Args:
        query: 检索问题
        top_k: 返回片段数，默认 5
    """
    return _format_results(vector_search_impl(query, top_k))


@tool("bm25_search")
def bm25_search(query: str, top_k: int = 5) -> str:
    """仅做关键词（BM25）检索（不依赖语义向量）。

    适用场景：问题含精确专有名词/术语、或 hybrid 语义召回不准时。

    Args:
        query: 检索问题
        top_k: 返回片段数，默认 5
    """
    return _format_results(bm25_search_impl(query, top_k))


@tool("filtered_search")
def filtered_search(
    query: str,
    filepath: str = "",
    heading: str = "",
    tag: str = "",
    top_k: int = 5,
) -> str:
    """按元数据过滤（filepath / heading / tag）后再做向量检索，缩小范围。

    适用场景：已知答案在某篇笔记、某章节或某标签下，想精准定位。

    Args:
        query: 检索问题
        filepath: 限定来源笔记路径（精确匹配）
        heading: 限定章节（heading_path 包含该字符串）
        tag: 限定标签（tags 包含该字符串）
        top_k: 返回片段数，默认 5
    """
    return _format_results(
        filtered_search_impl(query, filepath or None, heading or None, tag or None, top_k)
    )


@tool("get_note")
def get_note(filepath: str) -> str:
    """读取整篇笔记的全部片段（按 filepath 全量读取）。

    适用场景：已通过其它工具定位到某篇笔记，想一次性看全文上下文。

    Args:
        filepath: 笔记来源路径（来自其它工具返回的「来源路径」）
    """
    return _format_results(get_note_impl(filepath))


@tool("query_rewrite")
def query_rewrite(original_query: str, missing_aspect: str = "") -> str:
    """把原查询改写为更利于知识库检索的查询。

    适用场景：上一轮检索证据不足，需要换一种问法/聚焦缺失角度再查。

    Args:
        original_query: 原始查询
        missing_aspect: 上一轮缺失的角度（可选）
    """
    return query_rewrite_impl(original_query, missing_aspect)


# 暴露给 agent 构建器（bind_tools 需要 schema）
AGENT_TOOLS = [
    hybrid_search,
    graph_expand,
    vector_search,
    bm25_search,
    filtered_search,
    get_note,
    query_rewrite,
]


# ──────────────────────────────────────────────
# 统一执行入口（含重试 / 降级 / 兜底）
# ──────────────────────────────────────────────

def _dispatch(name: str, args: dict) -> Tuple[str, List[RetrievalResult]]:
    """真正分发到 impl，无重试。失败直接抛异常由上层捕获。"""
    if name == "hybrid_search":
        results = hybrid_search_impl(str(args.get("query", "")), int(args.get("top_k", 5)))
    elif name == "graph_expand":
        results = graph_expand_impl(args.get("filepaths", []) or [], int(args.get("hop", 1)))
    elif name == "vector_search":
        results = vector_search_impl(str(args.get("query", "")), int(args.get("top_k", 5)))
    elif name == "bm25_search":
        results = bm25_search_impl(str(args.get("query", "")), int(args.get("top_k", 5)))
    elif name == "filtered_search":
        results = filtered_search_impl(
            str(args.get("query", "")),
            args.get("filepath") or None,
            args.get("heading") or None,
            args.get("tag") or None,
            int(args.get("top_k", 5)),
        )
    elif name == "get_note":
        results = get_note_impl(str(args.get("filepath", "")))
    elif name == "query_rewrite":
        text = query_rewrite_impl(
            str(args.get("original_query", args.get("query", ""))),
            str(args.get("missing_aspect", "")),
        )
        return text, []
    else:
        return f"（未知工具: {name}）", []
    return _format_results(results), results


def _retry(fn):
    """重试包装：失败重试 settings.agent_max_tool_retry 次，全失败则抛出最后一次异常。"""
    last: Exception | None = None
    for _ in range(max(1, settings.agent_max_tool_retry)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last or RuntimeError("tool failed")


def run_tool_call(name: str, args: dict) -> Tuple[str, List[RetrievalResult]]:
    """工具统一执行入口：返回 (给 LLM 的观察文本, 结构化结果列表)。

    工业级加固：
        - 重试：单次调用失败按 settings.agent_max_tool_retry 重试。
        - 降级：hybrid_search 全部失败 → 自动降级为 vector_search。
        - 兜底：任何工具最终失败 → 返回空结果 + 友好提示，不让链路崩溃。
    """
    args = args or {}

    try:
        return _retry(lambda: _dispatch(name, args))
    except Exception as e:  # noqa: BLE001
        logger.warning("工具 %s 调用失败: %s", name, e)
        # hybrid 失败降级为纯向量检索
        if name == "hybrid_search":
            try:
                text, results = _retry(lambda: _dispatch("vector_search", args))
                return text + "\n（注：hybrid 检索失败，已降级为纯向量检索）", results
            except Exception as e2:  # noqa: BLE001
                logger.warning("hybrid 降级 vector 也失败: %s", e2)
        return f"（工具 {name} 调用失败，已跳过）", []

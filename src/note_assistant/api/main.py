# src/note_assistant/api/main.py
"""
FastAPI 入口 —— RAG 对外接口。

提供四个端点：
    POST /ask        — 非流式问答（返回完整 answer + sources + timing）
    POST /ask_stream — 流式问答（SSE，逐 token 输出）
    GET  /health     — 健康检查
    GET  /config     — 当前系统配置
    POST /reindex    — 增量索引

架构：
    应用启动时（lifespan）初始化 RAGChain，全局共享一个实例。
    RAGChain.ask() 返回内部 AskResponse 后，API 层负责：
        - 计时（timing）
        - 内部 SourceInfo → API SourceSchema 转换
        - 错误处理与限流
"""

import time
import logging
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from note_assistant.config import settings
from note_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    SourceSchema,
    HealthResponse,
    ReindexResponse,
    ConfigResponse,
)

logger = logging.getLogger(__name__)

# ─── 全局 RAG Chain ──────────────────────────────────────────
# lifespan 中初始化，启动后常驻内存
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动时初始化 RAG Chain（只加载一次）。

    【核心逻辑待实现】：组装 RAG 管线各组件。
    ```
    rag_chain = RAGChain(
        hybrid_retriever=HybridRetriever(dense, bm25, alpha=settings.dense_weight),
        reranker=LocalReranker(settings.reranker_model),
        graph=WikiGraph(),          # 网络加载 graph.gpickle
        generator=Generator(),      # LLM 生成器
    )
    ```

    面试考点：
    - 全局 rag_chain 是 Python module-level 变量，由 lifespan 管理生命周期
    - 为什么不每次请求新创建？组件（embedder/chroma）内有连接池和缓存，重复创建浪费
    - 重启 API 服务后需要重新 init——冷启动约 3-5s（reranker 加载耗时）
    """
    global rag_chain
    logger.info("🚀 正在初始化 RAG Chain...")

    # ═══════════════════════════════════════════════════
    # 参考 Day1-4 的 API（vault_loader, embedder, ingestor 等）
    # ═══════════════════════════════════════════════════
    from note_assistant.pipeline.rag_chain import RAGChain
    from note_assistant.retrieval.hybrid import HybridRetriever
    from note_assistant.retrieval.reranker import LocalReranker
    from note_assistant.retrieval.graph import WikiGraph
    from note_assistant.generation.generator import Generator

    # HybridRetriever 内部自行创建 embedder/ingestor/bm25，不需要外部传
    hybrid = HybridRetriever(alpha=settings.dense_weight)
    reranker = LocalReranker(str(settings.reranker_model))
    graph = WikiGraph()
    generator = Generator()

    rag_chain = RAGChain(hybrid, reranker, graph, generator)

    if rag_chain is None:
        logger.warning("⚠️ RAG Chain 未初始化，/ask 端点将返回 503")
    else:
        logger.info("✅ RAG Chain 初始化完成")

    yield


app = FastAPI(title="Obsidian RAG", lifespan=lifespan)

# CORS —— 允许 Streamlit 前端跨域访问（开发环境放开，生产应收窄）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# /ask — 非流式问答
# ═══════════════════════════════════════════════════════════════

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    非流式问答：检索 → 生成 → 返回完整结果。

    请求示例：
        {"question": "什么是 RAG？"}

    响应结构：
        {
            "answer": "...",
            "sources": [{"type": "text", "filepath": "...", ...}],
            "graph_expansion": 2,
            "timing": {"retrieve_ms": 120, "rerank_ms": 45, ...}
        }

    面试考点：
    - 为什么非流式和流式分开两个端点？—— 前端消费模式不同（一次性 vs SSE）
    - 计时为什么在 API 层而不是 RAGChain 内部？—— 关注点分离，Agent 层不加业务逻辑
    - 异常处理：rag_chain.ask() 内部异常应封装为 500，不要透传内部错误详情给用户

    【核心逻辑待实现】：调用 rag_chain + 计时 + 组装 response
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain 未初始化，请检查后端状态")

    t0 = time.time()
    ask_response = rag_chain.ask(req.question)

    t1 = time.time()



    answer_text = ask_response.answer
    source_list: list[SourceSchema] = []
    for s in ask_response.sources:
        source_list.append(SourceSchema(
            type=s.type,
            filepath=s.filepath,
            heading=s.heading,
            preview=s.preview,
            score=s.score,
        ))
    timing_dict = {"total_ms": (t1-t0) * 1000}

    return AskResponse(
        answer=answer_text,
        sources=source_list,
        graph_expansion=ask_response.graph_expansion,   # ← 用户填充实际值
        timing=timing_dict,
    )


# ═══════════════════════════════════════════════════════════════
# /ask_stream — 流式问答（SSE）
# ═══════════════════════════════════════════════════════════════

@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    """
    流式问答（Server-Sent Events）。

    输出格式（一行一个事件）：
        data: {"type": "meta", "retrieve_ms": 120, "graph_expansion": 2}
        data: {"type": "char", "content": "R"}
        data: {"type": "char", "content": "A"}
        data: {"type": "char", "content": "G"}
        ...
        data: {"type": "sources", "content": [...], "graph_expansion": 2}
        data: [DONE]
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain 未初始化")

    async def generate():
        """流式生成器：消费 rag_chain.ask_stream() 的事件，包装为 SSE。"""
        import json as _json

        try:
            async for event in rag_chain.ask_stream(req.question):
                event_type = event.get("type")

                if event_type == "sources":
                    # 将 SourceInfo dict 转为 API SourceSchema
                    raw_sources = event.get("content", [])
                    graph_exp = event.get("graph_expansion", 0)
                    schema_sources = []
                    for s in raw_sources:
                        schema_sources.append(SourceSchema(
                            type=s.get("type", "text"),
                            filepath=s.get("filepath", ""),
                            heading=s.get("heading", ""),
                            preview=s.get("preview", ""),
                            score=s.get("score"),
                        ).model_dump(mode="json"))
                    yield f"data: {_json.dumps({'type': 'sources', 'content': schema_sources, 'graph_expansion': graph_exp})}\n\n"

                elif event_type == "char":
                    yield f"data: {_json.dumps({'type': 'char', 'content': event['content']})}\n\n"

                else:
                    # meta 等事件直接透传
                    yield f"data: {_json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"ask_stream 异常: {e}")
            yield f"data: {_json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════
# /ask_trace — 跟踪式问答（含检索过程实时输出）
# ═══════════════════════════════════════════════════════════════

@app.post("/ask_trace")
async def ask_trace(req: AskRequest):
    """
    跟踪式问答：在 /ask_stream 的基础上，检索阶段按步骤输出 trace 事件。

    输出格式：
        data: {"type": "trace", "step": "embedding", "ms": 150}
        data: {"type": "trace", "step": "dense_retrieval", "results": 50, "ms": 200}
        data: {"type": "trace", "step": "sparse_retrieval", "results": 50, "ms": 10}
        data: {"type": "trace", "step": "hybrid_fusion", "results": 20, "ms": 5}
        data: {"type": "trace", "step": "rerank", "results": 5, "ms": 400}
        data: {"type": "trace", "step": "graph_expansion", "results": 2, "ms": 50}
        data: {"type": "meta", "retrieve_ms": 815, "graph_expansion": 2}
        data: {"type": "char", "content": "R"}
        ...
        data: {"type": "sources", "content": [...], "graph_expansion": 2}
        data: [DONE]

    前端可在收到 trace 事件时更新进度条/步骤列表，让用户看到"正在做什么"。
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG Chain 未初始化")

    async def generate():
        import json as _json

        try:
            async for event in rag_chain.ask_with_trace(req.question):
                yield f"data: {_json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"ask_trace 异常: {e}")
            yield f"data: {_json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════
# /health — 健康检查
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    健康检查端点。

    返回 ChromaDB 的 chunk 数和当前 embedding 模型名。
    用于：
    - Docker Compose healthcheck
    - 前端启动时检测后端是否就绪
    - 调试时确认索引是否已加载
    """
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        col = client.get_collection(settings.collection_name)
        chunk_count = col.count()
    except Exception as e:
        logger.warning(f"health check failed: {e}")
        return HealthResponse(status="degraded", chunks=0, model=settings.embed_model)

    return HealthResponse(status="ok", chunks=chunk_count, model=settings.embed_model)


# ═══════════════════════════════════════════════════════════════
# /config — 当前配置
# ═══════════════════════════════════════════════════════════════

@app.get("/config", response_model=ConfigResponse)
async def config():
    """
    返回当前系统配置。

    暴露给前端调试面板使用（/ask_stream 的 meta 事件中其实只需要部分字段，
    但 /config 提供一次性查看全部配置的能力）。
    """
    return ConfigResponse(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
        top_k_retrieve=settings.top_k_retrieve,
        top_k_rerank=settings.top_k_rerank,
        graph_hop=1,                       # 当前固定 1 跳，可在 settings 中配置
        embed_model=settings.embed_model,
        llm_model=settings.llm_model,
    )


# ═══════════════════════════════════════════════════════════════
# /reindex — 增量索引
# ═══════════════════════════════════════════════════════════════

@app.post("/reindex", response_model=ReindexResponse)
async def reindex():
    """
    增量索引 —— 重新扫描 vault 变化，只更新变更的文件。

    调用 scripts/reindex.py 的 incremental_reindex 函数。
    """
    try:
        from scripts.reindex import incremental_reindex
        result = incremental_reindex(str(settings.vault_path))
        return ReindexResponse(**result)
    except ImportError:
        raise HTTPException(status_code=500, detail="reindex 脚本未找到，请先运行 uv sync 或确认 scripts/reindex.py 存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增量索引失败: {e}")

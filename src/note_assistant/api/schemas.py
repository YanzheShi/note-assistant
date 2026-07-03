# src/note_assistant/api/schemas.py
"""
Pydantic 请求/响应模型 —— FastAPI 与外部通信的数据契约。

与内部 `rag_chain.AskResponse` 的区别：
- 内部：SourceInfo 只有 type/text 两个字段，用于 Pipeline 内部传递
- API：SourceSchema 区分四种类型（text/table/mermaid/image），带各自的 raw_* 字段
- API：额外携带 timing（端到端耗时）和 graph_expansion 统计
"""

from typing import Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# 请求
# ═══════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    """用户提问请求体。"""
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")


# ═══════════════════════════════════════════════════════════════
# 来源（响应中的单个来源片段）
# ═══════════════════════════════════════════════════════════════

class SourceSchema(BaseModel):
    """
    单个来源片段，按类型区分渲染字段。

    type 取值：
        text    — 普通文本段落（preview 即可）
        table   — Markdown 表格（preview + raw_table）
        mermaid — Mermaid 流程图（preview + raw_mermaid）
        image   — 图片（preview + img_path）
    """
    type: str = "text"                       # text | table | mermaid | image
    filepath: str = ""                       # 来源笔记路径（相对 vault 根）
    heading: str = ""                        # 标题路径，如 "二、RAG > 2.1 定义"
    preview: str = ""                        # 摘要预览（所有类型都有）
    score: Optional[float] = None            # Rerank 得分（0~1）

    # 仅 table 类型
    raw_table: Optional[str] = None

    # 仅 mermaid 类型
    raw_mermaid: Optional[str] = None

    # 仅 image 类型
    img_path: Optional[str] = None           # 图片在 Obsidian vault 中的路径


# ═══════════════════════════════════════════════════════════════
# 响应
# ═══════════════════════════════════════════════════════════════

class AskResponse(BaseModel):
    """
    /ask 的完整响应。

    设计考虑：
    - answer 和 sources 是必填字段，前端至少能显示"未找到答案"
    - timing 反映各阶段耗时，便于调试（调试面板用）
    - graph_expansion 告知用户是否做了双链扩展
    """
    answer: str
    sources: list[SourceSchema] = Field(default_factory=list)
    graph_expansion: int = 0
    timing: Optional[dict] = None            # {"retrieve_ms": ..., "rerank_ms": ..., "generate_ms": ..., "total_ms": ...}


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    chunks: int                              # ChromaDB 中的 chunk 总数
    model: str                               # 当前使用的 embedding 模型名


class ReindexResponse(BaseModel):
    """增量索引响应。"""
    status: str
    reindexed: int                           # 新索引/更新的文件数
    removed: int                             # 删除的文件数


class ConfigResponse(BaseModel):
    """当前系统配置——暴露给前端/调试用。"""
    chunk_size: int
    chunk_overlap: int
    dense_weight: float
    bm25_weight: float
    top_k_retrieve: int
    top_k_rerank: int
    graph_hop: int
    embed_model: str
    llm_model: str
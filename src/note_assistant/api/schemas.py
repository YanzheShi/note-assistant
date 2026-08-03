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
    """用户提问请求体，支持多轮对话历史与持久化上下文。"""
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    history: list[dict] = Field(
        default_factory=list,
        description="历史对话，[{\"role\": \"user\"|\"assistant\", \"content\": str}, ...]，按时间顺序，最新的在最后",
    )
    session_id: str = Field(
        default="",
        description="跨会话记忆的会话 id；提供后服务端按 id 维护历史，前端不必每轮带 history",
    )
    run_id: str = Field(
        default="",
        description="运行快照 id；提供已存在的 run_id 可断流续传（轮询或重订阅）",
    )


# ═══════════════════════════════════════════════════════════════
# 来源（响应中的单个来源片段）
# ═══════════════════════════════════════════════════════════════

class SourceSchema(BaseModel):
    """
    单个来源片段，按类型区分渲染字段。

    type 取值（内容类型）：
        text    — 普通文本段落（preview 即可）
        table   — Markdown 表格（preview + raw_table）
        mermaid — Mermaid 流程图（preview + raw_mermaid）
        image   — 图片（preview + img_path）

    origin 取值（来源渠道，与 type 正交）：
        direct  — 检索直接命中
        graph   — 双链扩展带出

    注意：内部 `SourceInfo` 的对应字段叫 `kind`（内容类型）/ `origin`（来源渠道）。
    两者曾同名为 `type` 且被直接透传，前端因此永远收到 "direct"，四个渲染分支全是死代码。
    """
    type: str = "text"                       # text | table | mermaid | image
    origin: str = "direct"                   # direct | graph
    filepath: str = ""                       # 来源笔记路径（相对 vault 根）
    heading: str = ""                        # 标题路径，如 "二、RAG > 2.1 定义"
    preview: str = ""                        # 摘要预览（所有类型都有）
    score: Optional[float] = None            # Rerank 得分（0~1）

    # 仅 table 类型
    raw_table: Optional[str] = None

    # 仅 mermaid 类型
    raw_mermaid: Optional[str] = None
    render_hint: Optional[str] = None        # "mermaid:inline"：前端可原生渲染（非幻觉）
    diagram_type: Optional[str] = None       # graph TD / sequenceDiagram / classDiagram ...

    # 仅 image 类型
    img_path: Optional[str] = None           # 图片在 Obsidian vault 中的路径
    asset_id: Optional[str] = None           # 图片资产内容哈希 id（供 /assets 端点定位）
    img_url: Optional[str] = None            # 图片服务 URL（/assets/{asset_id}），前端优先用此渲染


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


# ═══════════════════════════════════════════════════
# Agentic RAG 响应（/agent/* 端点）
# ═══════════════════════════════════════════════════

class AgentSource(BaseModel):
    """Agent 去重后的单个来源（来自 Context Accumulator）。

    设计 9.2 /agent 适配项：补齐图片渲染字段，使 Agent 主链路也能把命中图片
    透传给前端（/ask 路径已在 P0 修好，此处补齐 Agent 路径）。
    """
    filepath: str = ""
    title: str = ""
    heading: str = ""
    score: Optional[float] = None
    kind: str = "text"                       # text | table | mermaid | image
    img_url: Optional[str] = None            # 图片服务 URL（/assets/{asset_id}）
    render_hint: Optional[str] = None        # mermaid:inline / svg:inline / image:...


class AgentTrajectoryItem(BaseModel):
    """Agent 轨迹中的一个事件。"""
    type: str                                       # thought | tool_call | observation | answer | judge
    content: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[dict] = None
    verdict: Optional[str] = None                  # judge 节点的判定（sufficient/need_rewrite/need_more/give_up）
    reason: Optional[str] = None
    iteration: Optional[int] = None


class AgentAskResponse(BaseModel):
    """/agent/ask 的完整响应。"""
    answer: str
    sources: list[AgentSource] = Field(default_factory=list)
    trajectory: list[AgentTrajectoryItem] = Field(default_factory=list)
    cached: bool = False
    run_id: str = ""
    session_id: str = ""
    timing: Optional[dict] = None


class AgentRunStatus(BaseModel):
    """GET /agent/runs/{run_id} 的快照响应（流式中断后轮询取回）。"""
    run_id: str
    question: str
    status: str                                       # running | finished | interrupted
    answer: str
    sources: list[AgentSource] = Field(default_factory=list)
    trajectory: list[AgentTrajectoryItem] = Field(default_factory=list)


class AgentSessionHistory(BaseModel):
    """GET /agent/sessions/{session_id} 的会话记忆响应。"""
    session_id: str
    history: list[dict] = Field(default_factory=list)
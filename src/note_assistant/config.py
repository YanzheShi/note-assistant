from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # === Obsidian Vault ===
    vault_path: Path = Path("./vault")

    # === Embedding (bge-m3 via Ollama) ===
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3:latest"
    embed_dim: int = 1024

    # === Vector Store ===
    chroma_persist_dir: Path = Path("./data/chroma")
    collection_name: str = "obsidian_notes"

    # === Reranker ===
    reranker_model: str = str(PROJECT_ROOT / "models" / "BAAI" / "bge-reranker-v2-m3")
    reranker_top_k: int = 10

    # === RAGAS ===
    ragas_base_url: str = "http://localhost:11434/v1"
    ragas_api_key: str = "ollama"
    ragas_llm_model: str = "qwen2.5:0.5b"


    # === Agent / Generator LLM（统一走 AGENT_*）===
    agent_api_key: str
    agent_base_url: str
    agent_model: str
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-v4-flash"

    # === LangSmith (Tracing) ===
    langsmith_tracing_enabled: bool = True
    langsmith_api_key: str = ""
    langsmith_endpoint: str
    langsmith_project: str = "note-assistant"

    # === Retrieval ===
    bm25_index_path: Path = Path("./data/bm25.pkl")
    chunk_size: int = 800
    chunk_overlap: int = 150
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # === Agentic RAG ===
    agent_max_iter: int = 3                     # 检索循环最大轮次（硬性降级上限）
    agent_max_tool_retry: int = 2               # 单次工具调用失败重试次数
    agent_cache_enabled: bool = True            # 是否开启语义缓存
    agent_cache_ttl: int = 3600                 # 缓存 TTL（秒）
    agent_cache_max_size: int = 1000            # 缓存最大条数（FIFO 淘汰）
    agent_cache_semantic: bool = True           # 是否启用 embedding 近邻命中
    agent_cache_semantic_threshold: float = 0.92  # 近邻命中相似度阈值

    # === Agentic RAG 持久化（轻量 SQLite）===
    agent_session_enabled: bool = True           # 跨会话记忆 + 运行快照（session_turns / runs 表）
    agent_db_path: str = "data/agent.sqlite"     # SQLite 文件路径（相对 PROJECT_ROOT 或绝对路径）
    agent_run_orphan_ttl: int = 600              # run 未完成超过该秒数判定为 interrupted（孤儿检测）

    # === Agent Graph Expand（自动图扩展，默认关闭）===
    agent_graph_expand_enabled: bool = False     # 每轮检索后自动沿 [[wikilinks]] 扩展关联笔记
    agent_graph_expand_hop: int = 1              # 扩展跳数

    # === Agent Reranker（双层精排，可独立开关对比）===
    agent_reranker_loop_enabled: bool = True    # Rerank ①：循环内闸门（tools→reflect）
    agent_reranker_exit_enabled: bool = True    # Rerank ②：出口总安检（reflect→generate）
    agent_reranker_loop_top_k: int = 10          # 循环内保留条数（出口复用 top_k_rerank=5）

    # === 上下文管理（ContextManager）===
    # 总预算硬上限：历史 + 累积 + 工具观察三段之和不可超过此值（模型窗口 - 输出预留）。
    # ⚠️ 不变式：本值须 ≥ 三个子预算之和（2000+1500+800=4300）。
    #   obs 由 tools_node 按 agent_obs_token_budget 独立截断；history / accumulated 各自独立封顶，
    #   因此本值作为全局安全网存在（正常路径三段之和已 ≤ 本值）。调整任一段子预算时须同步上调本值。
    agent_total_context_token_budget: int = 4500
    # 各段默认子预算；超总预算时按 obs → accumulated → history 优先级压缩
    agent_history_token_budget: int = 2000
    agent_accumulated_token_budget: int = 1500
    agent_obs_token_budget: int = 800

    # 问题凝练（消指代）开关
    agent_condense_enabled: bool = True
    # 长程记忆（滚动摘要）开关
    agent_summary_enabled: bool = True
    # 触发滚动摘要的“原文 user/assistant 轮次 token 总和”阈值（非轮次数）
    agent_session_summary_threshold: int = 3000
    # 滚动摘要后保留的最近原文轮次数（不摘要，供 get_history 直接返回）
    agent_session_recent_keep: int = 6
    # session_turns 硬上限，超过删最旧轮次防无限增长
    agent_session_max_turns: int = 200
    # 相关性裁剪开关
    agent_history_relevance_enabled: bool = True
    # 相关性裁剪只看最近 N 轮时间窗口（窗口外不进候选，避免“全高分=没裁剪”）
    agent_history_relevance_window: int = 20
    # 与凝练问题的 embedding 相似度阈值，>= 该值才保留。
    # 注意：bge-m3 余弦分布下 0.3 偏松，易使「时间窗口内几乎全过=裁剪近乎失效」。
    # 建议跑真实对话数据后调到 0.5~0.6 区间，并按实际召回质量微调。
    agent_history_relevance_threshold: float = 0.3
    # 跨轮累积每跨一轮的 score 衰减系数（双重保险之一）
    agent_accumulated_decay: float = 0.9
    # 跨轮累积按 token 预算硬截断时保留的最大片段数（辅助上限）
    agent_accumulated_max_items: int = 30


settings = Settings()
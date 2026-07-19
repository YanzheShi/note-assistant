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


    # === LLM ===
    agnes_api_key: str
    agnes_base_url: str
    agnes_model: str
    deepseek_api_key: str
    longcat_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    longcat_base_url: str
    llm_model: str = "deepseek-v4-flash"
    longcat_model: str

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


settings = Settings()
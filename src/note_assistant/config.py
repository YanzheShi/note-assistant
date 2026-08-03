from pathlib import Path
from typing import Literal
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

    # === RAGAS ===
    ragas_base_url: str = "http://localhost:11434/v1"
    ragas_api_key: str = "ollama"
    ragas_llm_model: str = "qwen2.5:0.5b"


    # === Agent / Generator LLM（统一走 AGENT_*）===
    agent_api_key: str
    agent_base_url: str
    agent_model: str

    # === VLM / 多模态理解（P1-c，OpenAI 兼容视觉通道）===
    # 注意：字段名与 AGENT_* 完全独立，避免纯文本 LLM 与视觉模型互相串台。
    vlm_api_key: str = ""          # 为空 → 不启用 VLM 理解（索引降级为 alt+上下文）
    vlm_base_url: str = ""         # 视觉模型网关（如 https://api.agnes-ai.cn/v1）
    vlm_model: str = ""            # 视觉模型名（如 agnes-2.5-flash）

    # === 图片理解护栏（设计文档 5.1 / 5.4 / 5.5）===
    image_understand_enabled: bool = False    # 总开关（设计 G6）：默认关闭 → 零回归/逐字节等价，图片只走 alt+上下文（零 VLM 调用、零副作用）
    image_allow_remote_fetch: bool = True     # 隐私开关：关闭则只处理本地资产，不下载远程图
    image_max_bytes: int = 10 * 1024 * 1024   # 单图大小护栏（>10MB 跳过 VLM，省 token）
    image_vlm_max_calls_per_run: int = 500    # 单次索引 VLM 调用上限，超限剩余标记 pending
    image_vlm_concurrency: int = 4            # 并发上限（asyncio.Semaphore）
    vlm_prompt_version: str = "v2"            # 改 prompt 必须 bump，缓存据此失效重跑（v2=抗注入硬化版）
    assets_dir: Path = Path("./data/assets")          # 远程图/资产本地缓存目录
    vision_cache_path: Path = Path("./data/vision_cache.sqlite")  # VLM 结果缓存

    # === 图片检索/生成/展示闭环（设计文档 P2）===
    image_intent_boost: float = 0.15         # 图意图 query 命中时，image chunk 融合分 ×(1+boost)
    image_neighbor_expand: bool = True       # 命中 image chunk 时带出同 heading_path 的文本邻居（防脱离上下文误导）

    # === LangSmith (Tracing) ===
    langsmith_tracing_enabled: bool = True
    langsmith_api_key: str = ""
    langsmith_endpoint: str
    langsmith_project: str = "note-assistant"

    # === Retrieval ===
    bm25_index_path: Path = Path("./data/bm25.pkl")
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunking_strategy: Literal["v1", "v2", "v2b"] = "v2"  # 启动前切换切分策略（改后重建向量库生效）
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # === 父子双存切分（v2b，见 docs/父子双存切分（v2b）设计方案.md）===
    # 父块（整节返回给 LLM）的最大字符数；子块（800）仍负责精准检索
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    # 父块存储（不参与 embedding/BM25，仅在命中子块后按需取用）
    parent_docstore_path: Path = Path("./data/docstore.pkl")

    # === 结构优先检索（层级标题 boost，见 docs/层级检索与结构优先设计方案.md）===
    # β：结构命中叠加到融合分的权重。"高一点"——结构命中段比纯正文段多一个 β 增量，
    #    稳定靠前但不至于完全盖掉内容相关性；β=0 时完全退化回现状，零回归。
    structure_weight: float = 0.25
    # 结构分门控：低于此值不施加 boost，避免弱结构信号噪声翻车（提到目录名就排前但正文不相关）。
    structural_min_score: float = 0.5
    # query 精确命中某段「文档标题」时的额外硬兜底 bonus（如问"Code Agent 架构"命中文档名）。
    title_hit_bonus: float = 0.15

    # === Agentic RAG ===
    agent_max_iter: int = 3                     # 检索循环最大轮次（硬性降级上限）
    agent_max_tool_retry: int = 2               # 单次工具调用失败重试次数
    agent_cache_enabled: bool = True            # 是否开启语义缓存
    agent_cache_ttl: int = 3600                 # 缓存 TTL（秒）
    agent_cache_max_size: int = 1000            # 缓存最大条数（FIFO 淘汰）
    agent_cache_semantic: bool = True           # 是否启用 embedding 近邻命中
    agent_cache_semantic_threshold: float = 0.92  # 近邻命中相似度阈值

    # === 安全防御（docs/prompt-injection-defense-design.md，L0–L4）===
    # 全部关闭 = 与改造前逐字节等价（G6 式零回归约定）
    # L1 提示词硬化：system 提示追加安全护栏条款 + 不可信内容分隔符包裹
    security_guardrail_enabled: bool = True
    # L2 确定性输入清洗：注入形状启发式检测。flag=只记日志不改写；redact=遮蔽命中跨度
    prompt_injection_scan_enabled: bool = True
    prompt_injection_scan_action: Literal["flag", "redact"] = "flag"
    # L3 工具收敛：get_note / filtered_search 只能读本会话已浮现的笔记
    get_note_allowlist_enabled: bool = True
    filtered_search_allowlist_enabled: bool = True
    injection_escalation_threshold: int = 3     # 单会话注入命中 ≥ 该值 → 禁用读取类工具
    # L0 索引期供应链：远程图抓取主机策略（block_private 拒绝环回/私网/链路本地/元数据网段）
    image_remote_fetch_host_policy: Literal["block_private", "allowlist", "all"] = "block_private"
    image_remote_fetch_allowlist: list = []     # allowlist 模式下的域名白名单
    vlm_text_field_max_chars: int = 2000        # VLM description/ocr 单字段入库上限
    # L4 输出治理：远程图片中和（防渲染期外泄）+ system prompt 泄露指纹
    output_guard_enabled: bool = True
    output_guard_remote_media: Literal["neutralize", "allow"] = "neutralize"
    output_guard_media_allowlist: list = []     # 允许保留的远程图片域名（/assets 恒白名单）
    cache_skip_when_guarded: bool = True        # 输出护栏命中不入语义缓存（防投毒回放）

    # === Agentic RAG 持久化（轻量 SQLite）===
    agent_session_enabled: bool = True           # 跨会话记忆 + 运行快照（session_turns / runs 表）
    agent_db_path: str = "data/agent.sqlite"     # SQLite 文件路径（相对 PROJECT_ROOT 或绝对路径）
    agent_run_orphan_ttl: int = 600              # run 未完成超过该秒数判定为 interrupted（孤儿检测）

    # === Agent Graph Expand（自动图扩展，默认关闭）===
    agent_graph_expand_enabled: bool = False     # 每轮检索后自动沿 [[wikilinks]] 扩展关联笔记
    agent_graph_expand_hop: int = 1              # 扩展跳数

    # === 图扩展扇出护栏（审计修复：背链多的高连接笔记曾导致 context 膨胀/延迟失控）===
    graph_expand_max_files: int = 8              # 单次扩展最多取几个邻居文件（按扩展分取 top）
    graph_expand_max_chunks: int = 24            # 单次扩展返回的 chunk 总数上限

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

    # === 澄清 / 反问（clarify-as-terminal，方案 B）===
    # 方案0 前置拦截补漏：问题字数 < 该阈值且历史非空 → 也送 LLM 消解。
    # 修复「性能呢？/ 优缺点 / 怎么优化」这类零主语追问从未进入消解的漏洞。
    agent_condense_short_threshold: int = 8
    # 澄清总开关。关闭时所有澄清分支短路，行为与改造前逐字节等价。
    agent_clarify_enabled: bool = True
    # 消解置信度阈值：低于该值才允许进入澄清候选（级联终点约束）
    agent_clarify_confidence_threshold: float = 0.6
    # 上一轮召回 top1/top2 归一化分差低于此值 → 视为存在竞争主题
    agent_clarify_topic_margin: float = 0.15
    # last_entity 槽位记录的竞争主题条数上限
    agent_clarify_max_candidates: int = 3
    # 给 Judge 的证据片段条数 / 每条正文摘要字数（P0 盲判修复）
    # top_n 从 5 放宽到 8：既是兜底，也让 Judge 正文视图多覆盖几条低分但相关的片段；
    # 真正治本靠「覆盖概览」（见 agent.py::_format_judge_evidence）。
    agent_judge_evidence_top_n: int = 8
    agent_judge_evidence_chars: int = 200

    # === Agent 收敛闸门（防同文档空转，确定性停止）===
    # 连续多少轮改写后「新增独特文档数 = 0」就强制 sufficient 进生成，
    # 不再靠 LLM 自觉，直接切断「对同一篇文档换同义词反复重检」的死循环。
    agent_convergence_streak: int = 2
    # 生成窗口反向放宽：当靠覆盖视图 / 收敛闸门提前放行时，
    # 生成上下文从 top_k_rerank 放宽到该值，避免低分但相关的内容在生成端被裁掉。
    agent_generate_widen_top_k: int = 10


settings = Settings()
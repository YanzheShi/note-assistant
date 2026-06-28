# DECISIONS — note-assistant

> 每个模块的选型理由、拒掉的路、跟 vault 形态的绑定。
> 面试深聊时按这条索引讲，比背 README 值钱。

---

## [2026-06-26] loader: 只做 file-level，chunk 交给 splitter

- **做了啥**: VaultLoader 粒度 = 整篇笔记（scan / fm / wikilink / raw_md），标题层级 `# ## ###` 不在这解析
- **为什么**: loader 输出整篇、splitter 输出 chunk + heading_path，职责分开；若 loader 解析标题树，splitter 的 MarkdownHeaderTextSplitter 还得再解析一次，重复
- **拒掉的路**: 在 loader 里用正则抽 headings 带 line 号（另一 agent 版 DocNode 有）——留 Day 3 双链图 `[[target#anchor]]` 用，但 Day 1 不进 loader 返回结构
- **影响**: splitter 是唯一解析标题树的地方，metadata 缝 heading_path 在那做

---

## [2026-06-XX] loader: 双链 `[[wikilink]]` 正则

- **pattern**: `r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]'`
- **group(1)**: 目标名（`[[A|B]]` 取 A）；**group(2)**: 别名（B，当前忽略，Day 3 可扩展）
- **`[^\]\|]` 为什么排除 `]` 和 `|`**: `]` 是 `[[...]]` 闭合，`|` 是 `[[目标|别名]]` 分隔，不排除会吞别名段或越界
- **去重保序**: 同篇 `[[A]]` 出现多次（Obsidian 常见），set 去重 + list 保序，Day 3 双链图邻接表不重复
- **拒掉**: `re.findall` 裸用（不去重）→ 邻接表脏

---

## [2026-06-XX] loader: front matter 容错，不修用户笔记

- **问题**: vault 里 `项目说明.md` 这种 `# 标题` + `> 引述` + `---` 视觉分隔，python-frontmatter + PyYAML 误判进 block scalar 炸 ScannerError
- **做了啥**: `has_fm = text.lstrip().startswith("---\n")` 前置判断，无 fm 直接 fm={}, raw=全文，不进 PyYAML；有 fm 但 PyYAML 炸时降级同逻辑
- **为什么不做**: 改用户笔记是反模式，loader 扛脏数据才是真实场景（87 篇里 2 篇炸，降级后 87/87 加载）
- **影响**: title 回退 md_path.stem，tags=[], wikilinks 照样提，下游 splitter+embed 不受影响

---

## [2026-06-XX] config: load_dotenv() + pydantic-settings env_file 双保险

- **做了啥**: `config.py` 顶部 `load_dotenv(PROJECT_ROOT/".env")` + `load_dotenv(".env.local", override=True)`；`Settings.model_config.env_file` 也配双文件
- **为什么**: pydantic-settings 的 env_file 相对 cwd，PyCharm ▶️ / Docker / hermes wrapper 下 cwd 可能偏；load_dotenv 显式路径钉死 PROJECT_ROOT
- **PROJECT_ROOT**: `Path(__file__).resolve().parent.parent.parent`，config.py 位置算三层，cwd 怎么变不慌
- **影响**: PyCharm ▶️ / `uv run` / Day 6 Docker 三种场景 .env 都能读到

---

## [2026-06-XX] splitter: Heading-based + Hierarchical = v2（ stub 阶段）

> ⚠️ 下午实现，先占条目

- **选型**: 两层 — MarkdownHeaderTextSplitter（# ## ### ####）→ RecursiveCharacterTextSplitter（800/150）
- **为什么不选单层 Recursive**: 会切断 `## 检索` 下上下文，丢父 h1 "RAG 概述"，问"RAG 检索方法"时 chunk 缺 h1 上下文
- **separators 加 `。`**: 默认 `["\n\n","\n"," ",""]` 不含中文句界，不加会在 `。` 中间硬切一句劈两半——vault 中文为主必加
- **keep_separator=True**: `。` 切了粘回前 chunk 尾，LLM 读连贯
- **h1 空 skip**: vault 笔记 `## 一、xxx` 起手（无 h1），heading_path 拼 `"一、xxx > 检索"` 而非 `"> 一、xxx > 检索"`，跟 vault 形态绑定
- **wikilinks 整篇级缝**: loader 扫一次 → 每 chunk metadata 都有；Day 3 双链图可升级 chunk 级重扫
- **Parent-Child 双存（v2b）**: stub 占位，当前 vault 单篇 1-3KB，h2 段+下文≈400-800char 跟子 chunk 尺寸重叠，Double Chroma 收益<成本 → Day 3 长笔记再评估
- **拒掉的**: 单层 Recursive / return_each_line=True / wikilinks chunk 级 Day 1

---

## [待续] 后面每天追加

- embedder: bge-m3 Ollama 调用封装 + 稠密向量 only（稀疏 BM25 另路）
- Chroma: collection schema / metadata 过滤字段
- retrieval: BM25 加权融合权重 0.7:0.3 怎么来的
- reranker: bge-reranker-v2-m3 FlagReranker 本地跑 vs API
- query rewrite: LLM 转陈述句 prompt
- graph: `[[wikilink]]` → NetworkX 一跳扩展
- eval: Ragas 4 指标 + 100 QA 构造半自动
- Day 6 Docker / Day 7 云部署
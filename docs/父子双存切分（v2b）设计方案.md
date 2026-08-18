# 父子双存切分（v2b）设计方案

> 状态：✅ 已实现 | 日期：2026-07-29 | 关联：`splitter.py` · `ingestor.py` · `retrieval/hybrid.py` · `retrieval/docstore.py` · `config.py`
> 前置文档：[层级检索与结构优先设计方案.md](./层级检索与结构优先设计方案.md)（机制 A/B：结构前缀 + 结构分 boost，本次 v2b 与之正交）

---

## 一、背景与动机

Obsidian 笔记普遍偏长（实测 288 篇，篇均 ~13k 字，最长 7 万字）。当前生产路径 **v2**（header 拆分 + 800 字递归细切）把每个章节切成多个 800 字碎片入库。

**问题**：问"《Agent Skills》的最佳实践有哪些"时，检索命中了"八、最佳实践"章节里的 8.2 碎片（800 字），返回给 LLM 的**只有这 800 字**，8.1 / 8.3 直接丢失，答案不完整。

**v2b（Parent-Child 双存）** 的目标：**检索用细块保精准，返回用整节保完整**——命中 8.2 后，回退返回整个"八、最佳实践"章节（~2000 字），让 LLM 拿到完整上下文。

数据支撑（详见对话记录）：
- 单章节 > 1600 字（2×chunk）的文档占 **48%**（137/284）
- 单章节 > 2400 字（3×chunk）的文档占 **23%**

说明近一半文档的单个章节会被切碎，v2b 收益明确。

---

## 二、核心概念

| 概念 | 说明 |
|---|---|
| **child（子块）** | 800 字细切块，进 ChromaDB（embedding + BM25），用于**检索**。带 `parent_id` 指向其所属父块。 |
| **parent（父块）** | 一个"章节级"粗块（≤ `parent_chunk_size`，默认 2000 字），**不进索引**，只存 docstore，用于**返回**给 LLM。 |
| **docstore** | 父块存储（pickle 文件 `data/docstore.pkl`），`{parent_id: {page_content, metadata}}`，按 id 查询。 |
| **parent_id** | 父块稳定 id（`{filepath}::parent::{idx}`），child.metadata 与 parent.metadata 共用，建立关联。 |

子块负责"找得准"，父块负责"答得全"——这正是 v2b 相对 v2 的唯一增量价值。

---

## 三、存储结构（物理层）

```
┌─────────────────────────────── ChromaDB collection ───────────────────────────────┐
│ 每条 = { id, documents(结构前缀+正文), embeddings(1024维), metadatas }              │
│   ├─ child chunk (来自 v2b 切分，带 parent_id)  ← 检索只在这里跑（dense + BM25）    │
│   ├─ summary chunks（富结构摘要，与 v2 一致）                                      │
│   └─ fm chunks（front matter，与 v2 一致）                                         │
└───────────────────────────────────────────────────────────────────────────────────┘
                                          │  child.metadata.parent_id
                                          ▼
┌─────────────────────────────── docstore (data/docstore.pkl) ─────────────────────┐
│  { parent_id : { page_content: 整节正文(已 restore), metadata: {title,filepath,   │
│                     dir,heading_path,wikilinks,tags,parent_id,kind:"parent"} } }    │
│   ★ 纯文本存储，不 embedding、不进 BM25、不参与检索，仅在命中 child 后按需取用      │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**与 v2 的关键差异**：v2b 在 ChromaDB 之外多了一个 docstore；ChromaDB 里只有 child（与 v2 的 chunk 几乎一致），所以 **dense / BM25 / 结构分 boost 全部自动吃到，检索质量与 v2 持平，零回归**。

---

## 四、切分算法（split_v2b）

设计要点：**父块必须是"整节"，而不是"按字符硬切的 2000 字"**——否则 header 拆分后"八、最佳实践"会被切成 8.1/8.2/8.3 三个独立段，父块就回不到整节了。

算法（每个 DocNode）：

```
sections = header_sp.split_text(node.raw_md)        # 按 #/##/###/#### 拆成细章节
sec_list = [(hp, text, h1) for each section]        # hp=该章节 heading_path; h1=顶层章节

# 1) 同 h1 内、按 parent_chunk_size 预算，把连续细章节合并成父段
parent_segments = []
run = []
for s in sec_list:
    if run and (s.h1 != run[0].h1 or run_len + len(s.text) > parent_chunk_size):
        flush(run)                                  # 跨 h1 或超预算 → 另起父段
        run = []
    run.append(s)
flush(run)

# flush(run):
#   若合并文本 ≤ parent_chunk_size → 1 个父段
#   否则用 parent_splitter(递归, chunk_size=parent_chunk_size) 把该段再切成 bounded 父段
#   每个父段记录 hp = run[0].hp（所在章节名）

# 2) 每个父段 → 1 个 parent Chunk + N 个 child Chunk
for seg in parent_segments:
    pid = f"{filepath}::parent::{idx}"
    parent = Chunk(page_content=seg.text,
                   metadata={..., heading_path=seg.hp, parent_id=pid, kind="parent"})
    for cd in child_sp.split_text(seg.text):        # 800 字细切
        child = Chunk(page_content=cd.text,
                      metadata={..., heading_path=seg.hp, parent_id=pid, kind="text"})
        children.append(child)
parents.append(parent)

return {"children": children, "parents": parents}
```

**为什么合并而非直接 header 拆**：合并保证"八 + 8.1 + 8.2 + 8.3"作为整体成为父块，命中任一子块都能回退整节。

**取舍（heading_path 变粗）**：child 的 `heading_path` 取父段所在章节（如"八、最佳实践"），丢失了"8.1"这一级细节。这是"整节返回"的必要代价——子块语义检索与结构分 boost 仍按章节级生效，不影响排优先逻辑。

---

## 五、检索流程（HybridRetriever）

```
用户 query
  → embedding → ChromaDB(dense)   ┐
  → 分词       → BM25(sparse)     ├─→ _merge_results（α·dense + (1-α)·sparse + β·structural）
  → 结构分(机制B)                  ┘            │
                                             ▼
                                   merged[:top_k]  (仍是 child)
                                             │
                                  _expand_to_parents  ← 仅 chunking_strategy=="v2b" 且 docstore 非空时生效
                                             │
                                  对每个 child：按 parent_id 取 docstore 父块正文
                                  同 parent_id 去重（保留最高分 child 的分数）
                                  page_content 替换为父块整节正文，metadata 用父块 metadata
                                             │
                                  返回 parent 级结果 → agent reranker（在父块正文上重排）→ LLM
```

- **零回归保证**：`chunking_strategy != "v2b"`（v1/v2）时 `_expand_to_parents` 直接透传 child，行为完全等同改造前。
- **去重副作用**：top_k 个 child 若来自同一父块，展开后数量会 < top_k（返回更少但更完整的整节）。这是预期行为。

---

## 六、配置项（config.py）

| 字段 | 默认 | 说明 |
|---|---|---|
| `chunking_strategy` | `"v2"` | Literal 增加 `"v2b"`；启动前切换，改后重建向量库生效 |
| `parent_chunk_size` | `2000` | 父块（整节）最大字符数 |
| `parent_chunk_overlap` | `200` | 父块递归切分时的重叠 |
| `parent_docstore_path` | `./data/docstore.pkl` | 父块存储路径 |

切换方式（沿用"启动前切换"约定，非运行时热插拔）：在 `.env` 设 `CHUNKING_STRATEGY=v2b`，再执行全量重索引（见第九节）。

---

## 七、与现有机制的关系

- **与结构检索（机制 A/B）正交、横切其上**：child 仍带 `heading_path` 前缀 + 结构分 boost，v2b 只是把"返回内容"从 child 换成 parent。两者叠加 = 层级优先 + 整节返回。
- **与 v3（chunk 级 wikilink）正交**：本次未触碰 wikilink 粒度，仍整篇级。
- **与 v1/v2 切换共存**：`chunking_strategy` 三态可选，v2b 是其中一项；v1/v2 行为不变。

---

## 八、边界与取舍

1. **父块可能仍较大**：单 h1 章节超 `parent_chunk_size` 时，父块会被递归切成多个 bounded 段，child 映射到对应段——此时"整节"退化为"≤2000 字段"，但已远好于 800 字碎片。
2. **跨 h1 不合并**：父段不跨越顶层章节，避免把两章拼在一起。
3. **docstore 必须与 ChromaDB 同步重建**：切换 v2b 后必须重索引，docstore 才建立；否则 `_docstore` 为空，graceful 降级为返回 child。
4. **reranker 在父块正文上重排**：质量更好（整节上下文），但重排输入变长，耗时略增。
5. **存储成本**：docstore 额外占磁盘（纯 pickle，无 embedding，成本远低于向量）。

---

## 九、实现步骤

| 文件 | 动作 |
|---|---|
| `config.py` | `chunking_strategy` Literal 加 `"v2b"`；新增 `parent_chunk_size` / `parent_chunk_overlap` / `parent_docstore_path` |
| `retrieval/docstore.py` | **新建** `ParentDocstore`：add/get/save(pickle)/load |
| `indexing/splitter.py` | `split_v2b` 由 stub 实现为上述算法，返回 `{"children","parents"}` |
| `indexing/ingestor.py` | `index_vault` 分支 v2b：取 children+parents → restore → 补 metadata（children 加结构前缀，parents 不加）→ upsert children 到 ChromaDB → parents 写入 docstore 并 save；非 v2b 时清 stale docstore |
| `retrieval/hybrid.py` | `__init__` 加载 docstore；新增 `_expand_to_parents`；在 `search`/`search_with_trace`/`vector_search`/`bm25_search`/`filtered_search` 返回前调用（仅 v2b 生效） |
| `tests/indexing/test_splitter_v2b.py` 等 | 拆分逻辑单测 + docstore 往返 + hybrid 展开去重 |

重索引命令（切换 v2b 后必须执行）：

```bash
# .env 设 CHUNKING_STRATEGY=v2b 后
python -m note_assistant.indexing.ingestor          # 重建 ChromaDB + 生成 docstore
python -c "from note_assistant.retrieval.hybrid import HybridRetriever; HybridRetriever().build_bm25_from_chroma()"
```

---

## 十、测试与验证

- `test_splitter_v2b`：child 文本 ⊆ 其 parent 文本；parent_id 唯一；child/parent `heading_path` 一致。
- `test_docstore`：pickle save/load 往返一致。
- `test_hybrid_v2b`：mock ollama 下 `_expand_to_parents` 正确用父块正文替换、同 parent 去重、非 v2b 透传。
- 全量 `tests/indexing tests/retrieval tests/test_config.py` 无回归（默认 v2 行为不变）。

---

## 十一、风险

| 风险 | 缓解 |
|---|---|
| 父块过长撑爆 LLM 上下文 | `parent_chunk_size` 封顶 + 超长章节递归切；agent 侧 `agent_obs_token_budget` 仍有截断兜底 |
| 切换 v2b 后忘重索引 → docstore 空 | `_docstore` 为空时 graceful 降级为 child，不报错 |
| 父段跨 h1 拼错章节 | 合并仅在同 h1 内进行 |
| 与已有索引/BM25 不同步 | 重索引统一重建 ChromaDB + docstore + BM25 |

---

## 十二、实现记录（2026-07-29）

已实现并通过测试（代码改动，零回归）：

- **改动文件**：`config.py`（枚举加 `v2b` + 三个父块参数）、`retrieval/docstore.py`（新建 `ParentDocstore`）、`indexing/splitter.py`（`split_v2b` 由 stub 实现）、`indexing/ingestor.py`（v2b 分支 + docstore 落库）、`retrieval/hybrid.py`（`_expand_to_parents` 展开 + 5 个公开检索方法接入）。
- **测试**：`tests/indexing/test_splitter_v2b.py`（7）、`tests/retrieval/test_docstore.py`（2）、`tests/retrieval/test_hybrid_v2b.py`（4），合计 13 个；叠加原有用例 `tests/indexing tests/retrieval tests/test_config.py` 全量 **134 passed**。
- **实测行为**：child 文本 ⊆ parent 文本；parent_id 唯一；同 h1 内合并成整节父块；超长章节递归切父段；`_expand_to_parents` 在非 v2b / docstore 缺失时自动透传（零回归）。

### 实现中踩的坑

1. **`.env` 残留非法值 `CHUNKING_STRATEGY=v32`**：导致 `Settings()` 校验失败、所有 import config 的测试崩。已复位为默认值 `v2`（不改行为）。若你想启用 v2b，需显式设为 `v2b` 并重索引。
2. **`RecursiveCharacterTextSplitter.split_text` 返回 `List[str]`**（非 `Document`）：初版误用 `.page_content`，已改为直接用字符串。
3. **父块 heading_path 取"所在章节"而非"细分小节"**：这是"整节返回"的必要代价（见第四节取舍）。

### 如何启用

```bash
# .env 设 CHUNKING_STRATEGY=v2b
python -m note_assistant.indexing.ingestor          # 重建 ChromaDB + 生成 data/docstore.pkl
python -c "from note_assistant.retrieval.hybrid import HybridRetriever; HybridRetriever().build_bm25_from_chroma()"
```

> 注意：启用 v2b 会切换检索返回内容为"整节父块"，需重索引且 BM25 同步重建；评估效果建议用 RAGAS 对比 v2 vs v2b 的 answer_correctness / faithfulness。

---

## 十三、端到端验证记录（2026-07-29，切到 v2b 并真实重索引）

`.env` 设 `CHUNKING_STRATEGY=v2b` 后，真实 vault（288 篇）重索引 + 重建 BM25，验证通过：

- **ChromaDB**：9967 children（含 `parent_id` 元数据），BM25 重建（31MB，`data/bm25.pkl`）。
- **docstore**：2523 父块（整节正文，`data/docstore.pkl`）。
- **v2b 展开生效**：结构 query `Agent Skills 的 Skills 核心特点是什么` 命中 child → 返回整节父块（`page_content` 长度 1824/1366，与 docstore 父块**逐字一致**），不再是 800 字碎片。
- **结构分机制正常**：强结构 query 命中（标题命中段 score 高）；弱匹配（如纯内容 query `什么是按需加载机制`，struct=0.3 < 门限 0.5）被 gate 掉、不 boost，退化为 dense+sparse。
- **说明**：日志尾部出现一次 `Collection does not exist` 的 traceback，系 `wipe=True` 删旧集合+重建时的旧句柄残留（缓冲错序），最终 `INGEST_DONE`/`BM25_DONE` 均正常打印，当前集合 9967 + BM25 31MB，确认未影响结果。

> 结论：v2b 父子双存已端到端跑通——"小检索、大返回"成立，命中章节时返回整节上下文，解决了 v2 下长章节被切碎、答案不完整的痛点。

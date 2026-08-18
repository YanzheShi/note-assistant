# 自动增量索引（auto-reindex）设计方案

> 状态：✅ 已实现 | 日期：2026-08-10 | 关联：`indexing/sync.py` · `scripts/reindex.py` · `indexing/ingestor.py` · `api/main.py` · `config.py`
> 前置能力：[父子双存切分（v2b）设计方案.md](./父子双存切分（v2b）设计方案.md)（索引链路）· SyncDB 增量检测（mtime + sha256 双保险，已实现）

---

## 一、背景与现状

Obsidian vault 是持续演进的笔记库，笔记增删改后索引必须跟上，否则检索/回答读到的是旧内容。

**现状盘点**：

| 能力 | 状态 |
|---|---|
| 增量检测 | ✅ `SyncDB.need_reindex`（sync.py:72）mtime + sha256 双保险，秒级 |
| 增量执行 | ✅ `scripts/reindex.py::incremental_reindex()`，per-doc 流程与全量 `index_vault` 逐字节对齐（图片 enricher / v2b docstore / 结构前缀全通） |
| 触发方式 | ❌ 仅手动：`uv run python scripts/reindex.py` 或 `POST /reindex`（api/main.py:530） |
| 自动监听 | ❌ 无 watcher、无轮询、无后台任务 —— **本次要补的缺口** |
| 已知欠账 | ⚠️ 增量不重建 BM25（`data/bm25.pkl`）与 WikiGraph（`data/bm25.graph`），长期增量后两者滞后于 ChromaDB（reindex.py 头部注释自述） |

**目标**：笔记保存后，系统**自动**把变更同步进索引，全程无人工触发、不打断问答服务。

---

## 二、触发方式选型（为什么是事件监听，不是轮询）

| 方案 | 实时性 | 可靠性 | 复杂度 | 结论 |
|---|---|---|---|---|
| **文件系统事件监听**（watchfiles） | 秒级，Windows 底层 `ReadDirectoryChangesW` 内核回调，零轮询 | 本地 vault 可靠 | 低（watchfiles 已随 `uvicorn[standard]` 附带，零新依赖） | ✅ **唯一触发方式** |
| **Obsidian 插件主动推送** | 事件级精确 | 依赖第三方插件生态 | 高（多一个移动部件） | ❌ 二期可选，本期不做 |
| 轮询扫描（定时跑 need_reindex） | 分钟级 | 全场景可靠，但纯本地部署无必要 | 低 | ❌ 不做（无云盘场景） |
| 定时全量重建 | 落后一个周期 | 最稳但最贵 | 低 | ❌ 与增量能力重复，仅作二级索引校准 |

**结论**：vault 为本地目录，**只有 watchfiles 事件监听一条触发路径**，无轮询兜底。所有触发（watcher / 手动 API）由同一个「单飞队列」统一收口。

---

## 三、总体架构

```
                        ┌──────────────────────────────────────────────┐
                        │            AutoIndexService                   │
                        │  (src/note_assistant/indexing/autoindex.py)  │
                        │                                              │
  vault/*.md 变更 ─────►│  watchfiles.awatch()                         │
  (另存/增删/移动)        │    │ 过滤 .md + 排除隐藏目录                   │
                        │    │ debounce 合并 (默认 3s)                  │
                        │    ▼                                         │
                        │  change_queue (asyncio.Queue)                │
                        │    ▼                                         │
                        │  worker 任务（单飞，asyncio.Lock 串行）        │
                        │    │                                          │
                        │    ├─ modified/added → 单文件 reindex        │
                        │    ├─ deleted        → 删 chunks + sync 状态  │
                        │    └─ 批量合并窗口     → 一次批量 reindex      │
                        └────────┬─────────────────────────────────────┘
                                 │
                                 ▼
              incremental_reindex(filepaths=[...])        ← 复用现有 per-doc 流程
              （从 scripts/reindex.py 抽到包内 indexing/reindex.py）
                                 │
                                 ▼
        ChromaDB upsert/delete + SyncDB update + [v2b docstore]
                                 │
                                 ▼
              二级产物校准（可选）：累计 N 次增量后触发
              full_reindex（重建 BM25 + WikiGraph + docstore 全量）
```

**设计原则**：

1. **变化检测与执行解耦**：watch 只负责「发现变化、去抖、排队」；执行一律走 `incremental_reindex`，保证与手动 `/reindex`、`index_vault` 完全同一条流程，不复制第三套索引逻辑。
2. **单飞串行**：所有触发（watcher / 手动 API）汇入同一个队列，`asyncio.Lock` 串行执行，杜绝并发写 ChromaDB / Ollama embedding 抖动 / sync.db 竞争。
3. **单文件优先**：watcher 事件带精确路径，只重索引该文件（per-doc 流程天然支持），比整库扫描快一个量级。
4. **零回归约定**：`autoindex_enabled=False`（默认）时行为与现状逐字节等价（沿用项目 G6 式零回归约定）；开启后只多出自动触发，索引产物本身不变。

---

## 四、模块设计

### 4.1 新增 `src/note_assistant/indexing/autoindex.py`

```python
class AutoIndexService:
    """自动增量索引服务：watchfiles 事件监听 + debounce + 单飞队列。"""

    def __init__(self, vault_path, *, enabled: bool, ...): ...

    async def start(self) -> None    # 创建 watcher 任务 + worker 任务
    async def stop(self) -> None     # 取消任务，等待队列排空（优雅退出，不丢变更）
    async def enqueue(self, changes) -> None          # 事件去抖窗口合并后入队
    async def _worker(self) -> None                   # 单飞消费队列，串行执行 reindex
    # 状态（供 /reindex/status 观察）
    @property def stats(self) -> dict: ...  # {queue_len, running, last_run_at, last_run_files, total_runs, errors}
```

**事件处理规则**（`watchfiles.Change`）：

| 事件 | 处理 |
|---|---|
| `added` / `modified`（.md） | 加入待索引集合（去抖窗口内合并），非 .md 忽略（.obsidian / .trash / 隐藏路径按 `VaultLoader.scan` 同规则排除，vault_loader.py:19） |
| `deleted`（.md） | 加入待删除集合；批量执行时先删 ChromaDB chunks + `sync.remove_state` |
| 目录事件 / 其他扩展名 | 忽略，不触发任何索引成本 |

**去抖策略**：Obsidian 保存一次会触发多个事件（编辑器 flush 分段写）。collect 窗口默认 3s：窗口内的事件累积成一个批次，窗口结束入队一次。批量大于阈值（如 ≥5 个文件）或有目录结构大改动时可降级为**整库增量扫描**（直接调现有增量逻辑），避免 per-file 循环放大。

### 4.2 抽包：`src/note_assistant/indexing/reindex.py`

把 `scripts/reindex.py` 的 `incremental_reindex()` 迁入包内（函数体不变），并**增强一个参数**：

```python
def incremental_reindex(vault_path=None, *, filepaths: list[str] | None = None) -> dict:
    """filepaths 为 None → 全库变更比对（现状）；指定 → 只处理这些相对路径（watcher 单文件路径）。"""
```

- 单文件模式：跳过 `loader.load_all()` 全量扫描，直接 `loader.load_file` 指定文档 → 走原有 per-doc 流程（删旧 → 预处理 → 切分 → 入库 → sync.update_state）。
- 删除模式：`deleted_filepaths` 参数直接 `collection.delete(where={"filepath": ...})` + `sync.remove_state`。
- `scripts/reindex.py` 留薄壳：`from note_assistant.indexing.reindex import incremental_reindex`，CLI 入口不变。
- `api/main.py::/reindex` 改为 import 包内版本（行为不变）。

### 4.3 `config.py` 新增配置

```python
# === 自动增量索引（docs/自动增量索引（auto-reindex）设计方案.md）===
autoindex_enabled: bool = False        # 总开关，默认关闭（零回归约定）
autoindex_debounce_seconds: float = 3.0   # 事件收集窗口（Obsidian 保存多事件合并）
autoindex_batch_fallback_threshold: int = 5   # 单批待处理 ≥ 该值 → 降级整库增量
autoindex_full_sync_every: int = 20       # 每累计 N 次增量 → 触发一次全量校准（二级索引重建）
```

开关与去抖窗口写 `.env` 或 `.env.local`。

### 4.4 `api/main.py` 集成

- **FastAPI lifespan**：`autoindex_enabled=True` 时 `start()`，shutdown 时 `stop()`（等待队列排空再关，避免服务退出丢变更）。
- **新增端点**：

```
GET  /reindex/status  → {enabled, queue_len, running, last_run_at, last_run:{files,chunks}, total_runs, errors}
POST /reindex/run     → 手动触发一次增量（等价原 /reindex，保留原端点兼容）
```

- `/agent/ask*` 等问答路径不改：ChromaDB 读写分离，索引写入与检索查询天然兼容（ChromaDB 支持并发读 + 写）。

### 4.5 二级索引校准（BM25 / WikiGraph 滞后问题的收口）

现状欠账（reindex.py 注释自述）：增量更新 ChromaDB 后，`data/bm25.pkl` 与 `data/bm25.graph` 不跟随。

**本期方案**：计数器 `sync.db` 或内存累计增量次数，达到 `autoindex_full_sync_every`（默认 20 次）后自动在低峰（队列空时）触发一次全量刷新（复用 `scripts/full_reindex.py` 的两步重建 + `index_vault` 幂等逻辑），并重置计数。开关 `autoindex_full_sync_every=0` 可完全关闭校准。

**后续可选**（本期不做，写进清单）：BM25 按文件增量更新（删除旧文件 token 明细 + 插入新文件），WikiGraph 同理——收益是免全量，成本是两套增量逻辑维护。

---

## 五、执行流程（时序）

### 5.1 单文件变更（主路径）

```
1. Obsidian Ctrl+S → OS 文件事件（ReadDirectoryChangesW）
2. watchfiles awatch() 收到 (added|modified, path)
3. 过滤：.md + 非隐藏目录 → 丢进 3s debounce 窗口
4. 窗口结束 → enqueue([filepath])（相对 vault 根）
5. worker 取队列（lock 保护，串行）：
   a. sync.need_reindex(filepath) 复核（mtime/sha256，防重复事件空转）
   b. incremental_reindex(filepaths=[filepath])
       - loader.load_file → preprocess → split（按 chunking_strategy）→ restore
       - 补 metadata + 结构前缀 → upsert（先删旧 chunk）→ docstore 父块（v2b）
       - sync.update_state
   c. 更新 stats；若累计增量次数达阈值 → 排入全量校准
6. 结束。全程 < 数秒，问答服务不中断
```

### 5.2 文件删除

```
watcher 收到 deleted.md → 队列 → worker：
  collection.delete(where={"filepath": path}) → sync.remove_state → [v2b] docstore 清理该文件父块
```

### 5.3 手动触发

`POST /reindex`（兼容保留）与 `POST /reindex/run` 同样汇入 worker 队列，与自动触发互斥，不会并发写库。

---

## 六、并发与一致性保障

| 风险 | 对策 |
|---|---|
| watcher 与手动 API 并发 reindex | 全局 `asyncio.Lock` 单飞队列，任何时刻只有一个索引执行者 |
| Obsidian 高频保存（连续 Ctrl+S） | debounce 合并 + `need_reindex` 复核（mtime 未变 → 跳过），幂等 |
| 索引中途文件再次变更 | per-doc 流程「先删旧再写新」，最终以最新内容为准；sync.db 存的是最后处理时的 hash |
| 服务重启丢失变更 | watcher 启动瞬间会收到 `watchfiles` 的初始全量扫描事件（RustNotify watch 启动即 emit 当前文件树），天然补差；无需额外机制 |
| 队列积压（连续大改） | batch fallback：积压 ≥ `autoindex_batch_fallback_threshold` → 降级整库增量，一次收敛 |
| 失败/异常 | worker 捕获异常记入 `stats.errors`，不退出；单个文件失败不阻塞后续文件 |
| 二级索引滞后 | 累计增量阈值触发全量校准（4.5） |

---

## 七、文件清单与改动点

| 文件 | 动作 | 说明 |
|---|---|---|
| `src/note_assistant/indexing/autoindex.py` | ✨ 新增 | WatchService：awatch + debounce + 单飞队列 + stats |
| `src/note_assistant/indexing/reindex.py` | ✨ 新增 | `incremental_reindex` 从 scripts/ 迁入，新增 `filepaths`/`deleted_filepaths` 参数与 per-file 分支 |
| `scripts/reindex.py` | ✏️ 改 | 变薄壳，import 包内版本，CLI 不变 |
| `src/note_assistant/config.py` | ✏️ 改 | 新增 4.3 节配置组 |
| `src/note_assistant/api/main.py` | ✏️ 改 | lifespan 启停服务；新增 `GET /reindex/status`、`POST /reindex/run`；`/reindex` 改 import 包内 |
| `src/note_assistant/api/schemas.py` | ✏️ 改 | 新增 ReindexStatusResponse |
| `pyproject.toml` | ✏️ 改 | `watchfiles` 从传递依赖提升为显式依赖（`uvicorn[standard]` 已带，仅声明不增包） |
| `tests/test_autoindex.py` | ✏️ 新 | 见 §八 |
| `.env.local.template`（如存在） | ✏️ 改 | 样例配置 |

**不改动**：`ingestor.py`（upsert/delete 已够用）、`sync.py`（检测逻辑已完备）、检索/生成/agent 全链路（零影响）。

---

## 八、测试方案

| 用例 | 内容 |
|---|---|
| debounce 合并 | 模拟 3s 内 5 个事件 → 只入队 1 次 |
| 过滤规则 | `.obsidian/x.md`、`.trash/x.md`、图片/附件改动 → 不触发；`foo.md` → 触发 |
| 单文件增量 | mini vault 改 1 篇 → 队列执行后 ChromaDB 中该文件 chunks 数量与内容正确，其他文件 untouched（checkpoint count 不变） |
| 删除 | 删 1 篇 → collection 无该 filepath 残留，sync.db 状态清除 |
| 幂等 | 相同内容重复触发 → `need_reindex` 拦截，零 embedding 调用 |
| 并发互斥 | watcher 触发 + 手动 `/reindex` 并发 → 串行执行，无异常 |
| 服务重启补差 | 重启后 watcher 初始扫描事件触发全库比对 → 只处理期间变更的文件 |
| 全部走 `tmp_path` mini vault + monkeypatch 关闭 Ollama 调用（stub embedder），沿用现有测试基建 |

---

## 九、部署与运维

- **启动**：`autoindex_enabled=True` 后正常 `uv run uvicorn ...` 即在 lifespan 内自启；独立运行不需要额外进程。
- **日志**：每次自动索引一行结构化日志（`[autoindex] files=3 removed=1 duration=2.1s`），并入现有 python-json-logger。
- **观测**：`GET /reindex/status` 可查队列/上次执行/错误计数；前端可加状态徽标（二期）。

---

## 十、二期候选（本期明确不做）

1. **Obsidian 插件主动推送**：保存时 POST `filepath` 到本地，事件级精确、省文件扫描（多一个插件依赖）。
2. **BM25 / WikiGraph 按文件增量更新**：消除定期全量校准。
3. **前端索引状态面板**：Streamlit 显示最近同步时间/变更数/错误。
4. **重命名/移动优化**：现按 delete+add 处理（正确但多一次 delete-all 成本），可优化为 metadata 更新。

---

## 附录：与现有手动链路的兼容性

| 场景 | 行为 |
|---|---|
| `autoindex_enabled=False`（默认） | 与现状完全一致：手动命令行 / `POST /reindex` 照旧 |
| watcher 开着同时手跑 reindex | 单飞锁保证串行，手动请求或自动请求先到先执行 |
| 全量 `index_vault` 后 | full reindex 重置 sync.db（`index_vault` 当前不写 sync.db，此为既有行为边界，本次不改，另列待办：全量索引后应同步刷新 sync.db 全部状态，否则自动索引会把全库当变更重复处理） |

> **发现并列入待办的一个既有缺口**：`index_vault`（全量）不更新 sync.db，导致全量索引后首次自动/手动增量会把全库误判为「新文件」重索引一遍。建议在本次实现中顺带修复（全量后 `sync.update_state` 整库）。
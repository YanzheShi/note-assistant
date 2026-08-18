# 评测框架扩展方案：多轮剧本 + Token 统计（含 LLM 缓存命中）+ 语义缓存命中

> 目标：在现有 `evaluation/` 评测框架上，补齐三块缺失能力——**多轮对话自动化**、**Token 使用统计（含 LLM 网关 token 缓存命中）**、**语义缓存命中统计**——同时保留已有的检索/生成质量指标（MRR / recall@k / ROUGE / faithfulness / RAGAS 等）。
>
> 修订说明（v2）：Token 捕获方式从「手动逐点读 `usage_metadata`」升级为「**零侵入 LangChain callback handler**」，并明确纳入 **LLM 网关 token 缓存命中**（`cache_read`/`cache_creation`）。参考实现见 `code-tutor-agent/src/code_tutor_agent/token_usage/`（`callback.py` / `sink.py`）。

---

## 0. 两种「缓存命中」必须区分（关键澄清）

用户问的「缓存命中是否统计了 LLM TOKEN 那个缓存命中」——答案是：这是**两层不同的缓存**，本方案**两个都报**，但口径不同：

| 维度 | 语义缓存 SemanticCache | LLM 网关 token 缓存（prompt caching / KV cache） |
|---|---|---|
| 缓存的是什么 | 整条问答结果（answer+sources+trajectory） | 重复的 prompt 前缀 token |
| 命中后 | 从内存返回，**跳过整条 graph**（检索+rerank+生成 LLM） | 仍调 LLM，但前缀 token 不重算，计费/延迟更低 |
| 命中率口径 | `hits/(hits+misses)`（问答级） | `cache_read_tokens / prompt_tokens`（token 级） |
| 本系统现状 | ✅ 有 `SemanticCache.stats()`，方案接进评测 | ❌ 之前未捕获；**v2 新增**（网关已确认支持返回 `cached_tokens`） |

两者正交、互补：SemanticCache 命中省的是「整条链路」，LLM token 缓存命中省的是「同一链路里 prompt 前缀的重复计算」。报告里分别用 `semantic_cache_stats` 与 `llm_cache_hit_rate` 两个字段呈现。

---

## 1. 现状盘点（对齐诉求）

| 诉求 | 现状 | 落点 |
|---|---|---|
| 自动化提问/回答（批量） | ✅ 已成熟 | `Evaluator.run()` 遍历 `EvalDataset` 逐条 `rag_chain.ask()`；内置 10 条 + JSON 自定义；入口 `scripts/run_eval.py` |
| 回答质量分析 | ✅ 已成熟 | 检索侧 MRR/recall@k/precision@k/nDCG@k；生成侧 ROUGE-L/BLEU-1·4/语义相似度/faithfulness/answer_relevance；RAGAS 模式 |
| **多轮进行（上下文串联）** | ❌ 缺失 | `Evaluator` 每条问题独立 ask，无 turn 序列；`RAGChain.ask(question, history=...)` 与 `runner.ainvoke(question, session_id)` **已支持多轮**，但没接进评测 |
| **Token 使用分析** | ❌ 缺失 | 三处 LLM 调用（路由、`generator.generate/stream`、agent 节点）均未捕获 `usage_metadata`；报告无 token 字段 |
| **LLM token 缓存命中** | ❌ 缺失 | 网关已支持返回 `cached_tokens`，但代码从未读取/统计 |
| **语义缓存命中分析** | ⚠️ 半有 | `SemanticCache.stats()` 已能出 hits/misses/hit_rate，仅 agent 链路使用；评测从未调用 |

**关键结论**：多轮和 Token 截获的「基座」已经在生产链路里（`history` / `session_id` / 全局 `SemanticCache` / 网关 `usage`）。我们只要在**评测层**编排起来，并在**LLM 调用层**用一个零侵入 callback 把已返回的 `usage_metadata` 接住即可。

---

## 2. 参考实现（code-tutor-agent token_usage）

code-tutor-agent 已落地一套成熟、零侵入的 token 计量，本方案直接复用其思路：

- **`callback.py::TokenUsageCallbackHandler(BaseCallbackHandler)`**：重写 `on_llm_start`（按 `run_id` 缓存 metadata 归因）与 `on_llm_end`（提取 usage）。`on_llm_end` 从 `response.generations[0][0].message.usage_metadata` 优先取，回退 `response.llm_output.usage` / kwargs 里的 `usage`。
- **cache token 提取**（核心，`_extract_cache_tokens` + `_normalize_openai_usage`）：兼容三种厂商字段——
  - LangChain ≥0.3 标准：`input_token_details.cache_read`（OpenAI `prompt_tokens_details.cached_tokens` 映射到此）
  - 旧版 LangChain 顶层：`cache_read_input_tokens` / `cache_creation_input_tokens`
  - DeepSeek 风格顶层：`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
  产出 `cache_creation_tokens`（首次写入缓存）与 `cache_read_tokens`（命中读取）。
- **挂载方式**：`get_llm(purpose)` 用**构造参数** `metadata={purpose, model_alias, model_name}` 注入归因维度（因 `bind_tools`/`with_structured_output` 会丢弃 `with_config` 的 config，必须走构造参数）；callback handler 单例则挂到 `graph.invoke(config={"callbacks":[handler]})` 层（config 层 callbacks 不被 bind_tools 丢弃，故能覆盖整条链路）。
- **落库**：`sink.py` 异步队列 + 后台线程批量写 DB（note-assistant 评测不需要生产落库，仅用进程内 `TokenMeter` 累加）。

---

## 3. 总体设计

### 3.1 Token 截获：零侵入 callback（替代原方案的手动逐点读）
新增 `src/note_assistant/llm/usage.py`，含：
- `TokenMeter`：进程内累加器，字段 `prompt_tokens / completion_tokens / cache_creation_tokens / cache_read_tokens / total_tokens / llm_calls`。
- `TokenUsageCallbackHandler`：**移植** code-tutor-agent 的提取逻辑（含上述三种 cache token 字段来源），`on_llm_end` 把 usage 累加进 `self.meter`（`meter=None` 时直接返回，零副作用）。

模块级持有全局 handler 单例 `_token_handler`，由 `get_token_handler()` 返回；评测时 `get_token_handler().set_meter(meter)` 绑定，`clear` 后恢复 `None`。

### 3.2 注入点（3 处，业务代码几乎零改）
1. `llm/client.py::get_llm()`：返回 `base.with_config(callbacks=[_token_handler])`（每次新实例，不影响 `lru_cache` 缓存的基础模型）。覆盖路由等直接用 `get_llm` 的调用。
2. `generation/generator.py::__init__`：在 `(llm or init_chat_model(...))` 后 `.with_config(callbacks=[_token_handler])`。覆盖生成路径（naive target 主链路）。
3. `agent/runner.py::graph.ainvoke(state, config={"callbacks":[_token_handler]})`：**双保险**，覆盖 agent 链路中 `bind_tools` 后的模型节点（这些节点会丢弃 `with_config` 的 callback，但 config 层的不丢）。

> 同一全局 handler 实例不会重复计数（一个 LLM 调用只触发一次 `on_llm_end`，LangChain 对同实例 handler 去重）。

### 3.3 评测 target 可插拔（鸭子类型）
- **naive target** —— `RAGChain.ask(question, history=history)`，多轮靠 `history` 列表串联。无语义缓存可测。
- **agent target（默认推荐）** —— `runner.ainvoke(question, session_id=...)`，多轮靠 `session_id` 串联，且自带 `SemanticCache`，可产出 `semantic_cache_stats`。

### 3.4 数据集升级
`EvalQuestion` 增加可选 `turns: List[EvalTurn]`。`turns=None` 退化为现有单轮（向后兼容，内置 10 条数据集照常工作）。

### 3.5 报告扩展（EvalReport）
- `token_usage_total`：`{prompt_tokens, completion_tokens, cache_creation_tokens, cache_read_tokens, total_tokens, llm_calls}`
- `llm_cache_hit_rate`：`cache_read_tokens / prompt_tokens`（token 级，网关 prompt caching）
- `semantic_cache_stats`：`{hits, misses, hit_rate, size, enabled}`（仅 agent target；naive 为 `null`）
- `per_conversation`：多轮每轮明细（turn / question / answer_len / retrieval / generation / token_usage）

---

## 4. 详细改动清单

### 4.1 新增 `src/note_assistant/llm/usage.py`
```python
@dataclass
class TokenMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def add(self, *, prompt=0, completion=0, cache_creation=0, cache_read=0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cache_creation_tokens += cache_creation
        self.cache_read_tokens += cache_read
        self.total_tokens += prompt + completion
        self.llm_calls += 1

    def cache_hit_rate(self) -> float:
        return round(self.cache_read_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def to_dict(self) -> dict: ...

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """移植 code-tutor-agent：on_llm_end 提取 usage（含 cache token）累加进 self.meter。"""
    raise_error = False
    def __init__(self): self.meter = None
    def set_meter(self, m): self.meter = m
    def on_llm_end(self, response, *, run_id=None, **kwargs):
        if self.meter is None: return
        um = self._extract_usage(response, kwargs)   # 同参考：message.usage_metadata → llm_output.usage → kwargs
        if not um: return
        self.meter.add(
            prompt=um.get("input_tokens", 0) or um.get("prompt_tokens", 0),
            completion=um.get("output_tokens", 0) or um.get("completion_tokens", 0),
            cache_creation=_cache_creation(um),
            cache_read=_cache_read(um),
        )
    # _extract_usage / _cache_read / _cache_creation 同参考实现（兼容三种来源）
```
模块级：`_token_handler = TokenUsageCallbackHandler()`；`get_token_handler()` 返回它。

### 4.2 改造 `llm/client.py`
`get_llm()` 末尾 `return init_chat_model(**kwargs)` → `return init_chat_model(**kwargs).with_config(callbacks=[_token_handler])`。（`_token_handler` 从同模块 `usage` 导入，避免循环依赖：`usage` 不依赖 `client`。）

### 4.3 改造 `generation/generator.py`
`__init__`：`self.llm = llm or init_chat_model(...)` 之后 `self.llm = self.llm.with_config(callbacks=[get_token_handler()])`。`generate/stream` 业务逻辑无需改动（usage 已在 callback 层捕获）。

### 4.4 改造 `agent/runner.py`
`graph.ainvoke(state, config={"callbacks":[get_token_handler()]})`（非流式），以及 `graph.astream(state, config=..., stream_mode="updates")` 同步加 config。`get_cache_stats()` 新增 public 包装 `_get_cache().stats()`；评测每题前 `reset_cache()`。

### 4.5 数据集升级 `evaluation/eval_dataset.py`
```python
@dataclass
class EvalTurn:
    question: str
    golden_answer: str = ""
    relevant_files: List[str] = field(default_factory=list)

@dataclass
class EvalQuestion:
    question: str
    golden_answer: str = ""
    relevant_files: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    turns: Optional[List[EvalTurn]] = None
    def is_multiturn(self) -> bool: return bool(self.turns)
```
`EvalDataset.save/load` 的 JSON 序列化自动兼容嵌套 `turns`。

### 4.6 改造 `evaluation/evaluator.py`
- `__init__(self, ask_target, target_kind="agent", use_ragas=False)`：`ask_target` 鸭子类型；`target_kind` 决定多轮串联（history vs session_id）及是否收集 `semantic_cache_stats`。
- `run()`：若 `q.is_multiturn()`，维护 `history`/`session_id` 逐轮调用；每轮算 retrieval+generation 指标；末尾取 `meter.to_dict()` 与（agent）`get_cache_stats()` 汇总。否则现有单轮逻辑。
- 报告字段扩展：`token_usage_total`、`llm_cache_hit_rate`、`semantic_cache_stats`、`per_conversation`（含每轮 `token_usage`）。

### 4.7 入口 `scripts/run_eval.py`
- 新增 `--target {naive,agent}`（默认 `agent`）。多轮无需额外 flag。
- 输出新增：Token 汇总（含 `cache_creation`/`cache_read`）、`llm_cache_hit_rate`、`semantic_cache_stats.hit_rate`。
- `--ragas` 保持不变。

---

## 5. 报告结构示例（JSON 片段）
```json
{
  "dataset_name": "builtin_small",
  "total_questions": 10,
  "avg_elapsed_ms": 1820.0,
  "token_usage_total": {
    "prompt_tokens": 48213, "completion_tokens": 9120,
    "cache_creation_tokens": 2100, "cache_read_tokens": 33800,
    "total_tokens": 57333, "llm_calls": 47
  },
  "llm_cache_hit_rate": 0.7012,
  "semantic_cache_stats": { "hits": 3, "misses": 44, "hit_rate": 0.0638, "size": 44, "enabled": true },
  "retrieval_metrics_avg": { "mrr": 0.82, "recall@3": 0.91 },
  "generation_metrics_avg": { "rouge_l": 0.71, "faithfulness": 0.88 },
  "per_conversation": [
    { "turn": 0, "question": "...", "answer_len": 312,
      "retrieval_metrics": {...}, "generation_metrics": {...},
      "token_usage": {"prompt_tokens": 1200, "completion_tokens": 300, "cache_read_tokens": 0, "cache_creation_tokens": 800} }
  ]
}
```

---

## 6. 测试计划
- `tests/evaluation/test_usage.py`：mock `AIMessage(usage_metadata=...)` 验证 `TokenMeter` 累加 + callback 对 **三种 cache token 来源**（langchain `input_token_details.cache_read` / OpenAI `prompt_tokens_details.cached_tokens` / DeepSeek `prompt_cache_hit_tokens`）的兼容提取。
- `tests/evaluation/test_multiturn_evaluator.py`：用 `MagicMock` 作 `ask_target`（验证 `history`/`session_id` 逐轮累积 + meter 被回调累加 + `llm_cache_hit_rate`/`semantic_cache_stats` 落报告）。**不依赖真实 LLM/索引**。
- 既有 `tests/agent/test_cache.py`、`tests/evaluation/test_evaluator.py` 保持绿。

---

## 7. 风险与已知限制
1. **网关 usage 字段来源**：已确认网关支持返回 `cached_tokens`，但实际落在哪种结构需 P1 验证一次（`input_token_details.cache_read` vs `prompt_tokens_details.cached_tokens` vs DeepSeek `prompt_cache_hit_tokens`）。方案已**三种都兼容**，验证只需确认命中非空。
2. **语义缓存命中省 token 的边界**：`SemanticCache` 命中检查在 `_prepare_agent_context`（含 `condense_question` 小 LLM）之后；`condense_question` 有「方案0 透传」优化（问题独立无指代时跳过 LLM，见 `agent/context.py:305`），故多数明确问题不花该步 token，只有含指代的追问多调一次小 LLM（且命中前已发生，不省）。
3. **多轮质量评估**：中间轮通常无 golden answer；建议中间轮用不依赖 golden 的 `faithfulness`（对照 context）+ `answer_relevance`（对照问题），首/末轮再用 ROUGE/BLEU/语义相似度。
4. **agent target 状态管理**：依赖 `AgentStore`（SQLite）+ `session_id`；评测需每题为新 session，并在跑前 `reset_cache()`，避免跨题污染语义缓存命中率。

---

## 8. 实施顺序（phase）
- **P1 探测 + Token 截获**：先用一次真实/沙箱调用确认网关 `usage` 字段来源（见 7.1）；实现 `usage.py`（TokenMeter + Handler）、`get_llm`/`generator`/`runner` 三处注入。→ 单测 `test_usage.py`
- **P2 数据集 + Evaluator 多轮骨架**：`EvalTurn` + `Evaluator.run` 多轮分支 + 报告字段。→ 单测 `test_multiturn_evaluator.py`
- **P3 语义缓存 stats 接入**：`runner.get_cache_stats()` + `reset_cache()` 编排。
- **P4 入口与报告**：`run_eval.py --target` + 输出格式化（含 `llm_cache_hit_rate`）。
- **P5 集成验证**：真实 vault 跑一轮，确认 `cache_read` 非零、`llm_cache_hit_rate` 合理、语义命中率合理、质量指标正常。

每个 phase 完成后立即跑 `uv run pytest` 回归，遵循「小步快跑 + 先 plan 再 phase」。

---

## 9. 处理流程（逐步实施手册）

按 P1→P5 顺序，每步「改什么文件 → 怎么改 → 验证标准」。每步结束跑 `uv run pytest` 全量回归。

### P1 Token 截获（零侵入 callback）
- **新增 `src/note_assistant/llm/usage.py`**：
  - `TokenMeter` 进程内累加器（`prompt/completion/cache_creation/cache_read/total/llm_calls` + `cache_hit_rate()` + `to_dict()`）。
  - `TokenUsageCallbackHandler(BaseCallbackHandler)`：移植 `code-tutor-agent` 三路提取（`message.usage_metadata` → `llm_output.usage` → kwargs），含 `_extract_cache_tokens`（`input_token_details.cache_read` / 旧版顶层键 / DeepSeek `prompt_cache_hit_tokens`）。`meter=None` 时 `on_llm_end` 直接 return（零副作用，不影响线上）。
  - 模块级 `_token_handler` 单例 + `get_token_handler()`。
- **注入点**（业务代码几乎零改，仅挂同一个全局 handler）：
  1. `llm/client.py::get_llm()` 末尾 `init_chat_model(**kwargs)` → `.with_config(callbacks=[get_token_handler()])`（覆盖路由等 `get_llm` 调用）。
  2. `generation/generator.py::__init__` `self.llm = (...).with_config(callbacks=[get_token_handler()])`（覆盖 naive 生成，generator 用 `init_chat_model` 而非 `get_llm`，故单列）。
  3. `agent/runner.py::graph.ainvoke(state, config={"callbacks":[get_token_handler()]})` + `graph.astream(...)` 同步加 config（覆盖 agent 节点；`bind_tools` 会丢 `with_config` 的 callback，config 层兜底）。
  - 三处覆盖不同 LLM 实例，handler 全局单例，同一 run 的 `on_llm_end` 去重，**不会重复计数**。
- **验证**：`tests/evaluation/test_usage.py` 用 mock `LLMResult`（三种 cache 来源各造一条）调 `handler.on_llm_end`，断言 `TokenMeter` 正确累加 + `cache_hit_rate` 正确；`meter=None` 时断言不计。
- **网关探测（P1 前提）**：真实调一次 `get_llm().invoke([...])`，打印 `AIMessage.usage_metadata` 确认 `cached_tokens` 落点（三种已兼容，验证只需命中非空）。

### P2 数据集 + 多轮骨架
- **`evaluation/eval_dataset.py`**：新增 `EvalTurn(question, golden_answer, relevant_files, relevant_chunk_ids)`；`EvalQuestion` 加 `turns: Optional[List[EvalTurn]]` + `is_multiturn()` + `from_dict` 支持嵌套 turns（JSON 序列化 `asdict` 自动兼容）。`turns=None` 退化为现有单轮。
- **`evaluation/evaluator.py`**：
  - `__init__` 加 `target_kind="naive"`（默认 naive 保持向后兼容，不影响现有测试/调用）；鸭子类型 `ask_target`。
  - 新增 `_ask_one(question, history, session_id)`：naive → `ask_target.ask(question, history=history)`；agent → `runner.ainvoke(question, history=history, session_id=session_id, return_contexts=True)`（sync 内用 `asyncio.run` 包装）。统一返回 `(answer, retrieved_files, context)`（naive 用 `preview`、agent 用 `contexts` 完整正文）。
  - `run()` 多轮分支：维护 `history`/`session_id`，逐轮 `_ask_one` + 算检索/生成指标 + 把 `(user, assistant)` 累积进 `history`；每轮记 `turn_index` 与 `token_usage`（meter 前后快照差值）。单轮走原逻辑。
- **验证**：`tests/evaluation/test_multiturn_evaluator.py` 用 `MagicMock` 作 naive `ask_target`（断言 `history` 逐轮累积 + meter 被回调累加 + 报告含 `token_usage_total`/`llm_cache_hit_rate`）；agent 分支 mock `runner.ainvoke` 断言 `session_id` 串联。不依赖真实 LLM/索引。

### P3 缓存 stats 暴露
- **`agent/runner.py`** 新增 `get_cache_stats() -> dict` 包装 `_get_cache().stats()`（返回 `hits/misses/hit_rate/size/enabled/semantic`）。
- **`evaluator.run()`**：agent target 开头 `reset_cache()`（清跨题污染），结尾读 `get_cache_stats()` 写入 `semantic_cache_stats`。
- **验证**：复用 P2 测试 + 既有 `tests/agent/test_cache.py` 仍绿；单测 `get_cache_stats` 返回含 `hit_rate` 字段。

### P4 入口与报告字段
- **`evaluation/evaluator.py::EvalReport`** 加字段：`token_usage_total: Optional[dict]`、`llm_cache_hit_rate: float`、`semantic_cache_stats: Optional[dict]`、`per_conversation: List[dict]`。
- **`scripts/run_eval.py`** 加 `--target {naive,agent}`（默认 `agent`）；输出新增 Token 汇总（prompt/completion/cache_creation/cache_read/total/llm_calls）+ `llm_cache_hit_rate` + `semantic_cache_stats.hit_rate`；报告 JSON 含新字段。
- **验证**：`run_eval.py --help` 见 `--target`；内置 10 条单轮跑通，报告 JSON 含 `token_usage_total`。

### P5 集成验证
- 写/更新测试：`tests/evaluation/test_usage.py`、`tests/evaluation/test_multiturn_evaluator.py`、`tests/evaluation/test_eval_report.py`（报告字段序列化）。
- `uv run pytest` 全量回归（遵循 `--basetemp` 盘符路径规范，避免假红）。
- 真实 vault 跑一轮（需 Ollama + ChromaDB + 真实 LLM）：`uv run python scripts/run_eval.py --target agent`，确认 `cache_read_tokens` 非零、`llm_cache_hit_rate` 合理、`semantic_cache_stats` 合理、质量指标正常。

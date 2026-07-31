# 澄清与反问（Clarification）设计方案

> 状态：设计稿，待评审后实施
> 关联：`agentic-rag-design.md`（图结构）、`context-manager-design.md`（指代消解）

---

## 一、问题背景

当前系统在 `ContextManager.condense_question`（`agent/context.py:203-242`）做了指代消解，
把「它有什么缺点？」改写为独立完整问题。但**消解本质上是"猜"**：

- LLM 被 prompt 强制"若追问本身已完整则原样返回"，它**永远会给出一个答案**，
  不会说"我解不出来"。返回类型是裸 `str`，没有任何置信度或歧义信号。
- 降级路径 `_fallback_query`（`context.py:122-133`）用 `last_entity` 做正则替换，
  只取"最近高分 chunk 的标题末级"，历史里有多个候选实体时是**盲选**。
- 还有一整类歧义指代消解根本管不到：「讲讲 attention」没有代词、语法完整，
  但在知识库里可能对应 3 篇完全不相干的笔记。

猜错的代价是：整条检索链路白跑（3 轮工具调用 + 双层 rerank + 生成），
用户拿到一个答非所问的结果，还得自己重新组织提问。

## 二、现状核查：完全没有反问能力

不是"实现得不好"，是架构前置条件一项都不具备：

| 缺失项 | 位置 | 现状 |
|---|---|---|
| 图中断点 | `agent.py:559` | `g.compile()` 裸调用，无 checkpointer / interrupt |
| Judge 澄清档位 | `agent.py:190-194` | `_norm_verdict` 白名单硬编码四值，非法值**静默降级 sufficient** |
| 等待用户状态 | `agent.py:54-66` | `AgentState` 12 字段无任何 awaiting 语义 |
| SSE 事件类型 | `schemas.py:124-132` | `type` 枚举无 clarify，前端未知类型静默丢弃 |
| run 状态 | `store.py:53-61` | `runs.status` 仅 running/finished/interrupted |

全仓库 `clarify` / `澄清` / `ask_user` / `ambiguous` 零命中；`interrupt` 的全部命中
都是 run 孤儿超时检测，与 human-in-the-loop 无关。9 份设计文档也从未把澄清列入 roadmap。

### 2.1 已生成但被丢弃的歧义信号（白捡）

| 信号 | 产出位置 | 丢弃位置 |
|---|---|---|
| Router `confidence` (0.0-1.0) | `ROUTER_SYSTEM` L110 明确要求输出 | `router` L252-254 只读 needs_search |
| Judge `relevance_score` (0.0-2.0) | `JUDGE_SYSTEM` L130-139 明确要求输出 | `reflect` L353-355 只读 verdict |

### 2.2 前置 Bug：Judge 一直在盲判（必须先修）

`reflect` 节点（`agent.py:342-351`）传给 Judge 的 HumanMessage 是：

```python
f"用户问题：{q}\n\n"
f"当前已收集片段数：{len(state['accumulated'])}\n"     # 只有数量，没有内容
f"已检索轮次：{state['iteration']}/{MAX_ITER}（{label}）\n\n"
"请判断信息是否足够回答（输出 JSON）。"
```

而 `JUDGE_SYSTEM` 第一句是"给定用户问题**与已检索到的知识库片段**，评估这些片段是否足以回答"。
**片段内容从未传入**，`state["messages"]` 里的 ToolMessage 也没带上。
Judge 的 `relevance_score` 完全是幻觉产物——这也解释了为什么 prompt 里要用
"不要因为希望找到更详细的资料就判 need_rewrite"这种重措辞去压它。

本方案的 T2 主题歧义检测要复用这次 Judge 调用，**此 bug 不修则整个 T2 无法成立**。
修法：把 `_top_k_context(state["accumulated"])` 格式化后注入 HumanMessage
（复用现成的 `_format_context`，并按 `agent_obs_token_budget` 截断）。

---

## 三、判断：加反问合理，但不能一刀切

### 3.1 反对意见（先列，因为它们决定了设计形态）

1. **个人知识库场景天然不利于反问。** 用户是笔记的作者，对自己写过什么有 mental model。
   开放域客服的澄清价值 ≫ 个人 KB 的澄清价值。
2. **过度反问是不可逆的信任损失。** 澄清率超过 15% 用户就会形成"这系统很笨"的判断。
3. **直接违背现有产品取向。** `ROUTER_SYSTEM` 连写两遍"宁可检索，不要跳过"、
   "不要做可能判定"。澄清若放在检索前，等于把这个刻意的偏好反过来。
4. **存在两个零成本的更优替代**，很容易被忽略：
   - **猜 + 声明假设**：不问，直接答，开头写"我理解你问的是 X（若非请纠正）"。
     零额外轮次；用户纠错成本与反问相同（都是一轮）。
   - **多解并答**：检索发现两个不相干主题，两个都答，分节输出。零额外轮次。

### 3.2 主张：分层澄清（tiered clarification）

反问是**最后一档**，不是唯一档。

| 档位 | 触发条件 | 处置 | 判定位置 | 额外轮次 |
|---|---|---|---|---|
| T0 | 代词可解，历史锚点唯一 | 自动消解直接答（现状已有） | 检索前 | 0 |
| T1 | 消解置信偏低但检索后有唯一主导主题 | 生成前发 `assumption` 事件声明假设，答案照常流式输出，用户可打断 | **检索后** | 0 |
| T2 | 检索后结果分成 2 簇且主题互不相干 | 多解并答，分节输出 | 检索后 | 0 |
| T3 | 关键槽位缺失，或候选 ≥3 且互斥 | 反问，**必须带选项** | 检索后 | 1 |

**T2 是这套设计里唯一 RAG 独有、也最有价值的一档。**
通用 chatbot 判断歧义只能靠揣摩问题本身；而这里知识库提供了歧义的**客观证据**——
"这个问法在我的笔记里命中了几个不相干主题"是可测量的，不是猜的。

**所有歧义判定一律放在检索之后**（见 3.4），这是本设计的核心约束。

### 3.3 澄清问句的质量比"是否澄清"更重要

开放式反问（"你能说得更清楚一点吗？"）是把负担甩给用户，是最差的澄清。
**必须带选项**，且选项从检索到的真实笔记标题生成，用户点一下即可。
这恰好是 RAG 的天然优势：候选项是现成的。

### 3.4 为什么判定必须放在检索之后（延迟成本核算）

一个常见质疑：RAG 里猜错的代价是**整条链路跑完之后**才暴露，比直接反问慢得多，所以
"猜 + 声明假设"不划算。这个质疑指出了真问题，但结论要修正。

**先算账。** 反问不是免费终点——它后面同样要接一次完整 RAG。所以对比是：

- 猜错 = 2 次完整 run + 用户读完错答案的时间
- 反问 = 1 次完整 run + **用户的思考与操作时间（串行阻塞）**

取实测量级（检索+判定 4s / 生成 12s / 读澄清并点选项 6s / 读完错答案才反应 8s / 看到假设横幅后打断 3s）：

| 策略 | 端到端耗时 |
|---|---|
| 检索前盲猜且猜错 | 40s |
| 检索前就反问 | 24s |
| **检索后声明假设 · 猜对** | **16s** |
| **检索后声明假设 · 猜错并打断** | **23s** |

猜是否更快的条件是 `(1 - p) × T_run < T_澄清 + T_人`，代入即 `p > 50%`。

**但延迟不是主要风险。** RAG 猜错的真正代价是：错答案引用了真实存在的笔记段落，
格式工整、有出处，**看起来完全是对的**，用户可能根本意识不到它在回答另一个问题。
这比慢 16 秒严重得多。

**因此把判定点从"检索前"移到"检索后、生成前"**，三个问题一次解决：

1. **省掉的正好是最贵的一段** —— 生成 12s vs 检索 4s，最坏只浪费 4s。
2. **判断从猜变成看证据** —— 命中几个不相干主题是测出来的，猜对率显著抬高。
3. **反问那次检索完全不浪费** —— `ContextManager._accum` 按 session_id 缓存
   `RetrievalResult`，用户回答后的新请求会把它作为 seed 注入，第二轮**只需重跑生成**。

结论：检索后声明假设**严格占优** —— 最坏情况（23s）不比反问（24s）差，
最好情况（16s）快 33%。

### 3.5 投机执行：把猜错的代价压到用户反应时间

假设声明不混在答案正文里，而在生成开始**之前**单独发一个 SSE 事件，
前端渲染成可点横幅，**下方答案照常开始流式输出**：

```
event: assumption
{"assumed": "FlashAttention-2 的改进",
 "alternatives": ["FlashAttention-1 原理", "Paged Attention"],
 "run_id": "..."}
```

用户若发现猜错，点备选项 → 前端 abort 当前流 + 发新请求 → 后端复用 `_accum`
中的证据，只重跑生成。猜错成本 = 用户反应时间，而非一次完整 run。

副作用：该横幅同时消解了 3.4 提到的"错答案看起来是对的"风险——
用户不必读完全文才发现跑偏。

---

## 四、架构抉择：怎么"暂停等用户"

### 方案 A — LangGraph checkpointer + interrupt

教科书式 HITL，图内真暂停。但：
- 直接推翻 `store.py:10-13` 已明确记录的架构决策
- 要引入 `SqliteSaver`，`thread_id` / `session_id` / `run_id` 三套 ID 语义打架
- `astream` 事件映射、run 快照落盘、`build_graph` 的 `@lru_cache` 全要重构

### 方案 B — clarify-as-terminal（**选定**）

clarify 节点直接走 END，澄清问句作为 answer 返回，同时把待澄清上下文落进 SQLite。
用户下一轮回答作为**新请求**进来，runner 入口检出 pending 就合成完整问题，正常跑图。

选 B 的三条理由（不是为了省事）：

1. **澄清后信息量变了，本来就该重新路由、重新检索。**"从断点续跑"在这里是负价值——
   恢复出来的 `accumulated` 里那批旧证据，恰恰是导致歧义的噪声源。
2. **跨轮证据复用已经免费具备。** `ContextManager._accum` 按 session_id 缓存
   `RetrievalResult`，`_prepare_agent_context` 会作为 seed 注入下一轮。
   所以"上一轮的有效证据"什么都不用做就自动带过来了——方案 B 与现有机制天然契合。
3. 与项目既有 SQLite 快照哲学一致，跨进程、跨重启都可恢复。

**推论（很优雅）：API 不需要任何新字段。**
用户点选项 = 前端把 label 当普通 question 发；自由回答 = 一样。
runner 入口查 pending 表即可，`AskRequest` 保持不变。

---

## 五、详细设计

### 5.1 图结构变更（`agent/agent.py`）

新增 2 个节点、3 条边：

```
START → router
  ├─(chat)→ direct_chat → END
  └─(search)→ clarify_gate ──(ok)──→ agent → tools → rerank_loop → reflect
                    │                    ↑                            │
                    └──(need_clarify)────┼────────────────────────────┤
                                         │                            │
                              rewrite ←──┴──(need_rewrite/need_more)──┤
                                                                      │
                          clarify → END  ←──(need_clarify)────────────┤
                                                                      │
                          rerank_exit → generate → END ←(sufficient)──┘
```

- `clarify_gate`：router 之后、agent 之前。**极保守**，只拦硬歧义。默认 `enabled=False`。
- `clarify`：终止节点，产出澄清问句 + 选项，写 pending，走 END。

`_reflect_branch` 新增 `need_clarify` 出口；`_norm_verdict` 白名单加第五值。

### 5.2 AgentState 新增字段

```python
class AgentState(TypedDict):
    ...
    awaiting_clarify: bool          # 本轮以澄清收尾
    clarify_question: str           # 澄清问句
    clarify_options: list           # [{"label": str, "hint": str, "filepath": str}]
    clarify_attempts: int           # 本问题已澄清次数，>=1 时禁止再澄清
    ambiguity: dict                 # 可观测：{"type","score","signals":[...]}
```

`clarify_options` / `ambiguity` 为覆盖型，不用 reducer。

### 5.3 T2 检测：两级判定，零额外 LLM 调用

**L1 廉价过滤（无 LLM）** — 在 `reflect` 前用 `accumulated` 算：

```
1. 取 rerank 后 top-k，按 filepath 分组
2. 组内取最高分作为组分；按组分降序取前两组 A、B
3. 判定候选歧义 = (len(A)>=2 and len(B)>=2)
                and (A.score - B.score) < agent_clarify_score_gap   # 默认 0.15
                and title_token_overlap(A, B) < 0.3
```

分差小 = 两组都很像答案；标题词重叠低 = 两组不是一回事。零成本，能挡掉绝大多数正常查询。

**L2 LLM 判定 — 复用 reflect 那一次调用，不加延迟。**
L1 命中时，在 Judge 的 HumanMessage 里附上两组的 `heading_path` 列表，
`JUDGE_SYSTEM` 输出 schema 扩展：

```json
{
  "verdict": "sufficient|need_rewrite|need_more|give_up|need_clarify",
  "relevance_score": 0.0,
  "reason": "",
  "rewritten_query": "",
  "ambiguity": "none|multi_topic",
  "clarify_options": [{"label": "笔记标题", "hint": "一句话区分"}]
}
```

判定规则写进 prompt：**只有当两组笔记指向用户不可能同时想要的不同主题时**
才判 `need_clarify`；若两组只是同一主题的不同侧面，判 `sufficient` 并全部用于生成。

### 5.4 T1 / T2 的零成本处置

**T1 猜 + 声明假设（检索后判定 + 投机执行）** — 分两步。

第一步，改造 `condense_question` 返回结构体（保持向后兼容），只**记录**置信度，
**不在此处做任何分支决策**：

```python
@dataclass
class CondenseResult:
    question: str
    confidence: float          # 1.0=无需消解透传, 0.x=LLM 消解, 0.3=规则兜底
    assumption: str            # 低置信时填"我理解你问的是 X"
```

第二步，判定放在 `reflect` 之后、`generate` 之前。此时同时握有
`condense.confidence`（先验）与检索证据的主题分布（后验）：

| condense 置信 | 检索后主题分布 | 处置 |
|---|---|---|
| 高 | 任意 | 直接答，不声明 |
| 低 | 唯一主导主题 | **T1**：发 `assumption` 事件 + 照常生成 |
| 低 | 2 簇且均有充分证据 | T2 多解并答 |
| 低 | ≥3 簇 / 某簇证据不足 | T3 反问 |

`generate_node` 开始前 emit `assumption` SSE 事件（结构见 3.5），
**不阻塞生成**。假设文案同时写入答案首行，保证非流式客户端也能看到。

> 注意：不要在 `condense` 阶段就按 `confidence < 阈值` 直接决定声明或反问——
> 那时手上只有问题字面，与普通 chatbot 无异，且猜错要浪费整条链路。理由见 3.4。

**T2 多解并答** — 当 `ambiguity == "multi_topic"` 但选项只有 2 个且各自证据都充分时，
不走 clarify，改为在 `GENERATE_SYSTEM` 注入"分两节分别回答"的指令。
只有选项 ≥3 或某一组证据明显不足时才升级到 T3 反问。

### 5.5 选项生成规则

- 来源：`accumulated` 分组后每组的 `metadata["title"]` / `heading_path` 末级
- `hint`：取该组最高分 chunk 正文首句，截断 40 字（零 LLM）
- 数量：2-4 个
- **必须追加兜底项**：`{"label": "都不是 / 直接综合回答", "filepath": ""}`
  用户选它则下一轮强制 `clarify_attempts=99`，绝不再澄清

### 5.6 存储（`agent/store.py`）

新增一张表，风格与现有四张一致：

```sql
CREATE TABLE IF NOT EXISTS pending_clarify (
    session_id   TEXT PRIMARY KEY,
    question     TEXT NOT NULL,      -- 原始问题
    condensed    TEXT DEFAULT '',    -- 凝练后问题
    options_json TEXT DEFAULT '[]',
    attempts     INTEGER DEFAULT 0,
    created_at   REAL NOT NULL
)
```

方法：`set_pending(...)` / `get_pending(session_id)`（读时按 TTL 过期即删）/ `clear_pending(...)`。
`session_id` 做主键 = 天然只保留最新一条待澄清。

`runs.status` 增加 `awaiting_user` 取值，`schemas.py:150` 注释同步。

### 5.7 runner 入口合成（`agent/runner.py`）

在 `_prepare_agent_context` **之前**插入：

```python
pending = store.get_pending(session_id) if session_id and store else None
clarify_attempts = 0
if pending:
    store.clear_pending(session_id)              # 先清，避免异常时卡死
    clarify_attempts = pending["attempts"] + 1
    question = f"{pending['question']}（用户澄清：{question}）"
# 之后照常 condense → 路由 → 跑图
```

`clarify_attempts` 写入初始 state，`>=1` ��本轮**禁止任何澄清**——
用户回答完必须出答案，绝不二次反问。

### 5.8 缓存必须绕开（关键坑）

`runner.ainvoke:295` 语义缓存以 `question + ctx_key` 为键。
澄清问句若被当答案缓存，用户第二次问同样问题会直接命中缓存返回澄清问句，
且 `ctx_key` 相同 → **永远出不来答案的死循环**。

处置：`awaiting_clarify == True` 时**跳过 `cache.set`**；
合成后的问题因文本已变，天然不会误命中旧缓存。

### 5.9 SSE 与前端

新增事件类型：

```json
{"type": "clarify", "question": "你问的是哪一个？", "options": [{"label": "...", "hint": "..."}]}
```

- `runner.astream` 事件映射表加 `clarify` 节点分支
- `AgentTrajectoryItem.type` 枚举注释补 `clarify`
- `frontend/app.py:127-185` 事件分派链加分支：渲染问句 + `st.button` 选项列表；
  点击后把 label 作为新一轮 question 提交（复用现有提交路径，不新增接口）
- 用户也可以不点按钮直接打字，行为完全一致

### 5.10 配置项（`config.py`）

```python
agent_clarify_enabled: bool = True             # 总开关（T2/T3）
agent_clarify_gate_enabled: bool = False       # 入口拦截，默认关（最易滥用）
agent_clarify_ttl: int = 300                   # pending 过期秒数
agent_clarify_max_attempts: int = 1            # 同一问题最多澄清次数
agent_clarify_score_gap: float = 0.15          # L1 组分差阈值
agent_clarify_max_options: int = 4
agent_clarify_assume_threshold: float = 0.6    # T1 声明假设的置信阈值
```

---

## 六、熔断与防滥用（这套设计成败的关键）

1. `clarify_attempts >= agent_clarify_max_attempts` → 本轮硬禁澄清
2. session 级：最近 5 轮内已澄清过 1 次 → 本轮禁止（防连续骚扰）
3. `clarify_gate` 默认关闭，且只在两种极强信号下触发：
   - condense **之后**问题里仍含指代代词（说明消解失败），**且**历史里存在 ≥2 个候选实体
   - 问题 <6 字、无历史、且 router `confidence < 0.5`
4. Judge 异常 / JSON 解析失败 → 一律降级为非澄清路径（现有 except 分支语义不变）
5. 用户选了"都不是" → 该 session 后续强制不澄清
6. 澄清率上线后必须监控，超过 15% 视为设计失败，回退默认开关

---

## 七、评测（`Day4-评测体系.md` 扩展）

新增三个指标：

| 指标 | 定义 | 目标 |
|---|---|---|
| `clarify_rate` | 澄清轮次 / 总轮次 | < 10% |
| `clarify_necessity` | LLM judge 判"这次澄清是否必要" | > 80% |
| `clarify_recovery` | 澄清后答案质量 vs 不澄清基线 | 显著为正 |

**必须建负样本集**：20 条明确不该澄清的清晰问题，断言 `clarify_rate == 0`。
没有这个集合，系统一定会朝"多问总没错"的方向退化——这是本方案最大的失败模式。

A/B 实验：`agent_clarify_enabled` on/off 跑同一评测集，
对比 ragas `answer_relevancy` / `faithfulness` 提升幅度 vs 平均交互轮次代价。

---

## 八、实施顺序

| 阶段 | 内容 | 说明 |
|---|---|---|
| P0 | 修 `reflect` 不传片段的 bug | 前置依赖，独立可验证；T1/T2/T3 全部依赖它 |
| P1 | `CondenseResult` 结构体 + 检索后 T1 判定 + `assumption` SSE 事件 | 不改图结构，收益最高 |
| P1.5 | 前端假设横幅 + abort 重发（投机执行闭环） | 把猜错代价压到反应时间 |
| P2 | Judge `need_clarify` + `clarify` 节点 + pending 表 + SSE + 前端 | 核心闭环 |
| P3 | T2 多解并答 | 进一步压低澄清率 |
| P4 | `clarify_gate` 入口拦截（默认关） | 最易滥用，最后做 |
| P5 | 评测指标 + 负样本集 | 防退化 |

---

## 九、未决问题

1. `reflect` 注入片段后 token 开销上升（预计 +800~1500 token/轮），
   是否需要为 Judge 单独设更小的预算？
2. 澄清问句会作为 assistant 轮次写入 `session_turns`，进而进入下一轮 history。
   是否需要打标记，避免污染 `condense_question` 的历史输入？
3. 前端 Streamlit 按钮点击会触发 rerun，需确认与现有 SSE 流式渲染的交互不冲突。
4. 投机执行被用户 abort 后，已生成的半截答案是否入 `session_turns`？
   倾向不入，但需确认 run 快照落盘逻辑不会留下孤儿记录。
5. 3.4 的耗时数字为估算，需用真实 vault 实测检索/生成耗时后校准，
   并据此复核 T1 触发阈值。

---

## 十、方案 A / B 量化对比（代码核查实测）

第四节已定选型，本节补上支撑数据，供日后复盘或反悔时查阅。

### 10.1 改动量（中位估值，不含无法量化的风险）

| 文件 | 现有规模 | 方案 A | 方案 B |
|---|---|---|---|
| `agent/agent.py` | 559 行 | ~110 | ~85 |
| `agent/runner.py` | 615 行 | **~185**（85 行事件循环需重构） | ~75（+1 个 `elif` 分支） |
| `agent/store.py` | 307 行 | ~22 | ~45（唯一 B 更多的一项） |
| `api/`（2 文件） | 638 行 | ~62（新端点 + 契约变更） | ~11（SSE 端点 **0 行**） |
| `frontend/`（2 文件） | 338 行 | ~82 | ~32 |
| `config.py` | — | ~10 | ~9 |
| `tests/agent/` | 1707 行 | ~160（6 处断言失效） | ~105（4 处，均为参数扩展） |
| 依赖 | — | +1 包 + `uv.lock` 重生成 | **0** |
| **合计** | — | **500–790 行 / 14 文件** | **290–440 行 / 10 文件** |

B ≈ A 的 55–60%，且不含 A 的三项不可估算风险：依赖版本兼容
（本项目已在 langgraph 1.2.6 / checkpoint 4.1.1，`langgraph-checkpoint-sqlite`
未安装且版本约束需 spike）、`RetrievalResult` dataclass 的 checkpoint
序列化 round-trip、同一 SQLite 文件上同步 `sqlite3` 与异步 `aiosqlite`
双连接模型的写锁竞争。

### 10.2 架构清晰度

现有基线是「1 套持久化 / 2 套 ID / 1 种 DB 连接模型 / 图为零参 `lru_cache` 单例」。

- **方案 A 把它推到 2 / 3 / 2**：`checkpoints`+`writes` 表与 `runs` 表语义重叠
  （同一份运行状态两处存储）；引入第三套 `thread_id`，若映射到 `session_id`
  则该 thread 的 state 随会话无限增长，需额外清理策略；`build_graph` 的
  `@lru_cache` 必须拆除（saver 有生命周期，零参缓存无处挂载），连带打破
  `tests/agent/test_persistence.py:103` 的 `cache_clear()` 硬耦合。
- **方案 B 全部维持原样**：新增的 `pending_clarify` 是第 5 张表、走同一套同步
  `sqlite3` 机制，且语义不重叠（存「待澄清」而非运行态）；ID 体系不变；
  图只加 1 node + 2 edge。**新增的是叶子，不是并行主干。**

### 10.3 十一维优劣（A 赢 4 项，需诚实记录）

| 维度 | 胜方 | 说明 |
|---|---|---|
| 改动规模 / 新增依赖 / API 契约 / 架构一致性 | **B** | 见 10.1、10.2 |
| 状态保真度 | A | 完整 checkpoint；但 B 的 `_accum` 降级损失有限 |
| 进程鲁棒性 | **B** | A 要求协程与 SSE 连接存活，切标签页/锁屏即丢 |
| 交互连续性 | A | 同一条流内恢复；B 表现为新一轮气泡 |
| **HITL 可扩展性** | **A** | **唯一实质分歧点**，见下 |
| 语义正确性 | **B** | A 的 resume 带回的旧证据正是歧义噪声源 |
| 可测试性 | **B** | B 为纯函数 + 表 CRUD |
| 生态熟悉度 | A | 官方标准 HITL 范式 |

**唯一值得警惕的是 HITL 可扩展性**：若将来要做「工具执行前人工审批」这类
必须在图中间挂起的场景，B 的终止点交互模式无法覆盖。但那是独立需求，
届时可单独引入 checkpointer，不必现在为尚不存在的场景预付全部代价。
迁移成本亦可控——B 的 `clarify` 是终止节点，改成 `interrupt` 只需换掉
节点体与恢复入口，`clarify_decide` 的判定逻辑可全量复用。

### 10.4 核查中新发现的既存缺陷（两方案共同前置）

1. **前端未知事件静默丢弃**（`frontend/app.py:115-185`）：`if/elif` 链**无 `else`
   兜底**。新事件类型会先在 L119 清掉「思考中…」占位，再穿过整条链无渲染、
   无报错、无日志，最终 `full_answer=""` 写入历史。**最坏的失败模式**，
   两方案都必须先补 `else`（约 3 行）。
2. **`AgentTrajectoryItem` 缺 `options` 字段**（`api/schemas.py:124-132`）：
   `type` 是裸 `str` 无校验，新增类型不会报错，但 `main.py:313` 与 `main.py:391`
   两处 `AgentTrajectoryItem(**t)` 会**静默丢弃**选项数据。
3. **`run_id` 双格式**：`runner.py:269/440` 生成 8 位短 id，`store.py:110` 生成
   32 位 hex，`astream` 走后者、`ainvoke` 走前者，同一字段两种格式并存。
   方案 A 若复用为 `thread_id` 必须先统一；方案 B 可暂不处理。
4. **前端不回传 `run_id`**：`frontend/utils.py:61-97` 的 `_sse_events` 签名无该参数，
   导致后端续传能力（`astream` L454-465）目前是**死代码**。
5. **前端无 abort 能力**：主循环是同步阻塞生成器，Streamlit 在其执行期间不响应
   任何 widget 交互，`httpx` 读超时还被设为 `read=None`。P1.5 投机执行需要
   后台线程 + `st.session_state` 缓冲 + `st.fragment` 轮询的结构性改造——
   **这是与 A/B 选型正交的隐藏成本**。但 P2 核心闭环（clarify 走 END + 选项按钮）
   **不需要 abort**，流已自然结束，点击触发正常 rerun。

### 10.5 `_accum` 的两个坑（方案 B 落地前必须解决）

1. `ContextManager._accum`（`agent/context.py:193`）是**纯内存** `dict`，
   API 重启后 pending 还在（SQLite）、证据没了（内存）。会降级为重新检索
   而非报错，可接受，但需明确。
2. 澄清轮若为「不污染 `session_turns`」而跳过 `_post_run_context`，
   `record_turn` 就不执行，`_accum` **反而不会更新** → 证据丢失。
   当前 `_post_run_context`（`runner.py:376-398`）把 `record_turn` 与
   `maybe_summarize` 绑在一起，**这两个诉求互相冲突，必须先拆函数**。

---

## 十一、MVP：只要「一个反问环节」的最小实现（29 行）

### 11.1 为什么前面报的是 375 行

第八节的实施顺序回答的是「**分层澄清策略该怎么建**」，不是「加一个反问要多少代码」。
拆开看，前面 290–440 的估算里：

| 成分 | 行数 | 是不是「反问」 |
|---|---|---|
| MVP 反问闭环 | ~29 | ✅ 是 |
| 既存 bug 修复（`reflect` 盲判 / `else` 兜底 / 拆 `_post_run_context`） | ~60 | ❌ 地基，与反问无关 |
| T1 假设声明 + T2 多解并答 | ~90 | ❌ 为了**少**反问 |
| 前端横幅 + abort 重发（P1.5） | ~70 | ❌ 体验优化 |
| pending 表 + 评测负样本 + 测试 | ~120 | ❌ 其中 pending 已证明多余 |

**91% 不属于反问本身。**

### 11.2 关键发现：`pending_clarify` 表是多余的

原设计（5.5 节）要新建 `pending_clarify` 表 + runner 入口 hook 来做跨轮恢复。
代码核查后确认**这两样都不需要**：

- `runner.py:424` 附近，每次运行终止都会 `append_turn(session_id, "assistant", final_answer)`；
- `runner.py:230` 的 `cm.condense_question(question, effective_history, session_id)`
  吃的正是这份历史。

所以澄清问句作为**普通 assistant 轮次**落进 `session_turns` 后，第二轮的历史天然是：

```
user      : 那个 FA2 的改进
assistant : 你是指算法层面的改进，还是工程实现优化？
user      : 第一个
```

**指代消解本来就是 `condense` 的职责**，它面对这段历史能直接合成出完整问题。
跨轮恢复不需要任何新机制 —— 复用的是系统里已经跑通的那条路径。

推论：
- 0 张新表（`store.py` 改动从 45 行降到 **0**）
- 0 个 API 字段
- runner 入口 0 处 hook
- 10.5 节第 2 点（「澄清轮跳过 `_post_run_context` 会丢证据」）**自动消解** ——
  MVP 恰恰**需要**澄清轮写进 `session_turns`，不再有冲突诉求。

### 11.3 逐处改动清单

| # | 文件 | 位置 | 改动 | 行 |
|---|---|---|---|---|
| 1 | `agent/agent.py` | `AgentState` L54-66 | 加 `clarify_question: str` | +1 |
| 2 | `agent/agent.py` | `_norm_verdict` L190-194 | 白名单加 `"need_clarify"` | +1 |
| 3 | `agent/agent.py` | `JUDGE_SYSTEM` | 新增档位说明 + 输出字段 | +6 |
| 4 | `agent/agent.py` | `reflect` L342-355 | 从 `data` 取 `clarify_question` | +3 |
| 5 | `agent/agent.py` | 新 `clarify` 节点 | 问句写入 `answer` 即可 | +10 |
| 6 | `agent/agent.py` | `_reflect_branch` L515-522 | 多一个 verdict 分支 | +2 |
| 7 | `agent/agent.py` | `build_graph` L528-559 | `add_node` + 映射 + `→ END` | +3 |
| 8 | `agent/runner.py` | L549 | 终止节点元组加 `"clarify"` | +1 |
| 9 | `agent/runner.py` | L295 语义缓存 | 澄清答案**跳过写缓存** | +2 |

**合计 29 行 / 2 文件。前端零改动** —— `clarify` 复用 `answer` 事件类型，
UI 上表现为助手多说了一句话，用户直接打字回答即可（无选项按钮）。

### 11.4 MVP 的两条已知代价

1. **能跑通 ≠ 判得准。** `reflect` 盲判 bug 未修时，Judge 看不到片段内容，
   判「需要反问」等同抓阄。**功能 29 行，可用 ≈ 89 行**（+60 行地基）。
   这 60 行同时修好现有的 `need_rewrite` 判定，不是澄清专属开销。
2. **缓存死循环是唯一真陷阱。** 清单第 9 项不可省 —— 澄清问句若被
   `runner.ainvoke:295` 的语义缓存收录，同问题再问会命中缓存直接吐澄清问句，
   且 `ctx_key` 相同，**永远出不来答案**。

### 11.5 建议路径

```
第 1 步：29 行 MVP + 只在 Judge 里开一个很严的 need_clarify 阈值
第 2 步：真实用一周，统计澄清率与「澄清后答案是否变好」
第 3 步：澄清率过高 → 再上 T1/T2（90 行）把它压下去
        澄清率本就很低 → 说明个人 KB 场景确实不需要分层，省下 280 行
```

先量再建。分层策略是**给"澄清率过高"这个问题准备的解药**，
在没测出这个问题之前，它是投机性投资。

---

## 十二、方案 C 评估：把反问做成图内 HITL 循环节点

### 12.1 需求描述

> 反问作为图内 human-in-the-loop 节点，走循环直到问题明确，
> 然后继续流转到输出答案。

即：`reflect → clarify(interrupt) → 等用户 → resume → 判定是否明确
→ 不明确则回边再问 → 明确则继续 generate`。

### 12.2 反直觉结论：「循环」恰恰是方案 C 最吃亏的地方

直觉上「循环」听起来像是图内挂起的强项 —— 状态都在，转几圈都行。
实际相反：

| | 方案 B（终止重入） | 方案 C（图内挂起） |
|---|---|---|
| 第 1 轮澄清 | `reflect → clarify → END` | `interrupt()` + checkpoint 落盘 |
| 第 2 轮澄清 | **完全相同的代码路径** | 回边 + `clarify_count` + 明确性判定 |
| 第 N 轮澄清 | **完全相同的代码路径** | 同上，每轮一次序列化往返 |
| 循环的增量代码 | **0 行** | **约 70 行** |

原因：方案 B 的循环**由用户的对话轮次天然构成**。
第 2 轮澄清就是用户又发了一条消息，走的是和第 1 轮一模一样的入口。
图里从头到尾只有一条 `clarify → END` 边。

方案 C 必须把循环显式建模进图：回边、防死循环计数器、
「现在明确了吗」的判定分支 —— 而后者是个**新的 LLM 判定点**，
它要回答的问题恰恰是 `condense` 已经在回答的那个问题。

### 12.3 致命细节：`astream` 收尾逻辑在 interrupt 下静默出错

`graph.astream()` 撞上 `interrupt()` 时**正常结束迭代，不抛异常**。
于是 `runner.py:596-608` 那段收尾会照常执行，而此时 `final_answer` 是空串：

| 收尾语句 | B（answer=澄清问句） | C（answer=""） |
|---|---|---|
| `set_answer(rid, final_answer)` | 澄清问句正确落库 | 空答案覆盖 run 快照 |
| `finish_run(rid)` | 本次确实已结束 | 标记完成、实则挂起；轮询接口拿到空结果 |
| `append_turn(sid,"assistant",a)` | **成为下轮消解依据** | 空轮次污染会话历史 |
| `cache.put(q, final_answer, …)` | 需显式跳过（唯一改动） | **缓存空答案，该问题从此永远返回空** |

四个副作用**全部静默** —— 无异常、无日志、无告警。
方案 C 必须把这 20 行整体条件化，区分「真完成」与「因挂起而退出」。

这条同时揭示了方案 B 便宜的真正原因：
**`clarify` 走 END，它的终止语义与 `generate` 完全同构**，
所以整条已有的收尾链路不加改造就是对的。便宜不是偷懒，是同构。

### 12.4 方案 C 的三项隐藏成本

**1. `Command(resume=…)` 不经过 `_prepare_agent_context`。**
恢复时直接把值注入图内挂起点，因此 `condense` 不重跑、`ctx_key` 不重算、
`seed` 不重新注入。「原问题 + 用户回答 → 完整问题」的合成
必须**在图内重新实现一遍** —— 而 `condense_question` 已经具备这个能力。
方案 B 走正常入口，这三样全部自动正确。

**2. `@lru_cache` 必须拆除，且已有测试硬耦合。**
`build_graph`（`agent.py:528`）是零参 `lru_cache` 单例。
加 checkpointer 后图持有 DB 连接，全局单例与连接生命周期冲突。
拆除会直接打断 3 处 `build_graph.cache_clear()` 调用
（`test_persistence.py:103,174`、`test_runner_e2e.py:115`）。

**3. 依赖不确定，不能按「加一行」估。**
`.venv` 中只有 `langgraph_checkpoint-4.1.1`（base / memory / serde），
**`langgraph-checkpoint-sqlite` 与 `aiosqlite` 均未安装**。
项目在 langgraph 1.2.6 这条较新版本线上，安装 sqlite saver 存在反向降级风险；
且 `accumulated` 装的是 `RetrievalResult` dataclass，
checkpoint 序列化 round-trip 需实测。**必须先做 spike。**

**4. 悬挂 thread 需要 GC。** 用户问到一半跑掉，checkpoint 永久留存一个
未完成 thread。方案 B 无任何悬挂状态。

### 12.5 逐文件改动量

| 文件 | B（MVP 反问） | C（图内 HITL 循环） |
|---|---|---|
| `pyproject.toml` | 0 | 2（+ 一次 spike） |
| `agent/agent.py` | 26 | 96 |
| `agent/runner.py` | 3 | **160** |
| `agent/store.py` | 0 | 25 |
| `api/schemas.py` | 0 | 12 |
| `api/main.py` | 0 | 30 |
| `frontend/app.py` | 0 | 65 |
| `tests/agent/` | 30 | 140 |
| **合计** | **≈ 59 行 / 3 文件** | **≈ 530 行 / 8 文件** |

约 **9 倍**。最贵的是 `runner.py`：85 行事件映射要抽成可复用函数
（`__interrupt__` 是特殊 key 而非节点名）、20 行收尾要条件化、
新增一条 `Command(resume=…)` 入口流。

两方案若要达到同等「判得准」质量，各需再加约 60 行修 `reflect` 盲判 —— 共担项，未计入。

### 12.6 结论

方案 C 唯一真优势仍是 **HITL 可扩展性**（将来做「工具执行前人工审批」
这类必须在图中间挂起的场景）。但「循环直到明确」**不属于**这类场景 ——
它的每一轮都以「拿到用户新输入」为起点，天然就是一次新请求。

**建议：先做 59 行的 B，用真实数据量澄清率与平均澄清轮数。**
若实测多轮澄清（≥2 轮）频繁发生且体验确有问题，届时再评估 C；
迁移成本可控 —— `clarify_decide` 的判定逻辑可全量复用，
只需替换节点体与恢复入口。

---

## 十三、澄清是级联的终点：现有消解链路的失败信号缺失

### 13.1 前提澄清

方案 B 的 `clarify` **不是盲目提问**，它必须是级联兜底的最后一档：
先尝试全部消解手段，确实无解才反问。这个主张是对的，
也是澄清率能压到 <10% 的前提。

但代码核查后发现：**这条级联在当前实现里走不到 clarify —— 它没有「不行」这个信号。**

### 13.2 现有四档的真实触发条件

`ContextManager.condense_question`（`agent/context.py:203-242`）：

| 档 | 手段 | 触发条件 | 性质 |
|---|---|---|---|
| 方案0 | 无代词 / 无历史 → 透传 | `not _has_referring_pronoun(current)` | **省 LLM 调用的前置拦截** |
| 方案2 | LLM 改写 | 默认主路径 | 唯一实际消解手段 |
| 方案1 | 历史增强（拼上一轮 user 轮次） | `llm is None` / `except` / `not text` | 故障替补 |
| 方案3 | `last_entity` 规则替换代词 | 同上 | 故障替补 |

**三个降级条件全是「机器坏了」，没有一个是「消解得不好」。**

后果：
- 返回类型是 `str`，**无失败通道**；
- 改写器 prompt 写的是「若追问本身已完整则原样返回」，
  它没有拒绝出口，永远吐一个语法完整的字符串 ——
  哪怕把「那个」赌成了错的实体；
- 级联顺序还是反的：强手挂了才上弱手，而非由强到弱依次尝试。

**缺的不是 `clarify` 节点，是 failure signal。**

### 13.3 更值得警惕：方案0 让大量歧义追问根本没进消解

`_PRONOUN_RE`（`context.py:85-87`）只匹配显式代词：
`前者|后者|上述|这个|那个|这些|那些|这种|那种|它|她|这|那|此|该`。

而中文里最常见的省略式追问**不含任何代词**：

```
「性能呢？」      → 无代词 → 直接透传 → 检索 query 字面就是「性能呢？」
「优缺点」        → 无代词 → 直接透传
「怎么优化」      → 无代词 → 直接透传
「和 v1 比呢」    → 无代词 → 直接透传
```

这类零主语追问**连消解都没走**，直接以残缺 query 进入检索。
「即使做了指代消解仍然不清楚」的案例中，很可能相当一部分
**压根没做消解**。

修法比澄清便宜得多：在方案0 拦截条件里补一条
「问题过短（< N 字）且历史非空 → 也送 LLM 改写」。约 3 行。
**建议优先级高于 clarify 本身。**

### 13.4 设计修正：串行替补 → 并行投票，分歧即信号

「这次消解成功了吗」在 condense 阶段**不可判定** —— 判它本身又是一次猜。
但「几种手段给出的答案一不一样」是**可测的**。

改造思路（不改返回类型，旁路记录置信）：

1. **方案3 常态化执行**（当前仅故障时执行）。
   `last_entity` 存在时，规则替换结果与 LLM 改写结果比对：
   实体是否出现在改写结果中？不出现 = 两条独立路径分歧。
2. **`last_entity` 改存 top-N + 分差**。
   `_update_last_entity`（`context.py:364-378`）现在取 `max(acc, key=score)`
   的 `heading_path` 末级，上一轮若召回两个不同主题，
   它直接挑分最高那个，**完全不记录存在竞争者**。
   改存 top-3 主题 + 首尾分差；第一名领先不明显 = 客观歧义证据。约 6 行。
3. **置信度旁路槽位**：`self._last_condense_conf[session_id]`，
   不改 `condense_question -> str` 签名（`runner.py:230` 唯一调用点零改动）。
4. **低置信不拦截，只标记。** 仍照常检索，把 confidence + 候选集
   带进 `AgentState`，交由检索后的 `reflect` 综合判定 ——
   与 3.2 节「所有歧义判定一律在检索之后」的硬约束一致。

注：方案1（历史增强）产出的是 `参考上下文：…\n当前问题：…` 的拼接串，
不是候选实体，无法参与实体级比对，只作检索锚点。
真正可互证的是方案2 与方案3 两条路径。

### 13.5 增量成本

在第十一节 29 行 MVP 之上：

| 改动 | 行数 |
|---|---|
| 方案0 短问题补漏（13.3） | +3 |
| `_update_last_entity` 存 top-N + 分差 | +6 |
| 方案3 常态化 + 与方案2 比对 | +15 |
| 置信度旁路槽位 + 读取 | +8 |
| confidence / 候选集进 `AgentState` → `reflect` | +6 |
| **小计** | **+38** |

**29（反问闭环）+ 38（级联信号）≈ 67 行**，仍为 2–3 个文件。
若同时修 `reflect` 盲判（+60），「判得准的级联式反问」总量约 **127 行**。

对比第十二节方案 C 的 530 行：**即便把完整级联算进来，仍是其 1/4。**

### 13.6 执行顺序修正

```
① 方案0 短问题补漏（3 行）        ← 独立收益，可能直接消掉一批"不清楚"
② reflect 盲判修复（60 行）       ← 判定质量地基
③ MVP 反问闭环（29 行）           ← 功能可用
④ 级联信号（38 行）               ← 保证"确实无解才问"
⑤ 实测澄清率，决定是否上 T1/T2
```

①单独可验证，且很可能是投入产出比最高的一步 ——
它修的是「歧义追问从未被消解」，而不是「消解后仍歧义」。

---

## 十四、最终选型决议：为什么是 B，不是 A，也不是 C

> 本节是实施前的定论。前十三节是推导过程，本节只给结论与理由，
> 供后续维护者在不重读全文的情况下理解「为什么当初这么选」。

### 14.1 三个方案的一句话定义

| | 名称 | 澄清发生时 | 等待期间谁扛状态 |
|---|---|---|---|
| **A** | LangGraph `interrupt` + checkpointer（单次挂起） | 图在 `clarify` 处挂起 | checkpoint 表 + 活着的协程 |
| **B** | **clarify-as-terminal（终止重入）** | `clarify → END`，问句当答案返回 | **不需要**，`session_turns` 天然承接 |
| **C** | 图内 HITL 循环（A + 回边 + 明确性判定） | 同 A，且可循环多轮 | 同 A，每轮一次序列化往返 |

### 14.2 决定性理由：终止语义同构

`clarify` 走 END，它的终止语义与 `generate` / `direct_chat` **完全同构** ——
都是「本次请求已产出一段给用户看的文本，正常收尾」。

于是 `runner.py:596-608` 那段收尾逻辑不加任何改造就是对的：

| 收尾语句 | B（answer = 澄清问句） | A / C（answer = ""） |
|---|---|---|
| `set_answer(rid, answer)` | 澄清问句正确落库 | 空答案覆盖 run 快照 |
| `finish_run(rid)` | 本次确实已结束 | 标记完成、实则挂起 |
| `append_turn(sid,"assistant",a)` | **成为下一轮消解依据** | 空轮次污染会话历史 |
| `cache.put(q, answer, …)` | 需显式跳过（**唯一**改动） | 缓存空答案，该问题永远返回空 |

A / C 的四个副作用**全部静默** —— `graph.astream()` 撞上 `interrupt()`
是正常结束迭代、不抛异常，收尾照常执行。

**B 便宜不是因为偷懒，是因为同构。**

### 14.3 跨轮恢复不需要任何新机制

- `runner.py:424` 每次运行终止都 `append_turn(session_id, "assistant", final_answer)`；
- `runner.py:230` 的 `condense_question(question, effective_history, session_id)` 吃的正是这份历史。

澄清问句作为普通 assistant 轮次落库后，第二轮历史天然是：

```
user      : 那个 FA2 的改进
assistant : 你是指算法层面的改进，还是工程实现优化？
user      : 第一个
```

**把这三行合成完整问题，本来就是 `condense` 的职责。**
因此：0 张新表、0 个 API 字段、0 处 runner 入口 hook。
（原设计的 `pending_clarify` 表已于第十一节核销。）

### 14.4 「循环直到明确」不构成选 C 的理由

这是最反直觉的一点：**循环恰恰是 C 最吃亏的地方。**

B 的循环由用户的对话轮次天然构成 —— 第 N 轮澄清就是用户又发了一条消息，
走的是和第 1 轮**完全相同**的代码路径，图里始终只有一条 `clarify → END` 边，
**循环的增量代码是 0 行**。

C 必须把循环显式建模：回边 + `clarify_count` 防死循环 +
一个「现在明确了吗」的新 LLM 判定分支 —— 而后者要回答的问题，
`condense_question` 已经在回答了。**重复造轮子约 70 行。**

判据可以抽象成一条通则：

> **驱动下一步的信息由图内节点产出 → 可建图内循环；
> 由图外（人 / 外部系统）产出 → 必须先出图。**

`reflect → need_rewrite → agent` 这条已有自循环成立，是因为新信息（检索结果）
由图内节点产出。`clarify` 缺的信息在用户脑子里，图里没有任何节点能造出来。

### 14.5 公允地列出 A / C 赢的项

| 维度 | 赢家 | 在本项目的权重 |
|---|---|---|
| 状态保真（accumulated 原样恢复） | A/C | **低** —— 澄清后信息量已变，旧证据恰是歧义噪声源，重跑才正确 |
| 交互连续性（同一条 SSE 流） | A/C | **低** —— 聊天 UI 里多一条消息气泡本就是自然形态 |
| 生态熟悉度（教科书 HITL 写法） | A/C | **低** —— 单人项目 |
| **HITL 可扩展性** | A/C | **中** —— 见下 |

**唯一需要认真对待的是 HITL 可扩展性。**
将来若要做「工具执行前人工审批」这类**必须在图中间**挂起的场景，
B 的终止点交互模式覆盖不了。

但结论仍是先做 B，理由有二：

1. 「澄清」不属于那类场景 —— 它的每一轮都以「拿到用户新输入」为起点，
   天然就是一次新请求；
2. **迁移成本可控** —— B 的 `clarify` 是终止节点，将来换成 `interrupt`
   只需替换节点体与恢复入口，判定逻辑（`_should_clarify` + Judge 档位 +
   `CondenseSignal`）可 100% 复用。不必现在为不存在的场景预付全款。

### 14.6 量化对比（终局）

| 文件 | B | A | C |
|---|---|---|---|
| `pyproject.toml` | 0 | 2 + spike | 2 + spike |
| `agent/agent.py` | 26 | 60 | 96 |
| `agent/runner.py` | 3 | 185 | 160 |
| `agent/store.py` | 0 | 22 | 25 |
| `api/` | 0 | 42 | 42 |
| `frontend/app.py` | 0 | 55 | 65 |
| `tests/` | 30 | 120 | 140 |
| **反问闭环合计** | **≈ 59 行 / 3 文件** | ≈ 490 行 | **≈ 530 行 / 8 文件** |

三个方案都需要额外承担的**共担项**（不计入上表，因与选型无关）：

- 修 `reflect` 盲判（约 60 行）—— 判定质量地基；
- 消解级联失败信号（约 38 行）—— 保证「确实无解才问」。

**B 的最终落地量 ≈ 59 + 98 ≈ 157 行；C 同口径约 628 行。**

### 14.7 A/C 的三项非代码风险（选型时不可忽略）

1. **依赖不确定。** `.venv` 中只有 `langgraph_checkpoint-4.1.1`（base / memory / serde），
   `langgraph-checkpoint-sqlite` 与 `aiosqlite` **均未安装**。项目在 langgraph 1.2.6
   这条较新版本线上，装 sqlite saver 存在反向降级风险 —— 必须先 spike，不能按「加一行依赖」估。
2. **`@lru_cache` 必须拆除。** `build_graph`（`agent.py:528`）是零参单例，
   加 checkpointer 后图持有 DB 连接，与全局单例的生命周期冲突；
   拆除会直接打断 3 处 `build_graph.cache_clear()`
   （`test_persistence.py:103,174`、`test_runner_e2e.py:115`）。
3. **悬挂 thread 需要 GC。** 用户问到一半离开，checkpoint 永久留一个未完成 thread。
   B 无任何悬挂状态。

### 14.8 决议

> **采用方案 B（clarify-as-terminal），并配套「级联终点」约束：
> clarify 只在多档消解均给出低置信信号、且检索后证据仍指向多主题时触发。**

重新评估的触发条件（写死，避免凭感觉推翻）：

- 实测连续澄清（同一 session 内 ≥2 轮）频繁发生，且用户体验确有问题；
- 或出现「工具执行前人工审批」这类必须图中间挂起的新需求。

---

## 十五、实现规格（v1 落地）

### 15.1 总原则

1. **默认关闭。** `agent_clarify_enabled = False`，开关一关，
   全部新增分支短路，行为与改造前**逐字节等价**。
2. **级联终点，不是并列分支。** clarify 必须同时满足
   「消解侧低置信」**且**「Judge 判定需澄清」，任一不满足即降级为原有路径。
3. **不改任何已有函数签名。** 消解置信度走**旁路槽位**，
   `condense_question -> str` 保持不变（`runner.py:230` 唯一调用点零改动）。
4. **失败一律降级，不抛异常。** 新增逻辑全部包 try/except，
   异常时退回改造前行为。

### 15.2 数据结构：`CondenseSignal`

`agent/context.py` 新增（旁路，不进 `AgentState` 之外的任何持久层）：

```python
@dataclass
class CondenseSignal:
    confidence: float = 1.0      # 0.0–1.0，消解结果可信度
    method: str = "passthrough"  # passthrough | llm | fallback
    entity_agreement: bool = True   # 方案2（LLM）与方案3（规则）是否一致
    candidates: list[str] = field(default_factory=list)  # 竞争主题 top-N
    topic_margin: float = 1.0    # 上一轮召回 top1 与 top2 的归一化分差
```

存放位置：`ContextManager._condense_signal: dict[session_id, CondenseSignal]`，
读取接口 `get_condense_signal(session_id) -> CondenseSignal`。
内存态，与 `_accum` 同生命周期；重启丢失的唯一后果是「可能多问一次」，可接受。

### 15.3 置信度规则（确定性，零额外 LLM 调用）

| 情形 | confidence | 说明 |
|---|---|---|
| 方案0 透传（无需消解） | 1.0 | 问题本身完整 |
| LLM 改写成功 + 无 `last_entity` 可比对 | 0.7 | 单路径，无从互证 |
| LLM 改写成功 + 实体出现在结果中 + 主题领先明显 | 0.9 | 两条独立路径一致 |
| LLM 改写成功 + 实体出现 + **主题分差小** | 0.5 | 上一轮召回存在竞争主题 |
| LLM 改写成功 + **实体未出现在结果中** | 0.4 | 方案2 与方案3 分歧 |
| 方案1/3 兜底（LLM 不可用 / 异常 / 空返回） | 0.3 | 强手失效 |

阈值 `agent_clarify_confidence_threshold = 0.6`：
**低于**该值才允许进入澄清候选。

### 15.4 逐文件改动

#### `config.py`（+6）

```python
agent_condense_short_threshold: int = 8       # 短问题补漏字数阈值
agent_clarify_enabled: bool = False           # 澄清总开关（默认关）
agent_clarify_confidence_threshold: float = 0.6
agent_clarify_topic_margin: float = 0.15      # 主题分差低于此值视为存在竞争
agent_clarify_max_candidates: int = 3
```

#### `agent/context.py`（+~55）

1. **`_needs_condense(current, history)`（新，替代方案0 单一代词判定）**
   触发 LLM 改写的条件从「含代词」放宽为
   「含代词 **或** 问题长度 < `agent_condense_short_threshold`」。
   修复 13.3 指出的漏洞：`性能呢？` / `优缺点` / `怎么优化`
   这类零主语追问此前**从未进入消解**。
2. **`_update_last_entity` 扩展**：除 `_last_entity` 外，
   同时记录 `_last_topics[session_id] = [(标题末级, score) × top-N]`
   与归一化 `topic_margin = (s1 - s2) / s1`。
3. **方案3 常态化**：LLM 改写成功后，若存在 `last_entity`，
   检查其是否出现在改写结果中 → `entity_agreement`。
   （注：方案1 产出的是 `参考上下文：…\n当前问题：…` 拼接串，
   非候选实体，无法参与实体级比对，仅作检索锚点。真正互证的是方案2 与方案3。）
4. **澄清标记**：`mark_clarified(sid)` / `pop_clarified(sid)`，
   内存态，用于「上一轮刚反问过 → 本轮禁止再反问」，防连续追问。

#### `agent/agent.py`（+~75，含 P0 盲判修复）

| 改动 | 行 |
|---|---|
| `AgentState` 新增 `clarify_question` / `condense_confidence` / `condense_candidates` | +3 |
| `_norm_verdict` 白名单加 `need_clarify` | +1 |
| `JUDGE_SYSTEM` 新增 need_clarify 档位说明 + 输出字段 | +12 |
| **`_format_judge_evidence()`（P0 修盲判）** | +18 |
| `reflect` 传入片段证据 + 消解信号 + 取 `clarify_question` | +12 |
| `_should_clarify()` 级联守卫 | +14 |
| `clarify_node` | +12 |
| `_reflect_branch` 新分支 | +3 |
| `build_graph` 接线 | +2 |

**P0 盲判修复**是本次改动中价值最高的一项：
`reflect` 原先只把 `len(accumulated)` 这个**数字**传给 Judge，
而 `JUDGE_SYSTEM` 开头声称「给定用户问题与已检索到的知识库片段」——
Judge 一直在盲判，`relevance_score` 全靠猜。
修复后传入 top-N 片段的「标题 + heading_path + 正文摘要」，
同时修好现有的 `need_rewrite` 判定，**不是澄清专属开销**。

**`_should_clarify` 级联守卫**（把「多档消解都不行才问」编码成代码）：

```
开关关闭                      → False
Judge 未判 need_clarify       → False
无澄清问句                    → False
消解置信度 >= 阈值            → False   ← 消解成功，不该反问
上一轮刚反问过                → False   ← 防连续追问
否则                          → True
```

#### `agent/runner.py`（+~12）

1. `_initial_state` 补 3 个新字段（含默认值，向后兼容）；
2. `_prepare_agent_context` 末尾读取 `CondenseSignal` 并返回（内部函数，两处调用点同文件）；
3. `astream` 事件映射：终止节点元组 `("generate","direct_chat")` → 加 `"clarify"`；
4. **缓存防护（不可省）**：`clarify` 产出的答案**跳过** `cache.put`。
   否则澄清问句被语义缓存收录，同一问题再问会直接命中缓存吐出澄清问句，
   且 `ctx_key` 相同 → **永远出不来答案**。ainvoke / astream 两处都要。

#### `frontend/app.py`（+4，防御性）

事件分发 `if/elif` 链补 `else` 兜底：未知事件类型记 `st.caption` 调试信息，
不再静默穿过整条链（原行为：先清掉「思考中…」占位，然后无渲染、无报错、无日志）。
两个方案都会新增事件类型，不补必踩。

### 15.5 测试规格

新增 `tests/agent/test_clarify.py`，覆盖：

| # | 用例 | 断言 |
|---|---|---|
| 1 | `_norm_verdict("need_clarify")` | 不被静默降级为 sufficient |
| 2 | 开关关闭 | `_should_clarify` 恒 False（回归保护） |
| 3 | Judge 判 need_clarify 但置信度高 | False —— 消解成功不反问 |
| 4 | Judge 判 need_clarify + 低置信 | True |
| 5 | 上一轮刚反问过 | False —— 防连续追问 |
| 6 | `_reflect_branch` 返回 `clarify` | 且开关关闭时退回原路径 |
| 7 | `clarify_node` | `answer` == 澄清问句，且 `messages` 追加 AIMessage |
| 8 | 短问题补漏 | `性能呢？` 进入 LLM 消解（此前透传） |
| 9 | 实体分歧 | `entity_agreement=False` 且 confidence < 阈值 |
| 10 | 主题竞争 | `topic_margin` 小 → confidence 降档 |
| 11 | `reflect` 证据注入 | Judge 收到的 prompt 含片段标题与正文（P0 回归） |
| 12 | 兜底降级 | LLM 异常 → `method="fallback"`，confidence 0.3 |

回归保护：`uv run pytest` 全量必须通过，
且**开关关闭时**所有既有 agent 测试行为不变。

---

## 十六、落地记录（v1 已完成）

### 16.1 与 15.4 规格的偏差

规格是纸面推演，落地时有三处必须记下来，否则下次读代码会对不上：

| # | 规格原文 | 实际落地 | 原因 |
|---|---|---|---|
| 1 | `runner` 靠「answer == clarify_question」识别澄清轮 | `AgentState` 新增 **`clarified: bool`**，`clarify_node` 显式返回 `clarified=True` | 字符串比对脆弱：Judge 若把问句原样塞进 `generate` 的答案里就会误判，进而错误跳过缓存 |
| 2 | `_prepare_agent_context` 返回 4-tuple | 返回 **6-tuple**：`(condensed, history_messages, seed, ctx_key, signal, just_clarified)` | 旁路信号必须与 `condense_question` 同一次调用绑定；`pop_clarified` 也只能在这里读（读取即清除，一次性语义） |
| 3 | **前端零改动** | 前端改了 **2 处** | ① `vmap` 补 `need_clarify` 图标，否则轨迹里显示原始英文枚举；② `if/elif` 链补 `else` 兜底——这是 8.1 节自己列出的既有缺陷，新增事件类型必踩 |

`clarify` 复用 `answer` 事件类型这一条**成立**：UI 上表现为助手多说了一句话，
用户直接打字回答即可，无需选项按钮。

### 16.2 防连续追问的落地方式

`just_clarified` 不走 state 持久化，而是 `ContextManager._clarified: set[str]` 内存槽位：

- `runner` 在澄清轮结束后 `mark_clarified(session_id)`（`ainvoke` / `astream` 两处）；
- 下一轮 `_prepare_agent_context` 调 `pop_clarified(session_id)` —— **读取即清除**。

这保证连续反问最多发生一次：第二轮无论 Judge 怎么判都必须给答案。
进程重启后标记丢失，最坏情况是多问一次，可接受（跨轮累积 seed 也是同款降级策略）。

### 16.3 测试落地

`tests/agent/test_clarify.py`，**25 条**（规格 12 条 + 13 条补充）：

- 补充的 13 条主要覆盖：`_should_clarify` 五道闸门的**否定路径**（缺 verdict / 空问句 /
  阈值边界 `conf == 0.6` 不反问）、`clarified` 标记的一次性语义、`clarify_node` 空问句兜底，
  以及 3 条**端到端**（真实图 + fake LLM）。
- 端到端最关键的一条是 `test_e2e_clarify_answer_never_cached`：连问两次同一个模糊问题，
  断言第二次 `cached is False` **且答案不再是问句** —— 一次性验证了 11.4 节点名的
  「缓存死循环」陷阱与防连续追问守卫。
- 回归保护：`test_e2e_disabled_switch_never_clarifies` 断言开关关闭时
  Judge 就算判 `need_clarify` 也照常出答案。

### 16.4 回归结果

```
tests/agent   124 passed        （改造前 97，新增 25 + 修 2）
tests/ 全量   306 passed, 9 failed
```

9 个失败**全部是改造前既有的红**，与本次改动无关：

- `tests/pipeline/test_rag_chain.py`（8 个）：根因是 `pipeline/rag_chain.py:145/181`
  引用了 `settings.agnes_base_url`，该字段在 `config.py` 里**从来不存在**（HEAD 版本亦无），
  属于独立的历史 bug；
- `tests/indexing/test_chunking_strategy.py::test_default_strategy_is_v2`（1 个）：
  索引模块，与 agent 链路无交集。

`ruff check` 对本次新增/改动的 6 个文件全部通过（`runner.py` 的 8 条 E402/F401
是既有 import 布局问题，未在本次引入）。

### 16.5 默认关闭，先量再建

`agent_clarify_enabled` 默认 **False**。11.5 节的路径不变：

```
先真实用一周 → 统计澄清率与「澄清后答案是否变好」
澄清率过高 → 再上 T1/T2 分层（90 行）
澄清率本就低 → 说明个人 KB 场景不需要分层，省下 280 行
```

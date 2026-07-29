"""ContextManager 离线测试：全 mock（LLM / embedding / store），覆盖
token 精确计数、问题凝练降级、历史预算裁剪、相关性裁剪、跨轮累积 seed/record、
总预算兜底仲裁、观察截断、缓存指纹、长程摘要滚动、缓存 ctx_key 隔离。

不依赖 Ollama / DeepSeek / ChromaDB，全部离线可跑。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from note_assistant.agent.cache import SemanticCache
from note_assistant.agent.context import ContextManager
from note_assistant.agent.store import AgentStore
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


# ──────────────────────────────────────────────
# 测试辅助
# ──────────────────────────────────────────────

class FakeLLM:
    """最简异步 LLM 桩：固定回复，省去真实 API。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)


def _rr(score: float, content: str, filepath: str, heading: str = "h") -> RetrievalResult:
    return RetrievalResult(
        score=score,
        page_content=content,
        metadata={"filepath": filepath, "title": filepath, "heading_path": heading},
    )


def _hist(contents, roles=None):
    """构造 history dict 列表。"""
    roles = roles or (["user", "assistant"] * ((len(contents) + 1) // 2))
    return [
        {"role": roles[i % len(roles)], "content": c}
        for i, c in enumerate(contents)
    ]


# 简单 one-hot 式伪 embedding，便于离线验证相关性裁剪
_VOCAB = ["flash", "attention", "天气", "摘要"]


def _embed(q: str):
    ql = q.lower()
    return [1.0 if v in ql else 0.0 for v in _VOCAB]


# ──────────────────────────────────────────────
# Token 计数
# ──────────────────────────────────────────────

def test_count_tokens_string_and_messages():
    cm = ContextManager()
    assert cm.count_tokens("你好世界") > 0
    msgs = [HumanMessage("你好"), AIMessage("世界")]
    assert cm.count_tokens(msgs) == cm.count_tokens("你好") + cm.count_tokens("世界")


# ──────────────────────────────────────────────
# 问题凝练（condense）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_condense_success_uses_llm(monkeypatch):
    cm = ContextManager(condense_llm=FakeLLM("FlashAttention 的优点是什么"))
    out = await cm.condense_question("它有什么优点？", [{"role": "user", "content": "FlashAttention 是什么"}])
    assert out == "FlashAttention 的优点是什么"


@pytest.mark.asyncio
async def test_condense_degrades_without_llm(monkeypatch):
    # 真实 LLM 不可用（get_llm 返回 None）→ 降级走零模型兜底（方案1 历史增强），不抛异常
    monkeypatch.setattr("note_assistant.llm.client.get_llm", lambda *a, **k: None)
    cm = ContextManager(condense_llm=None)
    out = await cm.condense_question(
        "它有什么优点？", [{"role": "user", "content": "FlashAttention 是什么"}]
    )
    # 升级：不再裸返回原问题，而是带历史增强前缀（方案1）
    assert "它有什么优点？" in out
    assert "参考上下文" in out


@pytest.mark.asyncio
async def test_condense_degrades_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "agent_condense_enabled", False)
    cm = ContextManager(condense_llm=FakeLLM("改写"))
    out = await cm.condense_question("原问题", [])
    assert out == "原问题"


# ──────────────────────────────────────────────
# 方案0 / 方案1 / 方案3：指代消解降级增强（见面试笔记 B7）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan0_skip_without_pronoun_no_llm_call():
    # 无指代代词 → 直接透传，且根本不调用 LLM（省一次往返）
    llm = FakeLLM("不该被调用")
    cm = ContextManager(condense_llm=llm)
    hist = [
        {"role": "user", "content": "FlashAttention 是什么"},
        {"role": "assistant", "content": "是一种高效注意力机制"},
    ]
    out = await cm.condense_question("RAG 是什么", hist)
    assert out == "RAG 是什么"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_plan0_skip_when_history_empty():
    # 历史为空、无上下文可解指代 → 透传，不调用 LLM
    llm = FakeLLM("不该被调用")
    cm = ContextManager(condense_llm=llm)
    out = await cm.condense_question("它是什么？", [])
    assert out == "它是什么？"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_plan0_no_false_positive_on_qi_compound():
    # Bug1 回归：「其」在「其他/其实/其次/其中」里是词素而非指代，不应误判含指代，
    # 否则「其他RAG框架有哪些？」也会被白白调一次 LLM（匹配到了「其」）。
    llm = FakeLLM("不该被调用")
    cm = ContextManager(condense_llm=llm)
    hist = [{"role": "user", "content": "RAG 是什么"}]
    out = await cm.condense_question("其他RAG框架有哪些？", hist)
    assert out == "其他RAG框架有哪些？"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_plan1_history_augmented_fallback(monkeypatch):
    # LLM 不可用（get_llm 返回 None）+ 有历史 → 降级返回「历史增强」版 query（方案1）
    monkeypatch.setattr("note_assistant.llm.client.get_llm", lambda *a, **k: None)
    cm = ContextManager(condense_llm=None)
    hist = [
        {"role": "user", "content": "FlashAttention 比标准注意力快吗"},
        {"role": "assistant", "content": "是的，因为它避免了显存瓶颈"},
    ]
    out = await cm.condense_question("它比标准注意力快多少？", hist, session_id="s1")
    assert "参考上下文" in out
    # 方案1 取「最近一轮 user 问题」作参考上下文（而非 assistant 长回复），对检索更有价值
    assert "FlashAttention 比标准注意力快吗" in out


@pytest.mark.asyncio
async def test_plan3_last_entity_resolves_pronoun(monkeypatch):
    # 方案3：LLM 不可用 + 已记 last_entity → 代词被规则替换为实体（零模型）
    monkeypatch.setattr("note_assistant.llm.client.get_llm", lambda *a, **k: None)
    cm = ContextManager(condense_llm=None)
    cm.record_turn(
        "s1",
        [
            _rr(0.5, "低分内容", "a.md", "一、背景 > Foo"),
            _rr(2.0, "高分内容", "b.md", "二、方法 > FlashAttention"),
        ],
        "q", "a",
    )
    # 历史里不含实体，实体只能来自 last_entity，确保替换确实生效
    hist = [
        {"role": "user", "content": "之前我们聊过一个方法"},
        {"role": "assistant", "content": "对，那个方法很高效"},
    ]
    out = await cm.condense_question("它比标准注意力快多少？", hist, session_id="s1")
    assert "FlashAttention" in out                  # 来自 last_entity 规则替换
    assert "它比标准注意力快多少？" not in out       # 原代词已被替换
    assert "参考上下文" in out                        # 同时叠加方案1 历史增强


def test_record_turn_updates_last_entity():
    # 方案3：record_turn 后 last_entity 被更新为最高分 chunk 标题末级
    cm = ContextManager()
    cm.record_turn(
        "s1",
        [
            _rr(0.5, "低分", "a.md", "一、背景 > Foo"),
            _rr(2.0, "高分", "b.md", "二、方法 > FlashAttention"),
        ],
        "q", "a",
    )
    assert cm._last_entity["s1"] == "FlashAttention"


# ──────────────────────────────────────────────
# 历史预算 + 相关性裁剪
# ──────────────────────────────────────────────

def test_budget_history_truncates_by_token():
    cm = ContextManager()
    # 10 轮，每轮内容约 40 token，预算 60 → 应被裁剪到少数几条
    hist = _hist(["内容" * 40] * 10)
    out = cm.budget_history_messages(hist, "q", token_budget=60)
    assert len(out) < len(hist)
    assert cm.count_tokens(out) <= 60 + 50  # 留一点单条超预算的 slack


def test_relevance_prunes_unrelated_turns():
    cm = ContextManager(embed_fn=_embed)
    hist = _hist([
        "flash attention 是什么",   # 相关
        "今天天气真好",              # 无关
        "attention 机制介绍",        # 相关
    ])
    out = cm.budget_history_messages(
        hist, "flash attention 优点", token_budget=2000,
    )
    texts = [m.content for m in out]
    assert any("天气" in t for t in texts) is False
    assert len(out) == 2


def test_relevance_disabled_keeps_all():
    cm = ContextManager(embed_fn=None)  # 无 embed_fn → 纯时间窗口
    hist = _hist(["a" * 20, "b" * 20, "c" * 20])
    out = cm.budget_history_messages(hist, "q", token_budget=2000)
    assert len(out) == 3


def test_summary_prepended_as_system_message():
    cm = ContextManager()
    hist = _hist(["hello world content"] * 2)
    out = cm.budget_history_messages(hist, "q", token_budget=2000, summary="早期摘要")
    assert isinstance(out[0], HumanMessage) or out  # 至少返回历史
    # 摘要作为 SystemMessage 前置
    from langchain_core.messages import SystemMessage
    assert any(isinstance(m, SystemMessage) and "早期摘要" in m.content for m in out)


# ──────────────────────────────────────────────
# 跨轮知识累积（seed / record）
# ──────────────────────────────────────────────

def test_seed_applies_decay_compounding():
    cm = ContextManager()
    cm.record_turn("s1", [_rr(1.0, "A", "a")], "q", "a")
    seed1 = cm.seed_accumulated("s1")
    assert abs(seed1[0].score - 0.9) < 1e-6
    seed2 = cm.seed_accumulated("s1")  # 再跨一轮 → 复利衰减
    assert abs(seed2[0].score - 0.81) < 1e-6


def test_record_turn_dedup_keeps_higher_score():
    cm = ContextManager()
    cm.record_turn("s2", [_rr(1.0, "A" * 30, "a")], "q", "a")
    seed = cm.seed_accumulated("s2")  # 衰减为 0.9
    # 本轮新检索到同 key 但分数更低(0.5) + 另一条新片段(2.0)
    new = [_rr(0.5, "A" * 30, "a"), _rr(2.0, "B" * 30, "b")]
    cm.record_turn("s2", seed + new, "q", "a")
    stored = cm._accum["s2"]
    assert len(stored) == 2
    a_entry = next(r for r in stored if r.filepath == "a")
    assert abs(a_entry.score - 0.9) < 1e-6  # 保留衰减后的高分，丢弃 0.5


def test_record_turn_token_budget_truncates():
    cm = ContextManager()
    # 极小 token 预算 → 只保留极少片段
    import note_assistant.config as cfg
    old = cfg.settings.agent_accumulated_token_budget
    cfg.settings.agent_accumulated_token_budget = 30
    try:
        items = [_rr(float(i), "片段内容" * 20, f"f{i}") for i in range(5)]
        cm.record_turn("s3", items, "q", "a")
        kept = cm._accum["s3"]
        assert len(kept) < 5
        # 高分的被优先保留
        assert kept[0].score == max(i for i in range(5))
    finally:
        cfg.settings.agent_accumulated_token_budget = old


# ──────────────────────────────────────────────
# 观察文本截断
# ──────────────────────────────────────────────

def test_truncate_observation_marks_and_shortens():
    cm = ContextManager()
    long = "这是一段很长的观察文本用于测试截断逻辑" * 50
    out = cm.truncate_observation(long, token_budget=20)
    assert "截断" in out
    assert len(out) < len(long)


# ──────────────────────────────────────────────
# 总预算兜底仲裁
# ──────────────────────────────────────────────

def test_fit_total_budget_trims_low_priority_first():
    cm = ContextManager()
    # 3 条历史 + 8 条累积；预算设为「历史规模 * 1.5」：历史单独放得下，
    # 但历史 + 累积放不下 → 应裁累积（低优先级）先、历史（高优先级）保留。
    hm = [HumanMessage("内容" * 100) for _ in range(3)]
    acc = [_rr(float(i), "片段" * 50, f"f{i}") for i in range(8)]
    hist_tok = cm.count_tokens(hm)
    budget = int(hist_tok * 1.5)

    import note_assistant.config as cfg
    old = cfg.settings.agent_total_context_token_budget
    cfg.settings.agent_total_context_token_budget = budget
    try:
        new_hm, new_acc = cm.fit_total_budget(hm, acc)
        assert len(new_hm) == 3            # 历史先被保留
        assert len(new_acc) < 8            # 累积先被裁
        # 保留的累积按 score 降序（高分优先）
        scores = [r.score for r in new_acc]
        assert scores == sorted(scores, reverse=True)
    finally:
        cfg.settings.agent_total_context_token_budget = old


def test_fit_total_budget_noop_when_within():
    cm = ContextManager()
    hm = [HumanMessage("短内容")]
    acc = [_rr(1.0, "短", "a")]
    new_hm, new_acc = cm.fit_total_budget(hm, acc)
    assert len(new_hm) == 1 and len(new_acc) == 1


def test_total_budget_ge_sum_of_sub_budgets():
    # 不变式：三段子预算之和必须 ≤ 总预算，否则「三段之和 ≤ 总预算」设计约束被打破。
    sub_sum = (
        settings.agent_history_token_budget
        + settings.agent_accumulated_token_budget
        + settings.agent_obs_token_budget
    )
    assert sub_sum <= settings.agent_total_context_token_budget, (
        f"子预算之和 {sub_sum} > 总预算 {settings.agent_total_context_token_budget}，"
        f"需调大 agent_total_context_token_budget 或调小子预算"
    )



# ──────────────────────────────────────────────
# 缓存指纹（context_key）
# ──────────────────────────────────────────────

def test_context_key_fingerprint():
    cm = ContextManager()
    k1 = cm.context_key("问题A", "")
    k2 = cm.context_key("问题B", "")
    k3 = cm.context_key("问题A", "")
    assert k1 != k2
    assert k1 == k3                      # 同输入 → 同指纹（稳定）
    assert len(k1) == 16                 # sha256 前 16 位
    # 摘要变化 → 指纹变化（防串台）
    assert cm.context_key("问题A", "摘要1") != cm.context_key("问题A", "摘要2")


# ──────────────────────────────────────────────
# 长程摘要（滚动）
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_summarize_rolls_up_and_cleans(tmp_path, monkeypatch):
    store = AgentStore(tmp_path / "agent.sqlite")
    monkeypatch.setattr(settings, "agent_session_summary_threshold", 30)
    monkeypatch.setattr(settings, "agent_session_recent_keep", 2)
    cm = ContextManager(summarize_llm=FakeLLM("这是滚动摘要"))

    sid = "sess-sum"
    # 写入 10 轮，token 总和远超阈值 30
    for i in range(10):
        store.append_turn(sid, "user", f"这是一段较长的对话内容用于测试摘要触发机制第{i}轮")
        store.append_turn(sid, "assistant", f"这是对应的助手回复内容用于摘要测试第{i}轮")

    await cm.maybe_summarize(sid, store)

    latest = store.get_latest_summary(sid)
    assert latest is not None
    assert latest["summary"] == "这是滚动摘要"
    # 已摘要的最旧轮次被清理，仅保留最近 2 轮原文
    remaining = store.get_all_turns(sid)
    assert len(remaining) == 2


@pytest.mark.asyncio
async def test_maybe_summarize_disabled_is_noop(tmp_path, monkeypatch):
    store = AgentStore(tmp_path / "agent.sqlite")
    monkeypatch.setattr(settings, "agent_summary_enabled", False)
    cm = ContextManager(summarize_llm=FakeLLM("摘要"))
    sid = "sess-off"
    for i in range(4):
        store.append_turn(sid, "user", f"对话内容{i}")
    await cm.maybe_summarize(sid, store)
    assert store.get_latest_summary(sid) is None


# ──────────────────────────────────────────────
# 缓存 ctx_key 隔离（防多轮串台）
# ──────────────────────────────────────────────

def test_cache_ctx_key_isolates_sessions():
    c = SemanticCache(enabled=True, ttl=3600, semantic=False)
    c.put("它的优点是什么？", "A-会话1", [], [], ctx_key="sess1")
    c.put("它的优点是什么？", "B-会话2", [], [], ctx_key="sess2")

    assert c.get("它的优点是什么？", ctx_key="sess1").answer == "A-会话1"
    assert c.get("它的优点是什么？", ctx_key="sess2").answer == "B-会话2"
    # 不同上下文不串台
    assert c.get("它的优点是什么？", ctx_key="other") is None
    # 无 ctx_key（旧调用方）按旧 key 格式，不命中新写入的隔离条目
    assert c.get("它的优点是什么？") is None


def test_cache_semantic_neighbor_respects_ctx_key():
    c = SemanticCache(
        enabled=True, ttl=3600, semantic=True, semantic_threshold=0.9, embed_fn=_embed
    )
    c.put("Flash Attention 是什么", "答案", [], [], ctx_key="sess1")
    # 近义问题，但 ctx_key 不同 → 不应命中
    assert c.get("FlashAttention 定义", ctx_key="sess2") is None
    # 同 ctx_key → 近邻命中
    hit = c.get("FlashAttention 定义", ctx_key="sess1")
    assert hit is not None and hit.answer == "答案"

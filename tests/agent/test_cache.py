"""语义缓存测试（P7b）：精确命中 + 近邻命中 + TTL + 淘汰 + 兜底，全部离线。"""
import math

from note_assistant.agent.cache import SemanticCache, _cosine, _normalize


def _fake_embed(words: list[str]):
    """把查询映射成确定性的 one-hot 向量，便于离线测试近邻。"""
    vocab = ["flash", "attention", "rag", "检索", "切分"]
    vec = [0.0] * len(vocab)
    for w in words:
        if w in vocab:
            vec[vocab.index(w)] = 1.0
    return vec


def _embed_fn(q: str):
    """子串匹配式伪 embedding：词汇以子串形式出现在查询中即命中（离线可测近邻）。"""
    ql = q.lower()
    return [1.0 if (v in ql) else 0.0 for v in ["flash", "attention", "rag", "检索", "切分"]]


def test_normalize():
    assert _normalize("  Hello   WORLD ") == "hello world"
    assert _normalize("Flash\tAttention") == "flash attention"


def test_cosine():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([], []) == 0.0


def test_exact_hit_and_miss():
    c = SemanticCache(enabled=True, ttl=3600, semantic=False)
    assert c.get("同一个问题？") is None
    c.put("同一个问题？", "答案A", [{"filepath": "a.md"}], [{"type": "answer"}])
    hit = c.get("同一个问题？")
    assert hit is not None
    assert hit.answer == "答案A"
    assert c.get("另一个问题") is None
    # 归一化：大小写/空白差异视为同一问题
    assert c.get("  同一个问题？ ") is not None


def test_ttl_expiry():
    c = SemanticCache(enabled=True, ttl=1, semantic=False)
    c.put("q", "a", [], [])
    assert c.get("q") is not None
    import time
    time.sleep(1.1)
    assert c.get("q") is None


def test_semantic_neighbor_hit():
    c = SemanticCache(enabled=True, ttl=3600, semantic=True, semantic_threshold=0.9, embed_fn=_embed_fn)
    c.put("Flash Attention 是什么", "答案", [], [])
    # 近义但字面不同 → 近邻命中
    hit = c.get("FlashAttention 定义")
    assert hit is not None
    assert hit.answer == "答案"


def test_semantic_no_hit_below_threshold():
    c = SemanticCache(enabled=True, ttl=3600, semantic=True, semantic_threshold=0.9, embed_fn=_embed_fn)
    c.put("Flash Attention 是什么", "答案", [], [])
    # 语义不相关 → 不命中
    assert c.get("今天天气如何") is None


def test_semantic_disabled_falls_back_to_exact_only():
    c = SemanticCache(enabled=True, ttl=3600, semantic=False, embed_fn=_embed_fn)
    c.put("Flash Attention 是什么", "答案", [], [])
    assert c.get("FlashAttention 定义") is None


def test_embed_fn_failure_degrades_gracefully():
    def bad_embed(q):
        raise RuntimeError("embedder down")
    c = SemanticCache(enabled=True, ttl=3600, semantic=True, semantic_threshold=0.9, embed_fn=bad_embed)
    c.put("q", "a", [], [])
    # embed 失败不应抛异常，仅退化为精确命中
    assert c.get("q") is not None
    assert c.get("other") is None


def test_fifo_eviction():
    c = SemanticCache(enabled=True, ttl=3600, max_size=2, semantic=False)
    c.put("q1", "a1", [], [])
    c.put("q2", "a2", [], [])
    c.put("q3", "a3", [], [])  # 超过 max_size → q1 被淘汰
    assert c.get("q1") is None
    assert c.get("q2") is not None
    assert c.get("q3") is not None


def test_disabled_never_hits():
    c = SemanticCache(enabled=False)
    c.put("q", "a", [], [])
    assert c.get("q") is None


def test_stats():
    c = SemanticCache(enabled=True, ttl=3600, semantic=False)
    c.put("q", "a", [], [])
    c.get("q")
    c.get("miss")
    stats = c.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert math.isclose(stats["hit_rate"], 0.5)

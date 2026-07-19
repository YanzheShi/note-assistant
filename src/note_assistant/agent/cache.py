"""语义缓存：相同 / 近似问题命中缓存，省成本降延迟（P7b）。

两级命中：
    1. 精确缓存：归一化问题字符串 → SHA256 命中。零外部依赖，始终开启。
    2. 语义缓存：注入 ``embed_fn`` 时，用 query embedding 的余弦相似度做近邻命中
       （threshold 由 settings.agent_cache_semantic_threshold 控制）。

设计要点：
    - ``embed_fn`` 可注入：生产环境接 OllamaEmbedder.embed_one；测试环境可注入
      一个确定性的伪 embedding 函数，从而**完全离线**验证语义命中逻辑。
    - 所有外部调用（embed_fn）都包了 try/except，失败自动降级为「仅精确命中」，
      绝不因缓存组件导致主链路异常。
    - FIFO 淘汰 + TTL 过期，内存占用可控。
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional


def _normalize(q: str) -> str:
    """归一化问题：去首尾空白、折叠空白、小写（拉丁字符）。中文不区分大小写。"""
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class CacheEntry:
    answer: str
    sources: List[dict] = field(default_factory=list)
    trajectory: List[dict] = field(default_factory=list)
    ts: float = 0.0
    embedding: Optional[List[float]] = None
    cached: bool = True


class SemanticCache:
    """问答结果语义缓存。"""

    def __init__(
        self,
        enabled: bool = True,
        ttl: int = 3600,
        max_size: int = 1000,
        semantic: bool = True,
        semantic_threshold: float = 0.92,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.enabled = enabled
        self.ttl = ttl
        self.max_size = max_size
        self.semantic = bool(semantic and embed_fn is not None)
        self.semantic_threshold = semantic_threshold
        self.embed_fn = embed_fn
        self._exact: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── 内部工具 ──

    def _exact_key(self, q: str) -> str:
        return hashlib.sha256(_normalize(q).encode("utf-8")).hexdigest()

    def _safe_embed(self, q: str) -> Optional[List[float]]:
        if self.embed_fn is None:
            return None
        try:
            return self.embed_fn(q)
        except Exception:  # noqa: BLE001
            return None

    # ── 公共 API ──

    def get(self, q: str) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        now = time.time()
        k = self._exact_key(q)

        # 精确命中
        e = self._exact.get(k)
        if e is not None:
            if now - e.ts > self.ttl:
                self._exact.pop(k, None)
            else:
                self._exact.move_to_end(k)
                self._hits += 1
                return e

        # 语义近邻命中
        if self.semantic:
            emb = self._safe_embed(q)
            if emb is not None:
                best: Optional[CacheEntry] = None
                best_sim = self.semantic_threshold
                for e2 in self._exact.values():
                    if e2.embedding is None:
                        continue
                    sim = _cosine(emb, e2.embedding)
                    if sim >= best_sim:
                        best_sim = sim
                        best = e2
                if best is not None:
                    self._hits += 1
                    return best

        self._misses += 1
        return None

    def put(self, q: str, answer: str, sources: List[dict], trajectory: List[dict]) -> None:
        if not self.enabled:
            return
        emb = self._safe_embed(q) if self.semantic else None
        k = self._exact_key(q)
        self._exact[k] = CacheEntry(
            answer=answer,
            sources=sources,
            trajectory=trajectory,
            ts=time.time(),
            embedding=emb,
        )
        # FIFO 淘汰
        while len(self._exact) > self.max_size:
            self._exact.popitem(last=False)

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "size": len(self._exact),
            "enabled": self.enabled,
            "semantic": self.semantic,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def clear(self) -> None:
        self._exact.clear()
        self._hits = self._misses = 0

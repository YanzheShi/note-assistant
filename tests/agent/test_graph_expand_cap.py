# tests/agent/test_graph_expand_cap.py
"""图扩展扇出护栏（审计修复）：邻居文件数 + chunk 总数双上限。

背景：hop>=2 时背链多的高连接笔记会让邻居集爆炸，每个邻居再拉全量 chunks，
context 膨胀 + 延迟失控。本测试锁定两个上限的截断行为。
"""
from types import SimpleNamespace

import note_assistant.agent.tools as tools_mod
from note_assistant.agent.tools import graph_expand_impl
from note_assistant.config import settings


class FakeCollection:
    def __init__(self, docs_by_file):
        self.docs_by_file = docs_by_file
        self.calls = []

    def get(self, where=None, include=None):
        fp = where.get("filepath")
        self.calls.append(fp)
        docs = self.docs_by_file.get(fp, [])
        return {"documents": docs, "metadatas": [{} for _ in docs]}


class FakeGraph:
    def __init__(self, neighbors):
        self._n = neighbors

    def expand(self, files, hop=1):
        return self._n


def _patch(monkeypatch, neighbors, docs_by_file):
    coll = FakeCollection(docs_by_file)
    monkeypatch.setattr(tools_mod, "_wiki_graph", lambda: FakeGraph(neighbors))
    monkeypatch.setattr(
        tools_mod, "_hybrid_retriever",
        lambda: SimpleNamespace(ingestor=SimpleNamespace(collection=coll)),
    )
    return coll


def test_caps_neighbor_files_by_score(monkeypatch):
    """邻居文件数超上限时按扩展分取 top，低分文件不应被展开。"""
    neighbors = [(f"f{i}.md", 1.0 - i * 0.01) for i in range(20)]
    coll = _patch(monkeypatch, neighbors, {f"f{i}.md": [f"chunk-{i}"] for i in range(20)})
    monkeypatch.setattr(settings, "graph_expand_max_files", 5)
    monkeypatch.setattr(settings, "graph_expand_max_chunks", 100)

    out = graph_expand_impl(["seed.md"], hop=2)

    assert len(coll.calls) == 5
    assert "f19.md" not in coll.calls  # 分数最低的邻居被护栏挡掉
    assert "f0.md" in coll.calls
    assert len(out) == 5


def test_caps_total_chunks(monkeypatch):
    """chunk 总数超上限时截断（6 文件 × 10 chunks = 60 → 截到 25）。"""
    neighbors = [(f"f{i}.md", 0.5) for i in range(6)]
    coll = _patch(monkeypatch, neighbors,
                  {f"f{i}.md": [f"c{i}-{j}" for j in range(10)] for i in range(6)})
    monkeypatch.setattr(settings, "graph_expand_max_files", 8)
    monkeypatch.setattr(settings, "graph_expand_max_chunks", 25)

    out = graph_expand_impl(["seed.md"], hop=2)

    assert len(out) == 25
    assert len(coll.calls) <= 3  # 第 3 个文件内就会触顶，不会白拉其余文件


def test_stub_nodes_skipped(monkeypatch):
    """[[stub]] 节点不进 chunk 抓取。"""
    neighbors = [("[[不存在的笔记]]", 0.9), ("real.md", 0.8)]
    coll = _patch(monkeypatch, neighbors, {"real.md": ["c1"]})
    monkeypatch.setattr(settings, "graph_expand_max_files", 8)
    monkeypatch.setattr(settings, "graph_expand_max_chunks", 24)

    out = graph_expand_impl(["seed.md"])

    assert coll.calls == ["real.md"]
    assert len(out) == 1


def test_empty_neighbors(monkeypatch):
    coll = _patch(monkeypatch, [], {})
    assert graph_expand_impl(["seed.md"]) == []
    assert coll.calls == []

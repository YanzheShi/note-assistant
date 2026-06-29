# tests/retrieval/test_graph.py
"""
WikiGraph 测试。
"""
import sys
from pathlib import Path

from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.retrieval.graph import WikiGraph


# ----------------------------------------------------------------
# 工具
# ----------------------------------------------------------------

def _make_doc(filepath: str, wikilinks: list[str] | None = None, title: str | None = None, **kwargs):
    """构造一个轻量 mock DocNode（不用真的读文件）"""
    doc = MagicMock()
    doc.filepath = filepath
    doc.wikilinks = wikilinks or []
    doc.title = title or Path(filepath).stem
    doc.front_matter = kwargs.get("front_matter", {})
    return doc


# ================================================================
# build_from_docs
# ================================================================

class TestBuildFromDocs:
    def test_all_nodes_registered(self):
        """所有文档都应注册为节点"""
        docs = [
            _make_doc("a.md", ["B", "C"]),
            _make_doc("b.md", ["C"]),
            _make_doc("c.md", []),
        ]
        g = WikiGraph()
        g.build_from_docs(docs)
        assert g.node_count == 3

    def test_edges_created(self):
        """wikilinks 应创建对应的边"""
        docs = [
            _make_doc("a.md", ["B"]),
            _make_doc("b.md", []),
        ]
        g = WikiGraph()
        g.build_from_docs(docs)
        assert g.G.has_edge("a.md", "b.md")

    def test_self_loop_excluded(self):
        """自环应被排除（link 指向自己）"""
        docs = [
            _make_doc("a.md", ["A"]),  # 自己链自己
        ]
        g = WikiGraph()
        g.build_from_docs(docs)
        assert g.edge_count == 0

    def test_unresolved_link_creates_stub(self):
        """找不到目标的 wikilink 应建 stub 节点"""
        docs = [
            _make_doc("a.md", ["不存在的笔记"]),
        ]
        g = WikiGraph()
        # 需要让 _resolve_link 返回 None 来触发 stub
        def resolver(link, docs):
            return None
        g.build_from_docs(docs, link_resolver=resolver)
        # 应有 2 个节点：a.md + stub
        assert g.node_count == 2
        assert any(n.startswith("[[") for n in g.G.nodes())

    def test_multiple_links(self):
        """一个文档链多个笔记应创建多条边"""
        docs = [
            _make_doc("a.md", ["B", "C", "D"]),
            _make_doc("b.md", []),
            _make_doc("c.md", []),
            _make_doc("d.md", []),
        ]
        g = WikiGraph()
        g.build_from_docs(docs)
        assert g.edge_count == 3  # a→b, a→c, a→d


# ================================================================
# _resolve_link
# ================================================================

class TestResolveLink:
    def test_filename_match(self):
        """文件名匹配（去掉 .md，大小写不敏感）"""
        docs = [
            _make_doc("RAG 概念.md", []),
            _make_doc("其他笔记.md", []),
        ]
        g = WikiGraph()
        result = g._resolve_link_to_realpath("rag 概念", docs)
        assert result == "RAG 概念.md"

    def test_filename_case_insensitive(self):
        """文件名匹配应大小写不敏感"""
        docs = [
            _make_doc("FlashAttention.md", []),
        ]
        g = WikiGraph()
        result = g._resolve_link_to_realpath("flashattention", docs)
        assert result == "FlashAttention.md"

    def test_aliases_match(self):
        """front matter aliases 匹配"""
        docs = [
            _make_doc("RAG 概念.md", [], front_matter={"aliases": ["RAG", "检索增强生成"]}),
        ]
        g = WikiGraph()
        result = g._resolve_link_to_realpath("检索增强生成", docs)
        assert result == "RAG 概念.md"

    def test_no_match_returns_none(self):
        """未匹配应返回 None"""
        docs = [
            _make_doc("a.md", []),
        ]
        g = WikiGraph()
        result = g._resolve_link_to_realpath("不存在的笔记", docs)
        assert result is None


# ================================================================
# expand
# ================================================================

class TestExpand:
    def _build_graph(self) -> WikiGraph:
        """构造一个测试用图：A → B → C，A → D"""
        g = WikiGraph()
        g.G.add_edge("a.md", "b.md")
        g.G.add_edge("b.md", "c.md")
        g.G.add_edge("a.md", "d.md")
        return g

    def test_hop_1_returns_direct_successors(self):
        """hop=1 应返回直接后继"""
        g = self._build_graph()
        results = g.expand({"a.md"}, hop=1, max_neighbors=10)
        assert len(results) == 2  # b.md 和 d.md
        filepaths = {fp for fp, _ in results}
        assert filepaths == {"b.md", "d.md"}

    def test_hop_1_decay_is_1(self):
        """hop=1 的 decay_score 应为 1.0"""
        g = self._build_graph()
        results = g.expand({"a.md"}, hop=1)
        for _, decay in results:
            assert decay == 1.0

    def test_hop_2_decay_is_half(self):
        """hop=2 的 decay_score 应为 0.5"""
        g = self._build_graph()
        results = g.expand({"a.md"}, hop=2, max_neighbors=10)
        # hop=2 应命中 c.md（通过 b.md）
        c_results = [d for fp, d in results if fp == "c.md"]
        assert len(c_results) == 1
        assert c_results[0] == 0.5

    def test_max_neighbors_truncates(self):
        """max_neighbors 应截断结果"""
        g = self._build_graph()
        results = g.expand({"a.md"}, hop=1, max_neighbors=1)
        assert len(results) == 1

    def test_already_visited_not_expanded(self):
        """已访问节点不应重复出现"""
        g = self._build_graph()
        # a.md 已在 hit_files 中，不应被扩展回来
        results = g.expand({"a.md", "b.md"}, hop=2, max_neighbors=10)
        # b.md 已访问，不应再被 b→c 路径重复加入
        filepaths = [fp for fp, _ in results]
        assert "a.md" not in filepaths  # a.md 不会被扩展回来

    def test_nonexistent_node_returns_empty(self):
        """不存在的节点应返回空"""
        g = self._build_graph()
        results = g.expand({"不存在的.md"}, hop=1)
        assert results == []


# ================================================================
# save / load
# ================================================================

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        """save → load 后图应一致"""
        g = WikiGraph()
        g.G.add_edge("a.md", "b.md")
        g.G.add_node("c.md")

        pkl_path = tmp_path / "test.graph"
        g.save(pkl_path)

        g2 = WikiGraph()
        g2.load(pkl_path)

        assert g2.node_count == 3
        assert g2.edge_count == 1
        assert g2.G.has_edge("a.md", "b.md")


# ================================================================
# summary
# ================================================================

class TestSummary:
    def test_summary_contains_node_edge_count(self):
        """summary 应包含节点数和边数"""
        g = WikiGraph()
        g.G.add_edge("a.md", "b.md")
        s = g.summary()
        assert "2 nodes" in s
        assert "1 edges" in s

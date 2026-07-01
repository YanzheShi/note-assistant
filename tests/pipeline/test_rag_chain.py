# tests/pipeline/test_rag_chain.py
"""
RAGChain 管线编排器测试。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.pipeline.rag_chain import RAGChain
from note_assistant.retrieval.types import RetrievalResult


# ================================================================
# 测试用模拟对象
# ================================================================

def _make_retrieval_result(filepath: str = "test.md", content: str = "test content", score: float = 0.9):
    """构造一个 RetrievalResult mock"""
    from note_assistant.retrieval.types import RetrievalResult
    return RetrievalResult(
        score=score,
        page_content=content,
        metadata={"filepath": filepath, "heading_path": "测试标题"},
        dense_score=0.85,
        sparse_score=0.75,
    )


def _make_mock_retriever():
    """构造一个模拟的 HybridRetriever"""
    from note_assistant.retrieval.types import RetrievalResult
    retriever = MagicMock()
    retriever.search.return_value = [_make_retrieval_result()]
    retriever.dense.collection.query.return_value = {
        "documents": [["neighbor chunk 1", "neighbor chunk 2"]]
    }
    # _fetch_neighbor_chunks 走 ingestor.collection.get
    retriever.ingestor.collection.get.return_value = {
        "documents": ["chunk from ingestor"],
        "metadatas": [{"filepath": "test.md"}],
    }
    return retriever


def _make_mock_reranker():
    """构造一个模拟的 LocalReranker"""
    reranker = MagicMock()
    reranker.rerank.return_value = [_make_retrieval_result()]
    return reranker


def _make_mock_graph():
    """构造一个模拟的 WikiGraph"""
    graph = MagicMock()
    graph.expand.return_value = [("关联笔记.md", 1.0)]
    graph.node_count = 5
    graph.edge_count = 3
    return graph


# ================================================================
# init
# ================================================================

class TestInit:
    def test_init_with_required_args(self):
        """只传 required 参数应正常工作"""
        r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
        assert r.retriever is not None
        assert r.reranker is not None
        assert r.graph is None
        assert r.generator is None

    def test_init_with_all_args(self):
        """传所有参数也能工作"""
        g = _make_mock_graph()
        gen = MagicMock()
        r = RAGChain(_make_mock_retriever(), _make_mock_reranker(), graph=g, generator=gen)
        assert r.graph is g
        assert r.generator is gen


# ================================================================
# ask（核心管线，等你实现）
# ================================================================

class TestAsk:
    def test_ask_returns_valid_structure(self):
        """ask 应返回包含 answer/sources 等字段的字典"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=_make_mock_graph(),
            generator=MagicMock(return_value="测试答案"),
        )
        result = r.ask("测试问题")
        assert "answer" in result
        assert "sources" in result
        assert "graph_expansion" in result
        assert "retrieved" in result
        pass

    def test_ask_without_graph(self):
        """不传 graph 也能正常工作（跳过扩展）"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=None,
            generator=MagicMock(return_value="无图测试答案"),
        )
        result = r.ask("测试问题")
        assert result["graph_expansion"] == 0
        pass

    def test_ask_without_generator(self):
        """不传 generator 也能正常工作（只返回 context）"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=_make_mock_graph(),
            generator=None,
        )
        result = r.ask("无生成器测试")
        assert result["answer"] == ""
        pass


# ================================================================
# _fetch_neighbors（等你实现）
# ================================================================

class TestFetchNeighborChunks:
    def test_skip_stub_nodes(self):
        """stub 节点应跳过"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=None,
        )
        neighbors = [("[[不存在的笔记]]", 1.0), ("real_note.md", 1.0)]
        chunks = r._fetch_neighbor_chunks(neighbors)
        # stub 被跳过，只取 real_note 的 1 个 chunk
        assert len(chunks) == 1
        assert isinstance(chunks[0], RetrievalResult)
        pass

    def test_fetch_from_real_files(self):
        """真实文件应能拉取 chunks"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=None,
        )
        neighbors = [("test.md", 1.0)]
        chunks = r._fetch_neighbor_chunks(neighbors)
        # ingestor.get 返回 1 个 chunk
        assert len(chunks) == 1
        assert isinstance(chunks[0], RetrievalResult)
        pass

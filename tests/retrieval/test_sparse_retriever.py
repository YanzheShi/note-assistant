# tests/retrieval/test_sparse_retriever.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from note_assistant.indexing.types import Chunk
from note_assistant.retrieval.sparse_retriever import BM25Retriever
from note_assistant.retrieval.types import RetrievalResult

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock ollama before importing modules that create a client on init
_ollama_mock = MagicMock()
_ollama_mock.Client.return_value = MagicMock()
sys.modules.setdefault("ollama", _ollama_mock)



# ----------------------------------------------------------------
# 工具
# ----------------------------------------------------------------
def _make_corpus(n: int = 5) -> list[Chunk]:
    """构造一个小型测试 corpus"""
    chunks = []
    for i in range(n):
        chunks.append(Chunk(
            page_content=f"这是第{i}条测试文档，包含关键词{'ai' if i % 2 == 0 else 'rag'}。",
            metadata={"filepath": f"doc_{i}.md", "title": f"文档{i}"},
        ))
    return chunks


# ================================================================
# tokenize
# ================================================================
class TestTokenize:
    def test_tokenize_returns_list(self):
        """分词应返回 token 列表"""
        tokens = BM25Retriever.tokenize("RAG评测指标")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_chinese_words(self):
        """jieba 应把语义单元切在一起"""
        tokens = BM25Retriever.tokenize("评测指标")
        # "评测" 和 "指标" 应各自是一个 token，而不是逐字符
        assert "评测" in tokens or "指标" in tokens

    def test_tokenize_empty_string(self):
        """空字符串应返回空列表"""
        tokens = BM25Retriever.tokenize("")
        assert tokens == []


# ================================================================
# build_index
# ================================================================
class TestBuildIndex:
    def test_build_index_creates_bm25(self):
        """build_index 后 self.bm25 不应为 None"""
        corpus = _make_corpus(3)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        assert retriever.bm25 is not None
        assert len(retriever.corpus) == 3
        assert len(retriever.tokenized) == 3

    def test_build_index_stores_corpus(self):
        """build_index 应保存原始 corpus"""
        corpus = _make_corpus(2)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        assert retriever.corpus[0].page_content == corpus[0].page_content
        assert retriever.corpus[1].metadata["filepath"] == "doc_1.md"


# ================================================================
# search
# ================================================================
class TestSearch:
    def test_search_returns_results(self):
        """search 应返回 RetrievalResult 列表"""
        corpus = _make_corpus(5)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("ai", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_search_top_k_limits_results(self):
        """search 返回数量不应超过 top_k"""
        corpus = _make_corpus(10)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("测试", top_k=3)
        assert len(results) <= 3

    def test_search_results_sorted_by_score(self):
        """结果应按 score 降序排列"""
        corpus = _make_corpus(10)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("测试", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_metadata(self):
        """结果应携带原始 metadata"""
        corpus = _make_corpus(5)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("测试", top_k=5)
        for r in results:
            assert "filepath" in r.metadata
            assert "title" in r.metadata

    def test_search_returns_index(self):
        """结果应包含 corpus 中的位置 index"""
        corpus = _make_corpus(5)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("测试", top_k=5)
        for r in results:
            assert r.index is not None
            assert 0 <= r.index < len(corpus)

    def test_search_empty_bm25_returns_empty(self):
        """bm25 未建时 search 应返回空列表"""
        retriever = BM25Retriever()
        results = retriever.search("测试")
        assert results == []

    def test_search_unknown_query(self):
        """不存在的 query 也应返回结果（BM25 总能算出分数）"""
        corpus = _make_corpus(5)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        results = retriever.search("完全不存在的词xyz", top_k=3)
        # BM25 对任何 query 都能返回分数，只是分数可能很低
        assert len(results) == 3


# ================================================================
# save / load
# ================================================================
class TestPersistence:
    def test_save_and_load(self, tmp_path):
        """save → load 后应能正常检索"""
        corpus = _make_corpus(5)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        pkl_path = tmp_path / "test_bm25.pkl"
        retriever.save(pkl_path)

        # 加载到新实例
        loaded = BM25Retriever()
        loaded.load(pkl_path)

        assert len(loaded.corpus) == 5
        assert loaded.bm25 is not None

        # 加载后应能检索
        results = loaded.search("ai", top_k=3)
        assert len(results) == 3

    def test_save_creates_parent_dir(self, tmp_path):
        """save 应自动创建父目录"""
        corpus = _make_corpus(2)
        retriever = BM25Retriever()
        retriever.build_index(corpus)

        pkl_path = tmp_path / "sub" / "dir" / "test_bm25.pkl"
        retriever.save(pkl_path)
        assert pkl_path.exists()


# ================================================================
# from_chroma
# ================================================================
class TestFromChroma:
    def test_from_chroma_builds_index(self, tmp_path):
        """from_chroma 应从 ChromaDB 拉取 chunks 建索引"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "documents": ["文档A内容", "文档B内容"],
            "metadatas": [{"filepath": "a.md"}, {"filepath": "b.md"}],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection

        # from_chroma 内部 import Ingestor，patch 模块级引用
        with patch("note_assistant.indexing.ingestor.Ingestor", return_value=mock_ingestor):
            retriever = BM25Retriever.from_chroma()

        assert len(retriever.corpus) == 2
        assert retriever.bm25 is not None
        assert retriever.corpus[0].metadata["filepath"] == "a.md"


# ================================================================
# RetrievalResult
# ================================================================
class TestRetrievalResult:
    def test_filepath_property(self):
        """filepath 应从 metadata 中提取"""
        r = RetrievalResult(
            score=0.9,
            page_content="test",
            metadata={"filepath": "note.md", "title": "笔记"},
        )
        assert r.filepath == "note.md"
        assert r.title == "笔记"

    def test_filepath_missing(self):
        """metadata 无 filepath 时应返回空字符串"""
        r = RetrievalResult(score=0.5, page_content="x", metadata={})
        assert r.filepath == ""
        assert r.title == ""

    def test_repr(self):
        """repr 应包含 score 和 filepath"""
        r = RetrievalResult(
            score=0.1234,
            page_content="test",
            metadata={"filepath": "note.md"},
        )
        assert "0.1234" in repr(r)
        assert "note.md" in repr(r)

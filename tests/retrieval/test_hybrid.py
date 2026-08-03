# tests/retrieval/test_hybrid.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from note_assistant.config import settings
from note_assistant.indexing.types import Chunk
from note_assistant.retrieval.hybrid import HybridRetriever
from note_assistant.retrieval.types import RetrievalResult

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock ollama before importing modules that create a client on init
_ollama_mock = MagicMock()
_ollama_mock.Client.return_value = MagicMock()
sys.modules.setdefault("ollama", _ollama_mock)



# ----------------------------------------------------------------
# 工具
# ----------------------------------------------------------------
def _make_corpus(n: int = 4) -> list[Chunk]:
    """构造一个小型测试 corpus"""
    contents = [
        "RAG 系统结合向量检索和大语言模型",
        "BM25 是一种稀疏检索算法",
        "Faiss 是向量相似度搜索库",
        "混合检索结合 dense 和 sparse 两路",
    ]
    return [
        Chunk(
            page_content=contents[i],
            metadata={"filepath": f"doc_{i}.md", "title": f"文档{i}"},
        )
        for i in range(min(n, len(contents)))
    ]


# ================================================================
# __init__
# ================================================================
class TestInit:
    def test_default_alpha(self):
        """默认 alpha 应来自 config.dense_weight"""
        retriever = HybridRetriever()
        assert retriever.alpha == settings.dense_weight

    def test_custom_alpha(self):
        """自定义 alpha 应覆盖默认值"""
        retriever = HybridRetriever(alpha=0.5)
        assert retriever.alpha == 0.5

    def test_default_top_k(self):
        """默认 top_k 应来自 config.top_k_retrieve"""
        retriever = HybridRetriever()
        assert retriever.top_k == settings.top_k_retrieve


# ================================================================
# _dense_search
# ================================================================
class TestDenseSearch:
    @patch("note_assistant.retrieval.hybrid.Ingestor")
    def test_dense_search_returns_results(self, mock_ingestor_cls):
        """_dense_search 应返回 RetrievalResult 列表"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档A", "文档B", "文档C"]],
            "metadatas": [[{"filepath": "a.md"}, {"filepath": "b.md"}, {"filepath": "c.md"}]],
            "distances": [[0.1, 0.5, 0.9]],
            "ids": [["id_0", "id_1", "id_2"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        results = retriever._dense_search([0.1, 0.2, 0.3], n_results=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    @patch("note_assistant.retrieval.hybrid.Ingestor")
    def test_dense_search_score_ordering(self, mock_ingestor_cls):
        """distance 越小 → similarity 越高，排前面"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["最相关", "次相关", "最不相关"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.5, 0.9]],  # 0.1 最近
            "ids": [["id_0", "id_1", "id_2"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        results = retriever._dense_search([0.1, 0.2, 0.3], n_results=3)

        # 第一条应是最相关的（distance 最小 → similarity 最大）
        assert results[0].page_content == "最相关"
        assert results[0].score > results[-1].score

    @patch("note_assistant.retrieval.hybrid.Ingestor")
    def test_dense_search_sets_dense_score(self, mock_ingestor_cls):
        """结果应包含 dense_score 字段"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档A"]],
            "metadatas": [[{"filepath": "a.md"}]],
            "distances": [[0.3]],
            "ids": [["id_0"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        results = retriever._dense_search([0.1, 0.2, 0.3])

        assert results[0].dense_score is not None

    @patch("note_assistant.retrieval.hybrid.Ingestor")
    def test_dense_search_equal_distances(self, mock_ingestor_cls):
        """所有 distance 相同时，similarity 应为 0（无区分度）"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["A", "B"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.5, 0.5]],  # 全部相同
            "ids": [["id_0", "id_1"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        results = retriever._dense_search([0.1, 0.2, 0.3])

        # max_d == min_d → rng = 1.0, sim = 1 - 0 = 1
        for r in results:
            assert r.score == 1.0


# ================================================================
# _merge_results
# ================================================================
class TestMergeResults:
    def test_merge_union(self):
        """并集融合：两路结果不重叠时应全部保留"""
        corpus = _make_corpus(3)
        retriever = HybridRetriever(alpha=0.7)

        # dense 命中 doc_0, doc_1
        dense = [
            RetrievalResult(score=0.9, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
            RetrievalResult(score=0.7, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
        ]
        # sparse 命中 doc_2（dense 没命中）
        sparse = [
            RetrievalResult(score=10.0, page_content=corpus[2].page_content, metadata=corpus[2].metadata),
        ]

        merged = retriever._merge_results(dense, sparse)

        # 并集：3 条都应保留
        assert len(merged) == 3

    def test_merge_intersection(self):
        """交集场景：两路都命中时，final_score 应加权"""
        corpus = _make_corpus(2)
        retriever = HybridRetriever(alpha=0.7)

        content = corpus[0].page_content
        dense = [RetrievalResult(score=0.8, page_content=content, metadata=corpus[0].metadata)]
        # 两条 sparse 结果，不同分数，验证归一化后加权
        sparse = [
            RetrievalResult(score=10.0, page_content=content, metadata=corpus[0].metadata),
            RetrievalResult(score=5.0, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
        ]

        merged = retriever._merge_results(dense, sparse)

        assert len(merged) == 2
        # sparse 归一化: (10-5)/(10-5)=1.0, (5-5)/(10-5)=0.0
        # doc_0: final = 0.7*0.8 + 0.3*1.0 = 0.86
        doc0 = next(r for r in merged if r.page_content == content)
        assert doc0.dense_score == 0.8
        assert doc0.sparse_score == 1.0
        assert abs(doc0.score - (0.7 * 0.8 + 0.3 * 1.0)) < 1e-6

    def test_merge_empty_sparse(self):
        """sparse 为空时，final 应只剩 dense 贡献"""
        corpus = _make_corpus(2)
        retriever = HybridRetriever(alpha=0.7)

        dense = [
            RetrievalResult(score=0.9, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
            RetrievalResult(score=0.5, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
        ]

        merged = retriever._merge_results(dense, [])

        assert len(merged) == 2
        # sparse 为空 → sparse_norm = 0，final = alpha * dense_score
        for r in merged:
            assert r.sparse_score == 0.0

    def test_merge_empty_dense(self):
        """dense 为空时，final 应只剩 sparse 贡献"""
        corpus = _make_corpus(2)
        retriever = HybridRetriever(alpha=0.7)

        # 两条 sparse 结果（单条时 min==max 归一化为 0）
        sparse = [
            RetrievalResult(score=10.0, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
            RetrievalResult(score=5.0, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
        ]

        merged = retriever._merge_results([], sparse)

        assert len(merged) == 2
        doc0 = next(r for r in merged if r.page_content == corpus[0].page_content)
        # dense=0, sparse_norm=1.0, final = 0.7*0 + 0.3*1.0 = 0.3
        assert doc0.dense_score == 0.0
        assert doc0.sparse_score == 1.0
        assert abs(doc0.score - 0.3) < 1e-6

    def test_merge_sorted_by_final_score(self):
        """合并结果应按 final_score 降序"""
        corpus = _make_corpus(3)
        retriever = HybridRetriever(alpha=0.5)

        dense = [
            RetrievalResult(score=0.9, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
            RetrievalResult(score=0.3, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
        ]
        sparse = [
            RetrievalResult(score=10.0, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
            RetrievalResult(score=1.0, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
        ]

        merged = retriever._merge_results(dense, sparse)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_dense_hit_not_punished_by_sparse_miss(self):
        """dense 高分但 sparse 未命中的文档，不应被归一化成负分而挤出前列。

        回归：旧实现把 raw_sparse=0 直接代入 (0-min_s)/s_rng，当 min_s>0 时
        sparse_norm 为负数，导致 dense=1.0 的文档 final 为负。修复后未命中
        的 sparse_norm 应为 0，dense 高分的文档应排在首位。
        """
        corpus = _make_corpus(3)
        retriever = HybridRetriever(alpha=0.7)

        # dense 高度自信地命中 doc_0
        dense = [
            RetrievalResult(score=1.0, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
        ]
        # sparse 只命中 doc_1/doc_2，且分数都 > 0（这是触发旧 bug 的条件）
        sparse = [
            RetrievalResult(score=10.0, page_content=corpus[1].page_content, metadata=corpus[1].metadata),
            RetrievalResult(score=5.0, page_content=corpus[2].page_content, metadata=corpus[2].metadata),
        ]

        merged = retriever._merge_results(dense, sparse)
        # doc_0 应排第一
        assert merged[0].page_content == corpus[0].page_content
        assert merged[0].dense_score == 1.0
        assert merged[0].sparse_score == 0.0
        assert merged[0].score > 0.0

        # doc_1 的 sparse_norm=1.0，final=0.3；doc_0 的 final=0.7
        doc0 = next(r for r in merged if r.page_content == corpus[0].page_content)
        doc1 = next(r for r in merged if r.page_content == corpus[1].page_content)
        assert doc0.score > doc1.score


# ================================================================
# search (integration)
# ================================================================
class TestSearch:
    @patch("note_assistant.retrieval.hybrid.Ingestor")
    @patch("note_assistant.retrieval.hybrid.OllamaEmbedder")
    def test_search_returns_results(self, mock_embedder_cls, mock_ingestor_cls):
        """search 应返回 RetrievalResult 列表"""
        # Mock embedder
        mock_embedder_cls.return_value.embed_one.return_value = [0.1] * 1024

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档A"]],
            "metadatas": [[{"filepath": "a.md"}]],
            "distances": [[0.2]],
            "ids": [["id_0"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        # Mock BM25
        corpus = _make_corpus(1)
        retriever = HybridRetriever()
        retriever.bm25 = MagicMock()
        retriever.bm25.search.return_value = [
            RetrievalResult(score=5.0, page_content=corpus[0].page_content, metadata=corpus[0].metadata),
        ]

        results = retriever.search("测试", top_k=5)

        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    @patch("note_assistant.retrieval.hybrid.Ingestor")
    @patch("note_assistant.retrieval.hybrid.OllamaEmbedder")
    def test_search_truncates_to_top_k(self, mock_embedder_cls, mock_ingestor_cls):
        """search 应截断到 top_k"""
        mock_embedder_cls.return_value.embed_one.return_value = [0.1] * 1024

        corpus = _make_corpus(4)
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[c.page_content for c in corpus]],
            "metadatas": [[c.metadata for c in corpus]],
            "distances": [[0.1, 0.3, 0.5, 0.7]],
            "ids": [["id_0", "id_1", "id_2", "id_3"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        retriever.bm25 = MagicMock()
        retriever.bm25.search.return_value = [
            RetrievalResult(score=float(i), page_content=c.page_content, metadata=c.metadata)
            for i, c in enumerate(corpus)
        ]

        results = retriever.search("测试", top_k=2)
        assert len(results) == 2

    @patch("note_assistant.retrieval.hybrid.Ingestor")
    @patch("note_assistant.retrieval.hybrid.OllamaEmbedder")
    def test_search_calls_both_retrievers(self, mock_embedder_cls, mock_ingestor_cls):
        """search 应同时调用 dense 和 sparse"""
        mock_embedder_cls.return_value.embed_one.return_value = [0.1] * 1024

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档A"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
            "ids": [["id_0"]],
        }
        mock_ingestor = MagicMock()
        mock_ingestor.collection = mock_collection
        mock_ingestor_cls.return_value = mock_ingestor

        retriever = HybridRetriever()
        retriever.bm25 = MagicMock()
        retriever.bm25.search.return_value = []

        retriever.search("测试")

        # 验证 bm25.search 被调用
        retriever.bm25.search.assert_called_once()
        # 验证 collection.query 被调用
        mock_collection.query.assert_called_once()


# ================================================================
# build_bm25_from_chroma
# ================================================================
class TestBuildBm25FromChroma:
    @patch("note_assistant.retrieval.hybrid.BM25Retriever")
    def test_build_bm25_from_chroma(self, mock_bm25_cls):
        """build_bm25_from_chroma 应调用 BM25Retriever.from_chroma() 并 save"""
        mock_bm25 = MagicMock()
        mock_bm25_cls.from_chroma.return_value = mock_bm25

        retriever = HybridRetriever()
        retriever.build_bm25_from_chroma()

        mock_bm25_cls.from_chroma.assert_called_once()
        mock_bm25.save.assert_called_once()


# ================================================================
# _merge_results —— 结构优先 boost（机制 B）
# ================================================================
class TestMergeResultsStructural:
    def _two_chunks_equal_base(self, meta_a, meta_b):
        """构造两路分数完全相同的一对 chunk，使融合 base 分相等，便于隔离 boost 效果。"""
        retriever = HybridRetriever(alpha=0.7)
        dense = [
            RetrievalResult(score=0.8, page_content="A body", metadata=meta_a),
            RetrievalResult(score=0.8, page_content="B body", metadata=meta_b),
        ]
        sparse = [
            RetrievalResult(score=1.0, page_content="A body", metadata=meta_a),
            RetrievalResult(score=1.0, page_content="B body", metadata=meta_b),
        ]
        return retriever, dense, sparse

    def test_title_hit_boosts_target_chunk(self):
        """query 完整含文档标题时，目标 chunk 应因 title_hit 硬兜底获得 bonus 而排前。"""
        meta_a = {"title": "Code Agent 架构", "heading_path": "二、关键设计点", "dir": "AI/Agents"}
        meta_b = {"title": "其他文档", "heading_path": "一、前言", "dir": "Others"}
        retriever, dense, sparse = self._two_chunks_equal_base(meta_a, meta_b)

        query = "Code Agent 架构的关键设计点是什么"
        merged = retriever._merge_results(dense, sparse, query)

        a = next(r for r in merged if r.page_content == "A body")
        b = next(r for r in merged if r.page_content == "B body")
        # A 因 title 命中 +0.15 bonus，应高于 B
        assert a.score > b.score
        assert abs(a.score - b.score - settings.title_hit_bonus) < 1e-6
        # structural_score 字段应被记录
        assert a.structural_score is not None

    def test_no_query_no_boost(self):
        """不传 query（agent 旧调用路径）→ 不施 boost，base 相等（零回归）。"""
        meta_a = {"title": "Code Agent 架构"}
        meta_b = {"title": "其他文档"}
        retriever, dense, sparse = self._two_chunks_equal_base(meta_a, meta_b)

        merged = retriever._merge_results(dense, sparse)  # query=None
        a = next(r for r in merged if r.page_content == "A body")
        b = next(r for r in merged if r.page_content == "B body")
        assert abs(a.score - b.score) < 1e-9
        assert a.structural_score == 0.0

    def test_irrelevant_query_no_boost(self):
        """query 与结构信号无关 → 无 boost，两 chunk 分数相等。"""
        meta_a = {"title": "Code Agent 架构"}
        meta_b = {"title": "其他文档"}
        retriever, dense, sparse = self._two_chunks_equal_base(meta_a, meta_b)

        merged = retriever._merge_results(dense, sparse, "今天天气怎么样")
        a = next(r for r in merged if r.page_content == "A body")
        b = next(r for r in merged if r.page_content == "B body")
        assert abs(a.score - b.score) < 1e-9

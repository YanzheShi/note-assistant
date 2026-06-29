import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock FlagEmbedding before importing reranker
_flag_mock = MagicMock()
_flag_mock.FlagReranker.return_value = MagicMock()
sys.modules.setdefault("FlagEmbedding", _flag_mock)

from note_assistant.retrieval.reranker import LocalReranker
from note_assistant.retrieval.types import RetrievalResult


# ----------------------------------------------------------------
# 工具
# ----------------------------------------------------------------
_CONTENTS = [
    "RAG 系统结合向量检索和大语言模型",
    "BM25 是一种稀疏检索算法",
    "Faiss 是向量相似度搜索库",
    "混合检索结合 dense 和 sparse 两路",
    "向量检索基于 embedding 相似度",
    "稀疏检索基于词频和逆文档频率",
]


def _make_results(n: int = 4) -> list[RetrievalResult]:
    """构造测试用 RetrievalResult 列表（n 不应超过 len(_CONTENTS)）"""
    n = min(n, len(_CONTENTS))
    return [
        RetrievalResult(
            score=float(n - i),  # 降序分数
            page_content=_CONTENTS[i],
            metadata={"filepath": f"doc_{i}.md", "title": f"文档{i}"},
            dense_score=float(n - i) * 0.8,
            sparse_score=float(n - i) * 0.5,
        )
        for i in range(n)
    ]


# ================================================================
# __init__
# ================================================================
class TestInit:
    def test_default_model_path(self):
        """默认 model_path 应来自 config"""
        reranker = LocalReranker()
        from note_assistant.config import settings
        assert reranker.model_path == settings.reranker_model

    def test_custom_model_path(self):
        """自定义 model_path 应覆盖默认值"""
        reranker = LocalReranker(model_path="/custom/model")
        assert reranker.model_path == "/custom/model"

    def test_use_fp16_default_true(self):
        """use_fp16 默认为 True"""
        reranker = LocalReranker()
        assert reranker.use_fp16 is True

    def test_model_loaded_on_init(self):
        """init 时应加载 FlagReranker"""
        before = _flag_mock.FlagReranker.call_count
        reranker = LocalReranker(model_path="/test/model", use_fp16=False)
        assert _flag_mock.FlagReranker.call_count == before + 1
        _flag_mock.FlagReranker.assert_called_with(
            model_name_or_path="/test/model",
            use_fp16=False,
        )


# ================================================================
# rerank
# ================================================================
class TestRerank:
    def _make_reranker_with_mock(self, scores: list[float]) -> LocalReranker:
        """创建带 mock compute_score 的 reranker"""
        mock_model = MagicMock()
        mock_model.compute_score.return_value = scores
        _flag_mock.FlagReranker.return_value = mock_model
        reranker = LocalReranker(model_path="/mock/model")
        return reranker

    def test_rerank_returns_results(self):
        """rerank 应返回 RetrievalResult 列表"""
        results = _make_results(4)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5, 0.3])

        reranked = reranker.rerank("测试", results, top_k=10)

        assert len(reranked) == 4
        assert all(isinstance(r, RetrievalResult) for r in reranked)

    def test_rerank_truncates_to_top_k(self):
        """rerank 返回数量不应超过 top_k"""
        results = _make_results(4)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5, 0.3])

        reranked = reranker.rerank("测试", results, top_k=2)

        assert len(reranked) == 2

    def test_rerank_sorted_by_score_desc(self):
        """结果应按 rerank score 降序排列"""
        results = _make_results(4)
        # 乱序分数，验证 rerank 排序
        reranker = self._make_reranker_with_mock([0.3, 0.9, 0.5, 0.7])

        reranked = reranker.rerank("测试", results, top_k=10)

        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.7
        assert reranked[2].score == 0.5
        assert reranked[3].score == 0.3

    def test_rerank_updates_score(self):
        """rerank 后 score 应为 rerank 模型返回的分数"""
        results = _make_results(3)
        reranker = self._make_reranker_with_mock([0.95, 0.6, 0.3])

        reranked = reranker.rerank("测试", results, top_k=3)

        assert reranked[0].score == 0.95
        assert reranked[1].score == 0.6
        assert reranked[2].score == 0.3

    def test_rerank_preserves_metadata(self):
        """rerank 应保留原始 metadata"""
        results = _make_results(3)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5])

        reranked = reranker.rerank("测试", results, top_k=3)

        assert reranked[0].metadata["filepath"] == "doc_0.md"
        assert reranked[1].metadata["filepath"] == "doc_1.md"
        assert reranked[2].metadata["filepath"] == "doc_2.md"

    def test_rerank_preserves_page_content(self):
        """rerank 应保留 page_content"""
        results = _make_results(3)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5])

        reranked = reranker.rerank("测试", results, top_k=3)

        # 按 filepath 查找，不依赖顺序
        by_fp = {r.filepath: r for r in reranked}
        assert by_fp["doc_0.md"].page_content == "RAG 系统结合向量检索和大语言模型"
        assert by_fp["doc_2.md"].page_content == "Faiss 是向量相似度搜索库"

    def test_rerank_preserves_dense_sparse_scores(self):
        """rerank 应保留原始的 dense_score 和 sparse_score"""
        results = _make_results(3)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5])

        reranked = reranker.rerank("测试", results, top_k=3)

        by_fp = {r.filepath: r for r in reranked}
        # _make_results(3): doc_0 score=3.0, dense=2.4, sparse=1.5
        assert abs(by_fp["doc_0.md"].dense_score - 2.4) < 1e-6
        assert abs(by_fp["doc_0.md"].sparse_score - 1.5) < 1e-6
        # doc_1 score=2.0, dense=1.6, sparse=1.0
        assert abs(by_fp["doc_1.md"].dense_score - 1.6) < 1e-6

    def test_rerank_calls_compute_score(self):
        """rerank 应调用 model.compute_score"""
        results = _make_results(3)
        reranker = self._make_reranker_with_mock([0.9, 0.7, 0.5])

        reranker.rerank("测试 query", results, top_k=3)

        reranker.model.compute_score.assert_called_once()
        # 验证传入的是 [query, doc] 对
        call_args = reranker.model.compute_score.call_args[0][0]
        assert call_args[0] == ["测试 query", "RAG 系统结合向量检索和大语言模型"]
        assert call_args[1] == ["测试 query", "BM25 是一种稀疏检索算法"]

    def test_rerank_default_top_k(self):
        """top_k 默认值应从 config 读取"""
        from note_assistant.config import settings
        # 构造足够多的结果（不超过 _CONTENTS 长度 6）
        n = min(settings.top_k_rerank + 2, len(_CONTENTS))
        results = _make_results(n)
        scores = [float(i) for i in range(n, 0, -1)]
        reranker = self._make_reranker_with_mock(scores)

        reranked = reranker.rerank("测试", results)

        assert len(reranked) == settings.top_k_rerank

    def test_rerank_empty_results(self):
        """空结果应返回空列表"""
        reranker = self._make_reranker_with_mock([])
        reranked = reranker.rerank("测试", [])
        assert reranked == []

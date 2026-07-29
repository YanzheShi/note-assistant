# tests/retrieval/test_hybrid_v2b.py
"""v2b 父块展开（_expand_to_parents）测试。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

from note_assistant.config import settings
from note_assistant.retrieval.docstore import ParentDocstore
from note_assistant.retrieval.hybrid import HybridRetriever
from note_assistant.retrieval.types import RetrievalResult

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock ollama before importing modules that create a client on init
_ollama_mock = MagicMock()
_ollama_mock.Client.return_value = MagicMock()
sys.modules.setdefault("ollama", _ollama_mock)


def _child(pid: str, score: float, content: str) -> RetrievalResult:
    return RetrievalResult(
        score=score,
        page_content=content,
        metadata={"parent_id": pid, "filepath": "a.md", "title": "X", "heading_path": "二、关键设计点"},
    )


class TestExpandToParents:
    def _build_docstore(self, tmp_path: Path) -> ParentDocstore:
        ds = ParentDocstore(tmp_path / "ds.pkl")
        ds.add("p1", "整节P1 完整内容", {"title": "X", "heading_path": "二、关键设计点", "parent_id": "p1"})
        ds.add("p2", "整节P2 完整内容", {"title": "Y", "heading_path": "三、其他", "parent_id": "p2"})
        return ds

    def test_expands_children_to_parents(self, tmp_path):
        retriever = HybridRetriever()
        retriever._docstore = self._build_docstore(tmp_path)
        monkeypatch_settings("v2b")

        results = retriever._expand_to_parents([
            _child("p1", 0.9, "子块1片段"),
            _child("p2", 0.7, "子块2片段"),
        ])
        assert len(results) == 2
        assert results[0].page_content == "整节P1 完整内容"
        assert results[1].page_content == "整节P2 完整内容"
        # 元数据来自父块
        assert results[0].metadata["heading_path"] == "二、关键设计点"

    def test_dedup_same_parent_keeps_highest_score(self, tmp_path):
        retriever = HybridRetriever()
        retriever._docstore = self._build_docstore(tmp_path)
        monkeypatch_settings("v2b")

        results = retriever._expand_to_parents([
            _child("p1", 0.6, "低分片段"),
            _child("p1", 0.95, "高分片段"),
            _child("p2", 0.7, "子块2片段"),
        ])
        # p1 两个子块去重为 1 条，取最高分 0.95
        p1 = [r for r in results if r.metadata["parent_id"] == "p1"]
        assert len(p1) == 1
        assert p1[0].score == 0.95
        assert len(results) == 2

    def test_non_v2b_passthrough(self, tmp_path):
        retriever = HybridRetriever()
        retriever._docstore = self._build_docstore(tmp_path)
        monkeypatch_settings("v2")

        results = retriever._expand_to_parents([
            _child("p1", 0.9, "子块1片段"),
        ])
        # 非 v2b 应原样返回，不替换正文
        assert len(results) == 1
        assert results[0].page_content == "子块1片段"

    def test_missing_docstore_passthrough(self, tmp_path):
        retriever = HybridRetriever()
        retriever._docstore = None
        monkeypatch_settings("v2b")

        results = retriever._expand_to_parents([
            _child("p1", 0.9, "子块1片段"),
        ])
        assert results[0].page_content == "子块1片段"


def monkeypatch_settings(strategy: str):
    """临时把 chunking_strategy 切到指定值（用 pytest 的 monkeypatch 语义太重，这里直接 setattr）。"""
    settings.chunking_strategy = strategy  # noqa: B018  (测试内故意修改全局配置)

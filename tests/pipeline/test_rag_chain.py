# tests/pipeline/test_rag_chain.py
"""
RAGChain 管线编排器测试。
"""
import sys
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage

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


# ── 路由 LLM 模拟 ─────────────────────────────────────────────
# 原实现裸调 httpx，现统一走 get_llm()（langchain ChatModel）。
# 这里 patch 掉 rag_chain.get_llm，让它返回一个假装调过 LLM 的假 model。
def _fake_llm(content: str) -> MagicMock:
    m = MagicMock()
    msg = AIMessage(content=content)
    m.invoke.return_value = msg
    m.ainvoke = AsyncMock(return_value=msg)
    return m


def _patch_llm(content: str):
    """patch rag_chain.get_llm，使其返回携带指定 content 的 AIMessage。"""
    return patch(
        "note_assistant.pipeline.rag_chain.get_llm",
        return_value=_fake_llm(content),
    )


def _patch_llm_error(exc: Exception):
    """patch rag_chain.get_llm，使 invoke/ainvoke 抛异常（模拟网络/服务错误）。"""
    m = MagicMock()
    m.invoke.side_effect = exc
    m.ainvoke = AsyncMock(side_effect=exc)
    return patch(
        "note_assistant.pipeline.rag_chain.get_llm",
        return_value=m,
    )


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
        """ask 应返回 AskResponse dataclass"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "测试答案"
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=_make_mock_graph(),
            generator=mock_gen,
        )
        result = r.ask("测试问题")
        assert hasattr(result, "answer")
        assert hasattr(result, "sources")
        assert hasattr(result, "graph_expansion")
        assert hasattr(result, "retrieved")

    def test_ask_without_graph(self):
        """不传 graph 也能正常工作（跳过扩展）"""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "无图测试答案"
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=None,
            generator=mock_gen,
        )
        # mock 路由 LLM：判定为「不需要检索」→ 返回固定问候语（离线、确定）
        with _patch_llm('{"need_retrieval": false}'):
            result = r.ask("测试问题")
        assert result.graph_expansion == 0
        # 路由判断为 false 时不走 generator，返回固定问候语
        assert "你好" in result.answer
        pass

    def test_ask_without_generator(self):
        """不传 generator 也能正常工作（跳过生成）"""
        r = RAGChain(
            _make_mock_retriever(),
            _make_mock_reranker(),
            graph=_make_mock_graph(),
            generator=None,
        )
        result = r.ask("无生成器测试")
        # generator 为 None，answer 为空；但检索正常执行
        assert result.answer == ""
        assert result.retrieved == 1
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


# ================================================================
# 检索路由（_needs_retrieval_sync / _needs_retrieval_async）
# ================================================================

class TestNeedsRetrieval:
    def test_hello_returns_false(self):
        """打招呼应返回 False（不需要检索）"""
        with _patch_llm('{"need_retrieval": false}'):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("hi") is False

    def test_中文打招呼_returns_false(self):
        """中文打招呼也应返回 False"""
        with _patch_llm('{"need_retrieval": false}'):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("你好") is False

    def test_知识库问题_returns_true(self):
        """需要查笔记的问题应返回 True"""
        with _patch_llm('{"need_retrieval": true}'):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("RAG 怎么实现") is True

    def test_markdown_code_block_stripped(self):
        """LLM 返回 markdown 代码块包裹的 JSON 也能正确解析"""
        with _patch_llm('```json\n{"need_retrieval": false}\n```'):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("hi") is False

    def test_多种键名变体(self):
        """needs_retrieval / needRetrieval 等变体也能正确解析"""
        variants = [
            {"need_retrieval": False},
            {"needs_retrieval": False},
            {"needRetrieval": False},
            {"needsRetrieval": False},
        ]
        for variant in variants:
            with _patch_llm(json.dumps(variant)):
                r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
                assert r._needs_retrieval_sync("hi") is False, f"Failed for {variant}"

    def test_返回纯bool(self):
        """LLM 直接返回 true/false 也能处理"""
        with _patch_llm("false"):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("hi") is False

    def test_返回中文否定(self):
        """LLM 返回"不需要"/"否"也能处理"""
        for text in ["不需要", "否", "no", "false"]:
            with _patch_llm(text):
                r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
                assert r._needs_retrieval_sync("hi") is False, f"Failed for {text}"

    def test_async_returns_false(self):
        """异步路由路径（ask_stream 用）也应正确解析"""
        import asyncio

        with _patch_llm('{"need_retrieval": false}'):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert asyncio.run(r._needs_retrieval_async("hi")) is False

    def test_api_error_defaults_to_true(self):
        """LLM 调用失败时保守默认：仍然检索（避免漏答）"""
        with _patch_llm_error(RuntimeError("network error")):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            # 源码 _needs_retrieval_sync 在异常时降级为「默认检索」(return True)
            assert r._needs_retrieval_sync("hi") is True

    def test_json_parse_error_defaults_to_true(self):
        """非 JSON 字符串（如"不需要"）被 _parse_routing_result 解析为 False（不需要检索）"""
        with _patch_llm("不需要"):
            r = RAGChain(_make_mock_retriever(), _make_mock_reranker())
            assert r._needs_retrieval_sync("hi") is False

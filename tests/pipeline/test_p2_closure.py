# tests/pipeline/test_p2_closure.py
"""P2 检索/生成/展示闭环测试：

- source_kind 透传 asset_id/img_url
- 图意图 boost（hybrid 融合分对 image chunk 加权）
- 图片邻居扩展（命中图片带出同章节文本）
- [[IMG:]] 后处理（ask 链路把标记替换为真实图片 markdown）
- SourceSchema / SourceInfo 携带 asset_id/img_url
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from note_assistant.retrieval.types import RetrievalResult
from note_assistant.pipeline.source_kind import classify_source
from note_assistant.retrieval.hybrid import HybridRetriever
from note_assistant.pipeline.rag_chain import RAGChain, SourceInfo
from note_assistant.api.schemas import SourceSchema, AgentSource


def _result(content: str, metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(score=0.9, page_content=content, metadata=metadata or {})


async def _async_true(*args, **kwargs) -> bool:
    """替换异步检索路由，跳过真实 LLM 调用。"""
    return True


# ── source_kind 资产透传 ──
class TestSourceKindAsset:
    def test_asset_id_and_img_url_surfaced(self):
        meta = {
            "kind": "image",
            "img_src": "attach/p.png",
            "asset_id": "abc123def4567890",
            "img_url": "/assets/abc123def4567890",
        }
        r = classify_source("图片: p.png", meta)
        assert r["kind"] == "image"
        assert r["asset_id"] == "abc123def4567890"
        assert r["img_url"] == "/assets/abc123def4567890"

    def test_missing_asset_fields_empty(self):
        r = classify_source("普通文本", {})
        assert r["asset_id"] == ""
        assert r["img_url"] == ""


# ── 图意图 boost ──
class TestImageIntentBoost:
    def _merge(self, query):
        dense = [RetrievalResult(score=0.5, page_content="img", metadata={"kind": "image"})]
        sparse = [RetrievalResult(score=0.0, page_content="img", metadata={"kind": "image"})]
        fake = SimpleNamespace(alpha=0.7)
        return HybridRetriever._merge_results(fake, dense, sparse, query)

    def test_boost_on_image_intent(self):
        boosted = self._merge("架构图长什么样")
        plain = self._merge("RAG 是什么")
        assert boosted[0].score > plain[0].score

    def test_text_chunk_unaffected(self):
        dense = [RetrievalResult(score=0.5, page_content="txt", metadata={"kind": "text"})]
        sparse = [RetrievalResult(score=0.0, page_content="txt", metadata={"kind": "text"})]
        fake = SimpleNamespace(alpha=0.7)
        a = HybridRetriever._merge_results(fake, dense, sparse, "架构图长什么样")
        b = HybridRetriever._merge_results(fake, dense, sparse, "RAG 是什么")
        assert a[0].score == b[0].score


# ── 图片邻居扩展 ──
class TestImageNeighborExpansion:
    def _chain(self):
        return RAGChain(SimpleNamespace(), SimpleNamespace())

    def test_expands_same_heading_text(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", True)
        rerank = [_result("图", {"kind": "image", "heading_path": "H1", "filepath": "n.md",
                                 "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890"})]
        neighbors = [
            _result("正文A", {"kind": "text", "heading_path": "H1", "filepath": "n.md"}),
            _result("正文B", {"kind": "text", "heading_path": "H2", "filepath": "n.md"}),  # 不同章节，不带
            _result("图2", {"kind": "image", "heading_path": "H1", "filepath": "n.md"}),  # 自身图，跳过
        ]

        def fake_fetch(paths):
            return neighbors

        added = self._chain()._expand_image_neighbors(rerank, fetch_fn=fake_fetch)
        assert len(added) == 1
        assert added[0].page_content == "正文A"

    def test_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        rerank = [_result("图", {"kind": "image", "heading_path": "H1"})]
        added = self._chain()._expand_image_neighbors(rerank, fetch_fn=lambda p: [])
        assert added == []

    def test_no_image_no_expand(self):
        rerank = [_result("文本", {"kind": "text", "heading_path": "H1"})]
        added = self._chain()._expand_image_neighbors(rerank, fetch_fn=lambda p: [])
        assert added == []


# ── ask 链路 [[IMG:]] 后处理 ──
class TestAskPostprocess:
    def test_marker_replaced_in_answer(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        img = _result(
            "图摘要",
            {"kind": "image", "filepath": "n.md", "heading_path": "H1",
             "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
             "image_title": "架构图"},
        )

        class FakeRetriever:
            def search(self, q, top_k=None):
                return [img]

        class FakeReranker:
            def rerank(self, q, results, top_k=None):
                return results

        class FakeGenerator:
            def generate(self, q, context, history=None):
                return "见 [[IMG:abc123def4567890]] 所示。"
            def generate_stream(self, q, context, history=None):
                return iter([])

        chain = RAGChain(FakeRetriever(), FakeReranker(), None, FakeGenerator())
        chain._needs_retrieval_sync = lambda q: True  # 跳过 LLM 路由

        resp = chain.ask("记忆系统架构图长什么样")
        assert resp.answer == "见 ![架构图](/assets/abc123def4567890) 所示。"
        assert "[[IMG:" not in resp.answer
        # SourceInfo 也携带 asset 字段
        assert resp.sources[0].asset_id == "abc123def4567890"
        assert resp.sources[0].img_url == "/assets/abc123def4567890"


# ── ask_stream 流式 [[IMG:]] 后处理（回归：曾经算完 postprocess 却没用，标记裸奔到前端）──
class TestAskStreamPostprocess:
    def _chain(self, tokens):
        img = _result(
            "图摘要",
            {"kind": "image", "filepath": "n.md", "heading_path": "H1",
             "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
             "image_title": "架构图"},
        )

        class FakeRetriever:
            def search(self, q, top_k=None):
                return [img]

        class FakeReranker:
            def rerank(self, q, results, top_k=None):
                return results

        class FakeGenerator:
            def generate(self, q, context, history=None):
                return "".join(tokens)

            async def generate_stream(self, q, context, history=None):
                for t in tokens:
                    yield t

        chain = RAGChain(FakeRetriever(), FakeReranker(), None, FakeGenerator())
        chain._needs_retrieval = _async_true
        chain._needs_retrieval_sync = lambda q: True
        return chain

    async def _collect(self, agen):
        chars, others = [], []
        async for ev in agen:
            (chars if ev.get("type") == "char" else others).append(ev)
        return "".join(c["content"] for c in chars), others

    @pytest.mark.asyncio
    async def test_stream_replaces_marker_across_tokens(self):
        # 标记被切成多个 token —— 这正是老实现漏掉的场景
        tokens = ["见 ", "[[", "IMG:", "abc123def4567890", "]]", " 所示。"]
        chain = self._chain(tokens)
        text, _ = await self._collect(chain.ask_stream("架构图长什么样"))
        assert text == "见 ![架构图](/assets/abc123def4567890) 所示。"
        assert "[[IMG:" not in text

    @pytest.mark.asyncio
    async def test_stream_no_duplicate_emission(self):
        # 不能既流原文、又在末尾整段补发一次（会重复）
        tokens = ["纯", "文本", "回答"]
        chain = self._chain(tokens)
        text, _ = await self._collect(chain.ask_stream("架构图长什么样"))
        assert text == "纯文本回答"

    @pytest.mark.asyncio
    async def test_trace_stream_replaces_marker(self):
        # ask_with_trace 自己拆步检索，需要更完整的 retriever 假件
        tokens = ["见 [[IMG:", "abc123def4567890", "]] 所示。"]
        img = _result(
            "图摘要",
            {"kind": "image", "filepath": "n.md", "heading_path": "H1",
             "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
             "image_title": "架构图"},
        )

        class FakeTraceRetriever:
            top_k = 5
            embedder = SimpleNamespace(embed_one=lambda q: [0.1, 0.2])

            def _dense_search(self, emb, n):
                return [img]

            def _sparse_search(self, q, n):
                return []

            def _merge_results(self, dense, sparse):
                return dense

        class FakeReranker:
            def rerank(self, q, results, top_k=None):
                return results

        class FakeGenerator:
            async def generate_stream(self, q, context, history=None):
                for t in tokens:
                    yield t

        chain = RAGChain(FakeTraceRetriever(), FakeReranker(), None, FakeGenerator())
        chain._needs_retrieval_async = _async_true

        text, others = await self._collect(chain.ask_with_trace("架构图长什么样"))
        assert "![架构图](/assets/abc123def4567890)" in text
        assert "[[IMG:" not in text
        # 检索步骤 trace 仍然照常推送（后处理没破坏 trace 语义）
        steps = [e.get("step") for e in others if e.get("type") == "trace"]
        assert "embedding" in steps and "rerank" in steps


# ── ask 链路图片保位 + 确定性补图（「检索到了却不显示」的两道护栏）──
class TestAskImagePinAndAutoAppend:
    """rerank 把图片挤出 top-k（交叉编码器清零图意图 boost）时保位；
    LLM 未写 [[IMG:]] 标记时确定性补图。"""

    IMG_META = {
        "kind": "image", "filepath": "n.md", "heading_path": "H1",
        "asset_id": "abc123def4567890", "img_url": "/assets/abc123def4567890",
        "image_title": "架构图", "image_description": "三层架构",
    }

    def _chain(self, generator_answer: str):
        img = _result("图摘要", dict(self.IMG_META))
        t1 = _result("正文一", {"kind": "text", "filepath": "n.md", "heading_path": "H1"})
        t2 = _result("正文二", {"kind": "text", "filepath": "n.md", "heading_path": "H1"})

        class FakeRetriever:
            def search(self, q, top_k=None):
                return [t1, t2, img]

        class FakeCutReranker:
            """模拟交叉编码器：文本分 > 图片分，top_k 截断时图片被挤出。"""

            def rerank(self, q, results, top_k=None):
                scored = []
                for r in results:
                    s = 0.1 if r.metadata.get("kind") == "image" else 0.9
                    scored.append(RetrievalResult(score=s, page_content=r.page_content, metadata=r.metadata))
                scored.sort(key=lambda r: r.score, reverse=True)
                return scored[:top_k] if top_k is not None else scored

        class FakeGenerator:
            def generate(self, q, context, history=None):
                return generator_answer

            def generate_stream(self, q, context, history=None):
                return iter([])

        chain = RAGChain(FakeRetriever(), FakeCutReranker(), None, FakeGenerator())
        chain._needs_retrieval_sync = lambda q: True
        return chain

    def test_image_pinned_into_sources(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        chain = self._chain("回答中没有图标记。")
        resp = chain.ask("摄取与索引管线的架构图是什么样")
        # 图片被 rerank 挤出 top-2，但图意图保位把它补回 → 来源里必须有 image
        assert any(s.kind == "image" for s in resp.sources)
        assert any(s.img_url == "/assets/abc123def4567890" for s in resp.sources)

    def test_answer_without_marker_gets_images_appended(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        chain = self._chain("回答中没有图标记。")
        resp = chain.ask("摄取与索引管线的架构图是什么样")
        assert "![架构图](/assets/abc123def4567890)" in resp.answer
        assert "相关图片" in resp.answer

    def test_answer_with_marker_not_duplicated(self, monkeypatch):
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        chain = self._chain("见 [[IMG:abc123def4567890]] 所示。")
        resp = chain.ask("摄取与索引管线的架构图是什么样")
        assert resp.answer.count("/assets/abc123def4567890") == 1
        assert "相关图片" not in resp.answer

    def test_no_intent_no_pin(self, monkeypatch):
        # 非图意图 query：图片被 rerank 裁掉后不强塞（避免无关图干扰）
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        chain = self._chain("回答。")
        resp = chain.ask("RAG 的检索流程是什么")
        assert not any(s.kind == "image" for s in resp.sources)
        assert "/assets/" not in resp.answer

    @pytest.mark.asyncio
    async def test_stream_appends_images_when_no_marker(self, monkeypatch):
        # 流式同样闭环：答案 token 里没有标记 → 流末补发图片块
        monkeypatch.setattr("note_assistant.config.settings.image_neighbor_expand", False)
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        img = _result("图摘要", dict(self.IMG_META))
        t1 = _result("正文一", {"kind": "text", "filepath": "n.md", "heading_path": "H1"})
        t2 = _result("正文二", {"kind": "text", "filepath": "n.md", "heading_path": "H1"})

        class FakeRetriever:
            def search(self, q, top_k=None):
                return [t1, t2, img]

        class FakeCutReranker:
            def rerank(self, q, results, top_k=None):
                scored = []
                for r in results:
                    s = 0.1 if r.metadata.get("kind") == "image" else 0.9
                    scored.append(RetrievalResult(score=s, page_content=r.page_content, metadata=r.metadata))
                scored.sort(key=lambda r: r.score, reverse=True)
                return scored[:top_k] if top_k is not None else scored

        class FakeGenerator:
            def generate(self, q, context, history=None):
                return "无标记回答"

            async def generate_stream(self, q, context, history=None):
                for t in ["无标记", "回答"]:
                    yield t

        chain = RAGChain(FakeRetriever(), FakeCutReranker(), None, FakeGenerator())
        chain._needs_retrieval_async = _async_true
        chain._needs_retrieval_sync = lambda q: True

        chars = []
        async for ev in chain.ask_stream("架构图长什么样"):
            if ev.get("type") == "char":
                chars.append(ev["content"])
        text = "".join(chars)
        assert text.startswith("无标记回答")
        assert "![架构图](/assets/abc123def4567890)" in text


# ── Schema 资产字段 ──
class TestSchemaAssetFields:
    def test_source_schema_asset_fields(self):
        s = SourceSchema(
            type="image", kind="image", img_path="x.png",
            asset_id="abc123def4567890", img_url="/assets/abc123def4567890",
        )
        d = s.model_dump(mode="json")
        assert d["asset_id"] == "abc123def4567890"
        assert d["img_url"] == "/assets/abc123def4567890"

    def test_source_schema_asset_fields_optional(self):
        s = SourceSchema(type="text")
        d = s.model_dump(mode="json")
        assert d["asset_id"] is None
        assert d["img_url"] is None

    def test_agent_source_image_fields(self):
        a = AgentSource(
            filepath="n.md", title="标题", kind="image",
            img_url="/assets/abc123def4567890", render_hint="svg:inline",
        )
        d = a.model_dump(mode="json")
        assert d["kind"] == "image"
        assert d["img_url"] == "/assets/abc123def4567890"
        assert d["render_hint"] == "svg:inline"

    def test_source_info_asset_fields_passthrough(self):
        info = SourceInfo.from_result(
            _result("图", {"kind": "image", "asset_id": "abc123def4567890",
                           "img_url": "/assets/abc123def4567890", "filepath": "n.md"}),
            origin="direct",
        )
        assert info.asset_id == "abc123def4567890"
        assert info.img_url == "/assets/abc123def4567890"


# ── ask_with_trace 精排宽度（审计修复：旧实现把 top_k_retrieve(20) 当 rerank 截断）──
class TestTraceRerankWidth:
    @pytest.mark.asyncio
    async def test_trace_rerank_truncates_to_top_k_rerank(self, monkeypatch):
        """候选池保持 top_k_retrieve，但 rerank 截断必须用 top_k_rerank——
        旧实现上下文塞 20 条 chunk，偏离 ask() 路径。"""
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 5)
        results = [
            _result(f"正文{i}", {"kind": "text", "filepath": "n.md", "heading_path": f"H{i}"})
            for i in range(20)
        ]

        class FakeTraceRetriever:
            top_k = 20  # top_k_retrieve（候选池宽度）
            embedder = SimpleNamespace(embed_one=lambda q: [0.1])

            def _dense_search(self, emb, n):
                return results

            def _sparse_search(self, q, n):
                return []

            def _merge_results(self, dense, sparse):
                return dense

        class FakeReranker:
            def __init__(self):
                self.candidates_seen = None

            def rerank(self, q, rs, top_k=None):
                self.candidates_seen = len(rs)
                return rs[:top_k] if top_k else rs

        class FakeGenerator:
            async def generate_stream(self, q, context, history=None):
                yield "答"

        reranker = FakeReranker()
        chain = RAGChain(FakeTraceRetriever(), reranker, None, FakeGenerator())
        chain._needs_retrieval_async = _async_true

        sources_len = None
        async for ev in chain.ask_with_trace("任意问题"):
            if ev.get("type") == "sources":
                sources_len = len(ev["content"])

        assert reranker.candidates_seen == 20   # 候选池完整进入精排
        assert sources_len == 5                  # 输出按 top_k_rerank 截断（旧实现会是 20）

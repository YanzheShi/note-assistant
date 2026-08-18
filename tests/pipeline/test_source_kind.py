# tests/pipeline/test_source_kind.py
"""P0 修复点 4/5：来源内容类型判定（source_kind）与 SourceInfo 契约。

历史契约 bug：pipeline 把「来源渠道」(direct/graph) 直接透传给前端的「内容类型」
(text/table/mermaid/image)，导致前端永远收到 "direct"，四个渲染分支全是死代码。
修复拆成两套正交字段：origin（渠道）+ kind（内容类型），由 classify_source 判定 kind。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.retrieval.types import RetrievalResult
from note_assistant.pipeline.source_kind import classify_source, VALID_KINDS
from note_assistant.pipeline.rag_chain import SourceInfo


def _result(content: str, metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(score=0.9, page_content=content, metadata=metadata or {})


class TestClassifySource:
    def test_plain_text(self):
        r = classify_source("这是一段普通文本。", {})
        assert r["kind"] == "text"
        assert r["img_path"] == ""
        assert r["raw_table"] == ""
        assert r["raw_mermaid"] == ""

    def test_image_from_summary_meta(self):
        r = classify_source("图片: p.png", {"kind": "image", "img_src": "attach/p.png"})
        assert r["kind"] == "image"
        assert r["img_path"] == "attach/p.png"

    def test_image_from_body(self):
        r = classify_source("参考 ![[shot.png]] 如下。", {"has_image": True})
        assert r["kind"] == "image"
        assert r["img_path"] == "shot.png"

    def test_table_from_body(self):
        content = "正文\n| A | B |\n|---|---|\n| 1 | 2 |\n尾部"
        r = classify_source(content, {})
        assert r["kind"] == "table"
        assert "| A | B |" in r["raw_table"]

    def test_mermaid_from_body(self):
        content = "```mermaid\ngraph TD; A-->B\n```"
        r = classify_source(content, {})
        assert r["kind"] == "mermaid"
        assert "graph TD" in r["raw_mermaid"]

    def test_mermaid_summary_meta_yields_raw_mermaid_and_hints(self):
        """P1-b：mermaid summary chunk 的 page_content 是结构化文本（无 ```fence），
        raw_mermaid 需回退到 metadata 的 mermaid_src；render_hint/diagram_type 也透传。"""
        content = "Mermaid flowchart 图: 节点 A[输入]... 边 A->B..."
        meta = {
            "kind": "mermaid",
            "mermaid_src": "```mermaid\ngraph TD\n A[输入] --> B[输出]\n```",
            "render_hint": "mermaid:inline",
            "diagram_type": "flowchart",
        }
        r = classify_source(content, meta)
        assert r["kind"] == "mermaid"
        assert "graph TD" in r["raw_mermaid"]
        assert r["render_hint"] == "mermaid:inline"
        assert r["diagram_type"] == "flowchart"

    def test_image_priority_over_table(self):
        """图片必须走专门的渲染分支，判错代价最大，因此优先于表格/流程图；
        但表格载荷仍被抽出（不受 kind 约束，避免漏渲染）。"""
        content = "| A | B |\n|---|---|\n| 1 | 2 |\n![[x.png]]"
        r = classify_source(content, {"has_image": True})
        assert r["kind"] == "image"
        assert r["img_path"] == "x.png"
        assert "| A | B |" in r["raw_table"]

    def test_declared_kind_wins(self):
        # summary chunk 的 metadata["kind"] 最权威，优先于正文嗅探
        r = classify_source("![[x.png]]", {"kind": "table"})
        assert r["kind"] == "table"


class TestSourceInfoContract:
    def test_from_result_sets_origin_and_kind(self):
        """来源渠道(origin)与内容类型(kind)是两套正交字段。"""
        res = _result("参考 ![[shot.png]] 如下。", {"filepath": "n.md", "heading_path": "章节"})
        info = SourceInfo.from_result(res, origin="direct")
        assert info.origin == "direct"
        assert info.kind == "image"
        assert info.img_path == "shot.png"
        d = info.to_dict()
        assert "origin" in d and "kind" in d
        assert d["origin"] == "direct"
        assert d["kind"] == "image"

    def test_from_result_graph_origin(self):
        res = _result("普通文本段落。", {"filepath": "n.md"})
        info = SourceInfo.from_result(res, origin="graph")
        assert info.kind == "text"
        assert info.origin == "graph"

    def test_from_result_carries_render_hint_and_diagram_type(self):
        """P1-b：mermaid summary 命中的 render_hint/diagram_type 经 classify_source
        一路透传到 SourceInfo，前端才得以原生渲染。"""
        meta = {
            "kind": "mermaid",
            "mermaid_src": "```mermaid\ngraph TD\n A-->B\n```",
            "render_hint": "mermaid:inline",
            "diagram_type": "flowchart",
        }
        res = _result("Mermaid flowchart 图: ...", meta)
        info = SourceInfo.from_result(res, origin="direct")
        assert info.kind == "mermaid"
        assert info.render_hint == "mermaid:inline"
        assert info.diagram_type == "flowchart"
        d = info.to_dict()
        assert d["render_hint"] == "mermaid:inline"
        assert d["diagram_type"] == "flowchart"

    def test_valid_kinds_constant(self):
        assert VALID_KINDS == frozenset({"text", "table", "mermaid", "image"})

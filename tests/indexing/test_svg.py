"""P1-d SVG 原生解析层 测试（离线、零 VLM）。

验证：SVG 抽 <text> + 图形结构（rect 节点 + 带箭头 line 边），输出统一 DiagramGraph；
非法 SVG 降级抛错；经 make_image_enricher 的 svg 路由零 VLM 富化。
"""
from note_assistant.indexing.svg import SVGParseError, SVGParser
from note_assistant.indexing.understanding import make_image_enricher
from note_assistant.indexing.types import ExtractedChunk


SAMPLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
  <rect id="a" x="10" y="30" width="60" height="40"/>
  <text x="25" y="55">输入</text>
  <rect id="b" x="120" y="30" width="60" height="40"/>
  <text x="135" y="55">处理</text>
  <line x1="70" y1="50" x2="120" y2="50" stroke="black" marker-end="url(#arrow)"/>
  <text x="85" y="45">流程</text>
</svg>"""


def test_svg_parse_nodes_edges():
    dg = SVGParser.parse(SAMPLE_SVG)
    labels = {n.label for n in dg.nodes}
    assert labels == {"输入", "处理"}, labels
    assert len(dg.edges) == 1
    e = dg.edges[0]
    assert e.label == "流程", e.label
    # 节点 id 应与 from/to 对应
    ids = {n.id for n in dg.nodes}
    assert e.from_id in ids and e.to_id in ids
    # 正文可检索：text + 边都进了 raw_text
    assert "输入" in dg.raw_text and "处理" in dg.raw_text and "流程" in dg.raw_text
    assert dg.diagram_type == "flowchart"
    assert dg.source_format == "svg"
    assert dg.render_hint == "svg:inline"


def test_svg_invalid_degrades():
    import pytest
    with pytest.raises(SVGParseError):
        SVGParser.parse("not an svg <<<")


def test_svg_no_structure_text_only():
    """纯文本 SVG（无图形结构）→ 仍产出 raw_text，不报错。"""
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><text x="0" y="0">纯文字节点</text></svg>'
    dg = SVGParser.parse(svg)
    assert "纯文字节点" in dg.raw_text
    assert dg.diagram_type in ("unknown", "architecture")


def test_enricher_svg_route_zero_vlm(tmp_path, monkeypatch):
    """mime=svg 的图片走原生解析，零 VLM 调用，富化为 SVG 图摘要。"""
    p = tmp_path / "flow.svg"
    p.write_text(SAMPLE_SVG)
    # G6：enricher 总开关默认 False，这里显式开启以验证 svg 路由
    monkeypatch.setattr("note_assistant.config.settings.image_understand_enabled", True)
    enricher = make_image_enricher(tmp_path)
    ext = ExtractedChunk(
        uid="[IMAGE_UID_00000009]", kind="image",
        placeholder="[IMAGE_UID_00000009]",
        raw=f"![]({p.name})", context="前文",
        meta={"src": str(p), "alt": "流程图"},
    )
    out = enricher(ext, "章节")
    assert out is not None
    summary, meta = out
    assert summary.startswith("SVG 图:")
    assert "输入" in summary and "处理" in summary
    assert meta["render_hint"] == "svg:inline"
    assert meta["diagram_type"] == "flowchart"
    assert meta["has_diagram"] is True
    assert meta["source_format"] == "svg"
    assert "svg_raw" in meta  # 小文件应内联原始 SVG 供前端渲染

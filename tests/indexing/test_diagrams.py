"""P1 内联 Mermaid 原生解析层测试（5.B.4.4）。

验证点：
- MermaidParser 能从 flowchart 抽取节点/边/形状，并产出结构化 raw_text + render_hint
- 非流程图类型（sequence 等）类型识别正确，整段源码兜底为可检索文本
- 富化后 generate_summaries 的 mermaid summary chunk 带 render_hint/diagram_type
- 解析失败（空 mermaid）降级为旧弱摘要，绝不中断索引
"""
import pytest

from note_assistant.indexing.diagrams import MermaidParser, _strip_fences
from note_assistant.indexing.preprocessor import RichPreprocessor


# ──────────────────────────────────────────────
# MermaidParser 单测
# ──────────────────────────────────────────────

def test_parse_flowchart_extracts_nodes_edges():
    src = """```mermaid
graph TD
    A[开始] --> B{判断}
    B -->|是| C[处理]
    B -->|否| D[结束]
```"""
    dg = MermaidParser.parse(src)
    assert dg.diagram_type == "flowchart"
    assert {n.id for n in dg.nodes} == {"A", "B", "C", "D"}
    # 边 label 抽取
    assert any(e.label == "是" for e in dg.edges)
    assert any(e.label == "否" for e in dg.edges)
    # 结构化 raw_text 含节点 label 与边（A→B 边也应被捕获，不漏）
    assert "开始" in dg.raw_text
    assert "判断" in dg.raw_text
    assert "处理" in dg.raw_text
    assert "结束" in dg.raw_text
    assert "开始->判断" in dg.raw_text
    assert "判断->处理" in dg.raw_text
    # 渲染契约
    assert dg.render_hint == "mermaid:inline"
    assert dg.source_format == "mermaid"


def test_parse_flowchart_node_shapes():
    src = """```mermaid
graph LR
    R[矩形] --> O(圆角)
    O --> D{菱形}
    D --> C((圆))
    C --> Y[(圆柱)]
```"""
    dg = MermaidParser.parse(src)
    by_id = {n.id: n for n in dg.nodes}
    assert by_id["R"].shape == "rect"
    assert by_id["O"].shape == "round"
    assert by_id["D"].shape == "diamond"
    assert by_id["C"].shape == "circle"
    assert by_id["Y"].shape == "cylinder"


def test_parse_sequence_fallback_raw_text():
    """非流程图类型：类型识别正确，整段源码兜底为可检索文本（不强行解析节点/边）。"""
    src = "sequenceDiagram\n    A->>B: 请求\n    B-->>A: 响应"
    dg = MermaidParser.parse(src)
    assert dg.diagram_type == "sequence"
    assert dg.render_hint == "mermaid:inline"
    assert "请求" in dg.raw_text and "响应" in dg.raw_text


def test_strip_fences_handles_fenced_and_bare_and_comments():
    assert _strip_fences("```mermaid\ngraph TD\n A-->B\n```") == "graph TD\n A-->B"
    # 裸源码不变
    assert _strip_fences("graph TD\n A-->B") == "graph TD\n A-->B"
    # %% 注释行被去除
    out = _strip_fences("%% 标题\n graph TD\n A-->B")
    assert "%% 标题" not in out


def test_parse_empty_raises():
    with pytest.raises(Exception):
        MermaidParser.parse("```mermaid\n   \n```")


# ──────────────────────────────────────────────
# generate_summaries 富化集成测试
# ──────────────────────────────────────────────

def test_mermaid_summary_enriched_with_render_hint():
    """mermaid 代码块经原生解析后，summary chunk 应带 render_hint + 结构化文本。"""
    pp = RichPreprocessor()
    pp.process("# 标题\n```mermaid\ngraph TD\n A[输入] --> B[输出]\n```\n正文")
    summaries = pp.generate_summaries()

    mermaid = [c for c in summaries if c.metadata.get("kind") == "mermaid"]
    assert mermaid, "应生成 mermaid summary chunk"
    m = mermaid[0]
    assert m.metadata["render_hint"] == "mermaid:inline"
    assert m.metadata["diagram_type"] == "flowchart"
    assert m.metadata.get("has_diagram") is True
    # 结构化文本（节点 label + 边）进了索引，而非旧式「图类型 + 前一行 caption」
    assert "输入" in m.page_content and "输出" in m.page_content
    assert "输入->输出" in m.page_content


def test_mermaid_summary_degrades_on_empty():
    """空 mermaid 触发解析异常 → 结构解析降级为旧弱摘要，但源码仍可渲染，索引不中断。"""
    pp = RichPreprocessor()
    pp.process("```mermaid\n   \n```")
    summaries = pp.generate_summaries()  # 必须不抛
    mermaid = [c for c in summaries if c.metadata.get("kind") == "mermaid"]
    assert mermaid, "降级路径仍应产出 mermaid summary chunk"
    assert "Mermaid" in mermaid[0].page_content
    # 源码仍是合法 mermaid → render_hint 恒置（P1-b 透传到前端原生渲染）
    assert mermaid[0].metadata["render_hint"] == "mermaid:inline"
    # 解析失败：无结构化 diagram_type 时退化为弱检测类型（此处为通用"图"）
    assert mermaid[0].metadata.get("diagram_type") == "图"


def test_mermaid_summary_carries_source_for_render():
    """P1-b：summary chunk 必须把原始 mermaid 源码存入 metadata（mermaid_src），
    供 classify_source → SourceInfo → API 透传到前端原生渲染（此前 summary 命中时
    前端拿不到 raw_mermaid，永远只显示文本摘要、图渲染不出来）。"""
    pp = RichPreprocessor()
    pp.process("# 标题\n```mermaid\ngraph TD\n A[输入] --> B[输出]\n```\n正文")
    summaries = pp.generate_summaries()
    m = [c for c in summaries if c.metadata.get("kind") == "mermaid"][0]
    assert "mermaid_src" in m.metadata
    src = m.metadata["mermaid_src"]
    assert "graph TD" in src
    assert "A[输入]" in src

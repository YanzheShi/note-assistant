"""结构化图表原生解析层（5.B 节）。

凡是「源文件本身就是结构化图」的，绝不送 VLM——零 token、零幻觉，节点/边 100% 保真。
本模块先把**最常见的内联 Mermaid 代码块**落地（本 vault 29 篇笔记有命中），产出统一的
`DiagramGraph` 中间表示，供检索（节点/边文本入 BM25 + dense）与原生渲染
（`render_hint="mermaid:inline"`）复用。

解析器是**离线启发式**实现（不依赖 `mermaid` 官方 JS parser，便于单元测试与零成本部署）：
覆盖 flowchart/graph 的节点/边语法，以及 sequence/class 等类型的类型识别（整段源码兜底为
可检索文本）。解析失败时抛 `MermaidParseError`，调用方应降级为旧摘要，绝不中断索引。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DiagramNode:
    id: str
    label: str
    shape: str = "rect"   # rect|round|diamond|circle|cylinder|subprocess
    group: str = ""        # 子图/分组（如有）


@dataclass
class DiagramEdge:
    from_id: str
    to_id: str
    label: str = ""        # 边上的条件/说明文字


@dataclass
class DiagramGraph:
    diagram_type: str          # flowchart|sequence|class|state|er|gantt|pie|unknown
    title: str
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]
    raw_text: str              # 节点标签 + 边标签拼成的可读文本（喂 BM25 + dense）
    render_hint: str           # "mermaid:inline" 等，前端据此原生渲染
    source_format: str         # mermaid|drawio|excalidraw|plantuml|svg


class MermaidParseError(ValueError):
    """mermaid 解析失败（调用方应降级，不要向上抛）。"""


# ── 节点定义：id 后跟形状括号（覆盖常见 mermaid 形状）─────────────
# 用具名组，且按特异性排序：圆柱 (( )/[ ] )、圆 (( )) 必须排在通用 [label]/(label) 之前，
# 否则 [(圆柱)] 会被通用的 [label] 误匹配成 rect。
_NODE_DEF_RE = re.compile(
    r'(?P<id>[A-Za-z0-9_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*'
    r'(?:'
    r'\[\[(?P<sub>[^\]]*)\]\]|'          # [[label]] 子流程
    r'\[\("(?P<rs>[^"]*)"\)\]|'         # [("label")]
    r'\[\((?P<cyl>[^)]*)\)\]|'          # [(label)] 圆柱（先 [ 后 (）
    r'\(\((?P<circ>[^)]*)\)\)|'         # ((label)) 圆
    r'\[(?P<rect>[^\]]*)\]|'            # [label] 矩形
    r'\((?P<round>[^)]*)\)|'            # (label) 圆角
    r'\{(?P<diam>[^)]*)\}'              # {label} 菱形
    r')'
)

# ── 可选的节点形状后缀（供边正则消费内联标签语法 A[label] --> B[label]）──
_SHAPE_OPT = (
    r'(?:\[\[[^\]]*\]\]|'
    r'\[\("[^"]*"\)\]|'
    r'\[\([^)]*\)\]|'           # [(label)] 圆柱（先 [ 后 (，必须排在 [label] 前）
    r'\(\([^)]*\)\)|'           # ((label)) 圆
    r'\[[^\]]*\]|'              # [label] 矩形
    r'\([^)]*\)|'               # (label) 圆角
    r'\{[^\}]*\})?'
)

# ── 边箭头（单行，无内嵌换行）：--> / --- / -.-> / ==> / ->> / -> / <- / .-> / .- ─
_EDGE_ARROWS = r'(?:-{1,2}>|-\.->|-\.-|==>|===|->>|->|<-|\.->|\.-)'
_EDGE_RE = re.compile(
    r'([A-Za-z0-9_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*'
    + _SHAPE_OPT
    + r'\s*' + _EDGE_ARROWS + r'\s*'
    + r'(?:\|([^|]*)\|)?\s*'
    + r'([A-Za-z0-9_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*'
    + _SHAPE_OPT
)

# ── 边（内联 label）：A -- text --> B ────────────────────────────
_EDGE_INLINE_RE = re.compile(
    r'([A-Za-z0-9_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*'
    + _SHAPE_OPT
    + r'\s*--\s*([^>|][^>]*?)\s*-->'
    + r'\s*([A-Za-z0-9_\u4e00-\u9fff][\w\u4e00-\u9fff]*)\s*'
    + _SHAPE_OPT
)

# 流程图头：graph TD / flowchart LR 等
_HEADER_RE = re.compile(r'^(?:flowchart|graph)\s+(?:TB|TD|BT|RL|LR)\b', re.IGNORECASE)

# 跳过的脚手架行
_SKIP_RE = re.compile(
    r'^(?:subgraph|end|direction|classDef|class\s|click|style\s|linkStyle|'
    r'graph\s|flowchart\s)', re.IGNORECASE
)


def _strip_fences(source: str) -> str:
    """去掉 ```mermaid ... ``` 围栏与 %% 注释行，返回纯 mermaid 源码。"""
    s = source.strip()
    m = re.match(r'^```(?:mermaid)?\s*\n(.*?)\n```$', s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    else:
        s = re.sub(r'^```(?:mermaid)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
    return "\n".join(
        line for line in s.splitlines() if not line.strip().startswith("%%")
    )


def _detect_type(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    low = first.lower()
    if _HEADER_RE.match(first):
        return "flowchart"
    for kw, t in (
        ("sequencediagram", "sequence"),
        ("classdiagram", "class"),
        ("statediagram", "state"),
        ("erdiagram", "er"),
        ("gantt", "gantt"),
        ("pie", "pie"),
        ("mindmap", "mindmap"),
        ("timeline", "timeline"),
    ):
        if low.startswith(kw):
            return t
    return "unknown"


def _node_label_shape(m: re.Match) -> tuple:
    """从节点定义匹配的具名组里取（label, shape）。按特异性降序判定。"""
    if m.group("sub") is not None:
        return m.group("sub"), "subprocess"
    if m.group("rs") is not None:
        return m.group("rs"), "rect"
    if m.group("cyl") is not None:
        return m.group("cyl"), "cylinder"
    if m.group("circ") is not None:
        return m.group("circ"), "circle"
    if m.group("rect") is not None:
        return m.group("rect"), "rect"
    if m.group("round") is not None:
        return m.group("round"), "round"
    if m.group("diam") is not None:
        return m.group("diam"), "diamond"
    return "", "rect"


def _ensure_node(
    nid: str,
    label: str,
    shape: str,
    group: str,
    nodes: List[DiagramNode],
    index: dict,
) -> None:
    if nid not in index:
        # 首次出现（多为边端点先注册）：label 为空时先以 id 兜底占个位，
        # 稍后节点定义行会用真实 label 覆盖。
        node = DiagramNode(id=nid, label=label or nid, shape=shape, group=group)
        index[nid] = node
        nodes.append(node)
    else:
        # 节点定义行总是比边注册的 id 兜底更权威 → label 直接覆盖。
        if label:
            index[nid].label = label
        # 仅在提供更具体形状时升级（避免边注册的 rect 覆盖 diamond/circle 等）
        if shape != "rect":
            index[nid].shape = shape
        if group:
            index[nid].group = group


def _parse_flowchart(
    text: str,
    nodes: List[DiagramNode],
    edges: List[DiagramEdge],
    index: dict,
) -> None:
    current_group = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("subgraph"):
            m = re.match(r'subgraph\s+(?:\[?([^\]]*)\]?)?', line, re.IGNORECASE)
            current_group = (m.group(1) if m and m.group(1) else "").strip()
            continue
        if low == "end":
            current_group = ""
            continue
        if _SKIP_RE.match(line):
            continue

        # 边（先于节点定义处理，保证端点先注册）
        for em in _EDGE_RE.finditer(line):
            frm, lbl, to = em.group(1), em.group(2), em.group(3)
            _ensure_node(frm, "", "rect", current_group, nodes, index)
            _ensure_node(to, "", "rect", current_group, nodes, index)
            edges.append(DiagramEdge(from_id=frm, to_id=to, label=(lbl or "").strip()))
        for em in _EDGE_INLINE_RE.finditer(line):
            frm, lbl, to = em.group(1), em.group(2), em.group(3)
            _ensure_node(frm, "", "rect", current_group, nodes, index)
            _ensure_node(to, "", "rect", current_group, nodes, index)
            edges.append(DiagramEdge(from_id=frm, to_id=to, label=(lbl or "").strip()))

        # 节点定义
        for nm in _NODE_DEF_RE.finditer(line):
            nid = nm.group("id")
            label, shape = _node_label_shape(nm)
            _ensure_node(nid, label.strip(), shape, current_group, nodes, index)


class MermaidParser:
    """离线启发式 Mermaid 解析器，输出统一的 DiagramGraph。"""

    @staticmethod
    def parse(source: str, title: str = "") -> DiagramGraph:
        text = _strip_fences(source).strip()
        if not text:
            raise MermaidParseError("empty mermaid source")

        diagram_type = _detect_type(text)
        nodes: List[DiagramNode] = []
        edges: List[DiagramEdge] = []
        index: dict = {}

        if diagram_type == "flowchart":
            _parse_flowchart(text, nodes, edges, index)
        # 非流程图类型：整段源码兜底为可检索文本（不强行解析节点/边）

        # 组装 raw_text：节点标签 + 边（from->to [label]）
        parts: List[str] = []
        for n in nodes:
            if n.label:
                parts.append(n.label)
        for e in edges:
            f = index.get(e.from_id)
            t = index.get(e.to_id)
            fl = (f.label if f else e.from_id) or e.from_id
            tl = (t.label if t else e.to_id) or e.to_id
            seg = f"{fl}->{tl}"
            if e.label:
                seg += f"({e.label})"
            parts.append(seg)
        raw_text = " ".join(p for p in parts if p)
        if not raw_text:
            raw_text = text  # 兜底：至少源码可检索

        return DiagramGraph(
            diagram_type=diagram_type,
            title=title,
            nodes=nodes,
            edges=edges,
            raw_text=raw_text,
            render_hint="mermaid:inline",
            source_format="mermaid",
        )

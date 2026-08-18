"""SVG 流程图原生解析层（设计文档 5.B.4.5，P1-d）。

SVG 是「部分可读」的结构化图：抽 `<text>` 是免费的高质量文本信号（比 VLM 准且便宜），
还要解析图形结构（rect/ellipse/path 节点 + 带箭头的 line/path 边）。整条路径**零 VLM 调用、零幻觉**。

路由（见 understanding.make_image_enricher）：mime==image/svg+xml 的图片先走本解析器，
只有解析彻底失败才降级回默认摘要（不送 VLM）。

解析失败抛 `SVGParseError`，调用方应降级，绝不中断索引。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

from note_assistant.indexing.diagrams import DiagramEdge, DiagramGraph, DiagramNode


class SVGParseError(ValueError):
    """SVG 解析失败（调用方应降级，不要向上抛）。"""


_SVG_NS = "{http://www.w3.org/2000/svg}"


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _text_content(el) -> str:
    """取 <text>/<tspan> 的可见文字（含子 tspan），跨换行拼成一行。"""
    parts = []
    if el.text and el.text.strip():
        parts.append(el.text.strip())
    for child in el:
        if _local(child.tag) in ("tspan", "text") and child.text and child.text.strip():
            parts.append(child.text.strip())
    return " ".join(parts).strip()


def _shape_center(attrib: dict) -> Optional[Tuple[float, float]]:
    """从形状属性推断中心点（用于把 text 绑到节点、把边端点匹配到节点）。"""
    tag = attrib.get("_tag", "")
    if tag in ("rect",):
        x, y, w, h = _f(attrib.get("x")), _f(attrib.get("y")), _f(attrib.get("width")), _f(attrib.get("height"))
        if w or h:
            return (x + w / 2, y + h / 2)
    if tag in ("circle", "ellipse"):
        return (_f(attrib.get("cx")), _f(attrib.get("cy")))
    if tag == "polygon":
        pts = _parse_points(attrib.get("points", ""))
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
    if tag == "path":
        # 取首个 moveto 坐标作近似中心（path 真实 bbox 解析成本高，近似足够做端点匹配）
        m = re.search(r"[Mm]\s*([-\d.]+)[, ]\s*([-\d.]+)", attrib.get("d", ""))
        if m:
            return (_f(m.group(1)), _f(m.group(2)))
    return None


def _parse_points(s: str) -> List[Tuple[float, float]]:
    nums = re.findall(r"([-\d.]+)", s)
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append((_f(nums[i]), _f(nums[i + 1])))
    return pts


def _shape_kind(attrib: dict) -> str:
    tag = attrib.get("_tag", "")
    return {"rect": "rect", "circle": "circle", "ellipse": "ellipse",
            "polygon": "polygon", "path": "path"}.get(tag, "rect")


def _nearest_node(x: float, y: float, nodes: List[DiagramNode],
                  centers: dict, max_dist: float = 1e9) -> Optional[str]:
    best, best_d = None, max_dist
    for n in nodes:
        cx, cy = centers.get(n.id, (0.0, 0.0))
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best, best_d = n.id, d
    return best


class SVGParser:
    """离线 SVG 解析器，输出统一的 DiagramGraph。"""

    @staticmethod
    def parse(source: str) -> DiagramGraph:
        try:
            root = ET.fromstring(source)
        except ET.ParseError as e:
            raise SVGParseError(f"invalid SVG XML: {e}")

        # 去掉命名空间前缀，统一按 local tag 处理
        texts: List[Tuple[float, float, str]] = []   # (x, y, content)
        shapes: List[dict] = []                        # {id,tag,attrib,center}
        edges_raw: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for el in root.iter():
            tag = _local(el.tag)
            attrib = dict(el.attrib)
            attrib["_tag"] = tag
            if tag == "text":
                content = _text_content(el)
                if not content:
                    continue
                x = _f(attrib.get("x"))
                y = _f(attrib.get("y"))
                texts.append((x, y, content))
            elif tag in ("rect", "circle", "ellipse", "polygon"):
                center = _shape_center(attrib)
                shapes.append({"id": attrib.get("id") or f"n{len(shapes)}",
                               "tag": tag, "attrib": attrib, "center": center})
            elif tag == "path":
                # 仅当带有箭头 marker（marker-end）时才当作「边」候选
                if attrib.get("marker-end") or attrib.get("marker-start"):
                    pts = _parse_path_endpoints(attrib.get("d", ""))
                    if len(pts) >= 2:
                        edges_raw.append((pts[0], pts[-1]))
                else:
                    # 无箭头的 path 当作形状节点（如流程图方框用 path 画的）
                    center = _shape_center(attrib)
                    if center is not None:
                        shapes.append({"id": attrib.get("id") or f"n{len(shapes)}",
                                       "tag": "path", "attrib": attrib, "center": center})
            elif tag in ("line",):
                x1, y1 = _f(attrib.get("x1")), _f(attrib.get("y1"))
                x2, y2 = _f(attrib.get("x2")), _f(attrib.get("y2"))
                edges_raw.append(((x1, y1), (x2, y2)))
            elif tag in ("polyline",):
                pts = _parse_points(attrib.get("points", ""))
                if len(pts) >= 2:
                    edges_raw.append((pts[0], pts[-1]))

        # 形状 → 节点：把最近的 text 绑成节点 label
        nodes: List[DiagramNode] = []
        centers: dict = {}
        for s in shapes:
            label = ""
            if s["center"] is not None:
                cx, cy = s["center"]
                # 找中心最近的 text（阈值 60 单位）
                best_d = 60 ** 2
                for (tx, ty, content) in texts:
                    d = (tx - cx) ** 2 + (ty - cy) ** 2
                    if d < best_d:
                        best_d, label = d, content
            nid = s["id"]
            nodes.append(DiagramNode(id=nid, label=label, shape=_shape_kind(s["attrib"])))
            if s["center"] is not None:
                centers[nid] = s["center"]

        # 边：端点匹配到最近节点
        edges: List[DiagramEdge] = []
        for (p1, p2) in edges_raw:
            frm = _nearest_node(p1[0], p1[1], nodes, centers, max_dist=120 ** 2)
            to = _nearest_node(p2[0], p2[1], nodes, centers, max_dist=120 ** 2)
            if frm and to and frm != to:
                # 边 label：中点附近的 text
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                lbl = ""
                best_d = 60 ** 2
                for (tx, ty, content) in texts:
                    d = (tx - mx) ** 2 + (ty - my) ** 2
                    if d < best_d:
                        best_d, lbl = d, content
                edges.append(DiagramEdge(from_id=frm, to_id=to, label=lbl))

        # raw_text：所有 text + 边（from->to [label]）
        text_blob = " ".join(t for (_, _, t) in texts)
        edge_blob = " ".join(
            f"{e.from_id}->{e.to_id}" + (f"({e.label})" if e.label else "")
            for e in edges
        )
        raw_text = " ".join(p for p in (text_blob, edge_blob) if p).strip()
        if not raw_text:
            raw_text = text_blob or "SVG 图"

        diagram_type = "flowchart" if edges else ("architecture" if len(nodes) > 1 else "unknown")

        return DiagramGraph(
            diagram_type=diagram_type,
            title="",
            nodes=nodes,
            edges=edges,
            raw_text=raw_text,
            render_hint="svg:inline",
            source_format="svg",
        )


def _parse_path_endpoints(d: str) -> List[Tuple[float, float]]:
    """粗略取 path 的起止坐标（M 起点与最后的 L/Z 终点）。"""
    pts = _parse_points(d)
    return pts

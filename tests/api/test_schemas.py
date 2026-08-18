# tests/api/test_schemas.py
"""P0 修复点 5：SourceSchema 契约（type=内容类型, origin=来源渠道，两套正交）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.api.schemas import SourceSchema


class TestSourceSchemaContract:
    def test_fields_are_orthogonal(self):
        s = SourceSchema(
            type="image", origin="graph", filepath="n.md", heading="章节",
            preview="...", score=0.9, img_path="attach/p.png",
        )
        assert s.type == "image"        # 内容类型
        assert s.origin == "graph"      # 来源渠道（与 type 正交）
        assert s.img_path == "attach/p.png"

    def test_defaults(self):
        s = SourceSchema()
        assert s.type == "text"
        assert s.origin == "direct"

    def test_kind_values_accepted(self):
        for k in ("text", "table", "mermaid", "image"):
            s = SourceSchema(type=k)
            assert s.type == k

    def test_none_payloads_allowed(self):
        # 非 image 类型不应强制 img_path
        s = SourceSchema(type="text", img_path=None, raw_table=None, raw_mermaid=None)
        assert s.img_path is None

    def test_mermaid_render_hint_fields(self):
        """P1-b：mermaid 来源的 render_hint / diagram_type 经 SourceSchema 透传。"""
        s = SourceSchema(
            type="mermaid", origin="direct",
            preview="Mermaid flowchart 图: ...",
            raw_mermaid="```mermaid\ngraph TD\n A-->B\n```",
            render_hint="mermaid:inline",
            diagram_type="flowchart",
        )
        assert s.type == "mermaid"
        assert s.render_hint == "mermaid:inline"
        assert s.diagram_type == "flowchart"
        dumped = s.model_dump(mode="json")
        assert dumped["render_hint"] == "mermaid:inline"
        assert dumped["diagram_type"] == "flowchart"

    def test_mermaid_render_hint_optional(self):
        # 无 render_hint 时应为 None（前端可据此降级为代码展示）
        s = SourceSchema(type="mermaid", raw_mermaid="```mermaid\ngraph TD\n```")
        assert s.render_hint is None
        assert s.diagram_type is None

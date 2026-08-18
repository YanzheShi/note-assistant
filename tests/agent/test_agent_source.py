# tests/agent/test_agent_source.py
"""P2 /agent 适配项（设计 9.2）：AgentSource 图片字段透传。

runner._sources_from_results 应从 chunk metadata 填充 kind / img_url / render_hint，
使 Agent 主链路也能把命中图片传给前端。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.retrieval.types import RetrievalResult
from note_assistant.agent.runner import _sources_from_results


def _result(metadata: dict) -> RetrievalResult:
    return RetrievalResult(score=0.9, page_content="x", metadata=metadata)


class TestAgentSourceImageFields:
    def test_image_fields_passthrough(self):
        results = [_result({
            "filepath": "n.md", "title": "标题", "heading_path": "H1",
            "kind": "image", "img_url": "/assets/abc123def4567890", "render_hint": "svg:inline",
        })]
        out = _sources_from_results(results)
        assert len(out) == 1
        s = out[0]
        assert s["kind"] == "image"
        assert s["img_url"] == "/assets/abc123def4567890"
        assert s["render_hint"] == "svg:inline"

    def test_text_default_fields(self):
        results = [_result({"filepath": "n.md", "title": "标题"})]
        out = _sources_from_results(results)
        assert out[0]["kind"] == "text"
        assert out[0]["img_url"] is None
        assert out[0]["render_hint"] is None

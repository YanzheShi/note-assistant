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


def _result(metadata: dict, content: str = "x") -> RetrievalResult:
    return RetrievalResult(score=0.9, page_content=content, metadata=metadata)


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

    def test_preview_from_page_content(self):
        # 回归：agentic 来源不能只有标题，必须有正文摘要（与 naive /ask 对齐）
        results = [_result({"filepath": "n.md", "title": "标题"}, content="这是正文片段…")]
        out = _sources_from_results(results)
        assert out[0]["preview"] == "这是正文片段…"

    def test_preview_truncated_at_200(self):
        results = [_result({"filepath": "n.md"}, content="长" * 500)]
        out = _sources_from_results(results)
        assert len(out[0]["preview"]) == 200

    def test_preview_empty_content_safe(self):
        results = [_result({"filepath": "n.md"}, content="")]
        out = _sources_from_results(results)
        assert out[0]["preview"] == ""

    def test_mermaid_raw_passthrough(self):
        # 与 naive /ask 对齐：正文里的 ```mermaid 块抽成渲染载荷，前端才能画图
        content = "流程如下：\n```mermaid\ngraph TD\nA-->B\n```\n结束。"
        results = [_result({"filepath": "n.md", "title": "标题"}, content=content)]
        out = _sources_from_results(results)
        assert out[0]["kind"] == "mermaid"
        assert out[0]["raw_mermaid"] is not None
        assert "graph TD" in out[0]["raw_mermaid"]

    def test_mermaid_metadata_fallback(self):
        # summary chunk：正文是结构化文本无 ```fence，回退 metadata 里的原始源码
        results = [_result(
            {"filepath": "n.md", "title": "标题", "kind": "mermaid",
             "mermaid_src": "graph TD\nA-->B\n", "diagram_type": "graph TD"},
            content="出题主通道流程图摘要",
        )]
        out = _sources_from_results(results)
        assert out[0]["kind"] == "mermaid"
        assert out[0]["raw_mermaid"] == "graph TD\nA-->B\n"
        assert out[0]["diagram_type"] == "graph TD"

    def test_table_raw_passthrough(self):
        content = "| 列A | 列B |\n| --- | --- |\n| 1 | 2 |\n说明文字"
        results = [_result({"filepath": "n.md", "title": "标题"}, content=content)]
        out = _sources_from_results(results)
        assert out[0]["kind"] == "table"
        assert out[0]["raw_table"] is not None
        assert "列A" in out[0]["raw_table"]

    def test_ranked_by_score_then_truncated_to_top_k(self, monkeypatch):
        # 排序 + top_k_rerank 截断（与 _contexts_from_results / naive 链路同规则）
        monkeypatch.setattr("note_assistant.config.settings.top_k_rerank", 2)
        results = [
            _result({"filepath": "n1.md"}, content="内容1"),
            _result({"filepath": "n2.md"}, content="内容2"),
            _result({"filepath": "n3.md"}, content="内容3"),
        ]
        for r, s in zip(results, [0.1, 0.9, 0.5]):
            r.score = s
        out = _sources_from_results(results)
        assert [o["score"] for o in out] == [0.9, 0.5]   # 降序 + top-2 截断

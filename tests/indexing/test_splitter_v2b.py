# tests/indexing/test_splitter_v2b.py
"""v2b 父子双存切分逻辑单测（无需 Ollama / ChromaDB）。"""
from pathlib import Path

from note_assistant.config import settings
from note_assistant.indexing.splitter import make_splitters, split_v2b
from note_assistant.indexing.types import DocNode


def _make_node(content: str) -> DocNode:
    return DocNode(
        filepath="AI/Agents/code-agent.md",
        abs_path=Path("/tmp/x"),
        raw_md=content,
        front_matter={},
        title="Code Agent 架构",
        tags=[],
        wikilinks=[],
        headings=[],
    )


# 一、背景(短) + 二、关键设计点(含 2.1/2.2，合并后超预算) + 三、超长章节(强制递归切父段)
_SAMPLE = """# Code Agent 架构

## 一、背景
这是一段关于背景的简短说明，介绍整体来龙去脉。

## 二、关键设计点
### 2.1 子点A
子点A的较长描述内容，用于演示合并后超过父块预算从而被拆成多个父段的场景。这里多写一些字以便凑够长度。

### 2.2 子点B
子点B的描述内容，同样补充一些说明文字让这一段也具备一定长度，便于验证切分行为。

## 三、长章节
段落甲：这里是长章节里的第一段内容，需要足够长以超过父块预算。段落甲还有更多说明文字用来堆长度，继续补充一些描述让这一段稳定在较长的体量。
段落乙：这里是长章节里的第二段内容，同样需要足够长。段落乙继续补充描述以减少被截断的风险，并且要明显比前一段更长一些以累计长度。
段落丙：第三段内容继续扩展，确保整节明显超过父块上限从而触发递归切分逻辑，这里也加入不少冗余说明文字用来凑够长度。
段落丁：第四段作为补充，进一步拉长整个长章节的体量，使其必定超过三百字符的预算上限，从而在测试中稳定地被拆成多个父块。
段落戊：第五段继续补充，确保即便前几段被压缩也足以触发拆分，验证超长章节的递归切父段行为符合设计预期。
"""


class TestSplitV2b:
    def test_returns_children_and_parents(self):
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        assert "children" in res and "parents" in res
        assert len(res["children"]) > 0
        assert len(res["parents"]) > 0

    def test_child_has_parent_id_and_heading_path(self):
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        for c in res["children"]:
            assert c.metadata.get("parent_id"), "child 必须有 parent_id"
            assert c.metadata.get("heading_path"), "child 必须有 heading_path"
            assert c.metadata.get("kind") == "text"

    def test_parent_id_unique_and_kind(self):
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        pids = [p.metadata["parent_id"] for p in res["parents"]]
        assert len(pids) == len(set(pids)), "parent_id 必须唯一"
        for p in res["parents"]:
            assert p.metadata.get("kind") == "parent"

    def test_child_text_is_substring_of_its_parent(self):
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        parents_by_id = {p.metadata["parent_id"]: p for p in res["parents"]}
        for c in res["children"]:
            pid = c.metadata["parent_id"]
            parent = parents_by_id[pid]
            assert c.page_content in parent.page_content, "child 文本必须是其 parent 的子串"

    def test_child_and_parent_share_heading_path(self):
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        parents_by_id = {p.metadata["parent_id"]: p for p in res["parents"]}
        for c in res["children"]:
            pid = c.metadata["parent_id"]
            assert c.metadata["heading_path"] == parents_by_id[pid].metadata["heading_path"]

    def test_overlong_section_is_split_into_multiple_parents(self, monkeypatch):
        """单 h2 章节超 parent_chunk_size 时，应被递归切成多个父段。"""
        monkeypatch.setattr(settings, "parent_chunk_size", 300)
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        # 三、长章节 明显超过 300 字符，应被切成 ≥2 个父段
        long_parents = [
            p for p in res["parents"]
            if "长章节" in (p.metadata.get("heading_path") or "")
        ]
        assert len(long_parents) >= 2, "超长章节应递归切分为多个父块"

    def test_small_parent_chunk_size_still_produces_valid_mapping(self, monkeypatch):
        """极端小预算下，父子映射仍自洽（child ⊆ parent，parent_id 唯一）。"""
        monkeypatch.setattr(settings, "parent_chunk_size", 80)
        monkeypatch.setattr(settings, "parent_chunk_overlap", 20)
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_SAMPLE), hs, cs)
        parents_by_id = {p.metadata["parent_id"]: p for p in res["parents"]}
        for c in res["children"]:
            assert c.page_content in parents_by_id[c.metadata["parent_id"]].page_content

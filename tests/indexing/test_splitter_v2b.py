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


# 无 h1、从 ## 起手的笔记（Obsidian 子笔记 / MOC / Daily Notes 常见）
_NO_H1_SAMPLE = """## 一、背景
这是关于背景的简短说明，介绍整体来龙去脉，补充一些字数以便构成独立章节内容。

## 二、关键设计
### 2.1 子点A
子点A的较长描述内容，用于演示从二级标题起手的笔记也能正确按章节分段，而不是把所有章节合并进一个大父块。这里多写一些字以便凑够长度。

### 2.2 子点B
子点B的描述内容，同样补充一些说明文字让这一段也具备一定长度，便于验证切分行为。

## 三、另一话题
这是完全无关的第三个二级章节，应当成为独立的父块，不能和前两节混在一起。补充足够字数以便观察分段结果是否正确。
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
            parent_hp = parents_by_id[pid].metadata["heading_path"]
            child_hp = c.metadata["heading_path"]
            # child 保留自身子节标题，parent 用整节公共前缀；二者应同源（互为子串）
            assert parent_hp in child_hp or child_hp in parent_hp, \
                f"child 与 parent heading_path 应同源：child={child_hp!r} parent={parent_hp!r}"

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

    def test_no_h1_splits_by_h2_section(self):
        """无 h1、从 ## 起手的笔记：每个 ## 章节应成为独立父块。

        回归测试——原实现换段只看 h1，缺 h1 时所有 ## 章节被合并进一个
        大父块（heading_path 还错标成第一段）。修复后应按最高层级标题分段。
        """
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_NO_H1_SAMPLE), hs, cs)
        # 三个 ## 章节应分开成 ≥3 个父段（而不是混成 1 个）
        assert len(res["parents"]) >= 3, "无 h1 时每个 ## 应独立分段"
        texts = [p.page_content for p in res["parents"]]
        # 修复前：所有 ## 被合并进大父块，单段文本同时含「背景」和「另一话题」
        # 修复后：每个 ## 独立，不存在同时含两段的父块
        merged = [t for t in texts if "背景" in t and "另一话题" in t]
        assert not merged, "无 h1 不能把不同 ## 章节合并进同一父块"


# 单 ## 下挂多个 ### 子节（复现「查询侧检索链路」漏检根因：
# 合并父段时父段 heading_path 只取首个子节，3.2/3.3/3.4 永久丢失）
_MULTI_SUBSECTION = """# 多模态 RAG 与传统 RAG 对比

## 3. 核心做法
### 3.1 编码器选型
编码器选型说明内容，用于演示合并后第一个子节的标题如何保留。

### 3.2 索引策略
索引策略说明内容，补充字数以便构成独立子节。

### 3.3 重排设计
重排设计说明内容，继续补充描述文字让这一段也具备一定长度。

### 3.4 查询侧检索链路
查询侧检索链路（双塔 + RRF + VLM 重排）的详细说明，这是需要被正确命中 heading_path 的子节。
"""


class TestSplitV2bSubsectionHeading:
    def test_each_child_keeps_own_subsection(self):
        """根因回归：合并父段后，每个 child 必须携带自身子节标题，末级子节不能丢。"""
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_MULTI_SUBSECTION), hs, cs)
        child_hps = [c.metadata["heading_path"] for c in res["children"]]
        # 末级子节 3.4 必须出现在某个 child 的 heading_path 中
        assert any("3.4 查询侧检索链路" in hp for hp in child_hps), \
            "合并父段后最后一个子节的标题不能丢失：" + str(child_hps)
        # 3.1/3.2/3.3 也应各自出现
        for sub in ["3.1 编码器选型", "3.2 索引策略", "3.3 重排设计"]:
            assert any(sub in hp for hp in child_hps), \
                f"子节 {sub} 未出现在任何 child heading_path：" + str(child_hps)

    def test_parent_heading_is_section_not_first_subsection(self):
        """父段标题应为整节「3. 核心做法」，不应错标成首个子节「3.1 编码器选型」。"""
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_MULTI_SUBSECTION), hs, cs)
        parent_hps = [p.metadata["heading_path"] for p in res["parents"]]
        assert any("3. 核心做法" in hp for hp in parent_hps), \
            "父段标题应含整节标题：" + str(parent_hps)
        # 回归：原 bug 下父段被标成首个子节，且后续子节永久丢失
        assert not any(
            hp.strip().endswith("3.1 编码器选型") for hp in parent_hps
        ), "父段不应只标首个子节而丢失后续子节：" + str(parent_hps)


# 带 h1、含多个 ## 章节（复现「_top_header 取 h1 导致所有章节被合并、
# 父块 heading_path 塌缩成只剩文档标题」的分组 bug）
_MULTI_H2_WITH_H1 = """# 多模态 RAG 总览

## 1. 背景
背景说明内容，用于演示第一个二级章节。补充一些字数以便构成独立章节。

## 2. 定义
定义说明内容，补充字数以便构成独立章节。

## 3. 核心做法
### 3.1 编码器选型
编码器选型说明内容。

### 3.2 索引策略
索引策略说明内容。

### 3.3 重排设计
重排设计说明内容。

### 3.4 查询侧检索链路
查询侧检索链路（双塔 + RRF + VLM 重排）的详细说明，这是需要被正确命中 heading_path 的子节。

## 4. 落地注意
落地注意事项说明内容，补充字数以便构成独立章节。
"""


class TestSplitV2bH1SectionGrouping:
    def test_h1_doc_splits_by_h2_section(self):
        """带 h1 的文档：每个 ## 章节应成为独立父段，不能因共享 h1 全合并。"""
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_MULTI_H2_WITH_H1), hs, cs)
        # 4 个 ## 章节 → 至少 4 个父段（而非混成 1 个）
        assert len(res["parents"]) >= 4, "带 h1 文档应按 ## 独立分段：" + str(
            [p.metadata["heading_path"] for p in res["parents"]]
        )
        # 不应存在同时含「背景」和「落地注意」的父块（说明被错误合并）
        texts = [p.page_content for p in res["parents"]]
        merged = [t for t in texts if "背景" in t and "落地注意" in t]
        assert not merged, "带 h1 不能把所有 ## 章节合并进同一父块"

    def test_h1_doc_parent_hp_keeps_section_not_title_only(self):
        """带 h1 文档的父块 heading_path 应保留 ## 章节，而非塌缩成仅文档标题。"""
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_MULTI_H2_WITH_H1), hs, cs)
        parent_hps = [p.metadata["heading_path"] for p in res["parents"]]
        # §3 父块应含「3. 核心做法」，且不应只是文档标题
        assert any("3. 核心做法" in hp for hp in parent_hps), \
            "§3 父块应保留章节标题：" + str(parent_hps)
        # 回归：原 bug 下所有父块 heading_path 只剩文档标题
        title_only = [hp for hp in parent_hps if hp.strip() == "多模态 RAG 总览"]
        assert not title_only, "父块 heading_path 不应塌缩成只剩文档标题：" + str(parent_hps)

    def test_h1_doc_child_keeps_subsection(self):
        """带 h1 文档：§3.4 子节标题必须出现在某个 child 的 heading_path 中。"""
        hs, cs = make_splitters()
        res = split_v2b(_make_node(_MULTI_H2_WITH_H1), hs, cs)
        child_hps = [c.metadata["heading_path"] for c in res["children"]]
        assert any("3.4 查询侧检索链路" in hp for hp in child_hps), \
            "合并父段后末级子节不能丢失：" + str(child_hps)

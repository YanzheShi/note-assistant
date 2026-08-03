# tests/test_indexing.py
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock ollama before importing modules that create a client on init
_ollama_mock = MagicMock()
_ollama_mock.Client.return_value = MagicMock()
sys.modules.setdefault("ollama", _ollama_mock)

from note_assistant.indexing.vault_loader import VaultLoader  # noqa: E402
from note_assistant.indexing.preprocessor import RichPreprocessor  # noqa: E402
from note_assistant.indexing.splitter import make_splitters, split_v1, split_v2  # noqa: E402
from note_assistant.indexing.types import DocNode, Chunk  # noqa: E402


# ----------------------------------------------------------------
# 工具：tmp_path 里造迷你 vault / DocNode
# ----------------------------------------------------------------
def _write_md(p: Path, content: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_node(tmp_path: Path, filename: str, content: str) -> DocNode:
    """创建一个临时 .md 文件并返回解析后的 DocNode"""
    md = _write_md(tmp_path / filename, content)
    loader = VaultLoader(tmp_path)
    return loader.load_file(md)


# ================================================================
# RichPreprocessor — process
# ================================================================
class TestPreprocessorProcess:
    def test_extract_code_fence(self, tmp_path):
        """code fence 应被替换为占位符"""
        node = _make_node(tmp_path, "code.md", """---
title: Test
---

# 标题

```python
def hello():
    print("world")
```

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        assert "<CODE_UID_" in cleaned
        assert "def hello" not in cleaned
        assert len(pp.extracted) == 1
        assert pp.extracted[0].kind == "code"

    def test_extract_table(self, tmp_path):
        """Markdown 表格应被替换为占位符"""
        node = _make_node(tmp_path, "table.md", """---
title: Test
---

# 标题

| 列A | 列B |
|-----|-----|
| a1  | b1  |
| a2  | b2  |

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        assert "[TABLE_UID_" in cleaned
        assert "| 列A" not in cleaned
        assert len(pp.extracted) == 1
        assert pp.extracted[0].kind == "table"

    def test_mermaid_extracted_as_mermaid_not_code(self, tmp_path):
        """
        P1 修复：```mermaid 围栏不能被 code fence 保护吞掉，必须被 _extract_mermaid
        捕获为 kind="mermaid"（设计文档 5.B.4.4 的前提）。普通代码围栏仍应被当 code 捕获。
        """
        node = _make_node(tmp_path, "mermaid.md", """---
title: Test
---

# 标题

```mermaid
graph TD
    A[开始] --> B[结束]
```

```python
def f():
    return 1
```

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        # mermaid 块被抽成 MERMAID_UID 占位符（源存活在 ExtractedChunk 里），
        # 普通 python 代码被抽成 CODE_UID 占位符
        assert "[MERMAID_UID_" in cleaned
        assert "<CODE_UID_" in cleaned

        kinds = {e.kind for e in pp.extracted}
        assert "mermaid" in kinds, "mermaid 必须被抽成 kind=mermaid"
        assert "code" in kinds, "普通代码围栏仍应被当 code 捕获"
        mermaid_ext = [e for e in pp.extracted if e.kind == "mermaid"][0]
        assert "graph TD" in mermaid_ext.raw
        assert mermaid_ext.placeholder.startswith("[MERMAID_UID_")

    def test_extract_image(self, tmp_path):
        """Obsidian embed 图片应被替换为占位符"""
        node = _make_node(tmp_path, "img.md", """---
title: Test
---

# 标题

![[photo.png]]

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        assert "[IMAGE_UID_" in cleaned
        assert "![[photo.png]]" not in cleaned
        assert len(pp.extracted) == 1
        assert pp.extracted[0].kind == "image"

    def test_no_extraction_for_plain_text(self, tmp_path):
        """纯文本不应触发任何抽取"""
        node = _make_node(tmp_path, "plain.md", """---
title: Test
---

# 标题

这是一段纯文本。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        assert len(pp.extracted) == 0
        assert "这是一段纯文本" in cleaned


# ================================================================
# RichPreprocessor — restore
# ================================================================
class TestPreprocessorRestore:
    def test_restore_code_fence(self, tmp_path):
        """restore 应将占位符还原为原始 code 内容"""
        node = _make_node(tmp_path, "restore_code.md", """---
title: Test
---

# 标题

```python
def hello():
    print("world")
```

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        from langchain_core.documents import Document
        fake_chunk = Document(page_content=cleaned, metadata={})
        chunks = pp.restore([Chunk(page_content=c.page_content, metadata=c.metadata) for c in [fake_chunk]])

        assert len(chunks) == 1
        assert "def hello" in chunks[0].page_content
        assert chunks[0].metadata.get("has_code") is True

    def test_restore_multiple_kinds(self, tmp_path):
        """含多种富结构的 chunk 应正确还原，metadata 各标记位都正确"""
        node = _make_node(tmp_path, "multi.md", """---
title: Test
---

# 标题

```python
x = 1
```

| A | B |
|---|---|
| 1 | 2 |

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        from langchain_core.documents import Document
        fake_chunk = Document(page_content=cleaned, metadata={})
        chunks = pp.restore([Chunk(page_content=c.page_content, metadata=c.metadata) for c in [fake_chunk]])

        assert chunks[0].metadata.get("has_code") is True
        assert chunks[0].metadata.get("has_table") is True
        assert "x = 1" in chunks[0].page_content
        assert "| A | B |" in chunks[0].page_content

    def test_restore_no_placeholder_is_passthrough(self, tmp_path):
        """不含占位符的 chunk 应原样透传"""
        node = _make_node(tmp_path, "plain.md", """---
title: Test
---

# 标题

纯文本内容。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)

        from langchain_core.documents import Document
        fake_chunk = Document(page_content=cleaned, metadata={})
        chunks = pp.restore([Chunk(page_content=c.page_content, metadata=c.metadata) for c in [fake_chunk]])

        assert "纯文本内容" in chunks[0].page_content
        # 没有富结构，has_code 等字段不应存在
        assert "has_code" not in chunks[0].metadata


# ================================================================
# RichPreprocessor — generate_summaries
# ================================================================
class TestPreprocessorSummaries:
    def test_summary_for_table(self, tmp_path):
        """表格应生成描述性 summary chunk"""
        node = _make_node(tmp_path, "sum_table.md", """---
title: Test
---

# 标题

| 姓名 | 年龄 |
|------|------|
| 张三 | 25   |
| 李四 | 30   |

正文。
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        summaries = pp.generate_summaries()

        assert len(summaries) == 1
        assert summaries[0].metadata["kind"] == "table"
        assert summaries[0].metadata["source"] == "extracted_summary"
        assert "表格" in summaries[0].page_content

    def test_summary_for_code(self, tmp_path):
        """Code 应生成含语言标记的 summary chunk"""
        node = _make_node(tmp_path, "sum_code.md", """---
title: Test
---

# 标题

```rust
fn main() {
    println!("hello");
}
```

正文。
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        summaries = pp.generate_summaries()

        assert len(summaries) == 1
        assert summaries[0].metadata["kind"] == "code"
        assert "rust" in summaries[0].page_content.lower()

    def test_summary_for_image(self, tmp_path):
        """图片应生成含路径的 summary chunk"""
        node = _make_node(tmp_path, "sum_img.md", """---
title: Test
---

# 标题

![[diagram.png]]

正文。
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        summaries = pp.generate_summaries()

        assert len(summaries) == 1
        assert summaries[0].metadata["kind"] == "image"
        assert "diagram.png" in summaries[0].page_content


# ================================================================
# RichPreprocessor — front_matter chunks
# ================================================================
class TestPreprocessorFrontMatter:
    def test_tags_to_chunk(self, tmp_path):
        """tags 应生成独立的可检索 chunk"""
        node = _make_node(tmp_path, "tags.md", """---
title: Test
tags: [ai, rag]
---

# 标题

正文。
""")
        pp = RichPreprocessor()
        _, fm_chunks = pp.process_with_meta(node)

        assert len(fm_chunks) == 1
        assert fm_chunks[0].metadata["kind"] == "tags"
        assert "ai" in fm_chunks[0].page_content
        assert "rag" in fm_chunks[0].page_content

    def test_no_tags_no_chunk(self, tmp_path):
        """无 tags 时不应生成 front_matter chunk"""
        node = _make_node(tmp_path, "no_tags.md", """---
title: Test
---

# 标题

正文。
""")
        pp = RichPreprocessor()
        _, fm_chunks = pp.process_with_meta(node)

        assert len(fm_chunks) == 0


# ================================================================
# Splitter — v1
# ================================================================
class TestSplitterV1:
    def test_v1_produces_chunks(self, tmp_path):
        """v1 应产出至少 1 个 chunk"""
        node = _make_node(tmp_path, "v1.md", """---
title: Test
---

# 概述

这是一段测试内容，用于验证 v1 切分器是否能正确工作。需要有足够长的内容来触发切分。
这里添加更多文本以确保 RecursiveCharacterTextSplitter 能够产生多个 chunk。
""")
        _, sp = make_splitters()
        chunks = split_v1(node, sp)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_v1_has_chunk_index(self, tmp_path):
        """v1 chunk 应包含 chunk_index"""
        node = _make_node(tmp_path, "v1_idx.md", """---
title: Test
---

# 概述

内容。
""")
        _, sp = make_splitters()
        chunks = split_v1(node, sp)

        for i, c in enumerate(chunks):
            assert c.metadata.get("chunk_index") == i

    def test_v1_no_heading_path(self, tmp_path):
        """v1 不添加 heading_path"""
        node = _make_node(tmp_path, "v1_nohp.md", """---
title: Test
---

# 概述

内容。
""")
        _, sp = make_splitters()
        chunks = split_v1(node, sp)

        for c in chunks:
            assert "heading_path" not in c.metadata


# ================================================================
# Splitter — v2
# ================================================================
class TestSplitterV2:
    def test_v2_produces_chunks(self, tmp_path):
        """v2 应产出至少 1 个 chunk"""
        node = _make_node(tmp_path, "v2.md", """---
title: Test
---

## 一、背景

这是一段测试内容。

## 二、方法

这是方法部分的内容。
""")
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_v2_has_heading_path(self, tmp_path):
        """v2 chunk 应包含 heading_path"""
        node = _make_node(tmp_path, "v2_hp.md", """---
title: Test
---

## 一、背景

这是背景内容。

## 二、方法

这是方法内容。
""")
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)

        for c in chunks:
            assert "heading_path" in c.metadata
            assert c.metadata["heading_path"] != ""

    def test_v2_heading_path_preserves_hierarchy(self, tmp_path):
        """v2 的 heading_path 应保留 h1 > h2 层级"""
        node = _make_node(tmp_path, "v2_hier.md", """---
title: Test
---

# 顶层标题

## 子标题

这是子标题下的内容，需要有足够的长度来确保 RecursiveCharacterTextSplitter 不会把它和标题合并。
""")
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)

        # 至少有一个 chunk 应包含 "顶层标题 > 子标题" 的路径
        paths = [c.metadata.get("heading_path", "") for c in chunks]
        assert any("顶层标题" in p and "子标题" in p for p in paths)

    def test_v2_heading_path_no_h1_skips_h1(self, tmp_path):
        """## 起手的笔记，h1 为空时不应出现 "> 子标题" 畸形路径"""
        node = _make_node(tmp_path, "v2_noh1.md", """---
title: Test
---

## 直接二级标题

内容。
""")
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)

        for c in chunks:
            hp = c.metadata.get("heading_path", "")
            # 不应以 " > " 开头（空 h1 被 skip）
            assert not hp.startswith(" > ")

    def test_v2_propagates_node_metadata(self, tmp_path):
        """v2 chunk 应携带 node 的基础 metadata（filepath/title/wikilinks）"""
        node = _make_node(tmp_path, "v2_meta.md", """---
title: Test
---

## 标题

内容。
""")
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)

        for c in chunks:
            assert c.metadata.get("filepath") == "v2_meta.md"
            assert c.metadata.get("title") == "Test"
            assert "h2" in c.metadata or "h1" in c.metadata


# ================================================================
# Ingestor — _make_id (纯逻辑，不依赖 Ollama)
# ================================================================
class TestIngestorMakeId:
    def test_make_id_format(self):
        """_make_id 应生成 filepath::index::kind 格式（分隔符无损归一为 /）"""
        from note_assistant.indexing.ingestor import Ingestor

        assert Ingestor._make_id("folder/file.md", 3, "text") == "folder/file.md::3::text"
        # Windows 分隔符归一化（跨平台一致）
        assert Ingestor._make_id("folder\\file.md", 3, "text") == "folder/file.md::3::text"

    def test_make_id_no_collision_between_similar_paths(self):
        """a/b.md 与 a_b.md 不得碰撞（旧实现把 / 替换成 _，两者 ID 相同会互相覆盖）"""
        from note_assistant.indexing.ingestor import Ingestor

        assert Ingestor._make_id("a/b.md", 0, "text") != Ingestor._make_id("a_b.md", 0, "text")

    def test_make_id_kind_variants(self):
        """不同 kind 应体现在 ID 中"""
        from note_assistant.indexing.ingestor import Ingestor

        id_text = Ingestor._make_id("a.md", 0, "text")
        id_summary = Ingestor._make_id("a.md", 0, "extracted_summary")

        assert id_text != id_summary
        assert "text" in id_text
        assert "extracted_summary" in id_summary


# ================================================================
# 端到端流程（mock Ollama embedder，不依赖真实服务）
# ================================================================
class TestEndToEnd:
    @patch("note_assistant.indexing.ingestor.OllamaEmbedder")
    def test_full_pipeline(self, mock_embedder_cls, tmp_path):  # noqa: ARG001
        """完整流程：加载 → 预处理 → 切分 → restore → summary → 入库"""
        # Mock embedder：返回与输入数量匹配的向量
        def mock_embed(texts):
            return [[0.1] * 1024 for _ in texts]

        mock_instance = MagicMock()
        mock_instance.embed.side_effect = mock_embed
        mock_instance.embed_one.return_value = [0.1] * 1024
        mock_embedder_cls.return_value = mock_instance

        # 造一个含富结构的文档
        vault = tmp_path / "vault"
        _write_md(vault / "test_note.md", """---
title: RAG 系统
tags: [ai, rag]
---

# 概述

RAG 系统结合[[vector-search]]和LLM。

```python
def retrieve(query):
    return search(query)
```

| 组件 | 作用 |
|------|------|
| 检索 | 召回 |
| 生成 | 总结 |

正文段落，用于验证整个索引流程是否能正确处理。这是一个足够长的段落以确保切分器能正常工作。
""")

        from note_assistant.indexing.ingestor import Ingestor

        # 1. 加载
        loader = VaultLoader(vault)
        docs = loader.load_all()
        assert len(docs) == 1
        node = docs[0]
        assert node.title == "RAG 系统"
        assert "vector-search" in node.wikilinks

        # 2. 预处理
        pp = RichPreprocessor()
        cleaned, fm_chunks = pp.process_with_meta(node)
        assert "<CODE_UID_" in cleaned
        assert "[TABLE_UID_" in cleaned
        assert len(fm_chunks) == 1  # tags

        # 3. 切分
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)
        assert len(chunks) >= 1

        # 4. restore
        chunks = pp.restore(chunks)
        # 至少一个 chunk 含有还原后的代码
        assert any("def retrieve" in c.page_content for c in chunks)
        assert any("|" in c.page_content for c in chunks)

        # 5. summary
        summaries = pp.generate_summaries()
        assert len(summaries) >= 2  # code + table

        # 6. 补 metadata
        for c in chunks:
            c.metadata["wikilinks"] = node.wikilinks
            c.metadata["filepath"] = str(node.filepath)
            c.metadata["title"] = node.title

        for s in summaries:
            s.metadata["filepath"] = str(node.filepath)

        # 7. 入库
        ingestor = Ingestor(persist_dir=tmp_path / "chroma_test")
        all_chunks = chunks + summaries + fm_chunks
        n = ingestor.upsert(all_chunks)
        assert n == len(all_chunks)

        # 验证 collection 有数据
        assert ingestor.collection.count() == n


# ================================================================
# P0 修复点 1：cleaned 必须喂给 splitter（占位符保护机制空转回归）
# ================================================================
class TestCleanedFedToSplitter:
    """历史 bug：preprocessor 算出的 cleaned 没喂给 splitter（splitter 吃了 node.raw_md），
    导致占位符从未进入 chunk，restore() 恒为 no-op，has_code/has_table/has_image 等
    metadata 永远写不进 ChromaDB。修复用 dataclasses.replace(node, raw_md=cleaned)。"""

    @staticmethod
    def _split_on_cleaned(tmp_path, content):
        from dataclasses import replace
        node = _make_node(tmp_path, "doc.md", content)
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)
        hs, cs = make_splitters()
        node_for_split = replace(node, raw_md=cleaned)
        chunks = split_v2(node_for_split, hs, cs)
        return pp, chunks

    def test_split_on_cleaned_carries_placeholders(self, tmp_path):
        content = """---
title: T
---

## 代码段

```python
def f(): return 1
```

## 图片

![[photo.png]]
"""
        pp, chunks = self._split_on_cleaned(tmp_path, content)
        # restore 前：切分后 chunk 应含占位符（证明走的是 cleaned 而非 raw）
        assert any("<CODE_UID_" in c.page_content or "[IMAGE_UID_" in c.page_content
                   for c in chunks)
        restored = pp.restore(chunks)
        # restore 后：metadata 标记位应被写回
        assert any(c.metadata.get("has_code") for c in restored)
        assert any(c.metadata.get("has_image") for c in restored)
        # 且正文里还原出真实代码与图片语法
        joined = "\n".join(c.page_content for c in restored)
        assert "def f()" in joined
        assert "![[photo.png]]" in joined

    def test_split_on_raw_loses_flags(self, tmp_path):
        """回归守卫：若有人把 splitter 改回吃 node.raw_md，restore 找不到占位符，
        has_code/has_image 永远丢失。此测试锁定修复不可回退。"""
        content = """---
title: T
---

## 代码段

```python
def f(): return 1
```

## 图片

![[photo.png]]
"""
        node = _make_node(tmp_path, "doc2.md", content)
        pp = RichPreprocessor()
        pp.process_with_meta(node)  # 确保 self.extracted 已填充占位符
        hs, cs = make_splitters()
        chunks = split_v2(node, hs, cs)  # 故意复现旧 bug：切 raw node
        restored = pp.restore(chunks)
        assert all("has_code" not in c.metadata for c in restored)
        assert all("has_image" not in c.metadata for c in restored)


# ================================================================
# P0 修复点 2/3：ExtractedChunk.raw 存完整 markdown 语法 + context 洗占位符
# ================================================================
class TestExtractedChunkRaw:
    def test_image_raw_is_full_markdown_embed(self, tmp_path):
        node = _make_node(tmp_path, "img.md", """---
title: T
---

# 标题

![[photo.png]]

正文。
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        img = pp.get_extracted("image")[0]
        assert img.raw == "![[photo.png]]"          # 完整语法，不是裸路径
        assert img.meta.get("src") == "photo.png"
        assert "IMAGE_UID" in img.placeholder

    def test_image_raw_is_full_markdown_link(self, tmp_path):
        node = _make_node(tmp_path, "img2.md", """---
title: T
---

# 标题

![架构图](assets/arch.png)

正文。
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        img = pp.get_extracted("image")[0]
        assert img.raw == "![架构图](assets/arch.png)"
        assert img.meta.get("src") == "assets/arch.png"
        assert img.meta.get("alt") == "架构图"

    def test_image_raw_keeps_dim_suffix_in_meta(self, tmp_path):
        node = _make_node(tmp_path, "img3.md", """---
title: T
---

![[photo.png|300x200]]
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        img = pp.get_extracted("image")[0]
        assert img.raw == "![[photo.png|300x200]]"
        assert img.meta.get("src") == "photo.png"
        assert img.meta.get("dims") == "300x200"

    def test_image_context_has_no_code_placeholder(self, tmp_path):
        """图片紧邻 code fence 时，其 context 窗口会盖到 code 占位符；
        strip_placeholders 必须洗掉它，否则噪声串会污染 summary / VLM prompt。"""
        node = _make_node(tmp_path, "ctx.md", """---
title: T
---

```python
def f(): return 1
```

![[near.png]]
""")
        pp = RichPreprocessor()
        pp.process_with_meta(node)
        img = pp.get_extracted("image")[0]
        assert "<CODE_UID_" not in img.context
        assert "near.png" in img.context      # 真实邻近文本应保留


# ================================================================
# P0 修复点 8：summary chunk 回填 heading_path + img_src
# ================================================================
class TestSummaryHeadingPath:
    def test_image_summary_has_heading_path_and_src(self, tmp_path):
        from dataclasses import replace
        node = _make_node(tmp_path, "sumhp.md", """---
title: T
---

## 图表章节

![[diagram.png]]

正文。
""")
        pp = RichPreprocessor()
        cleaned, _ = pp.process_with_meta(node)
        hs, cs = make_splitters()
        node_for_split = replace(node, raw_md=cleaned)
        pp.restore(split_v2(node_for_split, hs, cs))   # restore 反查 placeholder → heading
        summaries = pp.generate_summaries()
        img_sums = [s for s in summaries if s.metadata.get("kind") == "image"]
        assert img_sums, "应生成 image summary chunk"
        s = img_sums[0]
        assert s.metadata.get("heading_path"), "summary 应带回填的 heading_path"
        assert "图表章节" in s.metadata["heading_path"]
        assert s.metadata.get("img_src") == "diagram.png"

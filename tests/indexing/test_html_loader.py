# tests/indexing/test_html_loader.py
"""HTML 加载链路：归一化质量 + 扫描分发 + 编码。"""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.indexing.html_loader import decode_html_bytes, load_html
from note_assistant.indexing.splitter import make_splitters, split_v2
from note_assistant.indexing.types import DocNode
from note_assistant.indexing.vault_loader import VaultLoader


def _write(p: Path, content: str, encoding: str = "utf-8") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return p


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>测试文档</title>
<meta name="keywords" content="RAG, 检索">
<style>.x { color: red; }</style>
</head><body>
<nav id="toc"><a href="#a">目录项</a></nav>
<script>alert('xss')</script>
<h1>标题</h1>
<h2 id="toc">目录</h2>
<details><summary>展开：参考话术</summary><p>折叠区答案</p></details>
<div class="code-pre"><code><span class="cm">line1
    line2</span></code></div>
<pre><code class="language-python">def f():
    return 1</code></pre>
<table>
  <tr><th>项目</th><th>值</th></tr>
  <tr><td rowspan="2">A</td><td>x</td></tr>
  <tr><td>y</td></tr>
</table>
<img src="data:image/png;base64,AAAA" alt="示意图">
<a href="other.html">其他文档</a>
</body></html>
"""


class TestLoadHtml:
    def test_basic_conversion(self, tmp_path):
        vault = tmp_path / "vault"
        path = _write(vault / "doc.html", HTML)
        d = load_html(path, vault)

        assert d.filepath == "doc.html"
        assert d.title == "测试文档"
        assert d.tags == ["RAG", "检索"]
        assert d.front_matter["source_type"] == "html"
        assert d.wikilinks == ["other"]

        md = d.raw_md
        assert "alert('xss')" not in md          # script 剥离
        assert "color: red" not in md            # style 剥离
        assert "目录项" not in md                # 导航剥离
        assert "## 目录" not in md               # id=toc 的目录标题剥离
        assert "折叠区答案" in md                # details 内容保留
        assert "展开：" not in md                # summary UI 文案去掉
        assert "**参考话术**" in md

        # 两种代码块都要成围栏，且保留缩进
        assert "line1\n    line2" in md
        assert "```python\ndef f():\n    return 1\n```" in md

        # rowspan 展开后每行保持列对齐
        assert "| A | x |" in md
        assert "| A | y |" in md

        # data URI 不进入正文，只保留 alt
        assert "data:image" not in md
        assert "示意图" in md

        # 标题树来自归一化后的 Markdown
        assert any(h["text"] == "标题" for h in d.headings)

    def test_decode_gbk_with_meta_charset(self):
        raw = '<html><head><meta charset="gbk"></head><body><h1>中文标题</h1></body></html>'.encode("gbk")
        text = decode_html_bytes(raw)
        assert "中文标题" in text

    def test_loader_scan_and_dispatch(self, tmp_path):
        vault = tmp_path / "vault"
        _write(vault / "a.md", "# A")
        _write(vault / "b.html", "<title>B</title><p>B 正文</p>")
        _write(vault / "c.htm", "<title>C</title><p>C 正文</p>")
        _write(vault / "d.txt", "plain")
        _write(vault / "pic.png", "not really png")
        _write(vault / ".hidden" / "h.html", "<p>hidden</p>")

        loader = VaultLoader(vault)
        names = sorted(p.name for p in loader.scan())
        assert names == ["a.md", "b.html", "c.htm"]

        docs = {d.filepath: d for d in loader.load_all()}
        assert docs["b.html"].title == "B"
        assert docs["c.htm"].raw_md.strip().endswith("C 正文")


def test_html_through_preprocessor_and_splitter(tmp_path):
    """HTML 归一化后必须能走通占位符保护 → 切分 → restore → summary。"""
    from note_assistant.indexing.preprocessor import RichPreprocessor

    vault = tmp_path / "vault"
    path = _write(vault / "doc.html", HTML)
    node = VaultLoader(vault).load_file(path)

    pp = RichPreprocessor()
    cleaned, fm_chunks = pp.process_with_meta(node)
    assert "<CODE_UID_" in cleaned          # 两种代码块都进了占位符
    assert "[TABLE_UID_" in cleaned
    assert len(fm_chunks) == 1               # meta keywords → Tags chunk
    assert fm_chunks[0].kind == "front_matter"

    hs, cs = make_splitters()
    chunks = split_v2(replace(node, raw_md=cleaned), hs, cs)
    chunks = pp.restore(chunks)
    assert any("def f()" in c.page_content for c in chunks)
    assert any(c.metadata.get("has_code") for c in chunks)
    assert any(c.metadata.get("has_table") for c in chunks)
    assert any("折叠区答案" in c.page_content for c in chunks)

    summaries = pp.generate_summaries()
    assert any(s.metadata.get("kind") == "code" for s in summaries)


def test_incremental_reindex_html_end_to_end(tmp_path, monkeypatch):
    """加载 → 增量索引 → ChromaDB 落库 → sync 幂等，全链路不依赖真实 Ollama。"""
    import chromadb

    from note_assistant.config import settings
    from note_assistant.indexing.reindex import incremental_reindex

    vault = tmp_path / "vault"
    _write(vault / "doc.html", HTML)

    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "collection_name", "html_e2e")
    monkeypatch.setattr(settings, "chunking_strategy", "v2")
    monkeypatch.setattr(settings, "image_understand_enabled", False)

    fake = MagicMock()
    fake.embed.side_effect = lambda texts: [[0.1] * 1024 for _ in texts]
    fake.embed_one.return_value = [0.1] * 1024
    with patch("note_assistant.indexing.ingestor.OllamaEmbedder", return_value=fake):
        first = incremental_reindex(str(vault), filepaths=["doc.html"])
        assert first["reindexed"] == 1

        second = incremental_reindex(str(vault), filepaths=["doc.html"])
        assert second["reindexed"] == 0      # mtime/sha 未变，幂等

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_collection("html_e2e")
    rows = col.get(where={"filepath": "doc.html"}, include=["documents", "metadatas"])
    docs = rows["documents"]
    metas = rows["metadatas"]
    assert docs and any("折叠区答案" in d for d in docs)
    assert any(m.get("title") == "测试文档" for m in metas)
    assert any(m.get("has_code") for m in metas)
    assert any(m.get("heading_path") for m in metas)


def test_h5_kept_in_heading_path(tmp_path):
    node = DocNode(
        filepath="x.md",
        abs_path=tmp_path / "x.md",
        raw_md="# 文档\n\n## 章节\n\n### 小节\n\n#### 细节\n\n##### 五级标题\n\n正文\n",
        front_matter={},
        title="文档",
        tags=[],
        wikilinks=[],
    )
    hs, cs = make_splitters()
    chunks = split_v2(node, hs, cs)
    assert any(c.metadata.get("heading_path", "").endswith("五级标题") for c in chunks)

# tests/test_loader.py
import sys
from pathlib import Path
import pytest

# 确保 src/ 能在 pytest 里被找到（pyproject.toml 没配 pythonpath 时的兜底）
sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.indexing.vault_loader import VaultLoader


# ----------------------------------------------------------------
# 工具：tmp_path 里造迷你 md，不依赖真实 vault
# ----------------------------------------------------------------
def _write_md(p: Path, content: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ================================================================
# scan()
# ================================================================
def test_scan_excludes_hidden_dirs(tmp_path):
    """扫 .md 应跳过 .obsidian/ .trash/ 等以 '.' 开头的目录"""
    vault = tmp_path / "vault"
    _write_md(vault / "normal.md", "# hi\n")
    _write_md(vault / ".obsidian" / "config.md", "# hidden\n")
    _write_md(vault / ".trash" / "deleted.md", "# gone\n")

    loader = VaultLoader(vault)
    paths = loader.scan()
    assert len(paths) == 1
    assert paths[0].name == "normal.md"


def test_scan_recursive(tmp_path):
    """子目录的 .md 应能扫到"""
    vault = tmp_path / "vault"
    _write_md(vault / "a.md", "# a\n")
    _write_md(vault / "ai" / "rag.md", "# rag\n")

    loader = VaultLoader(vault)
    assert len(loader.scan()) == 2


# ================================================================
# load_file() — 有 fm
# ================================================================
def test_load_file_with_fm(tmp_path):
    """标准 --- fm --- 结构：title/tags 能提，fm 段被剥掉"""
    vault = tmp_path / "vault"
    md = _write_md(vault / "rag.md", """---
title: RAG 入门
tags: [ai, rag]
---

# 概述
用 [[vector-search]] 检索。
""")
    loader = VaultLoader(vault)
    d = loader.load_file(md)

    assert d.filepath == "rag.md"
    assert d.title == "RAG 入门"
    assert d.tags == ["ai", "rag"]
    assert "vector-search" in d.wikilinks
    # fm 已被 frontmatter 剥掉，raw_md 不应再以 --- 开头
    assert not d.raw_md.lstrip().startswith("---")


# ================================================================
# load_file() — 无 fm（# 标题 + > 引述 + --- 视觉分隔，你那两篇炸的同类）
# ================================================================
def test_load_file_no_fm_visual_divider(tmp_path):
    """
    Obsidian 手写笔记常见形态：# 标题 + > 引述 + --- 视觉线
    has_fm=False → fm={}, title 回退 stem, raw_md=全文, 不炸
    """
    vault = tmp_path / "vault"
    md = _write_md(vault / "项目说明.md", """# 项目说明
> 这是描述段落

---

## 一、背景
用 [[obsidian]] 做 RAG。
""")
    loader = VaultLoader(vault)
    d = loader.load_file(md)

    assert d.title == "项目说明"          # stem 回退
    assert d.front_matter == {}           # 无 fm
    assert "obsidian" in d.wikilinks
    # has_fm=False 分支：raw_md = 全文，所以应以 "# 项目说明" 开头
    assert d.raw_md.startswith("# 项目说明")


# ================================================================
# wikilink 正则：目标 / 别名 / 去重
# ================================================================
def test_extract_wikilinks_basic(tmp_path):
    """[[A]] 基础"""
    vault = tmp_path / "vault"
    md = _write_md(vault / "a.md", "# t\n[[flash-attn]] 和 [[lora]]。")
    loader = VaultLoader(vault)
    d = loader.load_file(md)
    assert d.wikilinks == ["flash-attn", "lora"]


def test_extract_wikilinks_alias_and_dedup(tmp_path):
    """[[目标|别名]] 只取目标名；同一目标重复只出现一次"""
    vault = tmp_path / "vault"
    md = _write_md(vault / "b.md", """---
title: test
---

# 双链
[[A]] 和 [[B|别名B]]，还有 [[A]] 重复。
""")
    loader = VaultLoader(vault)
    d = loader.load_file(md)
    # 去重保序：A 先出现，B 后；别名"别名B"不进列表
    assert d.wikilinks == ["A", "B"]


# ================================================================
# load_all 计数
# ================================================================
def test_load_all_count(tmp_path):
    vault = tmp_path / "vault"
    _write_md(vault / "a.md", "---\ntitle: a\n---\n# A\n[[x]]")
    _write_md(vault / "b.md", "# B\n> desc\n---\n\n## s\n[[y]]")
    loader = VaultLoader(vault)
    docs = loader.load_all()
    assert len(docs) == 2
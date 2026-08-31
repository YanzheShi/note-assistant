# tests/indexing/test_ignore.py
"""
索引忽略规则测试（indexing/ignore.py）。

覆盖（对应 INDEX_IGNORE_DIRS 配置项）：
- 隐藏路径（.obsidian / .git / .workbuddy）始终忽略，无需配置
- 配置的目录名在任意层级命中，大小写不敏感
- 整段匹配：不误伤同名前缀的真实笔记目录
- 配置容错：空串 / 尾部分隔符 / 前后空格
- 分隔符兼容：Windows 反斜杠路径与 Path 对象
- 三处扫描入口确实接上了同一份规则（vault_loader / AttachmentIndex / autoindex）

测试基建沿用项目约定：tmp_path mini vault，不碰 Ollama / ChromaDB。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from note_assistant.config import settings  # noqa: E402
from note_assistant.indexing.ignore import ignored_dir_names, is_ignored, path_parts  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# 隐藏路径：无需配置，始终忽略
# ═══════════════════════════════════════════════════════════════

class TestHiddenAlwaysIgnored:
    @pytest.mark.parametrize("rel", [
        ".obsidian/app.json",
        ".obsidian/workspace.json",
        ".git/config",
        ".workbuddy/memory/x.md",
        "dir/.trash/x.md",
        "a/b/.hidden/deep/c.md",
    ])
    def test_dot_segments_ignored(self, rel):
        assert is_ignored(rel) is True

    @pytest.mark.parametrize("rel", ["a.md", "00-索引.md", "项目串联/x.md", "a/b/c.md"])
    def test_normal_paths_kept(self, rel):
        assert is_ignored(rel) is False


# ═══════════════════════════════════════════════════════════════
# INDEX_IGNORE_DIRS：目录名命中
# ═══════════════════════════════════════════════════════════════

class TestConfiguredDirs:
    @pytest.fixture(autouse=True)
    def _ignore_dirs(self, monkeypatch):
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates", "附件"])

    def test_matches_at_any_depth(self):
        assert is_ignored("templates/x.md") is True
        assert is_ignored("a/b/templates/c.md") is True
        assert is_ignored("附件/pic.png") is True

    def test_case_insensitive(self):
        assert is_ignored("TEMPLATES/x.md") is True
        assert is_ignored("Templates/Sub/x.md") is True

    def test_whole_segment_only(self):
        """整段匹配：前缀相同的笔记目录不该被牵连。"""
        assert is_ignored("templates-备份/x.md") is False
        assert is_ignored("我的附件/x.md") is False

    def test_other_paths_unaffected(self):
        assert is_ignored("面试QA/x.md") is False

    def test_empty_config_means_hidden_only(self, monkeypatch):
        """默认空列表 → 行为与加配置前逐字节等价（零回归约定）。"""
        monkeypatch.setattr(settings, "index_ignore_dirs", [])
        assert ignored_dir_names() == frozenset()
        assert is_ignored("templates/x.md") is False
        assert is_ignored(".obsidian/x.md") is True

    def test_config_tolerates_noise(self, monkeypatch):
        monkeypatch.setattr(settings, "index_ignore_dirs", ["  ", "", "a/", "b\\", " c "])
        assert ignored_dir_names() == frozenset({"a", "b", "c"})
        for d in ("a", "b", "c"):
            assert is_ignored(f"{d}/x.md") is True
        assert is_ignored("d/x.md") is False


# ═══════════════════════════════════════════════════════════════
# 分隔符 / 入参类型兼容
# ═══════════════════════════════════════════════════════════════

class TestPathNormalization:
    def test_backslash_and_forward_slash_equivalent(self, monkeypatch):
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates"])
        assert path_parts("templates\\x.md") == ("templates", "x.md")
        assert is_ignored("templates\\sub\\x.md") is True
        assert is_ignored("templates/sub/x.md") is True

    def test_accepts_path_object(self, monkeypatch):
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates"])
        assert is_ignored(Path("templates") / "x.md") is True
        assert is_ignored(Path("keep") / "x.md") is False

    def test_empty_path_not_ignored(self):
        assert is_ignored("") is False


# ═══════════════════════════════════════════════════════════════
# 三处扫描入口确实接上同一份规则
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def wired_vault(tmp_path, monkeypatch):
    """含隐藏目录 / 忽略目录 / 正常笔记与附件的 mini vault。"""
    monkeypatch.setattr(settings, "index_ignore_dirs", ["templates", "附件"])
    vault = tmp_path / "vault"
    files = {
        "note.md": "# 正常笔记",
        "templates/tpl.md": "# 模板",
        "nested/templates/deep.md": "# 深层模板",
        ".obsidian/workspace.md": "# obsidian 配置",
        ".workbuddy/memory/m.md": "# 别的工具的记忆",
        "附件/pic.png": "not-really-a-png",
        "images/keep.png": "not-really-a-png",
    }
    for rel, text in files.items():
        p = vault / Path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return vault


def test_vault_loader_scan_excludes_ignored_and_hidden(wired_vault):
    from note_assistant.indexing.vault_loader import VaultLoader

    scanned = {
        str(p.relative_to(wired_vault)).replace("\\", "/")
        for p in VaultLoader(wired_vault).scan()
    }
    assert scanned == {"note.md"}


def test_attachment_index_excludes_ignored(wired_vault):
    from note_assistant.indexing.assets import AttachmentIndex

    idx = AttachmentIndex(wired_vault)
    assert idx.resolve("keep.png") is not None
    assert idx.resolve("pic.png") is None


def test_autoindex_filter_uses_same_rule(wired_vault):
    from note_assistant.indexing.autoindex import is_indexable_md

    assert is_indexable_md("note.md") is True
    assert is_indexable_md("templates/tpl.md") is False
    assert is_indexable_md(".workbuddy/memory/m.md") is False

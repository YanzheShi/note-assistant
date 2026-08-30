# tests/indexing/test_autoindex.py
"""
自动增量索引（auto-reindex）测试。

覆盖（docs/自动增量索引（auto-reindex）设计方案.md §八）：
- 过滤规则：.md + 排除隐藏目录
- debounce 合并：N 个事件 → 只入队 1 次
- 批量降级：积压 ≥ 阈值 → 整库增量
- 单文件增量：只处理变更文件，其他 untouched
- 删除：chunks + sync.db 状态清理（含 v2b docstore 父块）
- 幂等：内容未变重复触发 → need_reindex 拦截，零重索引
- 手动触发 run_incremental 走单飞队列
- 并发互斥：两个手动触发串行执行

测试基建沿用项目约定：tmp_path mini vault；embedder 全部 stub（不碰 Ollama）。
"""
import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from watchfiles import Change  # noqa: E402

from note_assistant.config import settings  # noqa: E402
from note_assistant.indexing.autoindex import AutoIndexService, is_indexable_md  # noqa: E402
from note_assistant.indexing.sync import SyncDB  # noqa: E402


# ──────────────────────────────────────────────
# 基建：stub embedder + 隔离 chroma/sync 落盘
# ──────────────────────────────────────────────

class FakeEmbedder:
    """确定性伪 embedding（sha256 → 1024 维），避免测试碰 Ollama。"""

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts):
        return [
            [(h[i % 32] / 255.0) - 0.5 for i in range(settings.embed_dim)]
            for h in (hashlib.sha256(t.encode()).digest() for t in texts)
        ]


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """隔离持久化路径 + 关闭图片 VLM + 切 v2 策略 + stub embedder。"""
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "chunking_strategy", "v2")
    monkeypatch.setattr(settings, "image_understand_enabled", False)
    monkeypatch.setattr("note_assistant.indexing.ingestor.OllamaEmbedder", FakeEmbedder)
    return tmp_path


@pytest.fixture
def mini_vault(tmp_path):
    """两篇笔记的 mini vault。"""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nAAA 内容 AAA\n", encoding="utf-8")
    (vault / "b.md").write_text("# B\n\nBBB 内容 BBB\n", encoding="utf-8")
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "hidden.md").write_text("hidden", encoding="utf-8")
    return vault


def _chroma_count():
    client = __import__("chromadb").PersistentClient(path=str(settings.chroma_persist_dir))
    col = client.get_collection(settings.collection_name)
    return col.count()


def _chroma_filepaths():
    client = __import__("chromadb").PersistentClient(path=str(settings.chroma_persist_dir))
    col = client.get_collection(settings.collection_name)
    return {m.get("filepath") for m in col.get(include=["metadatas"])["metadatas"]}


def _sync_db(tmp_path) -> SyncDB:
    db = SyncDB(tmp_path / "sync.db")
    return db


# ═══════════════════════════════════════════════════════════════
# 过滤规则
# ═══════════════════════════════════════════════════════════════

class TestFilter:
    def test_indexable_md(self):
        assert is_indexable_md("a.md") is True
        assert is_indexable_md("dir/sub/a.md") is True

    def test_non_md_ignored(self):
        assert is_indexable_md("a.txt") is False
        assert is_indexable_md("assets/pic.png") is False
        assert is_indexable_md("dir/noext") is False

    def test_hidden_dirs_ignored(self):
        assert is_indexable_md(".obsidian/x.md") is False
        assert is_indexable_md(".trash/x.md") is False
        assert is_indexable_md("dir/.hidden/note.md") is False

    def test_configured_ignore_dirs(self, monkeypatch):
        """INDEX_IGNORE_DIRS 配置的目录名同样不进队列（与 VaultLoader.scan 同规则）。"""
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates", "附件"])
        assert is_indexable_md("templates/x.md") is False
        assert is_indexable_md("a/b/templates/c.md") is False
        assert is_indexable_md("Templates/x.md") is False
        assert is_indexable_md("a.md") is True

    def test_watch_filter_drops_noisy_paths(self, mini_vault, monkeypatch):
        """传给 watchfiles 的事件级过滤：忽略路径不进 Python。"""
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates"])
        svc = AutoIndexService(mini_vault, enabled=False)

        assert svc._watch_filter(Change.modified, str(mini_vault / "a.md")) is True
        assert svc._watch_filter(Change.modified, str(mini_vault / ".obsidian" / "workspace.json")) is False
        assert svc._watch_filter(Change.modified, str(mini_vault / "templates" / "t.md")) is False
        assert svc._watch_filter(Change.modified, str(mini_vault / "pic.png")) is False
        assert svc._watch_filter(Change.modified, str(mini_vault.parent / "outside.md")) is False

    def test_absorb_respects_ignore_dirs(self, mini_vault, monkeypatch):
        """_absorb 与 watch_filter 用同一判定：忽略目录事件不置 dirty、不入队。"""
        monkeypatch.setattr(settings, "index_ignore_dirs", ["templates"])
        svc = AutoIndexService(mini_vault, enabled=False, batch_fallback_threshold=99)
        tpl = mini_vault / "templates" / "t.md"
        tpl.parent.mkdir(parents=True, exist_ok=True)
        tpl.write_text("# T", encoding="utf-8")
        assert svc._absorb([(Change.added, tpl)]) is False
        assert svc._absorb([(Change.added, mini_vault / "a.md")]) is True
        assert svc._pending == {"a.md": "add"}


# ═══════════════════════════════════════════════════════════════
# debounce 合并 / 批量降级（不启动 watcher，直接喂事件）
# ═══════════════════════════════════════════════════════════════

class TestDebounce:
    @pytest.mark.asyncio
    async def test_events_merged_into_single_batch(self, mini_vault):
        """3s 窗口内的 N 个事件 → 只入队 1 次（threshold 设大，避免降级整库）。"""
        svc = AutoIndexService(mini_vault, enabled=False, debounce_seconds=99, batch_fallback_threshold=99)
        for i in range(5):
            p = mini_vault / f"f{i}.md"
            p.write_text(f"# F{i}", encoding="utf-8")
            assert svc._absorb([(Change.added, p)]) is True
        await svc._flush_pending()
        assert svc._queue.qsize() == 1
        item, fut = svc._queue.get_nowait()
        assert fut is None
        assert item["index"] == ["f0.md", "f1.md", "f2.md", "f3.md", "f4.md"]

    @pytest.mark.asyncio
    async def test_non_md_events_not_queued(self, mini_vault):
        """图片/附件/隐藏改动不产生排队。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        assert svc._absorb([(Change.added, mini_vault / "pic.png")]) is False
        assert svc._absorb([(Change.modified, mini_vault / ".obsidian" / "x.md")]) is False
        await svc._flush_pending()
        assert svc._queue.empty()

    @pytest.mark.asyncio
    async def test_delete_event_queued_as_delete(self, mini_vault):
        svc = AutoIndexService(mini_vault, enabled=False)
        assert svc._absorb([(Change.deleted, mini_vault / "a.md")]) is True
        await svc._flush_pending()
        item, _ = svc._queue.get_nowait()
        assert item["delete"] == ["a.md"]

    @pytest.mark.asyncio
    async def test_batch_fallback_to_full(self, mini_vault):
        """积压 ≥ 阈值 → 降级整库增量。"""
        svc = AutoIndexService(mini_vault, enabled=False, batch_fallback_threshold=3)
        for i in range(3):
            p = mini_vault / f"f{i}.md"
            p.write_text(f"# F{i}", encoding="utf-8")
            svc._absorb([(Change.added, p)])
        await svc._flush_pending()
        item, _ = svc._queue.get_nowait()
        assert item == "full"


# ═══════════════════════════════════════════════════════════════
# 自动索引执行（喂队列 → worker 单飞执行）
# ═══════════════════════════════════════════════════════════════

class TestAutoIndexExecution:
    @pytest.mark.asyncio
    async def test_single_file_incremental(self, isolated_env, mini_vault):
        """只重索引变更文件，其他文件 untouched。"""
        from note_assistant.indexing.reindex import incremental_reindex

        # 先全库增量（两个文件都入库）
        await asyncio.to_thread(incremental_reindex, str(mini_vault))
        count_before = _chroma_count()
        assert count_before > 0

        # 改 b.md → 单文件模式只处理 b.md
        (mini_vault / "b.md").write_text("# B2\n\nBBB 修改后内容\n", encoding="utf-8")
        result = await asyncio.to_thread(
            incremental_reindex, str(mini_vault), filepaths=["b.md"]
        )
        assert result["status"] == "ok"
        assert result["reindexed"] == 1

        # a.md 的 chunks 完全没被动（filepath 集合不变）
        filepaths = _chroma_filepaths()
        assert "b.md" in filepaths
        assert "a.md" in filepaths
        # 幂等复核：sync.db 已更新为最新，再触发 → 零重索引
        result2 = await asyncio.to_thread(
            incremental_reindex, str(mini_vault), filepaths=["b.md"]
        )
        assert result2["reindexed"] == 0

    @pytest.mark.asyncio
    async def test_delete_removes_all_traces(self, isolated_env, mini_vault):
        from note_assistant.indexing.reindex import incremental_reindex

        await asyncio.to_thread(incremental_reindex, str(mini_vault))

        result = await asyncio.to_thread(
            incremental_reindex, str(mini_vault), deleted_filepaths=["a.md"]
        )
        assert result["removed"] == 1
        assert "a.md" not in _chroma_filepaths()

        db = _sync_db(isolated_env)
        try:
            assert db.get_state("a.md") is None
        finally:
            db.close()

    @pytest.mark.asyncio
    async def test_run_incremental_through_queue(self, isolated_env, mini_vault):
        """手动触发（/reindex 语义）：汇入单飞队列并返回结果。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        await svc.start()
        try:
            result = await svc.run_incremental()
            assert result["reindexed"] == 2
            st = svc.stats
            assert st["total_runs"] == 1
            assert st["enabled"] is False
            assert st["queue_len"] == 0
            assert "a.md" in _chroma_filepaths()
        finally:
            await svc.stop()

    @pytest.mark.asyncio
    async def test_concurrent_triggers_serialized(self, isolated_env, mini_vault):
        """两个并发手动触发 → 串行执行（单飞），无并发写。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        await svc.start()
        try:
            r1, r2 = await asyncio.gather(svc.run_incremental(), svc.run_incremental())
            assert r1["reindexed"] == 2
            assert r2["reindexed"] == 0  # 第二次：need_reindex 幂等拦截
            assert svc.stats["total_runs"] == 2
        finally:
            await svc.stop()

    @pytest.mark.asyncio
    async def test_v2b_delete_cleans_docstore(self, isolated_env, mini_vault, monkeypatch):
        """v2b 策略下删除文件 → docstore 中该文件父块被清除。"""
        monkeypatch.setattr(settings, "chunking_strategy", "v2b")
        from note_assistant.indexing.reindex import incremental_reindex
        from note_assistant.retrieval.docstore import ParentDocstore

        # 造一个含 a.md 父块的 docstore（模拟 v2b 索引后的产物）
        docstore_path = settings.parent_docstore_path
        docstore_path.parent.mkdir(parents=True, exist_ok=True)
        store = ParentDocstore(docstore_path)
        store.add("a.md::parent::0", "parent of a", {"filepath": "a.md", "title": "A"})
        store.add("b.md::parent::0", "parent of b", {"filepath": "b.md", "title": "B"})
        store.save()

        result = await asyncio.to_thread(
            incremental_reindex, str(mini_vault), deleted_filepaths=["a.md"]
        )
        assert result["removed"] == 1
        reloaded = ParentDocstore.load(docstore_path)
        assert "a.md::parent::0" not in reloaded._store
        assert "b.md::parent::0" in reloaded._store

    @pytest.mark.asyncio
    async def test_worker_batch_runs_index_and_delete(self, isolated_env, mini_vault):
        """worker 消费 index+delete 混合批次。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        await svc.start()
        try:
            await svc._queue.put(({"index": ["b.md"], "delete": ["a.md"]}, None))
            await svc._queue.join()
            filepaths = _chroma_filepaths()
            assert "a.md" not in filepaths
            assert "b.md" in filepaths
            assert svc.stats["errors"] == 0
        finally:
            await svc.stop()


# ═══════════════════════════════════════════════════════════════
# 服务生命周期
# ═══════════════════════════════════════════════════════════════

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stop_drains_pending(self, mini_vault):
        """stop() 时冲刷残留事件，不丢变更。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        await svc.start()
        svc._absorb([(Change.added, mini_vault / "b.md")])
        await svc.stop()
        # pending 已被冲刷进队列 → stop 后队列有 1 个批次（或已执行）
        assert svc._pending == {}

    @pytest.mark.asyncio
    async def test_start_worker_only_when_disabled(self, mini_vault):
        """enabled=False → 不启 watcher，只启 worker（手动触发仍可单飞）。"""
        svc = AutoIndexService(mini_vault, enabled=False)
        await svc.start()
        try:
            assert svc._watcher_task is None
            assert svc._worker_task is not None
        finally:
            await svc.stop()
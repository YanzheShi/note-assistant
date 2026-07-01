# tests/indexing/test_sync.py
"""
SyncDB 增量索引状态数据库测试。

注意：Windows 上 SQLite 连接必须关闭，否则临时目录无法清理。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.indexing.sync import SyncDB


def _use_db(tmp_dir):
    """辅助函数：获取 SyncDB 实例并自动 close"""
    db = SyncDB(Path(tmp_dir) / "sync.db")
    return db


# ================================================================
# get_state / update_state
# ================================================================

class TestGetState:
    def test_get_state_returns_none_for_unknown(self):
        """未知文件应返回 None"""
        with tempfile.TemporaryDirectory() as tmp:
            db = _use_db(tmp)
            try:
                assert db.get_state("unknown.md") is None
            finally:
                db.close()

    def test_get_state_after_update(self):
        """更新后应能查到状态"""
        with tempfile.TemporaryDirectory() as tmp:
            db = _use_db(tmp)
            try:
                db.conn.execute(
                    "INSERT INTO file_state VALUES (?, ?, ?, ?)",
                    ("a.md", 1234567890.0, "abc123", "2026-01-01T00:00:00")
                )
                db.conn.commit()
                state = db.get_state("a.md")
                assert state is not None
                assert state["mtime"] == 1234567890.0
            finally:
                db.close()


# ================================================================
# need_reindex（等你实现）
# ================================================================

class TestNeedReindex:
    def test_new_file_needs_reindex(self):
        """sync.db 中无记录 → 需要索引"""
        with tempfile.TemporaryDirectory() as tmp:
            db = _use_db(tmp)
            try:
                # 未索引过 → need_reindex 应为 True
                result = db.need_reindex("new.md")
                assert result is True
                pass  # 占位，实现后删除
            finally:
                db.close()

    def test_unchanged_file_no_reindex(self):
        """mtime 和 sha256 都没变 → 不需要索引"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            test_file = tmpdir / "test.md"
            test_file.write_text("hello")

            db = _use_db(tmp)
            try:
                db.update_state("test.md", test_file)

                result = db.need_reindex("test.md", test_file)
                assert result is False
                pass
            finally:
                db.close()

    def test_changed_mtime_needs_reindex(self):
        """mtime 变了 → 需要索引"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            test_file = tmpdir / "test.md"
            test_file.write_text("hello")

            db = _use_db(tmp)
            try:
                # 先记录一个旧的 mtime
                db.conn.execute(
                    "INSERT INTO file_state VALUES (?, ?, ?, ?)",
                    ("test.md", 0.0, "old_hash", "2026-01-01T00:00:00")
                )
                db.conn.commit()

                result = db.need_reindex("test.md", test_file)
                assert result is True
                pass
            finally:
                db.close()


# ================================================================
# update_state（等你实现）
# ================================================================

class TestUpdateState:
    def test_update_state_writes_correct_values(self):
        """update_state 应写入 mtime 和 sha256"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            test_file = tmpdir / "test.md"
            test_file.write_text("hello world")

            db = _use_db(tmp)
            try:
                db.update_state("test.md", test_file)

                state = db.get_state("test.md")
                assert state is not None
                assert state["mtime"] == test_file.stat().st_mtime
                assert state["sha256"] == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            finally:
                db.close()


# ================================================================
# get_all_indexed
# ================================================================

class TestGetIndexed:
    def test_returns_all_indexed_files(self):
        """应返回所有已索引文件"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            files = ["a.md", "b.md", "c.md"]

            db = _use_db(tmp)
            try:
                for f in files:
                    fpath = tmpdir / f
                    fpath.write_text(f"content of {f}")
                    db.update_state(f, fpath)

                all_indexed = db.get_all_indexed()
                assert set(all_indexed) == set(files)
            finally:
                db.close()


# ================================================================
# remove_state
# ================================================================

class TestRemoveState:
    def test_remove_state_deletes_record(self):
        """remove_state 应删除记录"""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            test_file = tmpdir / "test.md"
            test_file.write_text("x")

            db = _use_db(tmp)
            try:
                db.update_state("test.md", test_file)
                assert db.get_state("test.md") is not None

                db.remove_state("test.md")
                assert db.get_state("test.md") is None
            finally:
                db.close()

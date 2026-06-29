# src/note_assistant/indexing/sync.py
"""
增量索引状态数据库：记录每篇笔记的索引状态，实现秒级增量重索引。

架构：
    vault/*.md → mtime + sha256 → sync.db
    reindex.py → 对比 → 只处理变更/新增/删除 → < 10 秒
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from note_assistant.config import settings


class SyncDB:
    """
    增量索引状态数据库。

    表结构：file_state (
        filepath TEXT PRIMARY KEY,    -- 相对 vault 根的路径
        mtime REAL NOT NULL,          -- 文件修改时间（Unix timestamp）
        sha256 TEXT NOT NULL,          -- 文件内容哈希（双重保险）
        indexed_at TEXT NOT NULL      -- 索引时间（ISO format）
    )
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else settings.chroma_persist_dir.parent / "sync.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_table()

    def _create_table(self) -> None:
        """创建表（如果不存在）"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_state (
                filepath TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                sha256 TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    # ──────────────────────────────────────────────
    # 状态查询
    # ──────────────────────────────────────────────

    def get_state(self, filepath: str) -> Optional[Dict]:
        """
        获取文件状态。

        Args:
            filepath: 相对 vault 根的路径

        Returns:
            {"mtime": float, "sha256": str, "indexed_at": str} 或 None（未索引）
        """
        cur = self.conn.execute(
            "SELECT mtime, sha256, indexed_at FROM file_state WHERE filepath = ?",
            (filepath,)
        )
        row = cur.fetchone()
        if row:
            return {"mtime": row[0], "sha256": row[1], "indexed_at": row[2]}
        return None

    def need_reindex(self, filepath: str, abs_path: Path | None = None) -> bool:
        """
        【核心逻辑待实现】判断文件是否需要重新索引。

        判断逻辑：
        1. sync.db 中没有记录 → 新文件 → 需要索引
        2. mtime 变了 → 文件被修改 → 需要索引
        3. mtime 没变但 sha256 变了 → 极少见的边界情况 → 需要索引
        4. 都没变 → 不需要索引

        Args:
            filepath: 相对 vault 根的路径
            abs_path: 绝对路径（可选，用于计算 mtime/sha256）
                     如果不传，需要调用者自行处理

        Returns:
            True 需要重新索引，False 不需要
        """
        # 提示：先用 get_state 查记录，None → True；比较 mtime；双重保险 sha256
        state = self.get_state(filepath)
        if not state:
            return True


        cur_mtime = os.path.getmtime(abs_path)
        if cur_mtime != state["mtime"]:
            return True

        cur_sha = SyncDB._compute_sha256(abs_path)
        if cur_sha != state["sha256"]:
            return True

        return False

    def update_state(self, filepath: str, abs_path: Path | None = None) -> None:
        """
        【核心逻辑待实现】更新文件状态（索引完成后调用）。

        Args:
            filepath: 相对 vault 根的路径
            abs_path: 绝对路径（可选，用于计算 mtime/sha256）
        """
        cur_mtime = os.path.getmtime(abs_path)
        cur_sha = SyncDB._compute_sha256(abs_path)
        now = datetime.now().isoformat()

        self.conn.execute(
            """INSERT OR REPLACE INTO file_state (filepath, mtime, sha256, indexed_at)
               VALUES (?, ?, ?, ?)""",
            (filepath, cur_mtime, cur_sha, now),
        )
        self.conn.commit()
        
        # 提示：从 abs_path 计算 mtime + sha256，写入 file_state（INSERT OR REPLACE）
        pass

    def remove_state(self, filepath: str) -> None:
        """删除文件状态（文件被删除时）"""
        self.conn.execute("DELETE FROM file_state WHERE filepath = ?", (filepath,))
        self.conn.commit()

    def get_all_indexed(self) -> List[str]:
        """获取所有已索引文件"""
        cur = self.conn.execute("SELECT filepath FROM file_state")
        return [row[0] for row in cur.fetchall()]

    def close(self):
        """关闭数据库连接（测试/清理时调用）"""
        if self.conn:
            self.conn.close()
            self.conn = None

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        """计算文件 sha256"""
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

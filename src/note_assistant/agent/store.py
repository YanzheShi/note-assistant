"""Agentic RAG 轻量持久层（标准库 sqlite3，无外部依赖，全离线可测）。

两张核心表：
  - ``runs`` + ``run_events``：一次问答运行的快照。流式输出时每个轨迹事件实时落盘，
    客户端若中途断流，可用 ``run_id`` 轮询 ``GET /agent/runs/{run_id}`` 取回完整结果
    （实现「流式中断/恢复」）。
  - ``session_turns``：跨会话对话记忆。按 ``session_id`` 累积 user/assistant 轮次，
    服务端持有历史，前端不必每轮搬运 ``history``（实现「跨会话长期记忆」）。

设计取舍：
  - 选 SQLite 而非 LangGraph checkpointer，是因为本地优先项目要的是「断流后轮询拿结果」
    而非「从断点续跑图」；轮询快照更简单、确定、易测，且天然跨进程（服务重启也能取回）。
  - 所有方法都是同步的（sqlite3 本地文件），调用方用 ``asyncio.to_thread`` 包一层即可不阻塞事件循环。
  - 孤儿检测：``runs.status == 'running'`` 且超过 ``agent_run_orphan_ttl`` 秒未结束，
    读取时降级为 ``interrupted``（服务崩溃 / 进程被杀的兜底）。
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Optional

from note_assistant.config import PROJECT_ROOT, settings


def _resolve_db_path() -> Path:
    p = Path(settings.agent_db_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


class AgentStore:
    """基于单个 SQLite 文件的 Agent 持久层。线程安全（每次操作独立连接）。"""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else _resolve_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ──────────────────────────────────────────
    # 内部
    # ──────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10)

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id     TEXT PRIMARY KEY,
                    question   TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'running',
                    answer     TEXT DEFAULT '',
                    sources    TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    finished_at REAL DEFAULT 0
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id     TEXT NOT NULL,
                    seq        INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS session_turns (
                    session_id TEXT NOT NULL,
                    idx        INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, idx)
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT NOT NULL,
                    idx_from   INTEGER NOT NULL,
                    idx_to     INTEGER NOT NULL,
                    summary    TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, idx_to)
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_run_events ON run_events(run_id)")
            c.execute("CREATE INDEX IF NOT EXISTS ix_session ON session_turns(session_id)")
            c.execute(
                "CREATE INDEX IF NOT EXISTS ix_summary ON session_summaries(session_id, idx_to)"
            )

    # ──────────────────────────────────────────
    # runs：流式续传快照
    # ──────────────────────────────────────────

    def create_run(self, question: str) -> str:
        """登记一次运行，返回 run_id（流式首事件、响应体都会带回）。"""
        run_id = uuid.uuid4().hex
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs(run_id, question, status, created_at) VALUES (?,?,?,?)",
                (run_id, question, "running", time.time()),
            )
        return run_id

    def ensure_run(self, run_id: str, question: str) -> None:
        """若 run_id 不存在则登记（续传场景：客户端带着自己/上次的 run_id 回来）。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO runs(run_id, question, status, created_at) VALUES (?,?,?,?)",
                (run_id, question, "running", time.time()),
            )

    def append_event(self, run_id: str, event: dict, seq: int) -> None:
        """落盘一个轨迹事件（流式输出时实时调用）。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO run_events(run_id, seq, event_json) VALUES (?,?,?)",
                (run_id, seq, json.dumps(event, ensure_ascii=False)),
            )

    def set_answer(self, run_id: str, answer: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET answer=? WHERE run_id=?", (answer, run_id))

    def set_sources(self, run_id: str, sources: list) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET sources=? WHERE run_id=?",
                (json.dumps(sources, ensure_ascii=False), run_id),
            )

    def finish_run(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status='finished', finished_at=? WHERE run_id=?",
                (time.time(), run_id),
            )

    def get_run(self, run_id: str) -> Optional[dict]:
        """读取运行快照（含完整轨迹）。running 超 TTL 自动降级为 interrupted。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT run_id, question, status, answer, sources, created_at, finished_at "
                "FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if not row:
                return None
            events = c.execute(
                "SELECT event_json FROM run_events WHERE run_id=? ORDER BY seq",
                (run_id,),
            ).fetchall()
        trajectory = [json.loads(e[0]) for e in events]
        _id, question, status, answer, sources_json, created_at, finished_at = row
        if status == "running" and (time.time() - created_at) > settings.agent_run_orphan_ttl:
            status = "interrupted"
        return {
            "run_id": _id,
            "question": question,
            "status": status,
            "answer": answer or "",
            "sources": json.loads(sources_json or "[]"),
            "trajectory": trajectory,
            "created_at": created_at,
            "finished_at": finished_at,
        }

    # ──────────────────────────────────────────
    # session_turns：跨会话记忆
    # ──────────────────────────────────────────

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """写入一轮对话（role ∈ {user, assistant}）。"""
        with self._conn() as c:
            cur = c.execute(
                "SELECT COALESCE(MAX(idx), -1) FROM session_turns WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            c.execute(
                "INSERT INTO session_turns(session_id, idx, role, content, created_at) "
                "VALUES (?,?,?,?,?)",
                (session_id, cur + 1, role, content, time.time()),
            )

    def get_history(self, session_id: str, limit: int = 20) -> List[dict]:
        """取回最近 limit 轮对话（按时间正序，可直接喂给 generate 节点）。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM session_turns WHERE session_id=? "
                "ORDER BY idx DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        # 倒序取回，翻转回正序
        return [{"role": r, "content": c_} for r, c_ in reversed(rows)]

    def get_recent_turns(self, session_id: str, n: int) -> List[dict]:
        """取回最近 n 轮对话（按时间正序），供 agent 路径上下文组装。"""
        return self.get_history(session_id, limit=n)

    def count_turns(self, session_id: str) -> int:
        with self._conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM session_turns WHERE session_id=?", (session_id,)
            ).fetchone()[0]

    def get_turns_in_range(self, session_id: str, from_idx: int, to_idx: int) -> List[dict]:
        """取回 idx ∈ [from_idx, to_idx] 的轮次（按 idx 正序），供滚动摘要。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT idx, role, content FROM session_turns WHERE session_id=? "
                "AND idx >= ? AND idx <= ? ORDER BY idx",
                (session_id, from_idx, to_idx),
            ).fetchall()
        return [{"idx": i, "role": r, "content": c_} for i, r, c_ in rows]

    def get_all_turns(self, session_id: str) -> List[dict]:
        """取回全部轮次（按 idx 正序），供 token 阈值统计。"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT idx, role, content FROM session_turns WHERE session_id=? ORDER BY idx",
                (session_id,),
            ).fetchall()
        return [{"idx": i, "role": r, "content": c_} for i, r, c_ in rows]

    def delete_turns_up_to(self, session_id: str, idx_to: int) -> None:
        """删除 idx <= idx_to 的轮次（已被摘要后清理原文）。"""
        with self._conn() as c:
            c.execute(
                "DELETE FROM session_turns WHERE session_id=? AND idx <= ?",
                (session_id, idx_to),
            )

    def enforce_session_cap(self, session_id: str, max_turns: int) -> None:
        """超过 max_turns 时删除最旧轮次，防无限增长。"""
        with self._conn() as c:
            cur = c.execute(
                "SELECT COUNT(*) FROM session_turns WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            if cur > max_turns:
                excess = cur - max_turns
                c.execute(
                    "DELETE FROM session_turns WHERE session_id=? AND idx IN ("
                    "SELECT idx FROM session_turns WHERE session_id=? ORDER BY idx LIMIT ?)",
                    (session_id, session_id, excess),
                )

    # ──────────────────────────────────────────
    # session_summaries：长程记忆滚动摘要
    # ──────────────────────────────────────────

    def save_summary(self, session_id: str, idx_from: int, idx_to: int, summary: str) -> None:
        """保存一段滚动摘要（覆盖同名 idx_to，幂等）。"""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO session_summaries"
                "(session_id, idx_from, idx_to, summary, created_at) VALUES (?,?,?,?,?)",
                (session_id, idx_from, idx_to, summary, time.time()),
            )

    def get_latest_summary(self, session_id: str) -> Optional[dict]:
        """取回最近一段摘要（idx_to 最大），无则返回 None。"""
        with self._conn() as c:
            row = c.execute(
                "SELECT idx_from, idx_to, summary, created_at FROM session_summaries "
                "WHERE session_id=? ORDER BY idx_to DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        idx_from, idx_to, summary, created_at = row
        return {
            "idx_from": idx_from,
            "idx_to": idx_to,
            "summary": summary,
            "created_at": created_at,
        }

    def get_history_with_summary(
        self, session_id: str, recent_n: int = 6, limit: int = 20
    ) -> tuple[Optional[dict], List[dict]]:
        """API 用：返回 (最近摘要 | None, 最近 N 轮原文)。"""
        return self.get_latest_summary(session_id), self.get_recent_turns(session_id, recent_n)

    # ──────────────────────────────────────────
    # 测试辅助
    # ──────────────────────────────────────────

    def reset(self) -> None:
        """清空全部表（测试隔离用）。"""
        with self._conn() as c:
            c.execute("DELETE FROM runs")
            c.execute("DELETE FROM run_events")
            c.execute("DELETE FROM session_turns")
            c.execute("DELETE FROM session_summaries")

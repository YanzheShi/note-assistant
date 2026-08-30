# src/note_assistant/indexing/autoindex.py
"""
自动增量索引服务：watchfiles 事件监听 + debounce 合并 + 单飞队列。

架构（docs/自动增量索引（auto-reindex）设计方案.md）：

    vault/*.md 变更 ──► watchfiles.awatch(watch_filter=忽略规则)
        │ 过滤 .md + 排除隐藏目录与 INDEX_IGNORE_DIRS（规则见 indexing/ignore.py）
        │ debounce 收集窗口（默认 3s，合并 Obsidian 保存的多事件）
        ▼
    change_queue (asyncio.Queue) ──► worker 单飞（asyncio.Lock）
        │  ├─ modified/added → 单文件 reindex（incremental_reindex(filepaths=[...])）
        │  ├─ deleted        → 删 chunks + sync 状态（incremental_reindex(deleted_filepaths=[...])）
        │  └─ 批量 ≥ 阈值     → 降级整库增量（incremental_reindex()）
        ▼
    ChromaDB upsert/delete + SyncDB 状态 + [v2b docstore]

设计原则：
1. 检测与执行解耦：watch 只负责「发现变化、去抖、排队」；执行一律走
   `indexing.reindex.incremental_reindex`，与手动 /reindex、index_vault 同一流程，
   不复制第三套索引逻辑。
2. 单飞串行：所有触发（watcher / 手动 API）汇入同一队列 + asyncio.Lock，
   任何时刻只有一个索引执行者，杜绝并发写 ChromaDB / Ollama / sync.db。
3. 单文件优先：watcher 事件带精确路径，只重索引该文件，比整库扫描快一个量级。
4. 零回归约定：autoindex_enabled=False（默认）时只提供手动触发能力（单飞队列），
   不启 watcher；索引产物本身与现状完全一致。
5. reindex（embedding / ChromaDB 写）是阻塞 IO 密集操作，worker 用
   asyncio.to_thread 丢到线程池，避免卡住事件循环（watcher / API 不受影响）。
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from note_assistant.config import settings
from note_assistant.indexing.ignore import is_ignored

logger = logging.getLogger(__name__)

# 任务标记：整库增量（批量降级 / 手动触发共用）
_FULL: str = "full"


def is_indexable_md(rel: str) -> bool:
    """watcher 过滤：只看 .md，且套用统一忽略规则（与 VaultLoader.scan 同源）。

    Args:
        rel: 相对 vault 根的路径（正斜杠或反斜杠分隔）

    Returns:
        True 表示该路径值得进入索引队列
    """
    if not rel.lower().endswith(".md"):
        return False
    return not is_ignored(rel)


class AutoIndexService:
    """自动增量索引服务：watchfiles 事件监听 + debounce + 单飞队列 + 状态统计。

    生命周期由 api lifespan 管理：start() 启 worker（+enabled 时启 watcher），
    stop() 优雅退出（取消 watcher → 等队列排空 → 停 worker，不丢变更）。
    """

    def __init__(
        self,
        vault_path: str | Path | None = None,
        *,
        enabled: Optional[bool] = None,
        debounce_seconds: Optional[float] = None,
        batch_fallback_threshold: Optional[int] = None,
        full_sync_every: Optional[int] = None,
    ):
        self.vault_path = Path(vault_path) if vault_path else settings.vault_path.resolve()
        self.enabled = settings.autoindex_enabled if enabled is None else enabled
        self.debounce_seconds = (
            settings.autoindex_debounce_seconds if debounce_seconds is None else debounce_seconds
        )
        self.batch_fallback_threshold = (
            settings.autoindex_batch_fallback_threshold
            if batch_fallback_threshold is None
            else batch_fallback_threshold
        )
        self.full_sync_every = settings.autoindex_full_sync_every if full_sync_every is None else full_sync_every

        self._queue: "asyncio.Queue[Tuple]" = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._watcher_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._debounce_task: Optional[asyncio.Task] = None
        # 去抖窗口内的待处理变更：rel path -> "add" | "delete"（latest 覆盖）
        self._pending: Dict[str, str] = {}

        self._stats = {
            "running": False,
            "last_run_at": None,       # ISO 时间字符串
            "last_run": {},            # {"reindexed": n, "removed": n, "duration_ms": n}
            "total_runs": 0,
            "errors": 0,
            "increments_since_full": 0,
            "last_full_sync_at": None,
        }

    # ──────────────────────────────────────────────
    # 生命周期（api lifespan 调用）
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        """启动 worker（始终启动，手动触发也走单飞队列）；enabled 时追加 watcher。"""
        self._worker_task = asyncio.create_task(self._worker(), name="autoindex-worker")
        if self.enabled:
            self._watcher_task = asyncio.create_task(self._watch_loop(), name="autoindex-watcher")
            logger.info(
                "🔄 自动增量索引已启用：监听 %s（debounce=%.1fs）",
                self.vault_path, self.debounce_seconds,
            )

    async def stop(self) -> None:
        """优雅退出：停 watcher → 冲刷残留事件 → 等队列排空 → 停 worker。"""
        if self._watcher_task:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None

        # 取消 debounce 睡眠时冲刷 pending，不丢已收集事件
        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None
        if self._pending:
            await self._flush_pending()

        if self._worker_task:
            await self._queue.join()  # 等剩余任务执行完
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    # ──────────────────────────────────────────────
    # 手动触发（api /reindex、/reindex/run 调用）
    # ──────────────────────────────────────────────

    async def run_incremental(self) -> dict:
        """手动触发一次全库增量：汇入单飞队列并等待执行完毕（与自动触发互斥）。

        Returns:
            incremental_reindex 的返回值 {"status", "reindexed", "removed"}
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._queue.put((_FULL, fut))
        return await fut

    # ──────────────────────────────────────────────
    # watcher：事件吸收 + debounce + 入队
    # ──────────────────────────────────────────────

    def _rel_if_indexable(self, path) -> Optional[str]:
        """绝对路径 → 值得索引的 vault 相对路径；vault 外或被忽略则 None。"""
        try:
            rel = Path(str(path)).resolve().relative_to(self.vault_path.resolve())
        except ValueError:
            return None
        rel_str = rel.as_posix()
        return rel_str if is_indexable_md(rel_str) else None

    def _watch_filter(self, _event_type, path: str) -> bool:
        """交给 watchfiles 的事件级过滤：忽略路径不进 Python，省掉无谓的 absorb 空转。"""
        return self._rel_if_indexable(path) is not None

    def _absorb(self, changes) -> bool:
        """吸收一批 watchfiles 事件到 pending；返回是否有值得入队的 .md 变更。"""
        dirty = False
        for change, path in changes:
            rel_str = self._rel_if_indexable(path)
            if rel_str is None:
                continue
            from watchfiles import Change

            self._pending[rel_str] = "delete" if change == Change.deleted else "add"
            dirty = True
        return dirty

    async def _watch_loop(self) -> None:
        """watchfiles 事件循环：事件 → pending，重置 debounce 窗口。"""
        from watchfiles import awatch

        # ignore_permission_denied：vault 内偶发的无权限目录（同步占位/临时锁目录）
        # 只跳过，不让它把整个 watcher 打崩（watcher 一崩，自动索引就静默失效）
        async for changes in awatch(
            self.vault_path,
            watch_filter=self._watch_filter,
            ignore_permission_denied=True,
        ):
            if self._absorb(changes) and self._debounce_task is None:
                self._debounce_task = asyncio.create_task(self._debounce_flush())

    async def _debounce_flush(self) -> None:
        """收集窗口结束 → 把 pending 入队一次（窗口内事件合并）。"""
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            # stop() 取消时由调用方负责冲刷（见 stop）
            return
        finally:
            self._debounce_task = None
        await self._flush_pending()

    async def _flush_pending(self) -> None:
        """把 pending 分类入队：批量 ≥ 阈值 → 降级整库增量；否则单文件批次。"""
        pending, self._pending = self._pending, {}
        adds = sorted(p for p, t in pending.items() if t == "add")
        dels = sorted(p for p, t in pending.items() if t == "delete")
        if not adds and not dels:
            return
        if len(pending) >= self.batch_fallback_threshold:
            logger.info("autoindex: 批量 %d 个变更 → 降级整库增量", len(pending))
            await self._queue.put((_FULL, None))
        else:
            await self._queue.put(({"index": adds, "delete": dels}, None))

    # ──────────────────────────────────────────────
    # worker：单飞串行执行
    # ──────────────────────────────────────────────

    async def _worker(self) -> None:
        while True:
            item, fut = await self._queue.get()
            try:
                result = await self._run_batch(item)
                if fut is not None and not fut.done():
                    fut.set_result(result)
            except Exception as e:  # noqa: BLE001
                self._stats["errors"] += 1
                logger.error("autoindex: 批次执行失败: %s", e, exc_info=True)
                if fut is not None and not fut.done():
                    fut.set_result({"status": "error", "reindexed": 0, "removed": 0})
            finally:
                self._queue.task_done()

    async def _run_batch(self, item) -> dict:
        """执行一个批次（单飞：整个批次持锁，任何时刻只有一个索引执行者）。"""
        from note_assistant.indexing.reindex import incremental_reindex

        async with self._lock:
            self._stats["running"] = True
            t0 = time.time()
            try:
                if item == _FULL:
                    result = await asyncio.to_thread(
                        incremental_reindex, str(self.vault_path)
                    )
                else:
                    result = await asyncio.to_thread(
                        incremental_reindex,
                        str(self.vault_path),
                        filepaths=item["index"] or None,
                        deleted_filepaths=item["delete"] or None,
                    )
                self._stats["total_runs"] += 1
                self._stats["last_run"] = {
                    "reindexed": result.get("reindexed", 0),
                    "removed": result.get("removed", 0),
                    "duration_ms": round((time.time() - t0) * 1000),
                }
                self._stats["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                logger.info(
                    "[autoindex] files=%s removed=%s duration=%.1fs",
                    result.get("reindexed", 0), result.get("removed", 0),
                    time.time() - t0,
                )
                await self._maybe_full_calibration()
                return result
            finally:
                self._stats["running"] = False

    # ──────────────────────────────────────────────
    # 二级索引校准（设计方案 4.5）：累计 N 次增量后全量刷新
    # ──────────────────────────────────────────────

    async def _maybe_full_calibration(self) -> None:
        """达到 autoindex_full_sync_every 后，在队列空（低峰）时全量重建一次。"""
        if not self.full_sync_every:
            return
        self._stats["increments_since_full"] += 1
        if self._stats["increments_since_full"] < self.full_sync_every:
            return
        if not self._queue.empty():
            return  # 队列非空，等下一批再校准
        self._stats["increments_since_full"] = 0
        try:
            await asyncio.to_thread(_full_calibration, str(self.vault_path))
            self._stats["last_full_sync_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            logger.info("[autoindex] 全量校准完成（BM25 + WikiGraph）")
        except Exception as e:  # noqa: BLE001
            self._stats["errors"] += 1
            logger.error("autoindex: 全量校准失败: %s", e, exc_info=True)

    # ──────────────────────────────────────────────
    # 状态（api /reindex/status 观察）
    # ──────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """当前状态快照：{enabled, queue_len, running, last_run_at, last_run, total_runs, errors}。"""
        st = dict(self._stats)
        st["enabled"] = self.enabled
        st["queue_len"] = self._queue.qsize()
        return st


def _full_calibration(vault_path: str) -> None:
    """全量校准：wipe 重建主索引 + 重建 BM25 / WikiGraph 两个派生产物。

    与 scripts/full_reindex.py 语义一致（两步重建独立 try/except，不互相拖累）。
    """
    from note_assistant.indexing.ingestor import Ingestor

    ingestor = Ingestor()
    ingestor.index_vault(vault_path=vault_path, wipe=True)

    # ① BM25：从 ChromaDB 全量拉 chunks 重建（与主索引同源，天然一致）
    try:
        from note_assistant.retrieval.sparse_retriever import BM25Retriever

        bm25 = BM25Retriever.from_chroma()
        bm25.save()
    except Exception as e:  # noqa: BLE001
        logger.warning("autoindex: BM25 重建失败（稀疏检索仍在旧索引上）: %s", e)

    # ② WikiGraph：从全库 wikilinks 重建
    try:
        from note_assistant.indexing.vault_loader import VaultLoader
        from note_assistant.retrieval.graph import WikiGraph

        docs = VaultLoader(vault_path).load_all()
        g = WikiGraph()
        g.build_from_docs(docs)
        g.save()
    except Exception as e:  # noqa: BLE001
        logger.warning("autoindex: WikiGraph 重建失败（图扩展仍在旧图上）: %s", e)
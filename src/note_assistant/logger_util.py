"""统一的 JSON 结构化日志。

设计原则（标准 logging 架构）：
- 业务模块只写 ``logger = logging.getLogger(__name__)``，只管「发日志」；
- 入口（FastAPI / CLI）调用一次 ``setup_logging()``，用 ``logging.config.dictConfig``
  统一挂 ``JsonFormatter`` + ``RequestIdFilter``，全项目生效；
- ``request_id`` 用 ``ContextVar`` 保存，由 ``RequestIdFilter`` 自动注入每条日志，
  无需手动传递，也自然跨 ``await`` 传播；
- 结构化字段（route / elapsed_ms / tool 等）直接走
  ``logger.info("msg", extra={...})``，``JsonFormatter`` 把 extra 字段拼进顶层 JSON。

输出样例（单行 JSON）：
    {"timestamp":"2026-07-27T02:52:46.123456+00:00","level":"INFO",
     "logger":"note_assistant.agent.runner","module":"runner","line":280,
     "message":"ainvoke.done","request_id":"req-abc","cached":true,
     "sources":3,"elapsed_ms":12}

后续接 LangSmith：把 ``request_id`` 通过 ``graph.ainvoke(state, config={
    "metadata": {"request_id": rid}})`` 注入即可，两套系统用 request_id 互相关联。
"""
from __future__ import annotations

import contextvars
import json as _json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any, Optional

# ──────────────────────────────────────────────
# 上下文：request_id 跨 await 传递
# ──────────────────────────────────────────────
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "note_assistant_request_id", default="-"
)


def set_request_id(rid: Optional[str]) -> None:
    """写入当前请求的 id，同请求的所有日志共享（ContextVar 跨 await 传递）。"""
    _request_id.set((rid or "-").strip() or "-")


def get_request_id() -> str:
    """读取当前请求 id，未设置返回 '-'。"""
    return _request_id.get()


# ──────────────────────────────────────────────
# 过滤器：把 request_id 自动注入每条日志记录
# ──────────────────────────────────────────────
class RequestIdFilter(logging.Filter):
    """为每条日志注入当前 request_id（无需手动传递）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _json_default(obj: Any) -> Any:
    """兜底：把 Path / set / 数据类 等不可序列化对象转成可读表示。"""
    if hasattr(obj, "__dict__"):
        return {
            k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
            for k, v in obj.__dict__.items()
        }
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return str(obj)
    return str(obj)


# LogRecord 内置属性，不应作为 extra 字段透传进 JSON
_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "lineno", "funcName", "created",
    "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "stack_info", "message", "taskName", "asctime",
    "request_id", "timestamp",
}


class JsonFormatter(logging.Formatter):
    """每条日志输出一行 JSON。

    - ``message`` 为格式化后的日志正文（含 %-args）；
    - ``module`` / ``line`` 取真实调用点（标准 logging，无需 stacklevel hack）；
    - 调用方通过 ``extra={...}`` 传入的字段原样透传（elapsed_ms / route / tool 等）；
    - ``request_id`` 由 ``RequestIdFilter`` 注入；
    - 异常栈以 ``exc_info`` 字段附带，不丢失。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, val in record.__dict__.items():
            if key in _RESERVED_ATTRS:
                continue
            payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return _json.dumps(payload, ensure_ascii=False, default=_json_default)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True,
) -> None:
    """入口统一配置（整个进程只调用一次）。

    Args:
        level: 根日志级别，默认 INFO。
        log_file: 额外写入的日志文件路径（JSON 格式）；不传则只打控制台。
        json_logs: True 用 JsonFormatter；False 用可读文本格式（本地调试用）。
    """
    fmt_id = "json" if json_logs else "text"
    text_fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": fmt_id,
            "filters": ["request_id"],
        }
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": log_file,
            "encoding": "utf-8",
            "formatter": fmt_id,
            "filters": ["request_id"],
        }
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {
                "json": {"()": JsonFormatter},
                "text": {"format": text_fmt},
            },
            "handlers": handlers,
            "root": {"handlers": list(handlers.keys()), "level": level},
        }
    )

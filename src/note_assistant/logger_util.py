"""统一的 JSON 结构化日志（基于 python-json-logger）。

使用方式：
    入口（FastAPI lifespan）调用一次 setup_logging()，
    业务模块只写 logger = logging.getLogger(__name__)，然后 logger.info("msg", extra={...})。

输出样例：
    {"timestamp": "2026-07-27T12:00:00.000000+00:00", "level": "INFO",
     "logger": "note_assistant.agent.runner", "message": "ainvoke.done",
     "request_id": "req-abc", "cached": true, "elapsed_ms": 12}
"""
from __future__ import annotations

import contextvars
import logging
import logging.config
from typing import Optional

from pythonjsonlogger import jsonlogger

# ── request_id 跨 await 传递 ──
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "note_assistant_request_id", default="-"
)


def set_request_id(rid: Optional[str]) -> None:
    _request_id.set((rid or "-").strip() or "-")


def get_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """自动注入 request_id 到每条日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True,
) -> None:
    """进程入口统一调用一次，配置全局日志。

    Args:
        level: 日志级别，默认 INFO。
        log_file: 额外写文件路径；不传只打控制台。
        json_logs: True 用 JSON 格式，False 用可读文本（本地调试）。
    """
    fmt_id = "json" if json_logs else "text"
    text_fmt = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": fmt_id,
            "filters": ["request_id"],
        }
    }
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": log_file,
            "when": "midnight",
            "encoding": "utf-8",
            "formatter": fmt_id,
            "filters": ["request_id"],
        }

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": (
                    "%(timestamp)s %(level)s %(logger)s %(message)s"
                ),
            },
            "text": {"format": text_fmt},
        },
        "handlers": handlers,
        "root": {"handlers": list(handlers.keys()), "level": level},
    })

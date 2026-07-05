# frontend/utils.py
"""
API 调用封装 —— 将前端与后端的网络通信抽象为纯函数。

对外暴露三个函数：
    ask_question(api_url, question, timeout) -> dict
        非流式调用 /ask，返回完整响应 dict

    ask_question_stream(api_url, question)
        流式调用 /ask_stream，返回 event generator

    ask_question_trace(api_url, question)
        追踪式流式调用 /ask_trace，额外输出检索过程 trace 事件
"""

import json
import httpx


def ask_question(api_url: str, question: str, history: list[dict] | None = None, timeout: float = 120.0) -> dict:
    """
    非流式调用 /ask，支持历史对话。

    Args:
        api_url: FastAPI 地址（如 "http://localhost:8000"）
        question: 用户问题
        history: 历史对话列表
        timeout: 超时秒数

    Returns:
        完整响应 dict，含 answer, sources, timing 等

    Raises:
        httpx.ConnectError: 后端未启动
        httpx.TimeoutException: 请求超时
        httpx.HTTPStatusError: 后端返回 4xx/5xx
    """
    import time
    t0 = time.time()
    body = {"question": question}
    if history:
        body["history"] = history
    resp = httpx.post(
        f"{api_url}/ask",
        json=body,
        timeout=timeout,
    )

    resp.raise_for_status()
    result = resp.json()

    if "timing" not in result or not result["timing"]:
        result["timing"] = {"total_ms": int((time.time() - t0) * 1000)}

    return result


def _sse_events(api_url: str, question: str, endpoint: str, history: list[dict] | None = None, timeout: float = 120.0):
    """
    通用的 SSE 事件流解析器。

    向指定 endpoint 发起 POST 请求，逐行解析 "data: " 前缀的 SSE 事件。
    """
    body = {"question": question}
    if history:
        body["history"] = history
    with httpx.stream(
        "POST", f"{api_url}/{endpoint}",
        json=body,
        timeout=httpx.Timeout(connect=30.0, read=None, write=timeout, pool=30.0),
    ) as resp:
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    yield {"type": "done"}
                    return
                yield json.loads(data_str)


def ask_question_stream(api_url: str, question: str, history: list[dict] | None = None, timeout: float = 120.0):
    """
    流式调用 /ask_stream（SSE 事件流），支持历史对话。

    事件类型: meta, char, sources
    """
    yield from _sse_events(api_url, question, "ask_stream", history, timeout)


def ask_question_trace(api_url: str, question: str, history: list[dict] | None = None, timeout: float = 120.0):
    """
    追踪式流式调用 /ask_trace（SSE 事件流），支持历史对话。

    在 ask_question_stream 的基础上，检索阶段额外输出 trace 事件：
        embedding → dense_retrieval → sparse_retrieval → hybrid_fusion → rerank → graph_expansion
    """
    yield from _sse_events(api_url, question, "ask_trace", history, timeout)

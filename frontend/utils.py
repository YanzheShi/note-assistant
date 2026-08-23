# frontend/utils.py
"""
API 调用封装 —— 将前端与后端的网络通信抽象为纯函数。

默认全部指向 Agentic RAG 端点（/agent/*）。后端旧的 /ask* 端点（传统 RAGChain）
仍保留可作对比通道，但前端不再使用。

对外暴露三个函数：
    ask_question(api_url, question, history, session_id) -> dict
        非流式调用 /agent/ask，返回完整响应 dict
        （前端页面已恒为流式，不再调用此函数；保留供脚本/调试直连后端使用）

    ask_question_stream(api_url, question, history, session_id)
        流式调用 /agent/ask_stream，返回 event generator

    ask_question_trace(api_url, question, history, session_id)
        追踪式流式调用 /agent/ask_stream（同一端点）；
        后端以事件形式实时输出推理过程（thought / tool_call / observation / judge），
        由 app.py 在 trace_mode 下渲染为步骤面板
"""

import json
import httpx


def rewrite_asset_urls(text: str, api_url: str) -> str:
    """把答案正文里的相对资产 URL ``](/assets/…)`` 重写为后端全地址。

    背景：后端图片闭环产出的 markdown 是 ``![title](/assets/{asset_id})``——
    相对路径。来源面板渲染时会拼后端地址，但答案正文直接进 ``st.markdown``，
    相对 URL 会解析到 Streamlit 自己的端口（如 8501）而非后端（8005），
    图片 404 不可见——这正是「标记替换成功了图却不显示」的最后一公里断点。

    - 存原文、渲染时重写：用户在侧边栏改 API 地址后历史消息依然可渲染。
    - api_url 为空（未配置）时原样返回，行为不劣于改造前。
    """
    if not text or "/assets/" not in text:
        return text
    base = (api_url or "").rstrip("/")
    if not base:
        return text
    return text.replace("](/assets/", f"]({base}/assets/")


def _post_json(api_url: str, endpoint: str, body: dict, timeout: float) -> dict:
    resp = httpx.post(f"{api_url}/{endpoint}", json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ask_question(
    api_url: str,
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
) -> dict:
    """
    非流式调用 /agent/ask，支持历史对话与跨会话记忆（session_id）。

    Returns:
        完整响应 dict，含 answer / sources / trajectory / cached / run_id / timing
    """
    import time

    t0 = time.time()
    body = {"question": question}
    if history:
        body["history"] = history
    if session_id:
        body["session_id"] = session_id

    result = _post_json(api_url, "agent/ask", body, timeout)

    if "timing" not in result or not result.get("timing"):
        result["timing"] = {"total_ms": int((time.time() - t0) * 1000)}
    return result


def _sse_events(
    api_url: str,
    question: str,
    endpoint: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
):
    """
    通用 SSE 事件流解析器。

    逐行解析 "data: " 前缀事件；遇到 "data: [DONE]" 结束。
    事件类型：run / thought / tool_call / observation / judge / answer / sources / cached / status / error
    """
    body = {"question": question}
    if history:
        body["history"] = history
    if session_id:
        body["session_id"] = session_id

    with httpx.stream(
        "POST", f"{api_url}/{endpoint}",
        json=body,
        timeout=httpx.Timeout(connect=30.0, read=None, write=timeout, pool=30.0),
    ) as resp:
        for line in resp.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                yield {"type": "done"}
                return
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


def ask_question_stream(
    api_url: str,
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
):
    """流式调用 /agent/ask_stream（SSE 事件流），支持历史对话与跨会话记忆。"""
    yield from _sse_events(api_url, question, "agent/ask_stream", history, session_id, timeout)


def ask_question_trace(
    api_url: str,
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
):
    """
    追踪式流式调用同一端点 /agent/ask_stream。

    后端以事件形式实时输出推理过程：
        thought     — 路由判定 / 推理
        tool_call   — 调用的检索工具与参数
        observation — 工具返回摘要
        judge       — 反思判定（sufficient / need_rewrite / need_more / give_up）
        answer      — 最终答案
        sources     — 去重后的来源列表
    app.py 在 trace_mode 下把 thought / tool_call / observation / judge 渲染为步骤面板。
    """
    yield from _sse_events(api_url, question, "agent/ask_stream", history, session_id, timeout)


def ask_question_classic_stream(
    api_url: str,
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
):
    """
    传统 RAG 流式调用 /ask_stream（SSE 逐字符事件流），支持多轮对话历史。

    事件协议（与 Agentic 不同）：
        meta    — 元信息（retrieve_ms / graph_expansion），前端忽略
        char    — 答案逐字符片段，累积为完整 answer
        sources — 检索来源列表（content 字段，非 sources 字段）
        status  — 非 finished 时告警
        error   — 错误
        [DONE]

    注意：传统 RAG 不经过 Router / 反思循环，无 thought / tool_call / judge 轨迹。
    """
    yield from _sse_events(api_url, question, "ask_stream", history, session_id, timeout)


def ask_question_classic_trace(
    api_url: str,
    question: str,
    history: list[dict] | None = None,
    session_id: str = "",
    timeout: float = 120.0,
):
    """
    传统 RAG 追踪式流式调用 /ask_trace（检索过程实时输出），支持多轮对话历史。

    在 /ask_stream 基础上，检索阶段额外逐步骤输出 trace 事件：
        trace    — 单步检索（step: embedding / dense_retrieval / sparse_retrieval /
                    hybrid_fusion / rerank / graph_expansion；附 ms / results）
        char     — 答案逐字符片段
        sources  — 检索来源列表
        [DONE]

    trace_mode 开启时，app.py 把 trace 步骤渲染为面板，让用户看到「正在检索什么」。
    """
    yield from _sse_events(api_url, question, "ask_trace", history, session_id, timeout)

# frontend/app.py
"""Streamlit 主入口 —— Obsidian RAG 聊天界面。

提供两个**完全独立**的问答通道（左侧 tab 切换，会话记忆互不共享）：
    📚 传统 RAG   — 走 /ask_stream（或 /ask_trace），Hybrid 检索 + Rerank + 生成，无反射循环
    🤖 Agentic RAG — 走 /agent/ask_stream，Router 路由 + 多轮检索反思 + 带引用生成

两种模式各自的 messages / session_id 互不引用：切 tab 不会串味，
传统模式的多轮上下文只在该 tab 内维护，不影响 Agentic 的跨会话记忆。

后端两套端点均已并存（见 src/note_assistant/api/main.py），本文件只负责前端接线与渲染分支。
"""

import sys
import time
import uuid

from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st  # noqa: E402
import httpx  # noqa: E402
from frontend.components import render_sources  # noqa: E402
from frontend.utils import rewrite_asset_urls  # noqa: E402

# ─── 页面配置 ───
st.set_page_config(page_title="Obsidian RAG", layout="wide", page_icon="📚")

# ─── 状态（两套会话各自独立，互不共享）───
for key, default in [
    # 传统 RAG 通道
    ("messages_rag", []),
    ("session_rag", str(uuid.uuid4())),
    # Agentic RAG 通道
    ("messages_agent", []),
    ("session_agent", str(uuid.uuid4())),
    # 共享显示偏好（非会话记忆，不影响隔离性）
    ("trace_mode", True),
    # 当前选中的问答通道（sidebar radio 写入；"rag" 传统 / "agent" Agentic）
    ("chat_mode", "rag"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── 侧边栏（共享：API 地址 + 追踪开关）───
with st.sidebar:
    st.markdown("## 🔧 设置")
    API_URL = st.text_input("API 地址", value="http://localhost:8005")
    st.session_state.trace_mode = st.checkbox(
        "追踪模式（显示推理/检索过程）", value=True,
        help="传统 RAG：展示检索各步骤耗时；Agentic RAG：展示路由→工具→观察→反思")

    # ── 模式选择（左侧单选，切换两种问答通道）──
    # Streamlit 的 sidebar 不支持 st.tabs 容器，故用 radio 实现「左侧选模式」诉求。
    # 两种模式的 messages / session_id 各自独立，切换不丢各自上下文。
    _MODE_MAP = {"📚 传统 RAG（更快速）": "rag", "🤖 Agentic RAG（更准确）": "agent"}
    _selected = st.radio(
        "问答模式",
        list(_MODE_MAP.keys()),
        index=0,
        help="左侧切换两种问答通道；会话记忆各自独立，互不影响。",
    )
    st.session_state.chat_mode = _MODE_MAP[_selected]
    st.caption("两种模式的会话记忆完全独立，互不影响。")


# ─── 追踪步骤渲染 helper（两模式共用）───
def _render_trace_step(container, icon: str, label: str, detail: str):
    """把一条推理/检索事件渲染为可折叠步骤（仅在 trace_mode 下调用）。"""
    if container is None:
        return
    with container:
        with st.expander(f"{icon} {label}", expanded=True):
            if detail:
                st.markdown(detail)


def _typewriter(placeholder, text: str, chunk: int = 4, delay: float = 0.008):
    """把整段答案做轻量打字机效果（后端一次性给出 answer，前端逐块揭示）。"""
    shown = ""
    for i in range(0, len(text), chunk):
        shown = text[: i + chunk]
        placeholder.markdown(shown + "▌")
        time.sleep(delay)
    placeholder.markdown(shown or text)


# ─────────────────────────────────────────────────────────────
# 渲染单个问答通道（整段逻辑按 mode 分支事件协议）
# ─────────────────────────────────────────────────────────────
def render_chat_pane(mode: str):
    """
    渲染一个问答通道的完整交互（历史回放 + 输入 + 流式回复）。

    Args:
        mode: ``"rag"`` 传统 RAG（/ask_stream | /ask_trace）；
              ``"agent"`` Agentic RAG（/agent/ask_stream）。

    隔离性：每个 tab 调用时只读取自己的一组 session_state key，
    不触碰另一模式的 messages / session_id。
    """
    if mode == "rag":
        msgs_key, sess_key = "messages_rag", "session_rag"
        title, caption = "📚 传统 RAG", "Hybrid 检索 + Rerank + 生成（无反射循环，单次检索即答）"
    else:
        msgs_key, sess_key = "messages_agent", "session_agent"
        title, caption = "🤖 Agentic RAG", "Router 路由 + 多轮检索反思 + 带引用生成"

    messages = st.session_state[msgs_key]
    session_id = st.session_state[sess_key]

    st.subheader(title)
    st.caption(caption)

    # ── 本通道独立的清空按钮 ──
    if st.button("🗑️ 清空对话", key=f"clear_{mode}"):
        st.session_state[msgs_key] = []
        st.rerun()

    # ── 历史回放 ──
    for msg in messages:
        with st.chat_message(msg["role"]):
            # 渲染时重写 /assets/ 相对 URL（存原文，API 地址变更后历史仍可渲染图片）
            st.markdown(rewrite_asset_urls(msg["content"], API_URL))
            if msg["role"] == "assistant" and "sources" in msg:
                render_sources(msg["sources"], backend_base_url=API_URL)

    # ── 输入（必须在 tab 内；空输入直接 return，不 st.stop，避免误伤另一 tab）──
    question = st.chat_input("问我你的笔记里有什么...", key=f"input_{mode}")
    if not question:
        return

    # ── 用户消息 ──
    st.session_state[msgs_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ── 助手回复（恒为流式 SSE）───
    with st.chat_message("assistant"):
        full_answer = ""
        result = {"sources": [], "timing": {}, "trajectory": [], "run_id": ""}

        # 本轮之前的历史（排除刚添加的问题），只取本通道的
        history = st.session_state[msgs_key][:-1]

        _t0 = time.time()
        streaming_placeholder = st.empty()
        streaming_placeholder.markdown("🤔 思考中…")

        trace_container = None
        if st.session_state.trace_mode:
            trace_container = st.container()

        try:
            from frontend.utils import (
                ask_question_stream,
                ask_question_trace,
                ask_question_classic_stream,
                ask_question_classic_trace,
            )

            if mode == "agent":
                fn = ask_question_trace if st.session_state.trace_mode else ask_question_stream
                event_iter = fn(API_URL, question, history=history, session_id=session_id)
            else:
                fn = ask_question_classic_trace if st.session_state.trace_mode else ask_question_classic_stream
                # 传统 RAG 的 session_id 后端忽略（无跨会话记忆），传空避免歧义
                event_iter = fn(API_URL, question, history=history, session_id="")

            answer_text = ""
            sources_list = []
            run_id = ""
            thinking_cleared = False

            for event in event_iter:
                etype = event.get("type")

                # 收到首个实质事件后清掉顶部「思考中…」占位
                # 传统 RAG 的 meta 事件不触发清屏（只是检索耗时元信息）
                if mode == "agent":
                    clear_skip = ("run", "status", "done")
                else:
                    clear_skip = ("meta", "status", "done")
                if not thinking_cleared and etype not in clear_skip:
                    streaming_placeholder.empty()
                    thinking_cleared = True

                # ── 轨迹事件收集（供历史记录 / 轨迹回放，仅 Agentic 有）──
                if etype in ("thought", "tool_call", "observation", "judge", "answer"):
                    result["trajectory"].append(event)

                if etype == "run":
                    run_id = event.get("run_id", "")

                elif etype == "thought":
                    _render_trace_step(trace_container, "🧠", "推理", event.get("content", ""))

                elif etype == "tool_call":
                    tool = event.get("tool", "")
                    args = event.get("args", {}) or {}
                    q = args.get("query") or args.get("question") or ""
                    _render_trace_step(trace_container, "🔧", f"工具调用：{tool}", q)

                elif etype == "observation":
                    _render_trace_step(trace_container, "👀", "观察结果", event.get("content", ""))

                elif etype == "judge":
                    verdict = event.get("verdict", "")
                    reason = event.get("reason", "")
                    vmap = {
                        "sufficient": ("✅", "信息充足，生成答案"),
                        "need_rewrite": ("🔄", "信息不足，改写法检索"),
                        "need_more": ("🔍", "信息不足，换策略检索"),
                        "give_up": ("⚠️", "多次无果，生成受限答案"),
                        "need_clarify": ("❓", "问题指向多个主题，反问澄清"),
                    }
                    icon, label = vmap.get(verdict, ("➡️", verdict))
                    detail = label
                    if reason:
                        detail += f"\n> {reason}"
                    _render_trace_step(trace_container, icon, "反思判定", detail)

                elif etype == "answer":
                    answer_text = event.get("content", "")
                    # 答案正文的图片 URL 是 /assets/{id} 相对路径，必须拼后端地址
                    render_text = rewrite_asset_urls(answer_text, API_URL)
                    if trace_container is not None:
                        with trace_container:
                            with st.expander("📝 答案", expanded=True):
                                _aw = st.empty()
                                _typewriter(_aw, render_text)
                    else:
                        _typewriter(streaming_placeholder, render_text)

                # ── 传统 RAG 检索步骤（仅 /ask_trace 产出）──
                elif etype == "trace":
                    step = event.get("step", "")
                    ms = event.get("ms", 0)
                    results = event.get("results", "")
                    detail = f"耗时 {ms}ms"
                    if results != "":
                        detail += f" ｜ 中间结果 {results} 条"
                    _render_trace_step(trace_container, "🔍", f"检索：{step}", detail)

                # ── 传统 RAG 逐字符答案（无独立 answer 事件，需累积）──
                elif etype == "char":
                    answer_text += event.get("content", "")

                elif etype == "sources":
                    # 传统 RAG：来源在 content 字段（SourceSchema 列表，带 type 字段）
                    # Agentic RAG：来源在 sources 字段（AgentSource 列表，带 kind 字段）
                    if mode == "rag":
                        sources_list = event.get("content", []) or []
                    else:
                        sources_list = event.get("sources", []) or []
                    result["sources"] = sources_list
                    ge = event.get("graph_expansion", 0)
                    if ge and trace_container is not None:
                        with trace_container:
                            st.caption(f"🔗 双链扩展带出 {ge} 条关联片段")

                elif etype == "cached":
                    if trace_container is not None:
                        with trace_container:
                            st.caption("⚡ 命中语义缓存，直接返回")

                elif etype == "status":
                    if event.get("status") != "finished":
                        st.warning(event.get("content", ""))

                elif etype == "error":
                    st.error(event.get("content", "未知错误"))

                elif etype == "done":
                    break

                else:
                    # 兜底：未知事件类型留痕迹，避免「界面卡住」却无任何提示
                    if trace_container is not None:
                        with trace_container:
                            st.caption(f"➡️ 未识别事件：{etype}")

            # ── 传统 RAG：答案以 char 累积，循环结束后统一渲染 ──
            if mode == "rag" and answer_text:
                render_text = rewrite_asset_urls(answer_text, API_URL)
                if trace_container is not None:
                    with trace_container:
                        with st.expander("📝 答案", expanded=True):
                            _aw = st.empty()
                            _typewriter(_aw, render_text)
                else:
                    _typewriter(streaming_placeholder, render_text)

            if trace_container is not None:
                with trace_container:
                    st.markdown("✅ 完成")

            if sources_list:
                render_sources(sources_list, backend_base_url=API_URL)

            full_answer = answer_text
            result["run_id"] = run_id

        except (ImportError, NotImplementedError, httpx.HTTPError) as e:
            st.warning(f"流式调用失败: {e}")
            full_answer = "（流式输出失败，请检查后端）"
            streaming_placeholder.markdown(full_answer)

        st.session_state[msgs_key].append({
            "role": "assistant",
            "content": full_answer,
            "sources": result.get("sources", []),
            "trajectory": result.get("trajectory", []),
            "run_id": result.get("run_id", ""),
        })


# ─── 主标题 + 按左侧 radio 渲染对应通道（各自独立会话）───
st.title("📚 个人知识库问答")
st.markdown("---")

# 左侧 radio 选定通道后，主区只渲染对应 pane；两会话状态各自保留，切换不串味。
render_chat_pane(st.session_state.chat_mode)

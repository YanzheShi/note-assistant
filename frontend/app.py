# frontend/app.py
"""Streamlit 主入口 —— Obsidian RAG 聊天界面，支持多轮连续对话（Agentic RAG）。

输出恒为流式（SSE）：普通问答与追踪模式都走 /agent/ask_stream，
不再提供非流式选项（后端 /agent/ask 仍保留，供 API 调用方/评估使用）。
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

# ─── 页面配置 ───
st.set_page_config(page_title="Obsidian RAG", layout="wide", page_icon="📚")

# ─── 状态 ───
for key, default in [
    ("messages", []),
    ("trace_mode", False), ("session_id", str(uuid.uuid4())),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── 侧边栏 ───
with st.sidebar:
    st.markdown("## 🔧 设置")
    API_URL = st.text_input("API 地址", value="http://localhost:8005")
    st.session_state.trace_mode = st.checkbox(
        "追踪模式（显示推理过程）", value=False,
        help="实时展示：路由判定 → 工具调用 → 观察结果 → 反思判定 → 答案")
    st.markdown("---")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ─── 主标题 ───
st.title("📚 个人知识库问答")
st.caption("基于 Obsidian 笔记库的 Agentic RAG：Router 路由 + 多轮检索反思 + 带引用生成")
st.markdown("---")


# ─── 追踪步骤渲染 helper ───
def _render_trace_step(container, icon: str, label: str, detail: str):
    """把一条推理事件渲染为可折叠步骤（仅在 trace_mode 下调用）。"""
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


# ─── 聊天历史 ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])

# ─── 输入 ───
question = st.chat_input("问我你的笔记里有什么...")
if not question:
    st.stop()

# ─── 用户消息 ───
st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

# ─── 助手回复（恒为流式 SSE）───
with st.chat_message("assistant"):
    full_answer = ""
    result = {"sources": [], "timing": {}, "trajectory": [], "run_id": ""}
    session_id = st.session_state.session_id

    # 本轮之前的历史（排除刚添加的当前问题）
    history = st.session_state.messages[:-1]

    _t0 = time.time()
    streaming_placeholder = st.empty()
    streaming_placeholder.markdown("🤔 思考中…")

    trace_container = None
    if st.session_state.trace_mode:
        trace_container = st.container()

    try:
        from frontend.utils import ask_question_stream, ask_question_trace

        fn = ask_question_trace if st.session_state.trace_mode else ask_question_stream
        answer_text = ""
        sources_list = []
        run_id = ""
        thinking_cleared = False

        for event in fn(API_URL, question, history=history, session_id=session_id):
            etype = event.get("type")

            # 收到首个实质事件后清掉顶部「思考中…」占位
            if not thinking_cleared and etype not in ("run", "status", "done"):
                streaming_placeholder.empty()
                thinking_cleared = True

            # 轨迹事件同步收集，供历史记录 / 轨迹回放使用
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
                }
                icon, label = vmap.get(verdict, ("➡️", verdict))
                detail = label
                if reason:
                    detail += f"\n> {reason}"
                _render_trace_step(trace_container, icon, "反思判定", detail)

            elif etype == "answer":
                answer_text = event.get("content", "")
                if trace_container is not None:
                    # 答案嵌进流程：显示在「信息充足，生成答案」之后、「✅ 完成」之前
                    with trace_container:
                        with st.expander("📝 答案", expanded=True):
                            _aw = st.empty()
                            _typewriter(_aw, answer_text)
                else:
                    _typewriter(streaming_placeholder, answer_text)

            elif etype == "sources":
                sources_list = event.get("sources", []) or []
                result["sources"] = sources_list

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

        if trace_container is not None:
            with trace_container:
                st.markdown("✅ 完成")

        if sources_list:
            render_sources(sources_list)

        full_answer = answer_text
        result["run_id"] = run_id

    except (ImportError, NotImplementedError, httpx.HTTPError) as e:
        st.warning(f"流式调用失败: {e}")
        full_answer = "（流式输出失败，请检查后端）"
        streaming_placeholder.markdown(full_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": result.get("sources", []),
        "trajectory": result.get("trajectory", []),
        "run_id": result.get("run_id", ""),
    })

# frontend/app.py
"""Streamlit 主入口 —— Obsidian RAG 聊天界面。"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from frontend.components import render_sources, render_retrieval_debug

# ─── 页面配置 ───
st.set_page_config(page_title="Obsidian RAG", layout="wide", page_icon="📚")

# ─── 状态 ───
for key, default in [("messages", []), ("debug", False), ("streaming", False), ("trace_mode", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── 侧边栏 ───
with st.sidebar:
    st.markdown("## 🔧 设置")
    API_URL = st.text_input("API 地址", value="http://localhost:8005")
    st.session_state.debug = st.checkbox("调试模式", value=False)
    st.session_state.streaming = st.checkbox("流式输出（SSE）", value=True)
    st.session_state.trace_mode = st.checkbox(
        "追踪模式（显示检索过程）", value=False,
        help="实时展示：Embedding → 稠密检索 → 稀疏检索 → 融合 → Rerank → 图扩展")
    st.markdown("---")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ─── 主标题 ───
st.title("📚 个人知识库问答")
st.caption("基于 Obsidian 笔记库的 RAG 系统，支持混合检索 + 双链扩展")
st.markdown("---")

# ─── 聊天历史 ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])
            if st.session_state.debug and "timing" in msg:
                st.sidebar.json(msg["timing"])

# ─── 输入 ───
question = st.chat_input("问我你的笔记里有什么...")
if not question:
    st.stop()

# ─── 用户消息 ───
st.session_state.messages.append({"role": "user", "content": question})
with st.chat_message("user"):
    st.markdown(question)

# ─── 助手回复 ───
with st.chat_message("assistant"):
    full_answer = ""
    result = {"sources": [], "timing": {}}

    if st.session_state.streaming:
        streaming_placeholder = st.empty()
        try:
            from frontend.utils import ask_question_stream, ask_question_trace

            fn = ask_question_trace if st.session_state.trace_mode else ask_question_stream
            text = ""
            r = {"sources": [], "timing": {}}

            trace_status = None
            trace_container = None
            if st.session_state.trace_mode:
                st.markdown("### 🔍 检索过程")
                trace_container = st.container()
                trace_status = st.status("准备检索...", expanded=True)

            for event in fn(API_URL, question):
                if event["type"] == "char":
                    text += event["content"]
                    streaming_placeholder.markdown(text + "▌")
                elif event["type"] == "sources":
                    streaming_placeholder.markdown(text)
                    if event.get("content"):
                        render_sources(event["content"])
                    r["sources"] = event.get("content", [])
                elif event["type"] == "trace":
                    step = event.get("step", "")
                    ms = event.get("ms", 0)
                    results = event.get("results")
                    preview = event.get("preview", "")
                    icons = {"embedding": "🧠", "dense_retrieval": "🔍", "sparse_retrieval": "📄",
                             "hybrid_fusion": "🔗", "rerank": "⚖️", "graph_expansion": "🕸️"}
                    labels = {"embedding": "向量化", "dense_retrieval": "稠密检索", "sparse_retrieval": "稀疏检索",
                              "hybrid_fusion": "融合排序", "rerank": "重排序", "graph_expansion": "图扩展"}
                    ico = icons.get(step, "➡️")
                    lbl = labels.get(step, step)
                    label = f"{ico} **{lbl}**  ✅ `{ms}ms`"
                    if results is not None:
                        label += f"  ({results} 条结果)"

                    if trace_container:
                        with trace_container:
                            with st.expander(label, expanded=True):
                                if preview:
                                    st.markdown(f"**第一条结果预览：**\n> {preview}")
                                else:
                                    st.caption("无检索结果")

                    if trace_status:
                        trace_status.update(label=label, state="running")

                elif event["type"] == "done":
                    break

            if trace_status:
                trace_status.update(label="✅ 检索完成", state="complete")
            full_answer, result = text, r

        except (ImportError, NotImplementedError) as e:
            st.warning(f"流式调用未实现: {e}")
            full_answer = "（流式输出待实现）"
            streaming_placeholder.markdown(full_answer)

    else:
        with st.spinner("检索中..."):
            try:
                from frontend.utils import ask_question
                result = ask_question(API_URL, question)
                full_answer = result.get("answer", "")
            except (ImportError, NotImplementedError):
                result = {"answer": "（非流式调用待实现）", "sources": [], "timing": {}, "graph_expansion": 0}
                full_answer = result["answer"]

        st.markdown(full_answer)
        if result.get("sources"):
            render_sources(result["sources"])
        if st.session_state.debug:
            render_retrieval_debug(result)
            if result.get("timing"):
                st.sidebar.json(result["timing"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": result.get("sources", []),
        "timing": result.get("timing", {}),
    })
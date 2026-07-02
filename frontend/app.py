# frontend/app.py
"""
Streamlit 主入口 —— Obsidian RAG 聊天界面。

功能：
    - 多轮对话历史
    - 来源折叠展示（按 type 区分渲染）
    - 调试模式（耗时 / 来源分数 / 图扩展）
    - 流式输出（逐字显示）
    - 清空对话
"""

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中（streamlit run 会把脚本所在目录设为 path[0]）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
from frontend.components import render_sources, render_retrieval_debug

# ─── 页面配置（必须在最前面） ───
st.set_page_config(page_title="Obsidian RAG", layout="wide", page_icon="📚")

# ─── 状态初始化 ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug" not in st.session_state:
    st.session_state.debug = False
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# ─── 侧边栏 ───
with st.sidebar:
    st.markdown("## 🔧 设置")
    API_URL = st.text_input("API 地址", value="http://localhost:8003")
    st.session_state.debug = st.checkbox("调试模式", value=False)
    st.session_state.streaming = st.checkbox("流式输出（SSE）", value=True)

    st.markdown("---")
    st.markdown("### ⏱️ 实时状态")
    timing_placeholder = st.empty()

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

# ─── 输入框 ───
question = st.chat_input("问我你的笔记里有什么...")

if question:
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 助手回复
    with st.chat_message("assistant"):
        full_answer = ""
        result = {"sources": [], "timing": {}}

        if st.session_state.streaming:
            # ── 流式路径 ──
            streaming_placeholder = st.empty()
            try:
                from frontend.utils import ask_question_stream
                text = ""
                r = {"sources": [], "timing": {}}
                for event in ask_question_stream(API_URL, question):
                    if event["type"] == "char":
                        text += event["content"]
                        streaming_placeholder.markdown(text + "▌")
                    elif event["type"] == "sources":
                        streaming_placeholder.markdown(text)
                        if event.get("content"):
                            render_sources(event["content"])
                        r["sources"] = event.get("content", [])
                    elif event["type"] == "meta":
                        pass
                    elif event["type"] == "done":
                        break
                full_answer, result = text, r
            except (ImportError, NotImplementedError):
                st.warning("流式调用未实现，请先实现 frontend/utils.py 的 ask_question_stream")
                full_answer = "（流式输出待实现）"
                streaming_placeholder.markdown(full_answer)
        else:
            # ── 非流式路径 ──
            with st.spinner("检索中..."):
                try:
                    from frontend.utils import ask_question
                    result = ask_question(API_URL, question)
                    full_answer = result.get("answer", "")
                except (ImportError, NotImplementedError):
                    result = {
                        "answer": "（非流式调用待实现——请实现 frontend/utils.py 的 ask_question）\n\n需要实现:\n1. frontend/utils.py 中的 ask_question()\n2. 在 app.py 的非流式分支中调用它",
                        "sources": [],
                        "timing": {},
                        "graph_expansion": 0,
                    }
                    full_answer = result["answer"]

            st.markdown(full_answer)

            if result.get("sources"):
                render_sources(result["sources"])

            if st.session_state.debug:
                render_retrieval_debug(result)
                if result.get("timing"):
                    timing_placeholder.json(result["timing"])

        # 存入会话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": result.get("sources", []),
            "timing": result.get("timing", {}),
        })
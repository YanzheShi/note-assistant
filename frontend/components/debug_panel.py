# frontend/components/debug_panel.py
"""
检索过程可视化面板 —— 调试模式下显示在侧边栏。

展示内容：
    - 各阶段耗时（检索 / 重排 / 生成 / 总计）
    - 来源分数条形图
    - 图扩展信息
"""

import streamlit as st
import pandas as pd


def render_retrieval_debug(result: dict):
    """
    完整的检索过程可视化面板。

    仅在 st.session_state.debug == True 时渲染。

    Args:
        result: AskResponse 的 dict 格式（含 timing, sources, graph_expansion）

    布局（侧边栏）：
        - 耗时表（DataFrame）
        - 来源分数条形图（字符画）
        - 图扩展提示
    """
    if not st.session_state.get("debug"):
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 检索过程")

    # ── 耗时 ——
    timing = result.get("timing", {})
    if timing:
        st.sidebar.markdown("### ⏱️ 耗时")
        timing_data = {
            "阶段": ["检索", "重排", "生成", "总计"],
            "耗时(ms)": [
                timing.get("retrieve_ms", "-"),
                timing.get("rerank_ms", "-"),
                timing.get("generate_ms", "-"),
                timing.get("total_ms", "-"),
            ],
        }
        st.sidebar.dataframe(pd.DataFrame(timing_data), hide_index=True)

    # ── 来源分数 ——
    sources = result.get("sources", [])
    if sources:
        st.sidebar.markdown("### 📊 来源分数")
        for i, src in enumerate(sources):
            score = src.get("score", 0)
            filepath = src.get("filepath", "")
            src_type = src.get("type", "text")
            bar = "█" * int(score * 20)
            st.sidebar.text(f"[{i+1}] {src_type} {bar} {score:.3f}")
            st.sidebar.caption(filepath)

    # ── 图扩展 ——
    graph_exp = result.get("graph_expansion", 0)
    if graph_exp > 0:
        st.sidebar.markdown("### 🕸️ 图扩展")
        st.sidebar.text(f"扩展了 {graph_exp} 个关联笔记")

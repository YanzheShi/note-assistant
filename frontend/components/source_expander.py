# frontend/components/source_expander.py
"""
来源折叠组件 —— 将检索到的 sources 按类型区分渲染。

支持四种来源类型：
    text    — 展开显示预览片段（preview）
    table   — 展开显示 preview + raw_table（Markdown 表格）
    mermaid — 展开显示 preview + raw_mermaid（流程图源码或渲染）
    image   — 展开显示图片路径 + 上下文描述（V1 不支持图片显示）
"""

import re
from itertools import count
import streamlit as st

_render_counter = count()


def render_sources(sources: list[dict]):
    """
    来源折叠展示，按类型区分渲染。

    Args:
        sources: 每个元素为 dict，包含 type / filepath / heading / preview / score
               以及可选的 raw_table / raw_mermaid / img_path

    布局：
        - 来源数量统计 + 类型徽章
        - 全部展开/折叠控制
        - 每个来源一个折叠项（expander），按类型渲染内部内容

    面试考点：
        - 为什么用 expander 而不是 checkbox 控制每个来源？
          因为 expander 原生支持 toggle，且不缩放页面
        - type 区分在前端还是后端做？
          后端负责标注 type 字段（预处理阶段已检测到表格/Mermaid），
          前端只负责按 type 渲染——职责分离
    """
    if not sources:
        return

    st.markdown("---")

    # ── 来源统计徽章 ──
    type_counts = {}
    for s in sources:
        t = s.get("type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    badges = " ".join(f"`{t}`×{c}" for t, c in type_counts.items())
    st.markdown(f"**📎 {len(sources)} 个来源** {badges}")

    # ── 全部展开/折叠 ──
    show_all = st.checkbox("全部展开", value=(len(sources) <= 3), key=f"expand_all_{next(_render_counter)}")

    type_icons = {"text": "📝", "table": "📊", "mermaid": "🔀", "image": "🖼️"}

    for i, src in enumerate(sources):
        src_type = src.get("type", "text")
        icon = type_icons.get(src_type, "📄")
        filepath = src.get("filepath", "未知")
        heading = src.get("heading", "")
        score = src.get("score", 0)

        # 标签：图标 + 文件名 > 标题 + 分数
        label = f"{icon} {filepath}"
        if heading:
            label += f" > {heading}"
        if score:
            label += f" `({score:.2f})`"

        with st.expander(label, expanded=show_all or i == 0):
            # ── 预览（所有类型都有） ──
            preview = src.get("preview", "")
            if preview:
                st.markdown(preview)

            # ── 按类型额外渲染 ──
            if src_type == "table" and src.get("raw_table"):
                st.markdown("**表格内容：**")
                st.markdown(src["raw_table"])

            elif src_type == "mermaid" and src.get("raw_mermaid"):
                st.markdown("**Mermaid 图：**")
                # 尝试用 streamlit-mermaid 渲染
                raw = src["raw_mermaid"]
                match = re.search(r"```mermaid\s*\n(.*?)```", raw, re.DOTALL)
                if match:
                    try:
                        from streamlit_mermaid import st_mermaid
                        st_mermaid(match.group(1))
                    except ImportError:
                        st.code(match.group(1), language="mermaid")
                else:
                    st.code(raw, language="markdown")

            elif src_type == "image" and src.get("img_path"):
                img_path = src["img_path"]
                st.markdown(f"**图片路径：** `{img_path}`")
                import os
                if os.path.exists(img_path):
                    st.image(img_path, caption=preview or "参考图片")
                else:
                    st.caption("图片文件不可见（路径不存在或未挂载）")

# frontend/components/source_expander.py
"""
来源折叠组件 —— 将检索到的 sources 按类型区分渲染。

支持四种来源类型：
    text    — 展开显示预览片段（preview）
    table   — 展开显示 preview + raw_table（Markdown 表格）
    mermaid — 展开显示 preview + raw_mermaid（流程图源码或渲染）
    image   — 展开显示图片（P2 起支持；经 /assets 端点 + 后端地址渲染）
"""

import os
import re
from itertools import count
from pathlib import Path

import streamlit as st

_render_counter = count()


def _render_image(img_path: str, caption: str = "", base_url: str = "") -> None:
    """
    渲染来源图片。

    三种形态分别处理：
      - /assets/{id} 相对 URL（P2 统一资产端点）→ 拼后端地址（由 render_sources 透传，
        或回退到 BACKEND_BASE_URL 环境变量）后交给 st.image
      - 远程 URL（本 vault 的主力形态）→ 直接交给 st.image，无需落盘
      - vault 内相对路径 → 拼 VAULT_PATH 再判断存在性。历史实现直接
        `os.path.exists(相对路径)`，是相对 Streamlit 进程 cwd 解析的，必然为 false
      - Obsidian 短名（`Pasted image xxx.png`）→ 需要全库附件索引才能定位，
        属 P1 的 AttachmentIndex 范畴，此处只做友好降级提示
    """
    st.markdown(f"**图片：** `{img_path}`")

    # P2：统一资产端点 /assets/{id}（内容哈希，前端零路径猜测）
    if img_path.startswith("/assets/"):
        base = (base_url or os.environ.get("BACKEND_BASE_URL", "")).rstrip("/")
        if base:
            st.image(base + img_path, caption=caption or "参考图片")
            return
        st.caption(
            "图片无法渲染：未提供后端地址。请在侧边栏填写 API 地址，"
            "或启动 Streamlit 时设置 BACKEND_BASE_URL 环境变量。"
        )
        return

    if img_path.startswith(("http://", "https://", "data:")):
        st.image(img_path, caption=caption or "参考图片")
        return

    candidates = [Path(img_path)]
    vault_root = os.environ.get("VAULT_PATH", "")
    if vault_root:
        candidates.append(Path(vault_root) / img_path)

    for c in candidates:
        try:
            if c.is_file():
                st.image(str(c), caption=caption or "参考图片")
                return
        except OSError:
            continue

    st.caption("图片文件不可见（vault 内短名引用需附件索引解析，见 P1）")


def render_sources(sources: list[dict], backend_base_url: str = ""):
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
        # /ask 链路发 type 字段；/agent 链路（runner._sources_from_results）发 kind 字段。
        # 前端统一兼容两种键名，避免图片来源被误标成 text、徽章统计失真。
        t = s.get("type") or s.get("kind") or "text"
        type_counts[t] = type_counts.get(t, 0) + 1

    badges = " ".join(f"`{t}`×{c}" for t, c in type_counts.items())
    st.markdown(f"**📎 {len(sources)} 个来源** {badges}")

    # ── 全部展开/折叠 ──
    show_all = st.checkbox("全部展开", value=(len(sources) <= 3), key=f"expand_all_{next(_render_counter)}")

    type_icons = {"text": "📝", "table": "📊", "mermaid": "🔀", "image": "🖼️"}

    for i, src in enumerate(sources):
        src_type = src.get("type") or src.get("kind") or "text"
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
            elif src.get("title"):
                # Agentic RAG 来源可能只有 title 没有 preview
                st.markdown(f"**标题：** {src['title']}")

            # ── 按类型额外渲染 ──
            # 注意：这一段必须与 preview 渲染**平级**。历史实现把它缩进进了
            # `elif src.get("title")` 内部，只有「preview 为空且有 title」才可能进入，
            # 事实上是死代码——table/mermaid/image 三个分支从来没有被执行过。
            if src.get("raw_table"):
                st.markdown("**表格内容：**")
                st.markdown(src["raw_table"])

            if src.get("raw_mermaid"):
                st.markdown("**Mermaid 图：**")
                raw = src["raw_mermaid"]
                match = re.search(r"```mermaid\s*\n(.*?)```", raw, re.DOTALL)
                mermaid_src = match.group(1) if match else raw
                # 解析层已确认这是真实提取的 mermaid（render_hint=mermaid:inline），
                # 可直接原生渲染；无 render_hint 时降级为代码展示，避免幻觉。
                if src.get("diagram_type"):
                    st.caption(
                        f"类型：{src['diagram_type']} ｜ "
                        f"render_hint：{src.get('render_hint', 'mermaid:inline')}"
                    )
                try:
                    from streamlit_mermaid import st_mermaid
                    st_mermaid(mermaid_src)
                except ImportError:
                    st.code(mermaid_src, language="mermaid")

            if src.get("img_url") or src.get("img_path"):
                # P2：优先用 /assets 统一 URL，其次退回原始 img_path（远程/本地）
                _render_image(
                    src.get("img_url") or src["img_path"],
                    caption=preview,
                    base_url=backend_base_url,
                )

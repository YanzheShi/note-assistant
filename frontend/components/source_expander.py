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
from urllib.parse import quote as _url_quote

import streamlit as st

# ── .env 加载 ──
# Streamlit 是独立进程，不走后端 pydantic-settings 的 env_file 加载，
# .env 里的 VAULT_PATH 不会自动注入到 os.environ。这里在模块导入时显式
# 加载项目根的 .env + .env.local（与后端加载顺序一致），使 vault_root() 可用。
# 仅当系统未设 VAULT_PATH 时才从 .env 读取，
# 这样用户在终端显式导出的 VAULT_PATH 优先，与后端 pydantic-settings 行为一致。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # frontend/components/.. = 项目根
_env_path = _PROJECT_ROOT / ".env"
_env_local = _PROJECT_ROOT / ".env.local"

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # 无 dotenv 依赖则跳过，保留原行为
    _load_dotenv = None

if _load_dotenv is not None:
    if _env_path.exists():
        _load_dotenv(_env_path)
    if _env_local.exists():
        _load_dotenv(_env_local)

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
    # 注意：局部变量不能命名成 vault_root，否则会遮蔽下面定义的 vault_root() 函数，
    # 在函数作用域内变成"赋值前引用局部变量"，运行时 UnboundLocalError，IDE 也会标红。
    root = vault_root()
    if root:
        candidates.append(Path(root) / img_path)

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
        filepath = src.get("filepath", "")
        heading = src.get("heading", "")
        score = src.get("score", 0)

        # 标签：图标 + 文件名 > 标题 + 分数
        label = f"{icon} {filepath}"
        if heading:
            label += f" > {heading}"
        if score:
            label += f" `({score:.2f})`"

        # 绝对路径与 obsidian:// 链接（点击跳到 Obsidian 或复制路径）
        abs_path = resolve_vault_path(filepath, vault_root=vault_root())
        obsidian_url = obsidian_open_url(filepath, vault_name=vault_name())

        with st.expander(label, expanded=show_all or i == 0):
            # 路径操作栏：复制绝对路径 + 在 Obsidian 中打开
            _render_path_actions(abs_path, obsidian_url, idx=i)
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


# ═══════════════════════════════════════════════════════════════
# 路径工具（obsidian:// 跳转 + 复制绝对路径，纯前端，后端零改动）
# ═══════════════════════════════════════════════════════════════

def vault_root() -> str:
    """
    返回 VAULT_PATH 对应的路径字符串。

    Streamlit 进程在模块导入时已通过 dotenv 加载项目根的 .env + .env.local
    （与后端 pydantic-settings 读取的是同一份配置文件，但加载时机不同）；
    这里与图片路径解析复用同一处来源，不重复配置。
    """
    return os.environ.get("VAULT_PATH", "")


def vault_name() -> str:
    """
    从 VAULT_PATH 末尾取目录名，作为 obsidian:// 协议的 vault 参数。

    约定：Obsidian 注册的 vault 名 == vault_path 最后一层目录名（绝大多数单机
    场景成立）。示例：VAULT_PATH=D:\\Note\\test → vault 名 = "test"。

    Returns:
        vault 名，无法解析时返回空串（obsidian:// 会回退到当前激活 vault）。
    """
    root = vault_root()
    if not root:
        return ""
    return Path(root).name


def resolve_vault_path(rel_path: str, vault_root: str = "") -> str:
    """
    把相对 vault 根的路径解析为绝对路径，便于用户复制后直接打开。

    Args:
        rel_path: SourceSchema.filepath（相对 vault 根的相对路径）
        vault_root: VAULT_PATH 绝对路径

    Returns:
        绝对路径字符串；无 vault_root 或 rel_path 时回退为原始 rel_path。
    """
    if not rel_path:
        return ""
    if not vault_root:
        return rel_path
    joined = Path(vault_root) / rel_path
    # normalize：吃掉 .. / 重复分隔符，避免路径穿越
    try:
        return str(joined.resolve())
    except OSError:
        return str(joined)


def obsidian_open_url(filepath: str, vault_name: str = "") -> str:
    """
    构造 obsidian:// 协议链接，点击后由本地 Obsidian 桌面版接住并打开对应笔记。

    协议格式（Obsidian 官方）：
        obsidian://open?vault=<vault>&file=<path/to/note>

    其中 file 是相对 vault 根的路径，文件名可选（不带 .md 后缀 Obsidian 同样认）。

    Args:
        filepath: 相对 vault 根的笔记路径（含 .md 后缀）
        vault_name: vault 名，为空则省略 vault 参数（回退到当前激活 vault）

    Returns:
        obsidian:// 协议链接字符串；无 filepath 时返回空串。
    """
    if not filepath:
        return ""
    # 去掉两端空格，URL 编码路径中的空格与特殊字符（Obsidian 协议要求）
    encoded_path = _url_quote(filepath, safe="")
    if vault_name:
        encoded_vault = _url_quote(vault_name, safe="")
        return f"obsidian://open?vault={encoded_vault}&file={encoded_path}"
    return f"obsidian://open?file={encoded_path}"


def _render_path_actions(abs_path: str, obsidian_url: str, idx: int = 0) -> None:
    """
    在 expander 顶部渲染路径操作栏：[复制路径] 按钮 + [在 Obsidian 中打开] 链接。

    两个能力互不依赖：
      - 复制路径：即使没装 Obsidian 也有用，复制后手动打开编辑器/资源管理器
      - obsidian://：装了 Obsidian 桌面版的用户点一下就跳过去，最省事

    没有绝对路径信息时（filepath 为空）整个操作栏不渲染，避免空占位。

    ----
    实现演进（踩坑记录）：

    最初用 ``clipboard_component.copy_component``，踩了三个雷：
      1. 它的 ``name`` 参数被前端 React render() 当成 **button children（按钮文字）**
         渲染，不是 streamlit 的组件 key —— 传 ``name="copy_0_1"`` 按钮就显示
         "copy_0_1"。
      2. wrapper 没把 streamlit 标准 ``key`` 参数透传给底层 CustomComponent，
         导致 element ID 只能用 (name, content, disabled) 三元组算，跨消息
         调用 render_sources 时同 abs_path 的按钮 element ID 撞车 →
         StreamlitDuplicateElementId。
      3. 前端 ``handleClick`` 只调 ``navigator.clipboard.writeText`` 然后什么都不做，
         点击后无任何 UI 反馈，用户以为按钮坏了。

    改用 ``st.components.v1.html`` 注入自定义 JS 按钮，完全掌控按钮文字、
    点击反馈、剪贴板写入逻辑。每次调用创建一个独立 iframe（自带唯一 DOM ID），
    天然避免 ID 撞车；同 abs_path 的多个 source 各自渲染独立按钮，互不干扰。
    """
    if not abs_path:
        return

    import json
    import uuid

    import streamlit.components.v1 as components

    # ── 复制路径按钮 ──
    # 唯一 DOM ID，避免 iframe 内 DOM 冲突（components.html 每次创建独立 iframe，
    # 但保险起见仍用 uuid 防止极端情况下的 ID 撞车）。
    btn_id = f"copy_btn_{uuid.uuid4().hex[:8]}"
    # json.dumps 自动正确处理反斜杠、引号、控制字符转义。Windows 路径
    # 如 "D:\Note\test\Python.md" → '"D:\\Note\\test\\Python.md"'，注入 JS 后
    # var text = "..." 能正确还原反斜杠，不会被解释成 \t（tab）等转义序列。
    # 额外把 </ 替换成 <\/ 防止路径含 </script> 时中断脚本标签（防御性）。
    js_text = json.dumps(abs_path).replace("</", "<\\/")

    components.html(f"""
    <style>
        .copy-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.25rem 0.75rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.4rem;
            background: rgb(255, 255, 255);
            color: rgb(49, 51, 63);
            font-size: 0.85rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            cursor: pointer;
            line-height: 1.6;
            transition: all 0.15s ease;
        }}
        .copy-btn:hover {{
            background: rgb(240, 242, 246);
            border-color: rgb(255, 75, 75);
            color: rgb(255, 75, 75);
        }}
        .copy-btn.copied {{
            background: rgb(220, 252, 231);
            color: rgb(22, 101, 52);
            border-color: rgb(134, 239, 172);
        }}
    </style>
    <button class="copy-btn" id="{btn_id}">
        📋 复制路径
    </button>
    <script>
        (function() {{
            var btn = document.getElementById("{btn_id}");
            var text = {js_text};
            function showCopied() {{
                btn.innerText = '✓ 已复制';
                btn.classList.add('copied');
                setTimeout(function() {{
                    btn.innerText = '📋 复制路径';
                    btn.classList.remove('copied');
                }}, 1500);
            }}
            function fallbackCopy() {{
                // navigator.clipboard 不可用（非 secure context，如局域网 IP 访问）
                // 时用 execCommand 降级。已 deprecated 但兼容性最好。
                var ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                try {{ document.execCommand('copy'); }} catch (e) {{}}
                document.body.removeChild(ta);
                showCopied();
            }}
            btn.addEventListener('click', function() {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(showCopied).catch(fallbackCopy);
                }} else {{
                    fallbackCopy();
                }}
            }});
        }})();
    </script>
    """, height=36)

    if obsidian_url:
        # obsidian:// 用 st.markdown 内联链接；浏览器默认把协议识别为可点击
        st.markdown(
            f"[🔗 在 Obsidian 中打开]({obsidian_url})",
        )


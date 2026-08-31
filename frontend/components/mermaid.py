# frontend/components/mermaid.py
"""
前端 mermaid 图渲染 —— 用 Streamlit 的 components.html 注入 mermaid.js 运行时，
在浏览器端把 mermaid 源码渲染成 SVG。

为什么不用 streamlit_mermaid：
    streamlit_mermaid 依赖 mermaid.ink 远程图床 API 把图渲染成图片再拉回来——
    答案里的流程图（可能含内部知识库结构、降级策略等业务信息）会发到第三方，
    有隐私/外泄风险，且必须联网。这里直接注入官方 mermaid.js（CDN 或本地），
    纯前端渲染，不走任何图床。

注入方式（2026-08-31 事故修复，重要）：
    源码必须经 **JS 字符串 + textContent** 注入，绝不能当 HTML 直接塞进
    <div class="mermaid">。老做法的坑：label 里的 ``<id>`` 之类文本会被浏览器
    解析成真 HTML 元素，mermaid 读 element.innerHTML 时被序列化回标签
    （孤立 <id> → <id></id>），源码被污染 → "Syntax error in text"。
    textContent 注入后，innerHTML 序列化把它转成 &lt;id&gt;，mermaid 内部的
    entityDecode 会原样还原，无损。

优雅降级（2026-08-31）：
    库里部分笔记的 mermaid 源码本身非法（边标签未引号括号、sequenceDiagram
    消息含 ``;`` 等，共 6/87 张）。mermaid.initialize 开 suppressErrorRendering
    阻止默认的报错弹窗注入，run().catch 里把原始源码降级显示为代码块。

延迟渲染（2026-08-31，重要）：
    来源面板默认折叠后，图表 div 处于 display:none 容器内，此时 mermaid 的
    dagre 布局量到 getBBox 全为 0 → 渲染出零尺寸空白 SVG（且不会自愈）。
    解法：IntersectionObserver 门控 —— 等 iframe 内图表真正可见时才调
    mermaid.run；展开后首帧触发渲染。不支持 IO 的老浏览器直接立即渲染（兜底）。
"""

import json
import uuid

import streamlit as st
import streamlit.components.v1 as components


# 默认走 jsdelivr CDN（联网即用）。若需完全离线，把 mermaid.min.js 放到
# frontend/static/mermaid.min.js，并把 MERMAID_SRC 改成相对引用或本地 path。
MERMAID_SRC = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


def render_mermaid(code: str, height: int = 800) -> None:
    """把一段 mermaid 源码渲染成流程图（SVG）。

    Args:
        code: mermaid 源码（不含 ```mermaid 围栏，纯图定义）。
        height: iframe 高度（px），默认 800。

    用法：
        在 st.expander / st.container 内直接调用，等价于 st.markdown 的位置。
    """
    if not code or not code.strip():
        return

    chart_id = f"mermaid_{uuid.uuid4().hex[:10]}"
    # 嵌入 JS 字符串字面量：json.dumps 处理引号/换行/反斜杠/非 ASCII；
    # 再把 </ 替换成 <\/ 防 </script> 截断（JS 字符串里 \/ 等价 /，无损）。
    js_src = json.dumps(code.strip()).replace("</", "<\\/")

    html = f"""
    <div id="{chart_id}" class="mermaid"></div>
    <script type="module">
        const SRC = {js_src};
        const el = document.getElementById("{chart_id}");
        // textContent 注入：不经过 HTML 解析，源码里的 <xxx> 原样保留为文本。
        el.textContent = SRC;

        // 解析失败降级：块内显示原始源码（不再弹 mermaid 默认红字报错）
        function fallbackCode(reason) {{
            el.innerHTML = "";
            const cap = document.createElement("div");
            cap.textContent = "⚠️ mermaid 解析失败，降级显示源码（" + reason + "）";
            cap.style.cssText = "color:#993c1d;font-size:12px;margin:0 0 4px;";
            const pre = document.createElement("pre");
            pre.textContent = SRC;
            pre.style.cssText = "background:#f6f6f6;border:1px solid rgba(49,51,63,0.2);"
                + "border-radius:6px;padding:10px;font-size:12px;overflow:auto;"
                + "max-height:520px;margin:0;white-space:pre-wrap;";
            el.appendChild(cap);
            el.appendChild(pre);
        }}

        // 库加载就绪 + 元素可见 两个条件都满足才渲染。
        // 隐藏容器（折叠的 expander）里渲染会因 getBBox 全 0 得到空白 SVG，
        // 且展开后不会自愈 —— 所以必须等真正可见。
        let libReady = false;
        let visible = false;
        let started = false;

        function tryRun() {{
            if (started || !libReady || !visible) return;
            started = true;
            window.mermaid.run({{ querySelector: "#" + "{chart_id}" }}).catch((err) => {{
                fallbackCode(String((err && err.message) || err).slice(0, 120));
            }});
        }}

        function markVisible() {{
            if (visible) return;
            visible = true;
            tryRun();
        }}

        if (typeof IntersectionObserver === "undefined") {{
            markVisible();  // 老浏览器兜底：退回立即渲染
        }} else {{
            const io = new IntersectionObserver((entries) => {{
                if (entries.some((e) => e.isIntersecting)) {{
                    io.disconnect();
                    markVisible();
                }}
            }}, {{ threshold: 0.01 }});
            io.observe(el);
        }}

        if (!window.__mermaidInit) {{
            window.__mermaidInit = true;
            const s = document.createElement("script");
            s.src = "{MERMAID_SRC}";
            s.onload = () => {{
                window.mermaid.initialize({{
                    startOnLoad: false,
                    securityLevel: "loose",
                    htmlLabels: true,
                    theme: "default",
                    suppressErrorRendering: true
                }});
                libReady = true;
                tryRun();
            }};
            s.onerror = () => fallbackCode("mermaid.js 加载失败，请检查网络/CDN");
            document.head.appendChild(s);
        }} else if (window.mermaid) {{
            libReady = true;
            tryRun();
        }}
    </script>
    """
    components.html(html, height=height, scrolling=True)

"""图片检索/生成/展示闭环的共享工具（设计文档 P2）。

把「图意图识别」「image chunk 上下文渲染」「[[IMG:asset_id]] 后处理」抽成纯函数，
供 /ask（Generator + rag_chain）与 /agent（agent.py + runner）两条链路复用，
保证图片在两条路径都能正确进入生成上下文并被渲染成真实图片。

设计要点（对齐 7.2 / 8.1 / 8.3）：
- 图意图识别：纯正则，零 LLM 开销，命中则 image chunk 加权。
- image chunk 上下文：结构化渲染（类型/描述/图中文字）+ [[IMG:asset_id]] 引用标记，
  由 postprocess_answer 在生成后替换为真实 markdown 图片语法。
- 后处理只在标记有对应资产时才替换，绝不凭空造图（防幻觉的核心护栏）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult

# 图意图关键词（设计 7.2）：命中则 image chunk 加权。零 LLM 开销。
_IMAGE_INTENT_RE = re.compile(
    r"(图|图片|截图|架构图|流程图|示意图|图表|画的|图中|示意|配图|"
    r"diagram|chart|screenshot|figure|image)",
    re.IGNORECASE,
)

# [[IMG:asset_id]] 标记（设计 8.3），asset_id 为 sha256[:16] 十六进制。
# 2026-09-01 放宽为「任意非空非 ] 内容」：LLM 会照抄教学文本里的字面
# [[IMG:asset_id]] 占位符（"asset_id" 非 hex），旧正则匹配不到 → 裸标记暴露
# 给用户。放宽后由 postprocess_answer 统一裁决：有资产替换、无资产删除。
_IMG_REF_RE = re.compile(r"\[\[IMG:([^\]\s]+)\]\]")


def has_image_intent(query: str) -> bool:
    """query 是否表现出对图片/图表的意图（决定 image chunk 是否加权）。"""
    return bool(_IMAGE_INTENT_RE.search(query or ""))


def ensure_image_selected(
    question: str,
    ranked_all: List[RetrievalResult],
    selected: List[RetrievalResult],
) -> List[RetrievalResult]:
    """rerank 截断的图示保位：图意图 query 的 top-k 里没有 image/mermaid chunk 时，
    从全量精排结果里把分数最高的可渲染图示 chunk 补进 selected（替换末位）。

    背景：hybrid 融合阶段的图意图 boost 会被后续交叉编码器 rerank 清零——
    rerank 纯按文本相关性重排，图片摘要块与 mermaid 代码块（代码文本 vs 自然语言
    query，交叉编码器打分天然低）常被长正文挤出 top-k，于是「检索明明命中了图，
    生成上下文里却没有图」。本函数是确定性护栏：只要 query 有图意图且候选里存在
    image / mermaid chunk，就保证 selected 里至少有一个（2026-09-01 从 image-only
    泛化到 mermaid——两者是同一症状：来源面板有图、答案正文没图）。

    - 非图意图 query / selected 已有可渲染图示 chunk / 候选无 → 原样返回。
    - ranked_all 须为同一次 rerank 的全量降序结果（selected 是其前缀截断），
      保位取的正是图示 chunk 的真实 rerank 分，不引入额外打分口径。
    - 替换 selected 末位（分数最低者）后重排，保持结果按分降序、长度不变。
    """
    if not has_image_intent(question or ""):
        return selected
    if any(_is_visual_chunk(r) for r in selected):
        return selected
    best = next((r for r in ranked_all if _is_visual_chunk(r)), None)
    if best is None:
        return selected
    if not selected:
        return [best]
    out = list(selected[:-1]) + [best]
    out.sort(key=lambda r: r.score, reverse=True)
    return out


def _is_image_chunk(item: RetrievalResult) -> bool:
    """根据 metadata 判断该 chunk 是否为「图片理解 chunk」。"""
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    if str(meta.get("kind")) == "image":
        return True
    # 没有明确 kind 时，靠 asset_id + (description 或 ocr) 组合判断
    return bool(meta.get("asset_id")) and (
        bool(meta.get("image_description")) or bool(meta.get("image_ocr_text"))
    )


_MERMAID_FENCE = "```mermaid"


def _is_mermaid_chunk(item: RetrievalResult) -> bool:
    """是否为 mermaid 图示 chunk：metadata kind 标记，或正文含 ```mermaid 围栏。"""
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    if str(meta.get("kind")) == "mermaid":
        return True
    return _MERMAID_FENCE in (getattr(item, "page_content", "") or "")


def _is_visual_chunk(item: RetrievalResult) -> bool:
    """是否为「可渲染图示 chunk」：图片理解块或 mermaid 块（保位护栏的判定口径）。"""
    return _is_image_chunk(item) or _is_mermaid_chunk(item)


def _chunk_key(result: RetrievalResult):
    """chunk 身份标识：优先 chunk_id，退化为 (filepath, heading_path, 正文前 64 字)。"""
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    cid = meta.get("chunk_id") or meta.get("id")
    if cid:
        return ("id", str(cid))
    return (
        "raw",
        meta.get("filepath"),
        meta.get("heading_path"),
        (getattr(result, "page_content", "") or "")[:64],
    )


def expand_image_neighbors(
    results: List[RetrievalResult],
    fetch_fn,
    *,
    budget: int = 6,
) -> List[RetrievalResult]:
    """命中 image chunk 时带出同 heading_path 的文本邻居（设计 7.3）。

    与 graph_expansion 并列、走同一套预算控制；邻居只进生成上下文，不进 sources/Judge 证据。
    DB 抓取通过 ``fetch_fn(heading_paths)`` 注入，便于离线测试（默认走 ChromaDB）。

    图片 chunk 判定用 :func:`_is_image_chunk`：
    先认 ``kind=="image"``（本仓库真实 VLM 图 chunk 经 classify_source 会带上该标记，
    旧判定 ``str(meta.get("kind")) == "image"`` 在生产环境能命中），再退化为
    ``asset_id`` + (``image_description``/``image_ocr_text``) 组合。后者是更鲁棒的兜底，
    可覆盖任何因某种原因未写入 ``kind`` 的图片富化 chunk；二者在本仓库现有数据上结果一致。
    """
    if not settings.image_neighbor_expand:
        return []
    image_headings: List[str] = []
    for r in results:
        if _is_image_chunk(r):
            hp = (r.metadata or {}).get("heading_path", "")
            if hp and hp not in image_headings:
                image_headings.append(hp)
    if not image_headings:
        return []
    candidates = fetch_fn(image_headings)
    target_headings = set(image_headings)
    # 去重必须按 chunk 身份，不能按 (filepath, heading_path)——
    # 图片 chunk 本身就占着目标 heading，按章节去重会把想要的正文邻居全过滤掉。
    seen = {_chunk_key(r) for r in results}
    added: List[RetrievalResult] = []
    for c in candidates:
        if _is_image_chunk(c):
            continue  # 只带文本邻居，避免自环
        if (c.metadata or {}).get("heading_path") not in target_headings:
            continue  # 防御：fetch_fn 可能返回越界结果
        key = _chunk_key(c)
        if key in seen:
            continue
        seen.add(key)
        added.append(c)
        if len(added) >= budget:
            break
    return added


def render_image_block(item: RetrievalResult) -> Optional[str]:
    """为 image chunk 生成专用上下文块；非 image chunk 返回 None（调用方用默认渲染）。

    返回的块内含 [[IMG:asset_id]] 标记，由 postprocess_answer 在生成后替换为真实 URL。
    调用方负责在块前拼 ``### [i] {title}`` 头，本函数只返回图片正文行。

    无理解内容兜底：VLM 字段（desc/ocr）与资产全缺时返回 None，让调用方退回
    page_content（至少含「图片: src (上下文)」摘要）——优于渲染一个空白
    「【图片·图】插图」块，那会让 LLM 既拿不到图信息、又看不到原始摘要。
    """
    if not _is_image_chunk(item):
        return None
    meta = item.metadata if isinstance(item.metadata, dict) else {}
    image_type = meta.get("image_type") or "图"
    title = meta.get("image_title") or meta.get("title") or "插图"
    desc = meta.get("image_description") or ""
    ocr = meta.get("image_ocr_text") or ""
    asset_id = meta.get("asset_id") or ""

    if not desc and not ocr and not asset_id:
        return None  # 空块无信息量，回退 page_content 渲染

    lines = [f"【图片·{image_type}】{title}"]
    if desc:
        lines.append(f"图中内容：{desc}")
    if ocr:
        lines.append(f"图中文字：{ocr}")
    if asset_id:
        lines.append(f"引用方式：如需引用此图，在回答中写 [[IMG:{asset_id}]]")
    return "\n".join(lines)


def collect_image_assets(chunks: List[RetrievalResult]) -> Dict[str, dict]:
    """从检索结果里收集 asset_id → {title, img_url} 映射，供后处理替换标记。"""
    out: Dict[str, dict] = {}
    for item in chunks or []:
        meta = item.metadata if isinstance(item.metadata, dict) else {}
        asset_id = meta.get("asset_id")
        img_url = meta.get("img_url")
        if asset_id and img_url and asset_id not in out:
            out[asset_id] = {
                "title": meta.get("image_title") or meta.get("title") or "插图",
                "img_url": img_url,
            }
    return out


def has_teachable_image_assets(chunks: List[RetrievalResult]) -> bool:
    """context 中是否存在「标记可被真实替换」的图片资产（asset_id + img_url 双全）。

    决定 system prompt 是否教 ``[[IMG:asset_id]]`` 语法（2026-08-31）：教学是
    软引导，LLM 会照着格式**编** id——context 里没有可解析资产时教学只会
    诱导幻觉标记（26/30 图 chunk 无 asset_id 的教训）。所以只在 collect_image_assets
    非空（即后处理真的能替换）时才教。
    """
    return bool(collect_image_assets(chunks or []))


def has_mermaid_chunks(chunks: List[RetrievalResult]) -> bool:
    """context 中是否存在 mermaid 图示 chunk（决定是否教 LLM 复现 mermaid）。

    与 ``has_teachable_image_assets`` 同一设计：教学是软引导，条件化追加避免
    对没有 mermaid 的上下文产生诱导（LLM 凭空画图 = 幻觉）。
    """
    return any(_is_mermaid_chunk(r) for r in (chunks or []))


def postprocess_answer(answer: str, chunks: List[RetrievalResult]) -> str:
    """把答案里的 [[IMG:asset_id]] 替换为 ![title](img_url)（设计 8.3）。

    标记无对应资产时**直接删除**（2026-08-31 修订）：此前「原样保留」的策略
    在 LLM 编造 asset_id / 资产未注册时给用户留下 [[IMG:xxx]] 裸标记噪音。
    现在绝不凭空造图（护栏不变），但也不保留无法解析的标记。
    """
    if not answer:
        return answer
    assets = collect_image_assets(chunks)

    def _repl(m: re.Match) -> str:
        info = assets.get(m.group(1))
        if not info:
            return ""  # 无对应资产：删除标记，不幻觉也不留噪音
        return f"![{info['title']}]({info['img_url']})"

    return _IMG_REF_RE.sub(_repl, answer)


# ── 确定性补图（用户偏好：有图且相关就尽量显示）────────────────────
# 图片是否显示不能全押在 LLM 自觉输出 [[IMG:]] 标记上（prompt 只是软引导）。
# 这里的护栏是：进入生成上下文的图片都是检索+精排筛选过的，视为「相关」；
# 若答案最终没有引用其中某些图（既无标记替换结果、也无手写 markdown），
# 在答案末尾确定性补上「相关图片」区块。绝不补 context 之外的图（不幻觉）。

# 单次最多自动补几张（context 里图片很多时防刷屏；检索 top-k 下通常 1~2 张）
MAX_AUTO_IMAGES = 4


def missing_images_block(answer: str, chunks: List[RetrievalResult]) -> str:
    """返回需要补在答案末尾的图片区块；无缺失时返回空串。

    「已引用」判定：asset 的 img_url 已出现在答案中——postprocess 替换出的
    ``![t](/assets/id)``、LLM 手写的 markdown 图片都会被识别，避免重复展示。
    只认 collect_image_assets 收到的真实资产（有 img_url），绝不凭空造图。
    """
    assets = collect_image_assets(chunks)
    if not assets:
        return ""
    text = answer or ""
    missing = [info for aid, info in assets.items() if info["img_url"] not in text]
    if not missing:
        return ""
    lines = ["\n\n---\n**相关图片：**"]
    for info in missing[:MAX_AUTO_IMAGES]:
        lines.append(f"\n![{info['title']}]({info['img_url']})")
    return "\n".join(lines) + "\n"


def append_missing_images(answer: str, chunks: List[RetrievalResult]) -> str:
    """答案里未引用的 context 图片确定性补到末尾（在 postprocess_answer 之后调用）。"""
    return (answer or "") + missing_images_block(answer or "", chunks)


# ── 确定性补 mermaid（2026-09-01，对称图片的确定性补齐）──────────────
# 与 [[IMG:]] 标记不同，mermaid 的「复现」由 LLM 直接把 ```mermaid 代码块写进
# 答案（无标记替换层），prompt 只是软引导——问「处理流程」这类不带图意图词的
# query 时 LLM 常常用文字概括而不复现代码块（实测：上下文含 mermaid 排第 2、
# 答案却无图）。护栏：进入生成上下文的 mermaid chunk 都是检索+精排筛过的，
# 视为「相关」；答案完全没有 mermaid 块时补分数最高的一张，已有块则不补
# （避免多图刷屏；LLM 已复现说明软引导生效）。
MAX_AUTO_MERMAID = 1

# 抽取 chunk 正文里的第一个完整 ```mermaid ... ``` 围栏块（chunk 可能含围栏外
# 的说明文字，不能整段照搬）
_MERMAID_BLOCK_RE = re.compile(r"(```mermaid[\s\S]*?```)", re.IGNORECASE)


def missing_mermaid_block(answer: str, chunks: List[RetrievalResult]) -> str:
    """返回需要补在答案末尾的 mermaid 区块；无缺失时返回空串。"""
    text = answer or ""
    if _MERMAID_FENCE in text:
        return ""  # 答案已有 mermaid 块：LLM 已复现，不重复补
    diagrams = [r for r in (chunks or []) if _is_mermaid_chunk(r)]
    if not diagrams:
        return ""
    best = max(diagrams, key=lambda r: r.score)
    m = _MERMAID_BLOCK_RE.search(getattr(best, "page_content", "") or "")
    if not m:
        return ""
    meta = best.metadata if isinstance(best.metadata, dict) else {}
    title = meta.get("title", "") or ""
    lines = ["\n\n---\n**相关流程图：**"]
    if title:
        lines.append(f"（来源：{title}）\n")
    lines.append(m.group(1))
    return "\n".join(lines) + "\n"


def append_missing_mermaid(answer: str, chunks: List[RetrievalResult]) -> str:
    """答案完全没有 mermaid 块时，把 context 里分数最高的 mermaid 补到末尾。"""
    return (answer or "") + missing_mermaid_block(answer or "", chunks)


def finalize_answer_images(answer: str, chunks: List[RetrievalResult]) -> str:
    """一步到位的生成后图示闭环：标记替换 + 未引用图片补齐 + 缺失 mermaid 补齐。"""
    out = append_missing_images(postprocess_answer(answer or "", chunks), chunks)
    return append_missing_mermaid(out, chunks)


# ── 流式后处理 ────────────────────────────────────────────────
# 流式场景不能等答案生成完再替换（那样就没有流式了），也不能逐 token 直接替换
# （[[IMG:abc]] 会被切成多个 token，跨边界匹配不到）。
# 解法：只在「可能正处于标记中」时缓冲，标记闭合即替换后吐出，其余照常实时流出。

_MARKER_OPEN = "[[IMG:"


class ImageMarkerStreamer:
    """标记感知的增量流式器：保留流式体验的同时正确替换 [[IMG:asset_id]]。

    用法::

        streamer = ImageMarkerStreamer(chunks)
        for token in stream:
            piece = streamer.feed(token)
            if piece:
                yield piece
        tail = streamer.flush()
        if tail:
            yield tail
    """

    def __init__(self, chunks: List[RetrievalResult]):
        self._assets = collect_image_assets(chunks)
        self._buf = ""

    @staticmethod
    def _partial_open_len(text: str) -> int:
        """text 末尾有多少字符可能是 ``[[IMG:`` 的不完整前缀（需要继续缓冲）。"""
        for n in range(len(_MARKER_OPEN) - 1, 0, -1):
            if text.endswith(_MARKER_OPEN[:n]):
                return n
        return 0

    def _replace(self, marker: str) -> str:
        m = _IMG_REF_RE.fullmatch(marker)
        if not m:
            return marker
        info = self._assets.get(m.group(1))
        if not info:
            return ""  # 无对应资产：直接删除，不幻觉也不留噪音
        return f"![{info['title']}]({info['img_url']})"

    def feed(self, token: str) -> str:
        """喂入一个 token，返回此刻可以安全吐出的文本（可能为空字符串）。"""
        if not token:
            return ""
        # 注意：即使没有可替换资产也不能透传 —— 未解析标记要在这里被剥离
        self._buf += token
        out: List[str] = []
        while True:
            start = self._buf.find(_MARKER_OPEN)
            if start == -1:
                hold = self._partial_open_len(self._buf)
                cut = len(self._buf) - hold
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            out.append(self._buf[:start])
            self._buf = self._buf[start:]
            end = self._buf.find("]]")
            if end == -1:
                break  # 标记未闭合，继续缓冲等后续 token
            out.append(self._replace(self._buf[: end + 2]))
            self._buf = self._buf[end + 2 :]
        return "".join(out)

    def flush(self) -> str:
        """流结束后吐出残留缓冲；其中被截断的未闭合标记直接删除（不留噪音）。

        feed 的缓冲区只可能残留「从 ``[[IMG:`` 开始的未闭合标记及其后文本」，
        所以锚定开头剥离截断的标记片段，其余文本原样保留。
        """
        rest, self._buf = self._buf, ""
        if rest.startswith(_MARKER_OPEN):
            # 与 _IMG_REF_RE 同口径（2026-09-01 放宽）：任意非空非 ] 内容均按标记
            # 处理，被截断的未闭合标记直接删除（不留噪音）
            m = re.match(r"\[\[IMG:[^\]\s]{0,64}(?:\]\])?", rest)
            if m:
                rest = rest[m.end():]
        return rest

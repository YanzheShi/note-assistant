"""上下文管理模块（ContextManager）：凝练 / 预算 / 累积 / 压缩 / 裁剪 / 长程摘要 / 缓存指纹。

单文件模块，不反向依赖 ``agent.py`` / ``runner.py``（避免循环引用）。``runner`` 与
各 agent 节点从本模块导入 ``get_context_manager()`` 单例。

所有外部依赖（LLM、embedding）均注入 + 失败降级，沿用项目防御式风格：
    - embedding 不可用 → 相关性裁剪降级为时间窗口截断；
    - LLM 不可用 / 调用异常 → 凝练降级为原问题、摘要降级跳过；
    - token 计数依赖 tiktoken，加载失败则回退为「长度 / 2」的粗近似。

术语对齐（见设计评审细化）：
    - **总预算硬上限** ``agent_total_context_token_budget``：历史 + 累积 + 观察三段之和不可超过；
      若三段之和超总预算，按 ``obs → accumulated → history`` 优先级压缩（history 最后才牺牲）。
    - **跨轮累积双重保险**：① 每跨一轮每个片段 ``score *= agent_accumulated_decay``；
      ② 按 token 预算硬截断，只保留有效分最高的若干条。
    - **长程摘要触发**：用原文 user/assistant 轮次的 **token 总和** 是否超过
      ``agent_session_summary_threshold`` 判定（非轮次数）。
    - **相关性裁剪**：先取「最近 N 轮」时间窗口，再在窗口内按与凝练问题的 embedding
      相似度 ``>= agent_history_relevance_threshold`` 裁剪；窗口外轮次不进候选，
      避免「全高分=没裁剪」。相关性关闭 / 不可用 → 纯时间窗口截断。
"""
from __future__ import annotations

import copy
import hashlib
import logging
import math
from typing import Callable, List, Optional, Sequence

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)

# 注入的 embed 函数签名： (text: str) -> list[float]
EmbedFn = Callable[[str], List[float]]


# ──────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _message_text(m) -> str:
    """把 LangChain 消息 / dict / 纯文本统一成可计数字符串。"""
    if isinstance(m, BaseMessage):
        c = m.content
        return c if isinstance(c, str) else (str(c) if c is not None else "")
    if isinstance(m, dict):
        return m.get("content", "") or ""
    return str(m)


def _strip_resp(resp) -> str:
    if hasattr(resp, "content"):
        c = resp.content
        return c if isinstance(c, str) else str(c)
    return str(resp)


def _default_llm():
    """延迟获取默认 LLM（凝练 / 摘要共用）；失败返回 None。"""
    try:
        from note_assistant.llm.client import get_llm

        return get_llm()
    except Exception:  # noqa: BLE001
        return None


# ──────────────────────────────────────────────
# TokenCounter：tiktoken 精确计数（编码单例缓存）
# ──────────────────────────────────────────────

class TokenCounter:
    _enc = None
    _tried = False

    @classmethod
    def encoding(cls):
        if not cls._tried:
            cls._tried = True
            try:
                cls._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                cls._enc = None
        return cls._enc

    @classmethod
    def count(cls, text_or_messages) -> int:
        enc = cls.encoding()
        if isinstance(text_or_messages, str):
            if enc is None:
                return max(1, len(text_or_messages) // 2)  # 粗近似降级
            return len(enc.encode(text_or_messages))
        total = 0
        for m in text_or_messages:
            total += cls.count(_message_text(m))
        return total


# ──────────────────────────────────────────────
# ContextManager
# ──────────────────────────────────────────────

class ContextManager:
    """统一承担上下文的凝练、预算、累积、压缩、裁剪、长程摘要、缓存指纹。"""

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        condense_llm=None,
        summarize_llm=None,
    ):
        self.embed_fn = embed_fn
        self._condense_llm = condense_llm
        self._summarize_llm = summarize_llm
        # 跨轮累积：按 session_id 缓存 RetrievalResult（内存，重启自然降级）
        self._accum: dict[str, list[RetrievalResult]] = {}

    # ── token 计数 ──
    def count_tokens(self, text_or_messages) -> int:
        return TokenCounter.count(text_or_messages)

    # ── 问题凝练（消指代）──
    async def condense_question(self, current: str, history: Sequence[dict]) -> str:
        """基于历史把追问改写为独立完整问题，供路由/检索/缓存指纹使用。

        失败或开关关闭时降级返回原问题（不抛异常）。
        """
        if not settings.agent_condense_enabled or not current:
            return current
        llm = self._condense_llm or _default_llm()
        if llm is None:
            return current
        try:
            recent = history[-settings.agent_history_relevance_window:]
            hist_text = _format_turns_for_prompt(recent)
            prompt = (
                "你是一个对话改写器。下面是一段多轮对话的历史与用户最新追问。\n"
                "请把「最新追问」改写成一个独立、完整、不依赖上下文就能理解的问题，"
                "消解「它/那/这个/前者」等指代；若追问本身已完整则原样返回。\n"
                "只输出改写后的问题，不要解释。\n\n"
                f"=== 对话历史 ===\n{hist_text}\n\n"
                f"=== 最新追问 ===\n{current}"
            )
            resp = await llm.ainvoke([HumanMessage(prompt)])
            text = _strip_resp(resp).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("condense_question 失败，降级用原问题: %s", e)
            logger.warning("condense.question_failed", extra={"error": str(e)[:120]})
            return current
        logger.info("condense.question", extra={"original": current[:40], "condensed": text[:40]})
        return text or current

    # ── 缓存指纹 ──
    def context_key(self, condensed: str, summary: str = "") -> str:
        """上下文指纹：凝练问题 + 长程摘要 hash。不同上下文 → 不同 key，防串台。"""
        raw = f"{condensed}|{summary}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # ── 历史预算 + 相关性裁剪 ──
    def budget_history_messages(
        self,
        history: Sequence[dict],
        condensed: str,
        token_budget: int,
        summary: Optional[str] = None,
    ) -> List[BaseMessage]:
        """返回预算后、按相关性裁剪过的历史消息列表（供 agent_node / direct_chat）。

        处理顺序：① 时间窗口（最近 N 轮）→ ② 相关性裁剪（窗口内相似度阈值）
        → ③ 组装消息（摘要作 SystemMessage 前置）→ ④ token 预算从最旧向前删。
        """
        turns = list(history)

        # ① 时间窗口
        window = turns
        if 0 < settings.agent_history_relevance_window < len(turns):
            window = turns[-settings.agent_history_relevance_window:]

        # ② 相关性裁剪（可选）
        if settings.agent_history_relevance_enabled and self.embed_fn is not None and condensed:
            try:
                q_emb = self.embed_fn(condensed)
                sims: List[float] = []
                for t in window:
                    t_emb = self.embed_fn(t.get("content", "") or "")
                    sims.append(_cosine(q_emb, t_emb))
                keep = [t for i, t in enumerate(window) if sims[i] >= settings.agent_history_relevance_threshold]
                if keep:
                    window = keep
            except Exception as e:  # noqa: BLE001
                logger.warning("相关性裁剪失败，降级时间窗口: %s", e)

        # ③ 组装消息
        prefix: List[BaseMessage] = []
        if summary:
            prefix.append(
                SystemMessage(content=f"以下是对话的长程摘要（早期上下文）：\n{summary}")
            )
        hist_msgs: List[BaseMessage] = []
        for t in window:
            role = t.get("role")
            content = t.get("content", "") or ""
            if not content:
                continue
            if role == "assistant":
                hist_msgs.append(AIMessage(content=content))
            else:
                hist_msgs.append(HumanMessage(content=content))

        # ④ token 预算：优先删历史，保住摘要
        all_msgs = prefix + hist_msgs
        while self.count_tokens(all_msgs) > token_budget and len(all_msgs) > 1:
            if hist_msgs:
                hist_msgs.pop(0)
            elif prefix:
                prefix.pop(0)
            all_msgs = prefix + hist_msgs
        return all_msgs

    # ── 跨轮知识累积 ──
    def seed_accumulated(self, session_id: str) -> List[RetrievalResult]:
        """取出上一轮累积的笔记片段，并对每条施加一轮衰减（双重保险之一）。

        seed 时就地衰减并写回，使跨轮自然复利衰减；本轮新检索的分数不衰减。
        """
        lst = self._accum.get(session_id, [])
        decayed: List[RetrievalResult] = []
        for r in lst:
            nr = copy.copy(r)
            nr.score = r.score * settings.agent_accumulated_decay
            decayed.append(nr)
        self._accum[session_id] = decayed
        logger.info("context.seed", extra={"session_id": session_id, "items": len(decayed)})
        return list(decayed)

    def record_turn(
        self,
        session_id: str,
        accumulated: Sequence[RetrievalResult],
        user_q: str,
        assistant_a: str,
    ) -> None:
        """运行结束后合并跨轮累积：确定性去重 + token 预算硬截断（双重保险之二）。

        ``accumulated`` 通常 = 本轮 seed（已衰减）+ 本轮新检索到的结果。
        """
        seen: dict[tuple, RetrievalResult] = {}
        merged: List[RetrievalResult] = []
        for r in accumulated:
            key = (r.filepath, r.metadata.get("heading_path", ""))
            if key in seen:
                if r.score > seen[key].score:
                    seen[key] = r
            else:
                seen[key] = r
                merged.append(r)

        # token 预算硬截断：按 score 降序贪心塞入
        merged.sort(key=lambda x: x.score, reverse=True)
        kept: List[RetrievalResult] = []
        used = 0
        for r in merged:
            t = self.count_tokens(r.page_content)
            if used + t > settings.agent_accumulated_token_budget and kept:
                break
            kept.append(r)
            used += t
            if len(kept) >= settings.agent_accumulated_max_items:
                break
        self._accum[session_id] = kept

    # ── 总预算兜底：已知两段（history + accumulated）之和超总上限时压缩 ──
    def fit_total_budget(
        self,
        history_msgs: Sequence[BaseMessage],
        accumulated: Sequence[RetrievalResult],
    ) -> tuple[List[BaseMessage], List[RetrievalResult]]:
        """总预算硬上限兜底（设计：obs → accumulated → history 逐级压缩）。

        obs 段由 ``tools_node`` 按 ``agent_obs_token_budget`` 独立截断，已是第一压缩位；
        此处对已确定的 history + accumulated 两段做总预算仲裁：超上限时先裁
        accumulated（低优先级），仍超再裁 history（高优先级，最后才牺牲）。
        """
        budget = settings.agent_total_context_token_budget
        hist_tok = self.count_tokens(history_msgs)
        acc_tok = self.count_tokens([r.page_content for r in accumulated])
        if hist_tok + acc_tok <= budget:
            return list(history_msgs), list(accumulated)

        # ① 裁 accumulated：按 score 降序贪心保留
        acc = sorted(accumulated, key=lambda r: r.score, reverse=True)
        acc_tok = self.count_tokens([r.page_content for r in acc])
        while acc and hist_tok + acc_tok > budget:
            acc.pop()
            acc_tok = self.count_tokens([r.page_content for r in acc])

        # ② 仍超则裁 history：从最旧向前删（history 最后才牺牲）
        hm = list(history_msgs)
        while hm and hist_tok + acc_tok > budget and len(hm) > 1:
            hm.pop(0)
            hist_tok = self.count_tokens(hm)
        return hm, list(acc)

    # ── 工具观察文本 token 截断 ──
    def truncate_observation(self, text: str, token_budget: int) -> str:
        if not text:
            return text
        n = self.count_tokens(text)
        if n <= token_budget:
            return text
        # 估算字符比例截断，再校准
        ratio = max(0.1, token_budget / n)
        cut = max(1, int(len(text) * ratio * 0.95))
        truncated = text[:cut]
        guard = 0
        while self.count_tokens(truncated) > token_budget and guard < 5:
            truncated = truncated[: int(len(truncated) * 0.9)]
            guard += 1
        return truncated + "\n…（观察内容已按 token 预算截断）"

    # ── 长程记忆：滚动摘要（异步触发）──
    async def maybe_summarize(self, session_id: str, store) -> None:
        """达到 token 阈值时，对最旧未摘要轮次做 LLM 滚动摘要并清理原文。

        失败 / 开关关闭 → 静默降级跳过，绝不拖垮主链路。
        """
        if not settings.agent_summary_enabled or store is None:
            return
        try:
            all_turns = store.get_all_turns(session_id)
            if not all_turns:
                return
            token_sum = self.count_tokens("\n".join(t["content"] for t in all_turns))
            if token_sum < settings.agent_session_summary_threshold:
                return

            latest = store.get_latest_summary(session_id)
            from_idx = (latest["idx_to"] + 1) if latest else 1
            max_idx = all_turns[-1]["idx"]
            to_idx = max_idx - settings.agent_session_recent_keep
            if to_idx < from_idx:
                return  # 最近轮次规模还不够，暂不摘要

            batch = store.get_turns_in_range(session_id, from_idx, to_idx)
            if not batch:
                return
            summary_text = await self._summarize_batch(
                batch, latest["summary"] if latest else None
            )
            if not summary_text:
                return
            store.save_summary(session_id, from_idx, to_idx, summary_text)
            store.delete_turns_up_to(session_id, to_idx)
            store.enforce_session_cap(session_id, settings.agent_session_max_turns)
            logger.info("context.summarize", extra={"session_id": session_id, "from_idx": from_idx, "to_idx": to_idx, "summary_len": len(summary_text)})
        except Exception as e:  # noqa: BLE001
            logger.warning("maybe_summarize 失败，降级跳过: %s", e)
            logger.warning("context.summarize_failed", extra={"error": str(e)[:120]})

    async def _summarize_batch(
        self, turns: Sequence[dict], prev_summary: Optional[str]
    ) -> Optional[str]:
        llm = self._summarize_llm or _default_llm()
        if llm is None:
            return None
        text = _format_turns_for_prompt(turns)
        prompt = (
            "你是一个对话摘要器。请把下面的对话记录压缩为简洁的要点摘要，"
            "保留事实、决策、用户偏好与未解决问题，去除寒暄与重复。\n"
        )
        if prev_summary:
            prompt += f"=== 已有摘要 ===\n{prev_summary}\n\n"
        prompt += f"=== 待摘要对话 ===\n{text}"
        try:
            resp = await llm.ainvoke([HumanMessage(prompt)])
            return _strip_resp(resp).strip() or None
        except Exception as e:  # noqa: BLE001
            logger.warning("_summarize_batch 失败: %s", e)
            return None


def _format_turns_for_prompt(turns: Sequence[dict]) -> str:
    lines = []
    for t in turns:
        role = "用户" if t.get("role") == "user" else "助手"
        lines.append(f"{role}: {t.get('content', '')}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 单例
# ──────────────────────────────────────────────

_cm_singleton: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """懒加载单例：默认尝试注入 OllamaEmbedder 做相关性裁剪，失败则降级时间截断。"""
    global _cm_singleton
    if _cm_singleton is None:
        embed_fn: Optional[EmbedFn] = None
        if settings.agent_history_relevance_enabled:
            try:
                from note_assistant.indexing.embedder import OllamaEmbedder

                embed_fn = OllamaEmbedder().embed_one
            except Exception as e:  # noqa: BLE001
                logger.info("embedding 不可用，相关性裁剪降级时间窗口: %s", e)
        _cm_singleton = ContextManager(embed_fn=embed_fn)
    return _cm_singleton


def set_context_manager_for_test(cm: Optional[ContextManager]) -> None:
    """测试注入自定义 ContextManager（含全离线 mock）。"""
    global _cm_singleton
    _cm_singleton = cm

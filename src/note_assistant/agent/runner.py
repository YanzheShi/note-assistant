"""Agent 运行器：把自写 StateGraph 封装成非流式/流式入口，并接入语义缓存（P7b）。

- ``ainvoke``：跑完整图，返回答案 + 去重来源 + 完整轨迹（命中缓存则直接返回）。
- ``astream``：逐项产出轨迹事件（thought / tool_call / observation / judge / answer / sources），
  同样命中缓存时先回放缓存轨迹。
- 缓存：精确 + 近邻（注入式 embedder），外部异常自动降级，绝不拖垮主链路。
"""
import asyncio
import logging
from dataclasses import dataclass, field
import time
import uuid
from typing import List, Optional

from note_assistant.logger_util import set_request_id, get_request_id
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from note_assistant.agent import agent as agent_mod
from note_assistant.agent.cache import SemanticCache
from note_assistant.agent.context import CondenseSignal, get_context_manager
from note_assistant.agent.store import AgentStore
from note_assistant.config import settings
from note_assistant.pipeline.image_answer import append_missing_images, postprocess_answer
from note_assistant.security.output_guard import check_prompt_leakage, neutralize_remote_media
from note_assistant.retrieval.types import RetrievalResult
from note_assistant.llm.usage import get_token_handler


logger = logging.getLogger(__name__)



OBS_TRUNCATE = 500  # observation 文本截断长度，避免轨迹过大

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    answer: str
    sources: List[dict]
    trajectory: List[dict]
    cached: bool = False
    run_id: str = ""                       # 运行快照 id（流式中断后可轮询 GET /agent/runs/{run_id}）
    timing: dict = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)  # 评测用：agent 实际检索到的完整 chunk 正文（top_k_rerank，不截断）


# ──────────────────────────────────────────────
# 缓存（懒加载，可注入 embed_fn）
# ──────────────────────────────────────────────

_cache: Optional[SemanticCache] = None


def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        embed_fn = None
        if settings.agent_cache_semantic:
            try:
                from note_assistant.indexing.embedder import OllamaEmbedder

                embed_fn = OllamaEmbedder().embed_one
            except Exception:  # noqa: BLE001
                embed_fn = None
        _cache = SemanticCache(
            enabled=settings.agent_cache_enabled,
            ttl=settings.agent_cache_ttl,
            max_size=settings.agent_cache_max_size,
            semantic=settings.agent_cache_semantic,
            semantic_threshold=settings.agent_cache_semantic_threshold,
            embed_fn=embed_fn,
        )
    return _cache


def reset_cache() -> None:
    """测试 / 重新配置时清空全局缓存（重新按当前 settings 构建）。"""
    global _cache
    _cache = None


def get_cache_stats() -> dict:
    """返回语义缓存统计（hits/misses/hit_rate/size/enabled/semantic）。

    评测脚本在跑完一轮后调用，写入 ``EvalReport.semantic_cache_stats``。
    仅 agent 链路有意义（naive 链路不使用 SemanticCache）。
    """
    return _get_cache().stats()


# ──────────────────────────────────────────────
# 持久层（懒加载，可注入 store）
# ──────────────────────────────────────────────

_store: Optional[AgentStore] = None


def get_store() -> Optional[AgentStore]:
    """返回全局持久层；``agent_session_enabled=False`` 时返回 None（退化为无状态）。"""
    global _store
    if not settings.agent_session_enabled:
        return None
    if _store is None:
        _store = AgentStore()
    return _store


def set_store_for_test(store: Optional[AgentStore]) -> None:
    """测试注入（含内存/临时文件 store），绕过全局单例与 settings 开关。"""
    global _store
    _store = store


def reset_store() -> None:
    global _store
    _store = None


async def _persist(store: AgentStore, coro_fn, *args):
    """同步 store 方法包一层 to_thread，避免阻塞事件循环。"""
    await asyncio.to_thread(coro_fn, *args)


# ──────────────────────────────────────────────
# 来源 / 轨迹 构造
# ──────────────────────────────────────────────

def _sources_from_results(results: List[RetrievalResult]) -> List[dict]:
    ranked = sorted(results, key=lambda r: r.score, reverse=True)[: settings.top_k_rerank]
    out = []
    for r in ranked:
        meta = r.metadata if isinstance(r.metadata, dict) else {}
        out.append({
            "filepath": r.filepath,
            "title": r.metadata.get("title", ""),
            "heading": r.metadata.get("heading_path", ""),
            "score": round(r.score, 4),
            # 设计 9.2 /agent 适配项：把图片渲染字段透传给前端
            "kind": str(meta.get("kind") or "text"),
            "img_url": meta.get("img_url") or None,
            "render_hint": meta.get("render_hint") or None,
        })
    return out


def _contexts_from_results(results: List[RetrievalResult]) -> List[str]:
    """取 top_k_rerank 个结果的完整正文，作为评测上下文（不截断）。

    与 ``_sources_from_results`` 同源排序，但保留 ``page_content`` 全文——
    评测 ragas 的 context_precision / context_recall 需要较完整的上下文，
    不能用 trajectory 里被 OBS_TRUNCATE 截断的 observation 片段。
    """
    ranked = sorted(results, key=lambda r: r.score, reverse=True)[: settings.top_k_rerank]
    return [r.page_content for r in ranked if r.page_content]


def _trajectory_from_messages(messages: List[BaseMessage]) -> List[dict]:
    """从消息列表抽取 tool_call / observation / thought 事件（不含判定与答案）。"""
    traj: List[dict] = []
    for m in messages:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    traj.append({
                        "type": "tool_call",
                        "tool": tc.get("name"),
                        "args": tc.get("args", {}),
                    })
            elif str(m.content).strip():
                traj.append({"type": "thought", "content": str(m.content)})
        elif isinstance(m, ToolMessage):
            traj.append({
                "type": "observation",
                "content": str(m.content)[:OBS_TRUNCATE],
            })
    return traj


def _trajectory_from_state(final: dict) -> List[dict]:
    """从最终 state 拼出完整轨迹：路由 → 工具/观察 → Judge 判定 → 答案。"""
    traj: List[dict] = []
    # 1. 路由判定
    route = final.get("route", "search")
    traj.append({
        "type": "thought",
        "content": f"路由判定：{'检索' if route == 'search' else '直接对话'}",
    })
    # 2. 消息中的 工具/观察/思考（跳过首条 HumanMessage）
    msgs = final.get("messages", [])
    if msgs and isinstance(msgs[0], HumanMessage):
        msgs = msgs[1:]
    traj.extend(_trajectory_from_messages(msgs))
    # 3. Judge 判定（每个 reflect 一次，按出现顺序）
    for entry in final.get("judge_log", []) or []:
        traj.append({
            "type": "judge",
            "verdict": entry.get("verdict"),
            "reason": entry.get("reason"),
            "iteration": entry.get("iteration"),
        })
    # 4. 答案
    ans = final.get("answer", "")
    if ans:
        traj.append({"type": "answer", "content": ans})
    return traj


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def _initial_state(
    question: str,
    history: Optional[list],
    condensed: str = "",
    history_messages: Optional[list] = None,
    accumulated: Optional[list] = None,
    signal: Optional[CondenseSignal] = None,
    just_clarified: bool = False,
) -> dict:
    sig = signal or CondenseSignal()
    return {
        "messages": [HumanMessage(question)],
        "accumulated": accumulated or [],
        "iteration": 0,
        "route": "",
        "question": question,
        "condensed_question": condensed,
        "history": history or [],
        "history_messages": history_messages or [],
        "answer": "",
        "judge_verdict": "",
        "rewritten_query": "",
        "judge_log": [],
        # 澄清（clarify-as-terminal）相关：消解置信度作为旁路信号注入，
        # 与 condense_question 的返回值签名解耦。
        "clarify_question": "",
        "condense_confidence": sig.confidence,
        "condense_candidates": list(sig.candidates),
        "just_clarified": just_clarified,
        "clarified": False,
        # 收敛闸门 / 反向放宽（2026-08-02 修复同文档空转）
        "doc_count_at_last_reflect": 0,
        "no_new_doc_streak": 0,
        "widen_context": False,
        "gate_overrode": False,
        # 安全（L2/L3）：会话注入命中计数 + 已浮现笔记白名单
        "allowed_files": set(),
        "injection_hits": 0,
    }


async def _prepare_agent_context(
    question: str,
    session_id: str,
    store: Optional[AgentStore],
    effective_history: list,
) -> tuple[str, List[BaseMessage], List[RetrievalResult], str, CondenseSignal, bool]:
    """入口上下文装配：凝练问题 → 取长程摘要 → 预算裁剪历史 → 跨轮累积 seed → 缓存指纹。

    返回 ``(condensed, history_messages, seed_accumulated, ctx_key, signal, just_clarified)``。

    后两项服务于澄清（clarify-as-terminal）：``signal`` 是本次指代消解的旁路置信度信号
    （不改 ``condense_question -> str`` 的签名），``just_clarified`` 表示上一轮刚反问过 ——
    两者共同构成 ``_should_clarify`` 级联守卫的输入，确保「反问是级联终点而非首选」。
    """
    cm = get_context_manager()
    # 1) 问题凝练（消指代），供路由/检索/缓存指纹使用
    condensed = await cm.condense_question(question, effective_history, session_id)

    # 2) 长程摘要（若有），作为 SystemMessage 前置 + 掺入缓存指纹
    summary_text = ""
    if store is not None and session_id:
        latest = await asyncio.to_thread(store.get_latest_summary, session_id)
        summary_text = latest["summary"] if latest else ""

    # 3) 预算裁剪后的历史消息（相关性 + token 预算），与 agent/direct_chat/generate 同源
    history_messages = cm.budget_history_messages(
        effective_history, condensed, settings.agent_history_token_budget, summary=summary_text
    )

    # 4) 跨轮累积 seed（上一轮检索片段，带一轮衰减）
    seed = cm.seed_accumulated(session_id) if session_id else []

    # 5) 总预算兜底（history + accumulated 之和不可超上限）
    history_messages, seed = cm.fit_total_budget(history_messages, seed)

    # 6) 缓存指纹：凝练问题 + 摘要 hash，不同上下文 → 不同 key，防串台
    ctx_key = cm.context_key(condensed, summary_text)

    # 7) 澄清旁路信号：condense 阶段记下的置信度/候选主题 + 「上一轮是否刚反问过」
    #    pop 语义（读取即清除）保证连续反问最多发生一次，不会把用户困在问句里。
    signal = cm.get_condense_signal(session_id)
    just_clarified = cm.pop_clarified(session_id)
    return condensed, history_messages, seed, ctx_key, signal, just_clarified


async def ainvoke(
    question: str,
    history: Optional[list] = None,
    session_id: str = "",
    run_id: str = "",
    return_contexts: bool = False,
) -> AgentRunResult:
    """非流式运行；命中缓存直接返回。可选持久化（run 快照 + 跨会话记忆）。

    Args:
        return_contexts: 为 True 时，在结果里附上 agent 实际检索到的完整 chunk
            正文（top_k_rerank 条，按 score 排序），用于评测（ragas 上下文指标）。
            默认 False，零副作用——不进轨迹、不撑大缓存/持久化、不影响 astream。
    """
    _t0 = time.perf_counter()
    rid = run_id or str(uuid.uuid4())[:8]
    set_request_id(rid)

    store = get_store()

    logger.info(
        "ainvoke.request",
        extra={
            "run_id": rid,
            "session_id": session_id,
            "question_preview": question[:60],
        },
    )
    # 历史来源：session 优先（服务端持有），否则用调用方传入的 history
    if session_id and store is not None:
        effective_history = await asyncio.to_thread(store.get_history, session_id)
    else:
        effective_history = history or []

    # 上下文装配：凝练问题 + 预算历史 + 跨轮累积 + 缓存指纹 + 澄清旁路信号
    condensed, history_messages, seed, ctx_key, signal, just_clarified = await _prepare_agent_context(
        question, session_id, store, effective_history
    )

    cache = _get_cache()
    if cache.enabled:
        hit = cache.get(question, ctx_key=ctx_key)
        if hit is not None:
            elapsed = (time.perf_counter() - _t0) * 1000
            logger.info(
                "ainvoke.done",
                extra={
                    "run_id": rid,
                    "session_id": session_id,
                    "cached": True,
                    "sources": len(hit.sources),
                    "elapsed_ms": round(elapsed),
                },
            )
            await _record_run(store, session_id, rid, question, AgentRunResult(
                answer=hit.answer, sources=hit.sources,
                trajectory=hit.trajectory, cached=True, run_id=rid,
            ), effective_history)
            _post_run_context(store, session_id, get_context_manager(), seed, question, hit.answer)
            return AgentRunResult(
                answer=hit.answer,
                sources=hit.sources,
                trajectory=hit.trajectory,
                cached=True,
                run_id=rid,
                timing={"total_ms": round(elapsed)},
            )

    graph = agent_mod.build_graph()
    final = await graph.ainvoke(
        _initial_state(
            question,
            effective_history,
            condensed=condensed,
            history_messages=history_messages,
            accumulated=seed,
            signal=signal,
            just_clarified=just_clarified,
        ),
        config={"callbacks": [get_token_handler()]},
    )
    elapsed = (time.perf_counter() - _t0) * 1000
    traj = _trajectory_from_state(final)
    acc = final.get("accumulated", seed)
    sources = _sources_from_results(acc)
    traj.append({"type": "sources", "sources": sources})
    contexts = _contexts_from_results(acc) if return_contexts else []
    # P2：把答案里的 [[IMG:asset_id]] 替换为真实图片 markdown；
    # 再确定性补齐 context 里 LLM 未引用的图片。
    # 仅检索生成路径补图：澄清问句与闲聊（direct_chat）不补，
    # 避免闲聊轮带着上一轮 seed 里的图片乱入。
    ans = postprocess_answer(final.get("answer", ""), acc)
    if not final.get("clarified") and final.get("route") != "chat":
        ans = append_missing_images(ans, acc)
    # L4 输出治理：远程图片中和（防渲染期外泄）+ system prompt 泄露指纹
    ans, media_hits = neutralize_remote_media(ans)
    leaked = check_prompt_leakage(ans)
    guarded = bool(media_hits) or bool(leaked)
    result = AgentRunResult(
        answer=ans,
        sources=sources,
        trajectory=traj,
        contexts=contexts,
        run_id=rid,
        timing={"total_ms": round(elapsed)},
    )
    logger.info(
        "ainvoke.done",
        extra={
            "run_id": rid,
            "session_id": session_id,
            "cached": False,
            "sources": len(sources),
            "elapsed_ms": round(elapsed),
        },
    )
    clarified = bool(final.get("clarified"))
    if clarified:
        # 反问轮：记下 session 标记，下一轮 _should_clarify 直接否决（不连续反问）
        get_context_manager().mark_clarified(session_id)
    # 澄清问句不入缓存：否则同一个模糊问题再问一次会命中缓存里的问句，永远吐反问
    # L4 门禁：输出护栏命中（远程图片被中和 / 泄露指纹）的答案不入缓存，防投毒回放
    if cache.enabled and not clarified and not (guarded and settings.cache_skip_when_guarded):
        cache.put(question, result.answer, result.sources, result.trajectory, ctx_key=ctx_key)
    await _record_run(store, session_id, rid, question, result, effective_history)
    _post_run_context(store, session_id, get_context_manager(), final.get("accumulated", seed), question, result.answer)
    return result


def _log_task_exception(task: asyncio.Task) -> None:
    """后台任务异常兜底：避免 asyncio『Task exception was never retrieved』被静默吞掉。

    ``maybe_summarize`` 内部已逐层 try/except 降级，正常不会抛到此处；这里仅作最后一道
    防线，把任何漏网的异常记成日志，便于排查而非无声消失。
    """
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # 取消不视为异常
    except Exception as e:  # noqa: BLE001
        logger.warning("后台 maybe_summarize 抛异常（已吞掉，不影响主链路）: %s", e)


def _post_run_context(
    store: Optional[AgentStore],
    session_id: str,
    cm,
    accumulated: list,
    question: str,
    answer: str,
) -> None:
    """运行后上下文收尾：更新跨轮累积，并在达到 token 阈值时后台触发长程摘要。

    摘要在后台任务里执行（不计入用户延迟）；无 session_id 时跨轮累积无意义，跳过。
    """
    if not session_id:
        return
    cm.record_turn(session_id, accumulated, question, answer)
    if settings.agent_summary_enabled and store is not None:
        try:
            # 后台任务：回答已返回，摘要不阻塞主链路
            task = asyncio.create_task(cm.maybe_summarize(session_id, store))
            task.add_done_callback(_log_task_exception)
        except RuntimeError:
            # 极端情况下无运行中的事件循环，静默降级跳过
            pass


async def _record_run(
    store: Optional[AgentStore],
    session_id: str,
    run_id: str,
    question: str,
    result: AgentRunResult,
    effective_history: list,
) -> None:
    """把一次运行结果落盘（run 事件 + 答案 + 来源 + 结束），并写会话记忆。"""
    if store is None:
        return
    if run_id:
        rid = run_id
        await asyncio.to_thread(store.ensure_run, rid, question)
    else:
        rid = await asyncio.to_thread(store.create_run, question)
    result.run_id = rid
    for i, ev in enumerate(result.trajectory):
        await asyncio.to_thread(store.append_event, rid, ev, i)
    await asyncio.to_thread(store.set_answer, rid, result.answer)
    await asyncio.to_thread(store.set_sources, rid, result.sources)
    await asyncio.to_thread(store.finish_run, rid)
    if session_id:
        await asyncio.to_thread(store.append_turn, session_id, "user", question)
        await asyncio.to_thread(store.append_turn, session_id, "assistant", result.answer)


async def astream(
    question: str,
    history: Optional[list] = None,
    session_id: str = "",
    run_id: str = "",
):
    """流式运行，逐项产出轨迹事件；命中缓存先回放；支持断流续传。

    每个响应流以 ``{"type": "run", "run_id": ...}`` 起始，客户端据此在断流后
    轮询 ``GET /agent/runs/{run_id}`` 取回完整结果。
    """
    _t0 = time.perf_counter()
    rid = run_id or str(uuid.uuid4())[:8]
    set_request_id(rid)
    store = get_store()
    if session_id and store is not None:
        effective_history = await asyncio.to_thread(store.get_history, session_id)
    else:
        effective_history = history or []
    cache = _get_cache()

    # 上下文装配（与 ainvoke 同源）：凝练 / 预算历史 / 跨轮累积 / 缓存指纹 / 澄清信号
    condensed, history_messages, seed, ctx_key, signal, just_clarified = await _prepare_agent_context(
        question, session_id, store, effective_history
    )

    # ── 续传：给定已存在的 run_id ──
    if run_id and store is not None:
        run = await asyncio.to_thread(store.get_run, run_id)
        if run is not None:
            yield {"type": "run", "run_id": run_id, "resumable": True}
            for ev in run["trajectory"]:
                yield ev
            yield {"type": "sources", "sources": run["sources"]}
            if run["status"] != "finished":
                yield {"type": "status", "status": run["status"],
                       "content": f"请轮询 GET /agent/runs/{run_id}"}
            return

    # ── 缓存命中：直接回放（并登记 run 以便后续轮询）──
    if cache.enabled:
        hit = cache.get(question, ctx_key=ctx_key)
        if hit is not None:
            rid = run_id or (await asyncio.to_thread(store.create_run, question) if store is not None else "")
            if store is not None:
                await _record_run(
                    store, session_id, rid, question,
                    AgentRunResult(answer=hit.answer, sources=hit.sources,
                                   trajectory=hit.trajectory, cached=True),
                    effective_history,
                )
                _post_run_context(store, session_id, get_context_manager(), seed, question, hit.answer)
            yield {"type": "run", "run_id": rid}
            for t in hit.trajectory:
                yield t
            yield {"type": "cached", "content": True}
            yield {"type": "answer", "content": hit.answer}
            yield {"type": "sources", "sources": hit.sources}
            return

    # ── 正常流式 ──
    rid = run_id or (await asyncio.to_thread(store.create_run, question) if store is not None else "")
    if store is not None and run_id:
        await asyncio.to_thread(store.ensure_run, rid, question)
    yield {"type": "run", "run_id": rid}

    graph = agent_mod.build_graph()
    state = _initial_state(
        question, effective_history,
        condensed=condensed, history_messages=history_messages, accumulated=seed,
        signal=signal, just_clarified=just_clarified,
    )
    seq = 0
    traj: List[dict] = []
    # ⚠️ 关键修复：用跨轮累积 seed 初始化，而非空列表。
    # 问题：当某一轮 agent 节点判定上下文已够、直接 generate（复用前几轮
    # seed，不再调工具），graph 不会经过 tools / graph_expand / rerank_* 节点，
    # 这些分支（runner.py:595/639/649/658）才更新局部 accumulated 变量。
    # 若不初始化为 seed，accumulated 一路保持 [] → 末尾 _sources_from_results([])
    # 返回空 → sources 事件为空 → 前端不渲染来源，表现为「检索命中了文件却没显示
    # 检索路径」。ainvoke 路径用 final.get("accumulated", seed) 兜底故无此问题，
    # 这里与之一致，保证流式与非流式行为对称。
    # 不会重复：tools / graph_expand / rerank_* 节点回写的是 state 完整 accumulated
    # （seed + 本轮新结果），会覆盖此处初始值；本初值仅在「完全没走这些节点」时兜底。
    accumulated: List[RetrievalResult] = list(seed)
    final_answer = ""
    clarified = False
    guarded = False  # L4：输出护栏是否命中（命中则不入缓存，防投毒回放）

    async for chunk in graph.astream(state, stream_mode="updates", config={"callbacks": [get_token_handler()]}):
        for node, update in chunk.items():
            update = update or {}
            ev = None
            if node == "router":
                route = update.get("route", "search")
                ev = {"type": "thought", "content": f"路由判定：{'检索' if route == 'search' else '直接对话'}"}
            elif node == "agent":
                ai = (update.get("messages") or [None])[-1]
                if isinstance(ai, AIMessage) and ai.tool_calls:
                    for tc in ai.tool_calls:
                        ev = {"type": "tool_call", "tool": tc.get("name"), "args": tc.get("args", {})}
                        yield ev
                        traj.append(ev)
                        if store is not None:
                            await asyncio.to_thread(store.append_event, rid, ev, seq)
                            seq += 1
                    # 已在循环内逐项 yield，置 None 避免底部 if ev is not None 重复发射最后一个
                    ev = None
                elif isinstance(ai, AIMessage) and str(ai.content).strip():
                    ev = {"type": "thought", "content": str(ai.content)}
            elif node == "tools":
                acc = update.get("accumulated")
                if acc is not None:
                    accumulated = acc
                for m in update.get("messages", []):
                    if isinstance(m, ToolMessage):
                        ev = {"type": "observation", "content": str(m.content)[:OBS_TRUNCATE]}
                        yield ev
                        traj.append(ev)
                        if store is not None:
                            await asyncio.to_thread(store.append_event, rid, ev, seq)
                            seq += 1
                # 已在循环内逐项 yield，置 None 避免底部 if ev is not None 重复发射最后一个
                ev = None
            elif node == "reflect":
                entry = (update.get("judge_log") or [{}])[-1]
                ev = {
                    "type": "judge",
                    "verdict": update.get("judge_verdict", "sufficient"),
                    "reason": entry.get("reason"),
                    "iteration": entry.get("iteration"),
                }
            elif node == "rewrite":
                ev = {"type": "thought", "content": "（反思改写）重新检索"}
            elif node in ("generate", "direct_chat", "clarify"):
                # clarify 复用 answer 事件类型：clarify-as-terminal 的终止语义与
                # generate / direct_chat 完全同构，前端零改动即可渲染澄清问句。
                ans = update.get("answer", "")
                if ans:
                    # P2：把答案里的 [[IMG:asset_id]] 替换为真实图片 markdown
                    ans = postprocess_answer(ans, accumulated)
                    if node == "generate":
                        # 确定性补图：LLM 没写标记时，context 里的相关图片也尽量显示
                        ans = append_missing_images(ans, accumulated)
                    # L4 输出治理：远程图片中和 + 泄露指纹（generate 路径计入门禁）
                    ans, media_hits = neutralize_remote_media(ans)
                    if node == "generate":
                        guarded = guarded or bool(media_hits) or bool(check_prompt_leakage(ans))
                    final_answer = ans
                    if update.get("clarified"):
                        clarified = True
                    ev = {"type": "answer", "content": ans}
            elif node == "graph_expand_node":
                # 静默节点：沿 [[wikilinks]] 扩展关联笔记，只改 accumulated。
                new_acc = update.get("accumulated")
                if new_acc is not None:
                    added = len(new_acc) - len(accumulated)
                    accumulated = new_acc
                    ev = {"type": "thought",
                          "content": f"🔗 图检索扩展：沿 wikilinks 关联，片段 {len(new_acc) - added} → {len(new_acc)}（新增 {added}）"}
                else:
                    ev = {"type": "thought", "content": "🔗 图检索扩展：本轮无新增关联笔记（未开启或未命中）"}
            elif node == "rerank_loop":
                # 静默节点：循环内闸门，每轮检索后精排并裁剪 top-k。
                new_acc = update.get("accumulated")
                if new_acc is not None:
                    accumulated = new_acc
                    ev = {"type": "thought",
                          "content": f"🔄 循环内重排：精排后保留 {len(new_acc)} 个片段"}
                else:
                    ev = {"type": "thought", "content": "🔄 循环内重排：未启用，跳过"}
            elif node == "rerank_exit":
                # 静默节点：出口总安检，Judge 通过后对多轮累积做全局精排。
                new_acc = update.get("accumulated")
                if new_acc is not None:
                    accumulated = new_acc
                    ev = {"type": "thought",
                          "content": f"🔄 出口重排：全局精排后保留 top-{len(new_acc)} 片段用于生成"}
                else:
                    ev = {"type": "thought", "content": "🔄 出口重排：未启用，跳过"}

            if ev is not None:
                yield ev
                traj.append(ev)
                if store is not None:
                    await asyncio.to_thread(store.append_event, rid, ev, seq)
                    seq += 1

    sources = _sources_from_results(accumulated)
    yield {"type": "sources", "sources": sources}
    traj.append({"type": "sources", "sources": sources})
    if store is not None:
        await asyncio.to_thread(store.append_event, rid, {"type": "sources", "sources": sources}, seq)
        await asyncio.to_thread(store.set_answer, rid, final_answer)
        await asyncio.to_thread(store.set_sources, rid, sources)
        await asyncio.to_thread(store.finish_run, rid)
        if session_id:
            await asyncio.to_thread(store.append_turn, session_id, "user", question)
            await asyncio.to_thread(store.append_turn, session_id, "assistant", final_answer)
    if clarified:
        # 反问轮：置位 session 标记，下一轮 _should_clarify 直接否决（不连续反问）
        get_context_manager().mark_clarified(session_id)
    # 澄清问句不入缓存：否则同一个模糊问题再问会命中缓存里的问句，形成反问死循环
    # L4 门禁：输出护栏命中的答案不入缓存，防投毒回放
    if cache.enabled and not clarified and not (guarded and settings.cache_skip_when_guarded):
        cache.put(question, final_answer, sources, traj, ctx_key=ctx_key)
    elapsed = (time.perf_counter() - _t0) * 1000
    logger.info(
        "astream.done",
        extra={
            "run_id": rid,
            "session_id": session_id,
            "cached": False,
            "sources": len(sources),
            "events": len(traj),
            "elapsed_ms": round(elapsed),
        },
    )
    _post_run_context(store, session_id, get_context_manager(), accumulated, question, final_answer)

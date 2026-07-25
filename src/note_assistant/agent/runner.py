"""Agent 运行器：把自写 StateGraph 封装成非流式/流式入口，并接入语义缓存（P7b）。

- ``ainvoke``：跑完整图，返回答案 + 去重来源 + 完整轨迹（命中缓存则直接返回）。
- ``astream``：逐项产出轨迹事件（thought / tool_call / observation / judge / answer / sources），
  同样命中缓存时先回放缓存轨迹。
- 缓存：精确 + 近邻（注入式 embedder），外部异常自动降级，绝不拖垮主链路。
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from note_assistant.agent import agent as agent_mod
from note_assistant.agent.cache import SemanticCache
from note_assistant.agent.context import get_context_manager
from note_assistant.agent.store import AgentStore
from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult

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
        out.append({
            "filepath": r.filepath,
            "title": r.metadata.get("title", ""),
            "heading": r.metadata.get("heading_path", ""),
            "score": round(r.score, 4),
        })
    return out


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
) -> dict:
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
    }


async def _prepare_agent_context(
    question: str,
    session_id: str,
    store: Optional[AgentStore],
    effective_history: list,
) -> tuple[str, List[BaseMessage], List[RetrievalResult], str]:
    """入口上下文装配：凝练问题 → 取长程摘要 → 预算裁剪历史 → 跨轮累积 seed → 缓存指纹。

    返回 ``(condensed, history_messages, seed_accumulated, ctx_key)``。
    """
    cm = get_context_manager()
    # 1) 问题凝练（消指代），供路由/检索/缓存指纹使用
    condensed = await cm.condense_question(question, effective_history)

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
    return condensed, history_messages, seed, ctx_key


async def ainvoke(
    question: str,
    history: Optional[list] = None,
    session_id: str = "",
    run_id: str = "",
) -> AgentRunResult:
    """非流式运行；命中缓存直接返回。可选持久化（run 快照 + 跨会话记忆）。"""
    store = get_store()

    # 历史来源：session 优先（服务端持有），否则用调用方传入的 history
    if session_id and store is not None:
        effective_history = await asyncio.to_thread(store.get_history, session_id)
    else:
        effective_history = history or []

    # 上下文装配：凝练问题 + 预算历史 + 跨轮累积 + 缓存指纹
    condensed, history_messages, seed, ctx_key = await _prepare_agent_context(
        question, session_id, store, effective_history
    )

    cache = _get_cache()
    if cache.enabled:
        hit = cache.get(question, ctx_key=ctx_key)
        if hit is not None:
            result = AgentRunResult(
                answer=hit.answer,
                sources=hit.sources,
                trajectory=hit.trajectory,
                cached=True,
            )
            await _record_run(store, session_id, run_id, question, result, effective_history)
            _post_run_context(store, session_id, get_context_manager(), seed, question, result.answer)
            return result

    graph = agent_mod.build_graph()
    final = await graph.ainvoke(
        _initial_state(
            question,
            effective_history,
            condensed=condensed,
            history_messages=history_messages,
            accumulated=seed,
        )
    )
    traj = _trajectory_from_state(final)
    sources = _sources_from_results(final.get("accumulated", seed))
    traj.append({"type": "sources", "sources": sources})
    result = AgentRunResult(
        answer=final.get("answer", ""),
        sources=sources,
        trajectory=traj,
    )
    if cache.enabled:
        cache.put(question, result.answer, result.sources, result.trajectory, ctx_key=ctx_key)
    await _record_run(store, session_id, run_id, question, result, effective_history)
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
    store = get_store()
    if session_id and store is not None:
        effective_history = await asyncio.to_thread(store.get_history, session_id)
    else:
        effective_history = history or []
    cache = _get_cache()

    # 上下文装配（与 ainvoke 同源）：凝练 / 预算历史 / 跨轮累积 / 缓存指纹
    condensed, history_messages, seed, ctx_key = await _prepare_agent_context(
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
    )
    seq = 0
    traj: List[dict] = []
    accumulated: List[RetrievalResult] = []
    final_answer = ""

    async for chunk in graph.astream(state, stream_mode="updates"):
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
            elif node in ("generate", "direct_chat"):
                ans = update.get("answer", "")
                if ans:
                    final_answer = ans
                    ev = {"type": "answer", "content": ans}

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
    if cache.enabled:
        cache.put(question, final_answer, sources, traj, ctx_key=ctx_key)
    _post_run_context(store, session_id, get_context_manager(), accumulated, question, final_answer)

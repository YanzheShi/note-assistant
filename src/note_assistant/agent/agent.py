"""Agentic RAG 编排（自写 langgraph StateGraph）。

相比之前 create_react_agent 黑盒版本，本实现显式管理状态，落地蓝图四大核心：
    1. Router 节点：意图识别，闲聊/人设问题直接 bypass 检索走 direct_chat，省钱可观测。
    2. Context Accumulator：每一轮检索的 RetrievalResult 按 (filepath, heading)
       确定性去重后累积进 state.accumulated，多跳结果不丢失。
    3. Reflection Judge（独立可观测节点）：生成前用 LLM 判定信息是否足够
       （sufficient / need_rewrite / need_more / give_up），决策离散可记录可回放；
       need_rewrite 走 rewrite 节点改写查询后重检，达到 max_iter 强制降级生成。
    4. 工具集：hybrid_search / graph_expand / vector_search / bm25_search /
       filtered_search / query_rewrite / get_note（含 retry/fallback 降级）。

状态图：
    START → router
    router --search--> agent ; router --chat--> direct_chat
    agent --tool_calls--> tools ; agent --回答--> generate
    tools → reflect（Judge）
    reflect --sufficient/give_up/达上限--> generate
    reflect --need_rewrite/need_more--> rewrite → agent
    generate/direct_chat → END
"""
import json
import operator
from functools import lru_cache
from typing import Annotated, List, TypedDict

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from note_assistant.agent.tools import AGENT_TOOLS, run_tool_call
from note_assistant.retrieval.reranker import get_reranker
from note_assistant.config import settings
from note_assistant.llm.client import get_llm
from note_assistant.retrieval.types import RetrievalResult

MAX_ITER = settings.agent_max_iter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    accumulated: List[RetrievalResult]          # Context Accumulator（去重后累积）
    iteration: int                               # 已执行的检索轮次
    route: str                                   # "search" | "chat"，由 router 写入
    question: str                                # 原始本轮问题（落库 / 展示用）
    condensed_question: str                      # 凝练后的独立完整问题（路由/检索/生成同源）
    history: list                                # 预算裁剪后的历史 dicts（generate_node）
    history_messages: list                       # 预算裁剪后的历史 BaseMessage（agent/direct_chat）
    answer: str
    judge_verdict: str                           # 最新 Judge 判定：sufficient/need_rewrite/need_more/give_up
    rewritten_query: str                         # Judge 建议的改写查询
    judge_log: Annotated[list, operator.add]     # 每轮 Judge 决策事件（可观测、可回放）


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "你是一个基于个人知识库（Obsidian 笔记）的智能问答助手。\n\n"
    "可用工具：\n"
    "- hybrid_search(query, top_k)：混合检索（语义向量 + BM25 关键词融合），通用首选入口。\n"
    "- graph_expand(filepaths, hop)：从已命中笔记出发，沿 [[双链]] 扩展关联笔记。\n"
    "- vector_search(query, top_k)：仅做语义向量检索。\n"
    "- bm25_search(query, top_k)：仅做关键词（BM25）检索。\n"
    "- filtered_search(query, filepath, heading, tag, top_k)：按元数据过滤后再检索。\n"
    "- get_note(filepath)：读取整篇笔记的全部片段。\n\n"
    "工作规则：\n"
    "1. 需要查知识库时，优先调用 hybrid_search。\n"
    "2. 若已命中笔记之间有双链关联、或需要更多上下文，调用 graph_expand 扩展。\n"
    "3. 当 hybrid 效果不佳时，尝试单独 vector_search / bm25_search 或 filtered_search 缩小范围。\n"
    "4. 一轮检索通常足够；只有证据明显不足或问题含多个子话题时，才再次检索"
    f"（最多 {MAX_ITER} 轮）。\n"
    "5. 证据充分后，直接作答：基于笔记内容、不编造、用 Markdown 结构化，并标注引用笔记标题。\n"
    "6. 若多次检索后仍无相关内容，明确告知用户知识库中缺少该信息。\n"
)

ROUTER_SYSTEM = (
    "你是意图分类器，决定用户问题是否需要检索个人知识库（Obsidian 笔记）才能回答。\n"
    "\n"
    "判定标准（关键）：\n"
    "- 只要问题中涉及的主题、概念、术语、人物、项目、方法论，有可能在个人知识库中被记录、注释或个性化阐述，就必须检索。\n"
    "- 个人知识库涵盖：技术概念笔记、读书摘要、项目记录、工具使用心得、个人总结与反思等。\n"
    "- 因此，大多数'XX 是什么'、'XX 有哪些组件'、'如何做 XX'这类知识性问题，都应检索——即使用户问的是通用话题，知识库中也可能有相关的个人笔记、摘录或见解。\n"
    "\n"
    "不需要检索的情况（仅限以下）：\n"
    "- 纯闲聊 / 问候（'你好'、'在吗'）\n"
    "- 询问你的身份或能力（'你是谁'、'你能做什么'）\n"
    "- 与知识库主题完全无关的通用问题（'今天天气'、'现在几点'）\n"
    "\n"
    "犹豫不决时，宁可检索，不要跳过。\n"
    "\n"
    "只输出一个 JSON 对象，不要有任何其他文字：\n"
    "{\"needs_search\": true 或 false, \"reason\": \"简短理由\", \"confidence\": 0.0到1.0之间}\n"
    "\n"
    "示例：\n"
    "输入：现代 LLM 的标配组件有哪些？\n"
    "输出：{\"needs_search\": true, \"reason\": \"涉及 LLM 架构概念，知识库可能有相关笔记\", \"confidence\": 0.9}\n"
    "\n"
    "输入：你好\n"
    "输出：{\"needs_search\": false, \"reason\": \"纯问候闲聊\", \"confidence\": 0.99}\n"
    "\n"
    "输入：RAG 系统中的 ReRank 应该怎么实现？\n"
    "输出：{\"needs_search\": true, \"reason\": \"RAG 技术细节，知识库应有相关实践笔记\", \"confidence\": 0.85}"
)

JUDGE_SYSTEM = (
    "你是检索质量评判器。给定用户问题与已检索到的知识库片段，评估这些片段是否足以回答。\n"
    "\n"
    "第一步：逐条检查（在脑内完成）\n"
    "- 用户问题的核心信息需求是什么？（例如问'路线图'，核心需求是：阶段划分、各阶段目标/产出）\n"
    "- 检索到的片段是否直接命中这些需求？请逐条列出命中情况。\n"
    "\n"
    "第二步：打出 relevance_score（0 到 2 的浮点数）\n"
    "  2.0 = 片段直接且完整地覆盖了问题的核心需求\n"
    "  1.0 = 片段相关，覆盖了核心需求的主要部分（即使细节不完全）\n"
    "  0.5 = 片段部分相关，但关键信息缺失\n"
    "  0.0 = 片段与问题无关\n"
    "\n"
    "第三步：按以下确定性规则映射到 verdict（严格遵守，不要凭感觉）：\n"
    "- relevance_score >= 1.0 → verdict = \"sufficient\"（直接生成，不要改写）\n"
    "- relevance_score == 0.5 → verdict = \"need_rewrite\"，并在 rewritten_query 给出改写\n"
    "- relevance_score == 0.0 → 若检索次数 < 2，verdict = \"need_more\"；否则 verdict = \"give_up\"\n"
    "\n"
    "关键原则：\n"
    "- 只要片段的标题或内容与问题的核心实体/主题直接匹配，且包含了回答问题所需的关键信息（如阶段、步骤、列表），就必须判 >= 1.0。\n"
    "- 不要因为'希望找到更详细/更完整的资料'就判 need_rewrite。sufficient 不代表完美，代表'足够回答'。\n"
    "- 检索次数已达上限时，即使觉得不够也判 give_up，不要无限循环。\n"
    "\n"
    "只输出 JSON 对象：\n"
    "{\"verdict\": \"sufficient|need_rewrite|need_more|give_up\", \"relevance_score\": 0.0到2.0, \"reason\": \"简短理由\", \"rewritten_query\": \"仅 need_rewrite 时填写，其余为空字符串\"}\n"
    "\n"
    "示例：\n"
    "问题：Agent 开发官方框架整体路线图是什么？\n"
    "片段：《Agent开发官方框架学习路径》中包含'整体路线图'章节，列出阶段一至阶段四，每阶段有目标、产出、核心。\n"
    "输出：{\"verdict\": \"sufficient\", \"relevance_score\": 2.0, \"reason\": \"片段直接命中'整体路线图'，包含完整的四阶段划分\", \"rewritten_query\": \"\"}\n"
    "\n"
    "问题：Agent 开发官方框架整体路线图是什么？\n"
    "片段：一篇讲 LangGraph 基础用法的笔记，未提及整体路线图。\n"
    "输出：{\"verdict\": \"need_rewrite\", \"relevance_score\": 0.5, \"reason\": \"片段仅涉及 LangGraph 单点技术，缺少路线图整体信息\", \"rewritten_query\": \"Agent 开发官方框架 学习路径 整体路线图 阶段规划\"}\n"
)

GENERATE_SYSTEM = (
    "你是基于个人知识库的问答助手。\n\n"
    "规则：\n"
    "1. 基于下面的「参考笔记」回答，不要编造笔记中不存在的内容。\n"
    "2. 如果参考笔记为空，说明知识库中缺少相关信息，请如实告知用户。\n"
    "3. 回答要简洁、结构化，使用 Markdown，并标注引用笔记标题。\n"
)

CHAT_SYSTEM = (
    "你是 Obsidian 个人知识库助手。用户现在在闲聊或询问你的能力，"
    "请用友好、简洁的语气直接回答，不需要检索知识库。"
)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """从 LLM 输出里抠出 JSON 子串（兼容 ```json 代码块）。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _norm_verdict(v: str) -> str:
    v = (v or "").strip().lower()
    if v not in ("sufficient", "need_rewrite", "need_more", "give_up"):
        return "sufficient"  # 默认充分，避免无限循环
    return v


def _format_context(results: List[RetrievalResult]) -> str:
    """把去重/裁剪后的结果格式化为生成上下文。"""
    if not results:
        return "（无参考笔记）"
    parts = []
    for i, r in enumerate(results, 1):
        title = r.metadata.get("title", "未知笔记")
        parts.append(f"### [{i}] {title}\n{r.page_content}")
    return "\n\n".join(parts)


def _top_k_context(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """生成前 Top-K 裁剪：按分数降序，截断到 top_k_rerank，防止超窗口。"""
    if not results:
        return []
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    return ranked[: settings.top_k_rerank]


def _fmt_history(history: list) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for turn in (history or [])[-20:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


# ──────────────────────────────────────────────
# 节点
# ──────────────────────────────────────────────

async def router(state: AgentState) -> dict:
    """意图识别节点：决定走检索循环还是直接对话。默认检索更稳妥。

    路由依据用户「原始问题」判意图——原始表述里的「笔记 / 我的 / 知识库」等词是判断
    是否要检索个人知识库的关键信号；若用凝练（消指代）后的问题，这些词会被改写步骤抹掉，
    导致明明问笔记却被误判为闲聊 / 常识。凝练问题仅在原问题疑似纯指代、需要消歧时作为
    参考附上（兼顾「它有什么缺点？」这类追问的消歧）。
    """
    llm = get_llm(temperature=0.0)
    q = state["question"]
    condensed = state.get("condensed_question") or ""
    if condensed and condensed != q:
        q = f"{q}\n（若原问题含指代或不完整，可参考已消歧的改写：{condensed}）"
    try:
        resp = await llm.ainvoke([
            SystemMessage(ROUTER_SYSTEM),
            HumanMessage(q),
        ])
        data = json.loads(_extract_json(str(resp.content)))
        needs = bool(data.get("needs_search", True))
        reason = data.get("reason", "")
    except Exception:
        needs = True
        reason = ""
    route = "search" if needs else "chat"
    logger.info("router.decision", extra={"route": route, "needs_search": needs, "reason": reason})
    return {"route": route}


async def agent_node(state: AgentState) -> dict:
    """决策节点：LLM 看上下文，决定调工具还是直接回答。

    用户当前问题用凝练后的独立完整问题（消指代），与 router/reflect/generate 同源，
    避免「它/那」类指代导致检索 query 错误；本轮内已有的工具调用 / 观察 / 改写提示
    保留，供 LLM 连贯决策。
    """
    llm = get_llm(temperature=0.3).bind_tools(AGENT_TOOLS)
    msgs: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT)]
    # 接入预算裁剪后的历史，避免失忆
    msgs.extend(state.get("history_messages") or [])
    # 当前问题用凝练版（消指代），替换原始未消解问题
    q = state.get("condensed_question") or state["question"]
    msgs.append(HumanMessage(q))
    # 保留本轮内已有的工具调用 / 观察 / 改写提示（messages[0] 为原始问题，跳过）
    rest = state["messages"][1:] if state["messages"] else []
    msgs.extend(rest)
    resp = await llm.ainvoke(msgs)
    return {"messages": [resp]}


async def tools_node(state: AgentState) -> dict:
    """工具节点：执行工具，把结构化结果去重后写入 Accumulator（含 retry/fallback）。

    观察文本回喂 LLM 前按 token 预算截断，避免撑爆窗口。
    """
    import asyncio

    from note_assistant.agent.context import get_context_manager

    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}
    cm = get_context_manager()
    new_messages: list[BaseMessage] = []
    accumulated = list(state["accumulated"])
    # 确定性去重：用 (filepath, heading) 作 key
    seen = {(r.filepath, r.metadata.get("heading_path", "")) for r in accumulated}
    new_before = len(accumulated)
    for tc in last.tool_calls:
        # 工具调用可能含同步 LLM（query_rewrite），放到线程避免阻塞事件循环
        obs_text, results = await asyncio.to_thread(
            run_tool_call, tc["name"], tc.get("args", {})
        )
        obs_text = cm.truncate_observation(obs_text, settings.agent_obs_token_budget)
        for r in results:
            key = (r.filepath, r.metadata.get("heading_path", ""))
            if key not in seen:
                seen.add(key)
                accumulated.append(r)
        new_messages.append(ToolMessage(content=obs_text, tool_call_id=tc["id"]))
    logger.info(
        "tools.summary",
        extra={
            "tools_executed": len(last.tool_calls),
            "results_added": len(accumulated) - new_before,
            "accumulated_total": len(accumulated),
            "iteration": state["iteration"] + 1,
        },
    )
    return {
        "messages": new_messages,
        "accumulated": accumulated,
        "iteration": state["iteration"] + 1,
    }


async def reflect(state: AgentState) -> dict:
    """Reflection Judge（独立可观测节点）：LLM 判定信息是否足够回答。

    生成前评估已收集上下文，决定：
      - sufficient / give_up → 进入生成（give_up 会触发降级提示）
      - need_rewrite → 写入改写查询，走 rewrite 节点重检
      - need_more    → 不改写，换检索策略重检
    达到 max_iter 时无论判定如何都强制进入生成（硬性降级）。
    """
    llm = get_llm(temperature=0.0)
    label = "达上限（强制生成）" if state["iteration"] >= MAX_ITER else "未达上限"
    q = state.get("condensed_question") or state["question"]
    try:
        msgs: list[BaseMessage] = [
            SystemMessage(JUDGE_SYSTEM),
            HumanMessage(
                f"用户问题：{q}\n\n"
                f"当前已收集片段数：{len(state['accumulated'])}\n"
                f"已检索轮次：{state['iteration']}/{MAX_ITER}（{label}）\n\n"
                "请判断信息是否足够回答（输出 JSON）。"
            ),
        ]
        resp = await llm.ainvoke(msgs)
        data = json.loads(_extract_json(str(resp.content)))
        verdict = _norm_verdict(data.get("verdict"))
        rewritten = str(data.get("rewritten_query") or "").strip()
    except Exception:
        # Judge 异常时保守处理：已达到上限就生成，否则再查一轮
        verdict = "sufficient" if state["iteration"] >= MAX_ITER else "need_more"
        rewritten = ""
        data = {}
    logger.info(
        "reflect.decision",
        extra={
            "iteration": state["iteration"],
            "verdict": verdict,
            "rewritten_query": rewritten,
        },
    )
    return {
        "judge_verdict": verdict,
        "rewritten_query": rewritten,
        "judge_log": [{
            "iteration": state["iteration"],
            "verdict": verdict,
            "reason": data.get("reason", ""),
            "rewritten_query": rewritten,
        }],
    }


async def rewrite_node(state: AgentState) -> dict:
    """改写节点：把 Judge 建议的改写查询（或换策略提示）注入下一轮检索。"""
    q = state.get("rewritten_query") or ""
    if q and q != state["question"]:
        hint = f"（反思改写）上一轮检索证据不足。请基于改写后的查询重新检索：{q}"
    else:
        hint = "（反思）上一轮检索证据不足，请尝试其它检索工具/策略（如单独 vector_search、bm25_search、filtered_search 或 graph_expand）再查一次。"
    logger.info("rewrite.decision", extra={"rewritten_query": q})
    return {"messages": [HumanMessage(hint)]}


async def rerank_loop(state: AgentState) -> dict:
    """Rerank ①：循环内闸门。每轮工具调用后，对 accumulated 做精排，保留 top-k。

    关闭时直接透传，不加载 reranker 模型。
    """
    if not settings.agent_reranker_loop_enabled:
        return {}
    if not state["accumulated"]:
        return {}
    reranker = get_reranker()
    question = state.get("condensed_question") or state["question"]
    results = reranker.rerank(question, state["accumulated"], top_k=settings.agent_reranker_loop_top_k)
    return {"accumulated": results}


async def rerank_exit(state: AgentState) -> dict:
    """Rerank ②：出口总安检。Judge 判定通过后，对多轮累积做全局精排。

    关闭时直接透传，不加载 reranker 模型。
    """
    if not settings.agent_reranker_exit_enabled:
        return {}
    if not state["accumulated"]:
        return {}
    reranker = get_reranker()
    question = state.get("condensed_question") or state["question"]
    results = reranker.rerank(question, state["accumulated"], top_k=settings.top_k_rerank)
    return {"accumulated": results}


async def generate_node(state: AgentState) -> dict:
    """生成节点：用累积去重 + Top-K 裁剪后的上下文生成带引用的答案。

    若 Judge 判 give_up 或达到 max_iter 仍证据不足，提示用户「部分信息可能不全」。
    """
    llm = get_llm(temperature=0.6, max_tokens=2048)
    context = _format_context(_top_k_context(state["accumulated"]))
    degraded = (
        state.get("judge_verdict") == "give_up"
        or (state["iteration"] >= MAX_ITER and not state["accumulated"])
    )
    if degraded:
        note = "\n\n（注：已多次检索但知识库中相关信息有限，以下回答可能不完整。）\n"
    else:
        note = ""
    msgs: list[BaseMessage] = [SystemMessage(GENERATE_SYSTEM)]
    # 预算裁剪后的历史（含长程摘要），与 agent/direct_chat 同源，避免重复
    history_msgs = state.get("history_messages") or _fmt_history(state["history"])
    msgs.extend(history_msgs)
    q = state.get("condensed_question") or state["question"]
    msgs.append(
        HumanMessage(f"## 参考笔记\n{context}\n\n## 问题\n{q}{note}")
    )
    resp = await llm.ainvoke(msgs)
    answer = str(resp.content)
    logger.info(
        "generate.summary",
        extra={"answer_len": len(answer), "degraded": degraded},
    )
    return {"answer": answer, "messages": [AIMessage(answer)]}


async def direct_chat(state: AgentState) -> dict:
    """直接对话节点：闲聊/人设类问题，不检索，友好回复。接入历史不再失忆。"""
    llm = get_llm(temperature=0.6, max_tokens=1024)
    msgs: list[BaseMessage] = [SystemMessage(CHAT_SYSTEM)]
    msgs.extend(state.get("history_messages") or [])
    q = state.get("condensed_question") or state["question"]
    msgs.append(HumanMessage(q))
    resp = await llm.ainvoke(msgs)
    answer = str(resp.content)
    logger.info("direct_chat.summary", extra={"answer_len": len(answer)})
    return {"answer": answer, "messages": [AIMessage(answer)]}


# ──────────────────────────────────────────────
# 分支函数
# ──────────────────────────────────────────────

def _route_branch(state: AgentState) -> str:
    return state["route"]


def _agent_branch(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "generate"


def _reflect_branch(state: AgentState) -> str:
    if state["iteration"] >= MAX_ITER:
        return "rerank_exit" if settings.agent_reranker_exit_enabled else "generate"
    verdict = state.get("judge_verdict", "sufficient")
    if verdict in ("sufficient", "give_up"):
        return "rerank_exit" if settings.agent_reranker_exit_enabled else "generate"
    return "rewrite"  # need_rewrite / need_more


# ──────────────────────────────────────────────
# 编译
# ──────────────────────────────────────────────

@lru_cache(maxsize=None)
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("rerank_loop", rerank_loop)
    g.add_node("reflect", reflect)
    g.add_node("rerank_exit", rerank_exit)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("direct_chat", direct_chat)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", _route_branch, {"search": "agent", "chat": "direct_chat"}
    )
    g.add_conditional_edges(
        "agent", _agent_branch, {"tools": "tools", "generate": "generate"}
    )
    g.add_edge("tools", "rerank_loop" if settings.agent_reranker_loop_enabled else "reflect")
    g.add_edge("rerank_loop", "reflect")
    g.add_conditional_edges(
        "reflect", _reflect_branch, {"generate": "generate", "rerank_exit": "rerank_exit", "rewrite": "rewrite"}
    )
    g.add_edge("rerank_exit", "generate")
    g.add_edge("rewrite", "agent")
    g.add_edge("generate", END)
    g.add_edge("direct_chat", END)
    return g.compile()

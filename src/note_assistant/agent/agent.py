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

from note_assistant.agent.tools import AGENT_TOOLS, graph_expand_impl, run_tool_call
from note_assistant.retrieval.reranker import get_reranker
from note_assistant.config import settings
from note_assistant.llm.client import get_llm
from note_assistant.retrieval.types import RetrievalResult
from note_assistant.pipeline.image_answer import ensure_image_selected, expand_image_neighbors

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
    judge_verdict: str                           # 最新 Judge 判定：sufficient/need_rewrite/need_more/give_up/need_clarify
    rewritten_query: str                         # Judge 建议的改写查询
    judge_log: Annotated[list, operator.add]     # 每轮 Judge 决策事件（可观测、可回放）
    clarify_question: str                        # Judge 给出的澄清问句（need_clarify 时）
    condense_confidence: float                   # 入口消解置信度（旁路信号，1.0=完全可信）
    condense_candidates: List[str]               # 上一轮召回的竞争主题，供澄清时作选项
    just_clarified: bool                         # 上一轮刚反问过 → 本轮禁止再反问
    clarified: bool                              # 本轮答案是澄清问句（runner 据此跳过缓存收录）
    # === 收敛闸门 / 反向放宽（2026-08-02 修复同文档空转）===
    doc_count_at_last_reflect: int               # 上一次 reflect 时的独特文档数（收敛闸门基线）
    no_new_doc_streak: int                       # 连续「新增独特文档 = 0」的轮数
    widen_context: bool                          # 生成窗口是否反向放宽（覆盖视图/闸门放行时置 True）
    gate_overrode: bool                          # 收敛闸门是否覆盖了 Judge 的 need_* 判定（诚实声明用）


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
    "不要做可能判定，宁可检索，不要因为可能没有就不检索。\n"
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
    "例外档位 need_clarify（极少数情况，判定标准极严）：\n"
    "- 仅当片段**同时命中两个及以上互不相干的主题**，且无法从问题本身判断用户指的是哪一个时，"
    "才可判 verdict = \"need_clarify\"，并在 clarify_question 给出一句带具体选项的澄清问句。\n"
    "- 澄清问句必须**从检索到的真实片段标题中取选项**，形如"
    "「你问的是 A 还是 B？」；严禁输出「你能说得更清楚一点吗」这类把负担甩给用户的空泛问法。\n"
    "- 若靠改写查询就能解决，一律判 need_rewrite，不要判 need_clarify。\n"
    "- 若片段虽多但都围绕同一主题，判 sufficient，不要判 need_clarify。\n"
    "- 若片段为空或全无关，判 need_more / give_up，不要判 need_clarify。\n"
    "\n"
    "关键原则：\n"
    "- 只要片段的标题或内容与问题的核心实体/主题直接匹配，且包含了回答问题所需的关键信息（如阶段、步骤、列表），就必须判 >= 1.0。\n"
    "- 证据末尾附有『知识库覆盖概览』：列出所有已命中笔记的标题与各章节（heading）。"
    "这是**广度判断**依据——若问题的某个子主题在概览中有对应章节标题，即视为该主题已被知识库覆盖，"
    "可判 sufficient，不必要求该片段正文完整出现在 top 正文里。覆盖即足够，不要求逐字精读。\n"
    "- 不要因为'希望找到更详细/更完整的资料'就判 need_rewrite。sufficient 不代表完美，代表'足够回答'。\n"
    "- 若多次改写检索后『知识库覆盖概览』始终没有新文档/新章节出现，说明知识库已穷尽，"
    "此时即使内容不完全也应判 sufficient（生成阶段会如实提示用户），不要继续空转。\n"
    "- 检索次数已达上限时，即使觉得不够也判 give_up，不要无限循环。\n"
    "\n"
    "只输出 JSON 对象：\n"
    "{\"verdict\": \"sufficient|need_rewrite|need_more|give_up|need_clarify\", \"relevance_score\": 0.0到2.0, \"reason\": \"简短理由\", \"rewritten_query\": \"仅 need_rewrite 时填写，其余为空字符串\", \"clarify_question\": \"仅 need_clarify 时填写带选项的澄清问句，其余为空字符串\"}\n"
    "\n"
    "示例：\n"
    "问题：Agent 开发官方框架整体路线图是什么？\n"
    "片段：《Agent开发官方框架学习路径》中包含'整体路线图'章节，列出阶段一至阶段四，每阶段有目标、产出、核心。\n"
    "输出：{\"verdict\": \"sufficient\", \"relevance_score\": 2.0, \"reason\": \"片段直接命中'整体路线图'，包含完整的四阶段划分\", \"rewritten_query\": \"\"}\n"
    "\n"
    "问题：Agent 开发官方框架整体路线图是什么？\n"
    "片段：一篇讲 LangGraph 基础用法的笔记，未提及整体路线图。\n"
    "输出：{\"verdict\": \"need_rewrite\", \"relevance_score\": 0.5, \"reason\": \"片段仅涉及 LangGraph 单点技术，缺少路线图整体信息\", \"rewritten_query\": \"Agent 开发官方框架 学习路径 整体路线图 阶段规划\", \"clarify_question\": \"\"}\n"
    "\n"
    "问题：那个的改进点是什么？\n"
    "片段：《FlashAttention-2 算法改进》与《Paged Attention 显存管理》两篇，主题互不相干，分数接近。\n"
    "输出：{\"verdict\": \"need_clarify\", \"relevance_score\": 0.5, \"reason\": \"片段命中两个互不相干主题，无法判断指代对象\", \"rewritten_query\": \"\", \"clarify_question\": \"你问的是 FlashAttention-2 的算法改进，还是 Paged Attention 的显存管理？\"}\n"
)

GENERATE_SYSTEM = (
    "你是基于个人知识库的问答助手。\n\n"
    "规则：\n"
    "1. 基于下面的「参考笔记」回答，不要编造笔记中不存在的内容。\n"
    "2. 如果参考笔记为空，说明知识库中缺少相关信息，请如实告知用户。\n"
    "3. 回答要简洁、结构化，使用 Markdown，并标注引用笔记标题。\n"
    "4. 参考笔记中标注【图片】的条目来自笔记里的插图，其内容由视觉模型解析得到。\n"
    "   - 引用图片信息时，说明\"根据笔记中的架构图/流程图\"，不要说\"根据文档描述\"\n"
    "   - 如果图片信息对回答有帮助，在相应位置插入 [[IMG:asset_id]] 标记，系统会自动替换为图片\n"
    "   - 严禁描述图片解析结果中不存在的细节\n"
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


_VALID_VERDICTS = ("sufficient", "need_rewrite", "need_more", "give_up", "need_clarify")


def _norm_verdict(v: str) -> str:
    v = (v or "").strip().lower()
    if v not in _VALID_VERDICTS:
        return "sufficient"  # 默认充分，避免无限循环
    return v


def _coverage_view(results: List[RetrievalResult]) -> str:
    """生成『知识库覆盖概览』：去重后的文档标题 + 各级 heading 清单（① 治本修复 Judge 盲判）。

    Judge 判断「某子主题在库里有没有被覆盖」本质是**广度问题**，不需要逐条读正文。
    给 heading 树，Judge 用极低 token 即可确认主题覆盖，不会被低分正文带偏——
    这正是「第一轮其实已命中预训练章节，却因正文排第 6 被 top-N 截断而误判 need_rewrite」的根因解法。
    """
    if not results:
        return "（无覆盖概览）"
    # 按文档标题聚合并去重 heading（heading_path 形如「一、背景 > 检索方法」）
    by_title: dict[str, list[str]] = {}
    for r in results:
        meta = r.metadata if isinstance(r.metadata, dict) else {}
        title = meta.get("title", "未知笔记") or "未知笔记"
        heading = (meta.get("heading_path", "") or "").strip()
        by_title.setdefault(title, [])
        if heading and heading not in by_title[title]:
            by_title[title].append(heading)
    lines = [
        "【知识库覆盖概览】以下为已检索笔记的标题与章节（用于判断主题是否已覆盖，"
        "不要求正文完整）："
    ]
    for title, headings in by_title.items():
        lines.append(f"· 《{title}》")
        for h in headings:
            lines.append(f"    - {h}")
    return "\n".join(lines)


def _coverage_has_extra_topics(results: List[RetrievalResult], top_n: int) -> bool:
    """覆盖概览里的 heading 是否超出 top-N 正文所覆盖的 heading（即存在低分但相关的章节）。

    返回 True 表示：Judge 可能基于覆盖概览放行，但生成若只取 top-N 正文会把这些
    低分相关内容裁掉——此时需要反向放宽生成窗口（④）。
    """
    if not results:
        return False
    ranked = sorted(results, key=lambda r: r.score, reverse=True)[:top_n]
    top_headings = {
        (r.metadata.get("heading_path", "") if isinstance(r.metadata, dict) else "")
        for r in ranked
    }
    all_headings = {
        (r.metadata.get("heading_path", "") if isinstance(r.metadata, dict) else "")
        for r in results
    }
    extra = all_headings - top_headings
    extra.discard("")
    return len(extra) > 0


def _format_judge_evidence(results: List[RetrievalResult]) -> str:
    """把已检索片段格式化成 Judge 的证据清单（P0：修复 Judge 盲判）。

    改造前 ``reflect`` 只把 ``len(accumulated)`` 这个**数字**传给 Judge，
    而 ``JUDGE_SYSTEM`` 开头声称「给定用户问题与已检索到的知识库片段」——
    Judge 一直在盲判，``relevance_score`` 全靠猜，这也是原 prompt 不得不用
    极重措辞去压 need_rewrite 的根因。

    这里按分数降序取 top-N 正文，每条给出「标题 + heading_path + 正文摘要」；
    末尾附「知识库覆盖概览」（① 治本）：去重后的文档标题 + 各级 heading 树，
    让 Judge 即使 top-N 正文没覆盖某主题，也能从 heading 确认其已被库覆盖。
    正文按 ``agent_judge_evidence_chars`` 截断，避免撑爆 Judge 的上下文
    （Judge 只需判断相关性，不需要读全文）。
    """
    if not results:
        return "（本轮未检索到任何片段）"
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    ranked = ranked[: settings.agent_judge_evidence_top_n]
    limit = settings.agent_judge_evidence_chars
    parts = []
    for i, r in enumerate(ranked, 1):
        meta = r.metadata if isinstance(r.metadata, dict) else {}
        title = meta.get("title", "未知笔记")
        heading = meta.get("heading_path", "")
        head = f"[{i}] 《{title}》" + (f" — {heading}" if heading else "")
        body = (r.page_content or "").strip().replace("\n", " ")[:limit]
        parts.append(f"{head}\n{body}")
    coverage = _coverage_view(results)
    return "\n\n".join(parts) + "\n\n" + coverage


def _format_context(results: List[RetrievalResult]) -> str:
    """把去重/裁剪后的结果格式化为生成上下文。"""
    if not results:
        return "（无参考笔记）"
    from note_assistant.pipeline.image_answer import render_image_block

    parts = []
    for i, r in enumerate(results, 1):
        title = r.metadata.get("title", "未知笔记")
        block = render_image_block(r)
        if block is not None:
            # image chunk：结构化渲染 + [[IMG:asset_id]] 引用标记
            parts.append(f"### [{i}] {title}\n{block}")
            continue
        parts.append(f"### [{i}] {title}\n{r.page_content}")
    return "\n\n".join(parts)


def _top_k_context(results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
    """生成前 Top-K 裁剪：按分数降序，截断到 top_k（默认 top_k_rerank），防止超窗口。

    ``top_k`` 可被调用方覆盖（④ 反向放宽：覆盖视图/闸门放行时放宽到 agent_generate_widen_top_k）。
    """
    if not results:
        return []
    k = top_k if top_k is not None else settings.top_k_rerank
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    return ranked[: k]


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
    # 确定性去重：用 identity_key（filepath, heading, kind, placeholder）。
    # 不能只用 (filepath, heading)——image summary chunk 与同节正文/父块共享
    # heading，旧键会让二者按分数竞速二选一，图片常被长文本挤掉（图不显示的根因之一）。
    seen = {r.identity_key() for r in accumulated}
    new_before = len(accumulated)
    for tc in last.tool_calls:
        # 工具调用可能含同步 LLM（query_rewrite），放到线程避免阻塞事件循环
        obs_text, results = await asyncio.to_thread(
            run_tool_call, tc["name"], tc.get("args", {})
        )
        obs_text = cm.truncate_observation(obs_text, settings.agent_obs_token_budget)
        for r in results:
            key = r.identity_key()
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
      - need_clarify → 片段命中多个不相干主题，写入澄清问句（需通过 ``_should_clarify`` 守卫）
    达到 max_iter 时无论判定如何都强制进入生成（硬性降级）。

    P0：这里必须把**片段内容**传给 Judge。改造前只传了片段数量，Judge 在盲判。
    """
    llm = get_llm(temperature=0.0)
    label = "达上限（强制生成）" if state["iteration"] >= MAX_ITER else "未达上限"
    q = state.get("condensed_question") or state["question"]
    evidence = _format_judge_evidence(state["accumulated"])
    try:
        msgs: list[BaseMessage] = [
            SystemMessage(JUDGE_SYSTEM),
            HumanMessage(
                f"用户问题：{q}\n\n"
                f"=== 已检索到的知识库片段（共 {len(state['accumulated'])} 条，"
                f"以下为分数最高的若干条）===\n{evidence}\n\n"
                f"已检索轮次：{state['iteration']}/{MAX_ITER}（{label}）\n\n"
                "请判断信息是否足够回答（输出 JSON）。"
            ),
        ]
        resp = await llm.ainvoke(msgs)
        data = json.loads(_extract_json(str(resp.content)))
        verdict = _norm_verdict(data.get("verdict"))
        rewritten = str(data.get("rewritten_query") or "").strip()
        clarify_q = str(data.get("clarify_question") or "").strip()
    except Exception:
        # Judge 异常时保守处理：已达到上限就生成，否则再查一轮
        verdict = "sufficient" if state["iteration"] >= MAX_ITER else "need_more"
        rewritten = ""
        clarify_q = ""
        data = {}
    logger.info(
        "reflect.decision",
        extra={
            "iteration": state["iteration"],
            "verdict": verdict,
            "rewritten_query": rewritten,
            "evidence_items": len(state["accumulated"]),
        },
    )

    # ── ① 反向放宽判定：覆盖概览里存在 top-N 正文未覆盖的章节（低分但相关的片段）──
    # 若 Judge 据此放行，生成时必须放宽窗口，否则这些内容会在生成端被裁掉（④ 的触发条件之一）。
    widen_context = _coverage_has_extra_topics(
        state["accumulated"], settings.agent_judge_evidence_top_n
    )

    # ── ③④ 收敛闸门：连续 streak 轮改写后「新增独特文档数 = 0」→ 确定性强制 sufficient ──
    # 直接切断「对同一篇文档换同义词反复重检」的死循环，不靠 LLM 自觉。
    # 仅在 Judge 本想 need_rewrite / need_more（继续检索）时才评估；sufficient/give_up 直接放行。
    gate_overrode = False
    current_docs = {r.filepath for r in state["accumulated"] if r.filepath}
    current_doc_count = len(current_docs)
    prev_doc_count = state.get("doc_count_at_last_reflect", 0)
    streak = state.get("no_new_doc_streak", 0)
    if verdict in ("need_rewrite", "need_more") and state["iteration"] >= 1:
        if current_doc_count <= prev_doc_count:
            streak += 1
        else:
            streak = 0  # 本轮检索到了新文档 → 重置，给 Judge 更多机会
        if streak >= settings.agent_convergence_streak:
            verdict = "sufficient"
            gate_overrode = True
            logger.info(
                "convergence.gate.forced",
                extra={"streak": streak, "unique_docs": current_doc_count},
            )
    else:
        streak = 0

    # 闸门覆盖（Judge 本想继续但被强制停）或覆盖视图有额外章节 → 都需反向放宽生成窗口
    widen_context = widen_context or gate_overrode

    return {
        "judge_verdict": verdict,
        "rewritten_query": rewritten,
        "clarify_question": clarify_q,
        "judge_log": [{
            "iteration": state["iteration"],
            "verdict": verdict,
            "reason": data.get("reason", ""),
            "rewritten_query": rewritten,
            "clarify_question": clarify_q,
        }],
        "doc_count_at_last_reflect": current_doc_count,
        "no_new_doc_streak": streak,
        "widen_context": widen_context,
        "gate_overrode": gate_overrode,
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


async def graph_expand_node(state: AgentState) -> dict:
    """自动图扩展节点：每轮检索后，从 accumulated 的 filepath 出发沿 [[wikilinks]] 扩展关联笔记。

    关闭时直接透传，不加载图。
    """
    if not settings.agent_graph_expand_enabled:
        return {}
    if not state["accumulated"]:
        return {}
    filepaths = list({
        r.filepath for r in state["accumulated"]
        if r.filepath and not r.filepath.startswith("[[")
    })
    if not filepaths:
        return {}
    new_chunks = graph_expand_impl(filepaths, hop=settings.agent_graph_expand_hop)
    if not new_chunks:
        return {}
    # 去重合并：与 tools_node 同一 identity_key（富结构 chunk 不因同章节被挤掉）
    accumulated = list(state["accumulated"])
    seen = {r.identity_key() for r in accumulated}
    added = 0
    for r in new_chunks:
        key = r.identity_key()
        if key not in seen:
            seen.add(key)
            accumulated.append(r)
            added += 1
    if added == 0:
        return {}
    return {"accumulated": accumulated}


async def rerank_loop(state: AgentState) -> dict:
    """Rerank ①：循环内闸门。每轮工具调用后，对 accumulated 做精排，保留 top-k。

    关闭时直接透传，不加载 reranker 模型。

    图片保位：图意图 query 的精排 top-k 里没有 image chunk 时，从全量精排结果
    里把最高分的图片补进（ensure_image_selected）——融合阶段的图意图 boost
    会被交叉编码器清零，不加护栏图片常被长正文挤出窗口。
    """
    if not settings.agent_reranker_loop_enabled:
        return {}
    if not state["accumulated"]:
        return {}
    reranker = get_reranker()
    question = state.get("condensed_question") or state["question"]
    full = reranker.rerank(question, state["accumulated"], top_k=len(state["accumulated"]))
    selected = full[: settings.agent_reranker_loop_top_k]
    results = ensure_image_selected(question, full, selected)
    return {"accumulated": results}


async def rerank_exit(state: AgentState) -> dict:
    """Rerank ②：出口总安检。Judge 判定通过后，对多轮累积做全局精排。

    关闭时直接透传，不加载 reranker 模型。
    ④ 反向放宽：若 ``widen_context`` 为真（覆盖视图/闸门放行），保留条数放宽到
    ``agent_generate_widen_top_k``，避免低分但相关的内容在生成端被裁掉。
    图片保位：同 rerank_loop——图意图 query 的 top-k 里没有 image chunk 时，
    从全量精排结果补入最高分图片，保证生成/来源端至少能看到一张相关图。
    """
    if not settings.agent_reranker_exit_enabled:
        return {}
    if not state["accumulated"]:
        return {}
    reranker = get_reranker()
    question = state.get("condensed_question") or state["question"]
    top_k = settings.agent_generate_widen_top_k if state.get("widen_context") else settings.top_k_rerank
    full = reranker.rerank(question, state["accumulated"], top_k=len(state["accumulated"]))
    selected = full[:top_k]
    results = ensure_image_selected(question, full, selected)
    return {"accumulated": results}


def _fetch_text_neighbors_by_heading(heading_paths: List[str]) -> List[RetrievalResult]:
    """从 ChromaDB 取同 heading_path 的文本 chunk（图片邻居扩展用，与 rag_chain 同源）。

    取不到 / 越界 / 异常 → 返回空列表，绝不中断生成。
    """
    from note_assistant.agent import tools as agent_tools

    try:
        collection = agent_tools._hybrid_retriever().ingestor.collection
        res = collection.get(
            where={"heading_path": {"$in": heading_paths}},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []
    out: List[RetrievalResult] = []
    for doc, meta in zip(res.get("documents") or [], res.get("metadatas") or []):
        out.append(RetrievalResult(score=0.0, page_content=doc, metadata=meta or {}))
    return out


async def generate_node(state: AgentState) -> dict:
    """生成节点：用累积去重 + Top-K 裁剪后的上下文生成带引用的答案。

    若 Judge 判 give_up 或达到 max_iter 仍证据不足，提示用户「部分信息可能不全」。
    ④ 反向放宽：``widen_context`` 为真时生成窗口放宽到 ``agent_generate_widen_top_k``，
    确保覆盖视图/闸门放行所依赖的低分相关内容真正进入生成上下文（不被裁掉）。
    ③ 诚实声明：仅当收敛闸门覆盖了 Judge 的 need_* 判定（gate_overrode）时，
    在末尾追加一句「知识库已穷尽」提示——绝因此缩小生成上下文。
    #4 图片邻居扩展（设计 7.3）：命中 image chunk 时带出同章节文本邻居，
    防图片脱离上下文被误解。邻居只进生成上下文、不回写 ``state["accumulated"]``
    （不污染 sources / Judge 证据），与 rag_chain 行为对齐。
    """
    llm = get_llm(temperature=0.6, max_tokens=2048)
    top_k = settings.agent_generate_widen_top_k if state.get("widen_context") else settings.top_k_rerank
    # #4：先 Top-K 裁剪，再把图片邻居补在末尾（邻居 score=0，确保不被截断出局）
    top_results = _top_k_context(state["accumulated"], top_k=top_k)
    expanded = expand_image_neighbors(state["accumulated"], _fetch_text_neighbors_by_heading)
    context = _format_context(top_results + expanded)
    degraded = (
        state.get("judge_verdict") == "give_up"
        or (state["iteration"] >= MAX_ITER and not state["accumulated"])
    )
    if state.get("gate_overrode"):
        # ③ 诚实声明：检索已穷尽但 Judge 本想继续 → 如实提示，不缩小上下文（④ 已放宽）
        note = (
            "\n\n（注：已多次检索但知识库未出现新的相关笔记，以上基于现有最相关内容作答，"
            "可能未涵盖该主题的全部方面。）\n"
        )
    elif degraded:
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


async def clarify_node(state: AgentState) -> dict:
    """澄清节点（clarify-as-terminal）：把澄清问句当答案返回，本次请求正常结束。

    刻意**不挂起**任何东西 —— 不 interrupt、不存 checkpoint、不占协程。
    走 END 后终止语义与 ``generate`` / ``direct_chat`` 完全同构，
    ``runner`` 已有的收尾链路（set_answer / finish_run / append_turn）不改就是对的。

    跨轮恢复不需要任何新机制：澄清问句作为普通 assistant 轮次落进 ``session_turns``，
    用户下一轮回答进来时，``condense_question`` 面对
    「user: 那个的改进 / assistant: 你问的是 A 还是 B？ / user: 第一个」
    这段历史本就能合成出完整问题 —— 指代消解本来就是它的职责。
    """
    q = (state.get("clarify_question") or "").strip()
    if not q:
        # 守卫兜底：理论上 _should_clarify 已拦截，这里再防一层空问句
        q = "你的问题可能指向多个不同主题，能补充说明一下具体想了解哪一方面吗？"
    candidates = state.get("condense_candidates") or []
    logger.info(
        "clarify.ask",
        extra={
            "question_preview": q[:60],
            "candidates": candidates[:3],
            "condense_confidence": state.get("condense_confidence", 1.0),
        },
    )
    # clarified=True 是给 runner 的显式标记：澄清问句**绝不能进语义缓存**，
    # 否则同一个模糊问题第二次问会永远命中缓存里的问句，形成「反问死循环」。
    return {"answer": q, "messages": [AIMessage(q)], "clarified": True}


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


def _should_clarify(state: AgentState) -> bool:
    """澄清级联守卫：把「多档消解都不行才问」编码成代码。

    clarify 是**级联的终点**，不是与 rewrite 并列的分支。五道闸门全过才反问：

        开关关闭                → False（默认关，行为与改造前逐字节等价）
        Judge 未判 need_clarify → False
        无澄清问句              → False（拒绝「你能说清楚点吗」这类空澄清）
        消解置信度 >= 阈值      → False（消解本就成功，不该打扰用户）
        上一轮刚反问过          → False（防连续追问）

    第四道闸门是关键：Judge 只看得到「检索结果指向多主题」，看不到「入口消解
    到底靠不靠谱」。若 LLM 改写与规则替换互相印证、上一轮主题也不存在竞争
    （confidence 0.9），那多主题很可能只是召回噪声，改写重检即可，不必反问。
    """
    logger.info(
        "CLARIFY_GATE_IN enabled=%s verdict=%s q=%r conf=%s thr=%s just=%s",
        settings.agent_clarify_enabled,
        state.get("judge_verdict"),
        (state.get("clarify_question") or "")[:20],
        state.get("condense_confidence", 1.0),
        settings.agent_clarify_confidence_threshold,
        state.get("just_clarified"),
    )
    if not settings.agent_clarify_enabled:
        return False
    if state.get("judge_verdict") != "need_clarify":
        return False
    if not (state.get("clarify_question") or "").strip():
        return False
    conf = state.get("condense_confidence", 1.0)
    if conf is None:
        conf = 1.0
    if conf >= settings.agent_clarify_confidence_threshold:
        return False
    if state.get("just_clarified"):
        return False
    return True


def _reflect_branch(state: AgentState) -> str:
    if _should_clarify(state):
        return "clarify"
    if state["iteration"] >= MAX_ITER:
        return "rerank_exit" if settings.agent_reranker_exit_enabled else "generate"
    verdict = state.get("judge_verdict", "sufficient")
    # 未通过澄清守卫的 need_clarify 降级为 sufficient —— 与改造前
    # （need_clarify 不在 _norm_verdict 白名单、被静默归一成 sufficient）行为一致，
    # 保证 agent_clarify_enabled=False 时整条链路逐字节等价。
    if verdict in ("sufficient", "give_up", "need_clarify"):
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
    g.add_node("graph_expand_node", graph_expand_node)
    g.add_node("rerank_loop", rerank_loop)
    g.add_node("reflect", reflect)
    g.add_node("rerank_exit", rerank_exit)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("direct_chat", direct_chat)
    g.add_node("clarify", clarify_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", _route_branch, {"search": "agent", "chat": "direct_chat"}
    )
    g.add_conditional_edges(
        "agent", _agent_branch, {"tools": "tools", "generate": "generate"}
    )
    g.add_edge("tools", "graph_expand_node" if settings.agent_graph_expand_enabled else               ("rerank_loop" if settings.agent_reranker_loop_enabled else "reflect"))
    g.add_edge("graph_expand_node", "rerank_loop" if settings.agent_reranker_loop_enabled else "reflect")
    g.add_edge("rerank_loop", "reflect")
    g.add_conditional_edges(
        "reflect",
        _reflect_branch,
        {
            "generate": "generate",
            "rerank_exit": "rerank_exit",
            "rewrite": "rewrite",
            "clarify": "clarify",
        },
    )
    g.add_edge("rerank_exit", "generate")
    g.add_edge("rewrite", "agent")
    g.add_edge("generate", END)
    g.add_edge("direct_chat", END)
    g.add_edge("clarify", END)  # clarify-as-terminal：不挂起，正常结束本次请求
    return g.compile()

"""
评测编排器：将检索 + 生成管线接入评测流程，批量跑评测集，输出指标报告。

架构（v2 扩展）：
    EvalDataset (评测集，单轮 / 多轮剧本)
            ↓
    Evaluator.run() → 对每条问题：
        1. 单轮：ask_target.ask(question) → AskResponse
           多轮：逐轮 ask，naive 用 history 串联、agent 用 session_id 串联
        2. 提取 retrieved_files + context
        3. 计算检索指标：compute_retrieval_metrics()
        4. 计算生成指标：
           use_ragas=False → compute_generation_metrics()（手写指标）
           use_ragas=True  → batch_compute_ragas()（RAGAS + 手写混合）
        5. 累计 TokenMeter（零侵入 callback handler 旁路采集）
           多轮场景额外累计语义缓存命中（agent target）
        ↓
    汇总所有问题的指标 + token 总量 + 缓存命中 → EvalReport

新增能力（v2）：
    - 多轮剧本（EvalQuestion.turns）
    - Token 使用统计（含 LLM 网关 token 缓存命中，经 TokenUsageCallbackHandler）
    - 语义缓存命中统计（agent target，经 runner.get_cache_stats）
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion, EvalTurn
from note_assistant.evaluation.retrieval_metrics import compute_retrieval_metrics, RetrievalMetrics
from note_assistant.evaluation.generation_metrics import compute_generation_metrics
from note_assistant.pipeline.rag_chain import AskResponse
from note_assistant.llm.usage import TokenMeter, get_token_handler

logger = logging.getLogger(__name__)

# 支持的评测目标链路
TARGET_KINDS = ("naive", "agent")


def _flatten_retrieval_metrics(rm: RetrievalMetrics) -> Dict[str, float]:
    """
    将 RetrievalMetrics dataclass 展平为 flat dict，方便 _aggregate_metrics 遍历。

    输入：
        RetrievalMetrics(mrr=0.5, recall_at_k={3: 1.0}, precision_at_k={3: 0.67}, ndcg_at_k={3: 0.75})
    输出：
        {"mrr": 0.5, "recall@3": 1.0, "precision@3": 0.67, "ndcg@3": 0.75}
    """
    d: Dict[str, float] = {"mrr": rm.mrr}
    for k, v in rm.recall_at_k.items():
        d[f"recall@{k}"] = v
    for k, v in rm.precision_at_k.items():
        d[f"precision@{k}"] = v
    for k, v in rm.ndcg_at_k.items():
        d[f"ndcg@{k}"] = v
    return d


@dataclass
class SingleEvalResult:
    """单条问题（或多轮剧本中单轮）的评测结果。"""
    question: str
    retrieved_files: List[str] = field(default_factory=list)
    generated_answer: str = ""
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    turn_index: int = -1  # 多轮剧本中的轮次序号（单轮为 -1）
    token_usage: Dict[str, int] = field(default_factory=dict)  # 该轮 token（含 cache 维度）
    # ── v3 新增（agent 链路过程量，naive 链路为 0 / []）──
    iterations: int = 0                 # agentic 循环轮数（Judge 决策次数）
    judge_verdicts: List[str] = field(default_factory=list)  # 每轮 Judge 的 verdict 序列
    # ── v4 新增：分环节耗时(ms)，agent 与 naive 统一口径便于对齐（见 _agent_canonical_stages）──
    stage_timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """完整评测报告（v2 扩展 token / 缓存字段）。"""
    dataset_name: str
    total_questions: int
    avg_elapsed_ms: float = 0.0
    retrieval_metrics_avg: Dict[str, float] = field(default_factory=dict)
    generation_metrics_avg: Dict[str, float] = field(default_factory=dict)
    per_question: List[Dict[str, Any]] = field(default_factory=list)
    # ── v2 新增 ──
    token_usage_total: Optional[Dict[str, Any]] = None  # LLM token 总量（含 cache 维度）
    llm_cache_hit_rate: float = 0.0  # token 级缓存命中率 = cache_read / prompt
    semantic_cache_stats: Optional[Dict[str, Any]] = None  # 问答级语义缓存命中（仅 agent）
    per_conversation: List[Dict[str, Any]] = field(default_factory=list)  # 多轮剧本明细
    # ── v3 新增（agent 链路过程量，naive 链路为 None）──
    iterations_avg: Optional[float] = None  # 平均 agentic 循环轮数
    judge_verdict_distribution: Optional[Dict[str, int]] = None  # verdict 频次分布

    def to_dict(self) -> dict:
        """转为字典，用于 JSON 序列化。"""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """保存为 JSON 文件。"""
        p = Path(path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class Evaluator:
    """
    RAG 管线评测器（支持 naive / agent 两种 target + 多轮剧本 + token/cache 统计）。

    用法：
        # naive（默认，单轮 / 多轮都用 history 串联）
        evaluator = Evaluator(rag_chain, llm=my_llm, target_kind="naive")
        # agent（推荐，多轮用 session_id 串联，自带语义缓存）
        evaluator = Evaluator(None, target_kind="agent")

        report = evaluator.run(dataset)
        report.save("eval_report.json")

    鸭子类型：
        - naive：ask_target 需有 ``ask(question, history=...) -> AskResponse``
        - agent：ask_target 不被使用，内部调 ``runner.ainvoke``
    """

    def __init__(self, ask_target, llm=None, use_ragas: bool = False, target_kind: str = "naive"):
        """
        Args:
            ask_target: naive 时为 RAGChain 实例（或任何有 ask 方法的对象）；
                        agent 时传 None（Evaluator 内部调 runner.ainvoke）。
            llm: 可选的 LLM 实例，用于 faithfulness + answer_relevance
                 需要有 invoke(messages) 方法。
            use_ragas: 是否使用 RAGAS 框架计算生成指标（默认 False，使用手写指标）。
            target_kind: "naive"（默认，向后兼容）| "agent"（推荐，含语义缓存）。
        """
        if target_kind not in TARGET_KINDS:
            raise ValueError(f"target_kind 必须是 {TARGET_KINDS} 之一，收到 {target_kind!r}")
        self.ask_target = ask_target
        self.rag_chain = ask_target  # 兼容旧 run_single() 的 self.rag_chain 引用
        self.llm = llm
        self.use_ragas = use_ragas
        self.target_kind = target_kind

    # ──────────────────────────────────────────────
    # 调用封装（统一 naive / agent 接口）
    # ──────────────────────────────────────────────

    @staticmethod
    def _run_async(coro):
        """在同步上下文里跑协程；已处于事件循环时开新 loop 避免冲突。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    def _ask_one(self, question: str, history: list, session_id: str) -> Tuple[str, List[str], str, int, List[str]]:
        """调一次问答，返回 (answer, retrieved_files, context_text, iterations, judge_verdicts)。

        - naive：``ask_target.ask(question, history=history)``，无 agent 过程量（iterations=0）
        - agent：``runner.ainvoke(question, history=history, session_id=session_id,
          return_contexts=True)``（用完整 chunk 正文作为 context，利于 faithfulness）；
          从轨迹里解析出 agentic 循环轮数与每轮 Judge 判定。
        """
        if self.target_kind == "naive":
            ans: AskResponse = self.ask_target.ask(question, history=history)
            retrieved_files = [s.filepath for s in ans.sources]
            context = " ".join(getattr(s, "preview", "") or "" for s in ans.sources)
            stages = dict(getattr(ans, "timing", {}) or {})
            return ans.answer, retrieved_files, context, 0, [], stages
        else:
            from note_assistant.agent.runner import ainvoke

            result = self._run_async(
                ainvoke(question, history=history, session_id=session_id, return_contexts=True)
            )
            retrieved_files = [s.get("filepath") for s in result.sources]
            context = " ".join(result.contexts)
            iterations, verdicts = self._parse_trajectory(result.trajectory)
            stages = self._agent_canonical_stages(getattr(result, "timing", None))
            return result.answer, retrieved_files, context, iterations, verdicts, stages

    @staticmethod
    def _parse_trajectory(trajectory: Optional[List[dict]]) -> Tuple[int, List[str]]:
        """从 agent 轨迹里提取 agentic 循环轮数与 Judge 判定序列。

        trajectory 中 ``type=="judge"`` 的条目对应每一次 reflect（Judge）决策，
        其数量即检索循环轮数（每轮 tools → reflect 一次）；verdict 字段为
        sufficient / need_rewrite / need_more / give_up / need_clarify。
        """
        judges = [e for e in (trajectory or []) if isinstance(e, dict) and e.get("type") == "judge"]
        iterations = len(judges)
        verdicts = [e.get("verdict") for e in judges if e.get("verdict")]
        return iterations, verdicts

    @staticmethod
    def _meter_snapshot(meter: TokenMeter) -> Dict[str, int]:
        """读取 meter 当前累计快照（用于算每轮 token 差值）。"""
        return {
            "prompt_tokens": meter.prompt_tokens,
            "completion_tokens": meter.completion_tokens,
            "cache_creation_tokens": meter.cache_creation_tokens,
            "cache_read_tokens": meter.cache_read_tokens,
            "total_tokens": meter.total_tokens,
            "llm_calls": meter.llm_calls,
        }

    @staticmethod
    def _agent_canonical_stages(timing) -> Dict[str, float]:
        """把 agent 图节点耗时归并成统一环节口径，便于与 naive 链路对齐比较。

        节点 → 环节映射：
            router / agent / rewrite + 凝练(condense_ms) → 规划决策
            tools                                     → 检索
            graph_expand_node                         → 图扩展
            rerank_loop + rerank_exit                 → 重排
            reflect                                   → 判定
            generate / direct_chat / clarify          → 生成
        naive 链路只产出 检索/重排/生成，其余环节缺省为 0。
        """
        timing = timing or {}
        stages = timing.get("stages", {}) or {}
        condense = timing.get("condense_ms", 0) or 0
        return {
            "规划决策": round(stages.get("router", 0) + stages.get("agent", 0)
                              + stages.get("rewrite", 0) + condense, 1),
            "检索": round(stages.get("tools", 0), 1),
            "图扩展": round(stages.get("graph_expand_node", 0), 1),
            "重排": round(stages.get("rerank_loop", 0) + stages.get("rerank_exit", 0), 1),
            "判定": round(stages.get("reflect", 0), 1),
            "生成": round(stages.get("generate", 0) + stages.get("direct_chat", 0)
                          + stages.get("clarify", 0), 1),
        }

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def run(self, dataset: EvalDataset, k_values: List[int] | None = None) -> EvalReport:
        """
        在评测集上运行完整评测（支持单轮 / 多轮 + token / 缓存统计）。

        Args:
            dataset: 评测数据集
            k_values: 检索指标截断点，默认 [3, 5, 10]

        Returns:
            EvalReport
        """
        from note_assistant.agent.runner import reset_cache, get_cache_stats

        meter = TokenMeter()
        handler = get_token_handler()
        prev_meter = handler.set_meter(meter)  # 绑定评测 meter，记录旧值便于恢复
        is_agent = self.target_kind == "agent"
        if is_agent:
            reset_cache()  # 清跨题 / 跨轮污染，保证命中率口径干净
        try:
            eval_results: List[SingleEvalResult] = []
            per_conversation: List[Dict[str, Any]] = []

            # RAGAS 模式：先收集扁平数据，最后批量调用 RAGAS evaluate()
            if self.use_ragas:
                ragas_questions: List[str] = []
                ragas_answers: List[str] = []
                ragas_contexts: List[List[str]] = []
                ragas_ground_truths: List[str] = []

            for qi, question in enumerate(dataset.questions):
                if question.is_multiturn():
                    conv = self._run_multiturn(question, qi, k_values, eval_results, meter)
                    per_conversation.append(conv)
                else:
                    self._run_single(
                        question, k_values, eval_results, meter,
                        (ragas_questions, ragas_answers, ragas_contexts, ragas_ground_truths)
                        if self.use_ragas else None,
                    )

            # RAGAS 模式：批量评估后填充生成指标（多轮场景暂不纳入 ragas）
            if self.use_ragas and ragas_questions:
                from note_assistant.evaluation.ragas_metrics import batch_compute_ragas

                try:
                    ragas_scores = batch_compute_ragas(
                        questions=ragas_questions,
                        answers=ragas_answers,
                        contexts=ragas_contexts,
                        ground_truths=ragas_ground_truths,
                    )
                    for r, score in zip(eval_results, ragas_scores):
                        r.generation_metrics = score
                except Exception as e:
                    logger.error(f"RAGAS 批量评估失败，报告将只包含检索指标: {e}")

            avg_retrieval, avg_generation = self._aggregate_metrics(eval_results)
            avg_elapsed = (
                sum(r.elapsed_ms for r in eval_results) / len(eval_results)
                if eval_results else 0.0
            )
            # v3：agentic 过程量（迭代轮数 / Judge 判定分布）
            iter_vals = [r.iterations for r in eval_results]
            iterations_avg = (
                sum(iter_vals) / len(iter_vals) if iter_vals else None
            )
            verdict_counter: Dict[str, int] = defaultdict(int)
            for r in eval_results:
                for v in r.judge_verdicts:
                    verdict_counter[v] += 1
            judge_verdict_distribution = (
                dict(verdict_counter) if verdict_counter else None
            )
            return EvalReport(
                dataset_name=dataset.name,
                total_questions=len(dataset.questions),
                avg_elapsed_ms=avg_elapsed,
                retrieval_metrics_avg=avg_retrieval,
                generation_metrics_avg=avg_generation,
                per_question=[asdict(r) for r in eval_results],
                token_usage_total=meter.to_dict(),
                llm_cache_hit_rate=meter.cache_hit_rate(),
                semantic_cache_stats=get_cache_stats() if is_agent else None,
                per_conversation=per_conversation,
                iterations_avg=iterations_avg,
                judge_verdict_distribution=judge_verdict_distribution,
            )
        finally:
            # 恢复 handler 现场（默认 None，零副作用，不影响线上）
            handler.set_meter(prev_meter)

    def _run_single(
        self,
        question: EvalQuestion,
        k_values: List[int] | None,
        eval_results: List[SingleEvalResult],
        meter: TokenMeter,
        ragas: Optional[Tuple[List[str], List[str], List[List[str]], List[str]]],
    ) -> None:
        """评测单轮问题（含 token 累计）。ragas 元组非空时收集 RAGAS 数据。"""
        start = time.time()
        snap0 = self._meter_snapshot(meter)
        try:
            answer, retrieved_files, context, iterations, verdicts, stages = self._ask_one(question.question, [], "")
        except Exception as e:
            logger.error(f"评测失败: {e}")
            answer, retrieved_files, context, iterations, verdicts = "", [], "", 0, []
            stages = {}
        snap1 = self._meter_snapshot(meter)
        elapsed = (time.time() - start) * 1000
        turn_tokens = {k: snap1[k] - snap0[k] for k in snap1}

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_files, question.relevant_files, k_values
        )

        if self.use_ragas:
            generation_metrics_dict: Dict[str, float] = {}
            if ragas is not None:
                ragas[0].append(question.question)
                ragas[1].append(answer)
                ragas[2].append([context] if context else [""])
                ragas[3].append(question.golden_answer)
        else:
            generation_metrics = compute_generation_metrics(
                answer,
                question.golden_answer,
                llm=self.llm,
                context=context,
                question=question.question,
            )
            generation_metrics_dict = generation_metrics.to_dict()

            eval_results.append(
                SingleEvalResult(
                    question=question.question,
                    retrieved_files=retrieved_files,
                    generated_answer=answer,
                    retrieval_metrics=_flatten_retrieval_metrics(retrieval_metrics),
                    generation_metrics=generation_metrics_dict,
                    elapsed_ms=elapsed,
                    turn_index=-1,
                    token_usage=turn_tokens,
                    iterations=iterations,
                    judge_verdicts=verdicts,
                    stage_timings=stages,
                )
            )

    def _run_multiturn(
        self,
        question: EvalQuestion,
        qi: int,
        k_values: List[int] | None,
        eval_results: List[SingleEvalResult],
        meter: TokenMeter,
    ) -> Dict[str, Any]:
        """评测多轮剧本：逐轮串联（naive 用 history、agent 用 session_id），每轮算指标 + token。"""
        session_id = f"eval_conv_{qi}_{uuid.uuid4().hex[:8]}"
        history: List[dict] = []
        turns_out: List[Dict[str, Any]] = []

        for ti, turn in enumerate(question.turns):
            start = time.time()
            snap0 = self._meter_snapshot(meter)
            try:
                answer, retrieved_files, context, iterations, verdicts, stages = self._ask_one(turn.question, history, session_id)
            except Exception as e:
                logger.error(f"多轮评测失败 (conv {qi} turn {ti}): {e}")
                answer, retrieved_files, context, iterations, verdicts = "", [], "", 0, []
                stages = {}
            snap1 = self._meter_snapshot(meter)
            elapsed = (time.time() - start) * 1000
            turn_tokens = {k: snap1[k] - snap0[k] for k in snap1}

            # 每轮检索指标（优先用该轮 golden/relevant，回退首轮）
            golden = turn.golden_answer or question.golden_answer
            relevant = turn.relevant_files or question.relevant_files
            retrieval_metrics = compute_retrieval_metrics(retrieved_files, relevant, k_values)

            if self.use_ragas:
                generation_metrics_dict: Dict[str, float] = {}
            else:
                generation_metrics = compute_generation_metrics(
                    answer, golden, llm=self.llm, context=context, question=turn.question
                )
                generation_metrics_dict = generation_metrics.to_dict()

            eval_results.append(
                SingleEvalResult(
                    question=turn.question,
                    retrieved_files=retrieved_files,
                    generated_answer=answer,
                    retrieval_metrics=_flatten_retrieval_metrics(retrieval_metrics),
                    generation_metrics=generation_metrics_dict,
                    elapsed_ms=elapsed,
                    turn_index=ti,
                    token_usage=turn_tokens,
                    iterations=iterations,
                    judge_verdicts=verdicts,
                    stage_timings=stages,
                )
            )
            # 累积多轮上下文
            history.append({"role": "user", "content": turn.question})
            history.append({"role": "assistant", "content": answer})
            turns_out.append({
                "turn": ti,
                "question": turn.question,
                "answer_len": len(answer),
                "retrieval_metrics": _flatten_retrieval_metrics(retrieval_metrics),
                "generation_metrics": generation_metrics_dict,
                "token_usage": turn_tokens,
            })

        return {
            "question_index": qi,
            "first_question": question.question,
            "turns": turns_out,
        }

    # ──────────────────────────────────────────────
    # 指标聚合
    # ──────────────────────────────────────────────

    def _aggregate_metrics(self, results: List[SingleEvalResult]) -> tuple[Dict[str, float], Dict[str, float]]:
        """
        将多条评测结果的指标汇总为平均值。

        按指标名分组收集所有值，对每组求平均。值为 None 时跳过（LLM 调用失败时）。
        """
        agg_retrieval = defaultdict(list)
        agg_generation = defaultdict(list)

        for r in results:
            for k, v in r.retrieval_metrics.items():
                if v is not None:
                    agg_retrieval[k].append(v)
            for k, v in r.generation_metrics.items():
                if v is not None:
                    agg_generation[k].append(v)

        avg_retrieval = {k: sum(vals) / len(vals) for k, vals in agg_retrieval.items()}
        avg_generation = {
            k: sum(vals) / len(vals) for k, vals in agg_generation.items()
        }

        return avg_retrieval, avg_generation

    def run_single(self, question: str, golden_answer: str, relevant_files: List[str], context: str = "") -> SingleEvalResult:
        """
        评测单条问题（调试用，naive target）。

        和 run() 的逻辑相同，只是不聚合，直接返回 SingleEvalResult。
        暂不支持 use_ragas=True 模式（调试用建议走手写指标，更轻量）。
        """
        from note_assistant.pipeline.rag_chain import AskResponse as _AskResponse

        start = time.time()
        try:
            ans = self.ask_target.ask(question)
        except Exception as e:
            logger.error(f"评测失败: {e}")
            ans = _AskResponse(answer="", sources=[], graph_expansion=0, retrieved=0)

        elapsed = (time.time() - start) * 1000

        retrieved_files = [source.filepath for source in ans.sources]
        context = " ".join(s.preview for s in ans.sources)

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_files, relevant_files
        )

        if self.use_ragas:
            from note_assistant.evaluation.ragas_metrics import batch_compute_ragas
            scores = batch_compute_ragas(
                questions=[question],
                answers=[ans.answer],
                contexts=[[s.preview for s in ans.sources]],
                ground_truths=[golden_answer],
            )
            generation_metrics_dict = scores[0] if scores else {}
        else:
            generation_metrics = compute_generation_metrics(
                ans.answer,
                golden_answer,
                llm=self.llm,
                context=context,
                question=question,
            )
            generation_metrics_dict = generation_metrics.to_dict()

        return SingleEvalResult(
            question=question,
            retrieved_files=retrieved_files,
            generated_answer=ans.answer,
            retrieval_metrics=_flatten_retrieval_metrics(retrieval_metrics),
            generation_metrics=generation_metrics_dict,
            elapsed_ms=elapsed,
        )

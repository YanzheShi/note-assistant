#!/usr/bin/env python3
"""
命令行评测入口（v2：多轮 + Token 统计 + 语义缓存命中）。

用法：
    # 快速评测（内置 10 条，默认 agent 链路，含语义缓存 + token 统计）
    python scripts/run_eval.py

    # 指定链路
    python scripts/run_eval.py --target naive
    python scripts/run_eval.py --target agent

    # 自定义数据集（支持多轮剧本：每条问题含 turns 列表）
    python scripts/run_eval.py --dataset my_eval.json

    # 指定 k 值
    python scripts/run_eval.py --k 3 5 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保 src 在 path 里
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_rag_chain():
    """构建 RAGChain 实例（按需调整）。"""
    from note_assistant.pipeline.rag_chain import RAGChain
    from note_assistant.retrieval.hybrid import HybridRetriever
    from note_assistant.retrieval.reranker import LocalReranker
    from note_assistant.generation.generator import Generator

    retriever = HybridRetriever()
    reranker = LocalReranker()
    try:
        generator = Generator()
    except Exception as e:
        logger.warning(f"Generator 初始化失败（答案将为空）: {e}")
        generator = None
    return RAGChain(hybrid_retriever=retriever, reranker=reranker, generator=generator)


def _print_token_report(report) -> None:
    """打印 Token 使用汇总（含 LLM 网关 token 缓存命中）。"""
    if not report.token_usage_total:
        return
    tu = report.token_usage_total
    logger.info("")
    logger.info("--- Token 使用（含 LLM 网关 token 缓存命中）---")
    logger.info(f"  prompt_tokens:      {tu['prompt_tokens']}")
    logger.info(f"  completion_tokens:  {tu['completion_tokens']}")
    logger.info(f"  cache_creation:     {tu['cache_creation_tokens']}")
    logger.info(f"  cache_read:         {tu['cache_read_tokens']}")
    logger.info(f"  total_tokens:       {tu['total_tokens']}")
    logger.info(f"  llm_calls:          {tu['llm_calls']}")
    logger.info(f"  llm_cache_hit_rate: {report.llm_cache_hit_rate}")


def _print_cache_report(report) -> None:
    """打印语义缓存命中统计（仅 agent 链路）。"""
    if not report.semantic_cache_stats:
        return
    cs = report.semantic_cache_stats
    logger.info("")
    logger.info("--- 语义缓存命中（问答级）---")
    logger.info(f"  enabled:  {cs['enabled']}")
    logger.info(f"  hits:     {cs['hits']}")
    logger.info(f"  misses:   {cs['misses']}")
    logger.info(f"  hit_rate: {cs['hit_rate']}")


def main():
    parser = argparse.ArgumentParser(description="RAG 评测工具（多轮 + Token + 缓存命中）")
    parser.add_argument(
        "--dataset", "-d",
        default="builtin",
        help="评测数据集路径（builtin 或用 JSON 文件路径）",
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/eval_report.json",
        help="评测报告输出路径",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="检索指标截断点（默认 3 5 10）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出每条问题的指标",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="使用 RAGAS 框架评估生成指标（默认使用手写指标）",
    )
    parser.add_argument(
        "--target",
        choices=["naive", "agent"],
        default="agent",
        help="评测目标链路：naive（RAGChain，history 串联多轮）/ agent（默认，含语义缓存 + session 串联）",
    )
    args = parser.parse_args()

    # 加载数据集
    from note_assistant.evaluation.eval_dataset import get_builtin_dataset, load_eval_dataset

    if args.dataset == "builtin":
        dataset = get_builtin_dataset()
    else:
        dataset = load_eval_dataset(args.dataset)

    logger.info(f"评测集: {dataset.name}, {dataset.size} 条问题")

    # 构建评测器（按 target 选择链路）
    from note_assistant.evaluation.evaluator import Evaluator

    if args.target == "naive":
        logger.info("构建 RAG 管线（naive）...")
        try:
            rag_chain = build_rag_chain()
        except Exception as e:
            logger.error(f"构建 RAG 管线失败: {e}")
            logger.info("使用 mock 管线进行演示...")
            from unittest.mock import MagicMock
            rag_chain = MagicMock()
            rag_chain.ask.return_value = MagicMock(answer="演示答案（mock）", sources=[])
        evaluator = Evaluator(rag_chain, use_ragas=args.ragas, target_kind="naive")
    else:
        # agent target：Evaluator 内部调 runner.ainvoke，不在此构建 RAG 管线
        logger.info("使用 agent 链路（含语义缓存 + session 串联）...")
        evaluator = Evaluator(None, use_ragas=args.ragas, target_kind="agent")

    # 运行评测
    logger.info("开始评测...")
    report = evaluator.run(dataset, k_values=args.k)

    # 输出报告
    logger.info("=" * 60)
    logger.info(f"评测报告: {report.dataset_name} ({report.total_questions} 条)")
    logger.info(f"平均耗时: {report.avg_elapsed_ms:.0f} ms/条")
    logger.info("")
    logger.info("--- 检索指标 ---")
    for k, v in sorted(report.retrieval_metrics_avg.items()):
        logger.info(f"  {k}: {v:.4f}")
    logger.info("")
    logger.info("--- 生成指标 ---")
    for k, v in sorted(report.generation_metrics_avg.items()):
        logger.info(f"  {k}: {v:.4f}")
    logger.info("")

    _print_token_report(report)
    _print_cache_report(report)

    if args.verbose:
        logger.info("--- 逐条详情 ---")
        for i, pq in enumerate(report.per_question):
            logger.info(f"  [{i+1}] {pq['question'][:50]}...")
            logger.info(f"      检索: {pq['retrieval_metrics']}")
            logger.info(f"      生成: {pq['generation_metrics']}")
            if pq.get("token_usage"):
                logger.info(f"      token: {pq['token_usage']}")

    # 保存报告
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)
    logger.info(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()

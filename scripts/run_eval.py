#!/usr/bin/env python3
"""
命令行评测入口。

用法：
    # 快速评测（内置 10 条）
    python scripts/run_eval.py
    
    # 自定义数据集
    python scripts/run_eval.py --dataset my_eval.json
    
    # 指定 k 值
    python scripts/run_eval.py --k 3 5 10
"""

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
    
    retriever = HybridRetriever()
    reranker = LocalReranker()
    return RAGChain(hybrid_retriever=retriever, reranker=reranker)


def main():
    parser = argparse.ArgumentParser(description="RAG 评测工具")
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
    args = parser.parse_args()

    # 加载数据集
    from note_assistant.evaluation.eval_dataset import get_builtin_dataset, load_eval_dataset

    if args.dataset == "builtin":
        dataset = get_builtin_dataset()
    else:
        dataset = load_eval_dataset(args.dataset)

    logger.info(f"评测集: {dataset.name}, {dataset.size} 条问题")

    # 构建管线
    logger.info("构建 RAG 管线...")
    try:
        rag_chain = build_rag_chain()
    except Exception as e:
        logger.error(f"构建 RAG 管线失败: {e}")
        logger.info("使用 mock 管线进行演示...")
        from unittest.mock import MagicMock
        rag_chain = MagicMock()
        rag_chain.ask.return_value = {
            "answer": "演示答案（mock）",
            "sources": [{"filepath": "mock.md", "preview": "...", "score": 0.5}],
        }

    # 运行评测
    logger.info("开始评测...")
    from note_assistant.evaluation.evaluator import Evaluator
    
    evaluator = Evaluator(rag_chain)
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

    if args.verbose:
        logger.info("--- 逐条详情 ---")
        for i, pq in enumerate(report.per_question):
            logger.info(f"  [{i+1}] {pq['question'][:50]}...")
            logger.info(f"      检索: {pq['retrieval_metrics']}")
            logger.info(f"      生成: {pq['generation_metrics']}")

    # 保存报告
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.save(args.output)
    logger.info(f"\n报告已保存到: {args.output}")


if __name__ == "__main__":
    main()

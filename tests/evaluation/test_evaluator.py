"""
评测编排器测试：Evaluator mock 管线。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion
from note_assistant.evaluation.evaluator import Evaluator
from note_assistant.pipeline.rag_chain import AskResponse, SourceInfo


def _make_mock_rag_chain():
    """构造一个 mock 的 RAGChain，返回真正的 AskResponse。"""
    chain = MagicMock()
    chain.ask.return_value = AskResponse(
        answer="这是测试答案",
        sources=[
            SourceInfo(type="direct", filepath="test.md", heading="标题", preview="预览", score=0.9),
            SourceInfo(type="direct", filepath="other.md", heading="其他", preview="预览", score=0.7),
        ],
        graph_expansion=0,
        retrieved=2,
    )
    return chain


class TestEvaluatorRun:
    def test_run_returns_report(self):
        """run 应返回 EvalReport。"""
        from note_assistant.evaluation.evaluator import EvalReport
        mock_chain = _make_mock_rag_chain()
        evaluator = Evaluator(mock_chain)
        
        dataset = EvalDataset(
            name="test_ds",
            questions=[
                EvalQuestion(question="q1", golden_answer="a1", relevant_files=["test.md"]),
            ],
        )
        
        report = evaluator.run(dataset)
        assert isinstance(report, EvalReport)
        assert report.dataset_name == "test_ds"
        assert report.total_questions == 1

    def test_retrieval_metrics_computed(self):
        """检索指标应被正确计算。"""
        mock_chain = _make_mock_rag_chain()
        evaluator = Evaluator(mock_chain)
        
        dataset = EvalDataset(
            name="test",
            questions=[
                EvalQuestion(question="q1", golden_answer="a1", relevant_files=["test.md"]),
            ],
        )
        
        report = evaluator.run(dataset)
        # test.md 在 retrieved_files 中，recall 应该 > 0
        assert report.retrieval_metrics_avg.get("mrr", 0) > 0
        assert report.retrieval_metrics_avg.get("recall@3", 0) > 0

    def test_generation_metrics_computed(self):
        """生成指标应被正确计算。"""
        mock_chain = _make_mock_rag_chain()
        evaluator = Evaluator(mock_chain)
        
        dataset = EvalDataset(
            name="test",
            questions=[
                EvalQuestion(question="q1", golden_answer="a1", relevant_files=["test.md"]),
            ],
        )
        
        report = evaluator.run(dataset)
        assert "rouge_l" in report.generation_metrics_avg
        assert "bleu_1" in report.generation_metrics_avg

    def test_multiple_questions(self):
        """多条问题应正常处理。"""
        mock_chain = _make_mock_rag_chain()
        evaluator = Evaluator(mock_chain)
        
        dataset = EvalDataset(
            name="multi",
            questions=[
                EvalQuestion(question="q1", golden_answer="a1", relevant_files=["test.md"]),
                EvalQuestion(question="q2", golden_answer="a2", relevant_files=["other.md"]),
                EvalQuestion(question="q3", golden_answer="a3", relevant_files=["test.md", "other.md"]),
            ],
        )
        
        report = evaluator.run(dataset)
        assert report.total_questions == 3
        assert len(report.per_question) == 3

    def test_error_handling(self):
        """管线报错时应记录错误而非崩溃。"""
        from unittest.mock import MagicMock
        bad_chain = MagicMock()
        bad_chain.ask.side_effect = RuntimeError("boom")
        
        evaluator = Evaluator(bad_chain)
        dataset = EvalDataset(
            name="err",
            questions=[EvalQuestion(question="q1", golden_answer="a1")],
        )
        
        report = evaluator.run(dataset)
        assert report.total_questions == 1
        # 出错的答案应为空
        assert report.per_question[0]["generated_answer"] == ""

    def test_run_single(self):
        """单条评测应返回 SingleEvalResult。"""
        from note_assistant.evaluation.evaluator import SingleEvalResult
        mock_chain = _make_mock_rag_chain()
        evaluator = Evaluator(mock_chain)
        
        result = evaluator.run_single("test", "gold", ["test.md"])
        assert isinstance(result, SingleEvalResult)
        assert result.question == "test"
        assert "test.md" in result.retrieved_files

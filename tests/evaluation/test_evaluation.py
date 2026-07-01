"""
评测模块测试：eval_dataset、retrieval_metrics、generation_metrics。
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from note_assistant.evaluation.eval_dataset import EvalDataset, EvalQuestion, get_builtin_dataset
from note_assistant.evaluation.retrieval_metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k, compute_retrieval_metrics, RetrievalMetrics
from note_assistant.evaluation.generation_metrics import rouge_l, bleu_1, bleu_4, semantic_similarity, compute_generation_metrics, GenerationMetrics


# ================================================================
# EvalDataset / EvalQuestion
# ================================================================

class TestEvalQuestion:
    def test_to_dict_and_from_dict(self):
        q = EvalQuestion(
            question="test",
            golden_answer="answer",
            relevant_files=["a.md", "b.md"],
            relevant_chunk_ids=["c1"],
        )
        d = q.to_dict()
        q2 = EvalQuestion.from_dict(d)
        assert q2.question == "test"
        assert q2.golden_answer == "answer"
        assert q2.relevant_files == ["a.md", "b.md"]
        assert q2.relevant_chunk_ids == ["c1"]

    def test_empty_fields(self):
        q = EvalQuestion(question="x", golden_answer="y")
        assert q.relevant_files == []
        assert q.relevant_chunk_ids == []


class TestEvalDataset:
    def test_builtin_dataset(self):
        ds = get_builtin_dataset()
        assert ds.name == "builtin_small"
        assert ds.size == 10
        assert isinstance(ds.questions[0], EvalQuestion)

    def test_save_load(self):
        ds = EvalDataset(
            name="test",
            questions=[
                EvalQuestion(question="q1", golden_answer="a1"),
                EvalQuestion(question="q2", golden_answer="a2"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_eval.json"
            ds.save(path)
            ds2 = EvalDataset.load(path)
            assert ds2.name == "test"
            assert ds2.size == 2
            assert ds2.questions[0].question == "q1"

    def test_subset(self):
        ds = get_builtin_dataset()
        sub = ds.subset(3)
        assert sub.size == 3
        assert "sample_3" in sub.name


# ================================================================
# Retrieval Metrics
# ================================================================

class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a.md", "b.md", "c.md"]
        relevant = {"a.md", "b.md"}
        assert recall_at_k(retrieved, relevant) == 1.0

    def test_partial_recall(self):
        retrieved = ["a.md"]
        relevant = {"a.md", "b.md"}
        assert recall_at_k(retrieved, relevant) == 0.5

    def test_no_hit(self):
        retrieved = ["x.md"]
        relevant = {"a.md", "b.md"}
        assert recall_at_k(retrieved, relevant) == 0.0

    def test_k_truncation(self):
        retrieved = ["z.md", "a.md", "b.md"]
        relevant = {"a.md", "b.md"}
        # k=1: only z.md, no hit
        assert recall_at_k(retrieved, relevant, k=1) == 0.0
        # k=2: z.md + a.md, 1 hit out of 2
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k([], set()) == 1.0
        assert recall_at_k(["a.md"], set()) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a.md", "b.md"]
        relevant = {"a.md", "b.md"}
        assert precision_at_k(retrieved, relevant) == 1.0

    def test_partial_precision(self):
        retrieved = ["a.md", "x.md"]
        relevant = {"a.md"}
        assert precision_at_k(retrieved, relevant) == 0.5

    def test_k_truncation(self):
        retrieved = ["x.md", "a.md", "b.md"]
        relevant = {"a.md", "b.md"}
        # k=1: only x.md
        assert precision_at_k(retrieved, relevant, k=1) == 0.0

    def test_k_hit(self):
        """k=1 且第一个命中时，分母应为 K=1 而不是 len(retrieved)。"""
        retrieved = ["a.md", "x.md"]
        relevant = {"a.md"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0


class TestMRR:
    def test_first_hit(self):
        retrieved = ["a.md", "b.md"]
        relevant = {"a.md"}
        assert mrr(retrieved, relevant) == 1.0  # 1/1

    def test_second_hit(self):
        retrieved = ["x.md", "a.md"]
        relevant = {"a.md"}
        assert mrr(retrieved, relevant) == 0.5  # 1/2

    def test_no_hit(self):
        retrieved = ["x.md", "y.md"]
        relevant = {"a.md"}
        assert mrr(retrieved, relevant) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a.md", "b.md"]
        relevant = {"a.md", "b.md"}
        assert ndcg_at_k(retrieved, relevant) > 0.9  # near perfect

    def test_no_relevant(self):
        retrieved = ["x.md", "y.md"]
        relevant = set()
        assert ndcg_at_k(retrieved, relevant) == 0.0

    def test_empty(self):
        assert ndcg_at_k([], {"a.md"}) == 0.0


class TestComputeRetrievalMetrics:
    def test_all_metrics_returned(self):
        retrieved = ["a.md", "b.md", "c.md"]
        relevant = {"a.md", "b.md"}
        m = compute_retrieval_metrics(retrieved, relevant, k_values=[3, 5])
        assert isinstance(m, RetrievalMetrics)
        assert m.mrr > 0.0
        assert 3 in m.recall_at_k
        assert 3 in m.precision_at_k
        assert 3 in m.ndcg_at_k
        assert 0 <= m.mrr <= 1.0
        assert 0 <= m.recall_at_k[3] <= 1.0


# ================================================================
# Generation Metrics
# ================================================================

class TestRougeL:
    def test_identical(self):
        assert rouge_l("hello world", "hello world") == 1.0

    def test_partial_match(self):
        r = rouge_l("the cat sat", "the cat")
        assert r > 0.0
        assert r < 1.0

    def test_no_match(self):
        # apple/banana share "a" char → small but non-zero
        assert rouge_l("apple", "banana") < 0.3
        assert rouge_l("xyz", "abc") < 0.3

    def test_empty(self):
        assert rouge_l("", "hello") == 0.0
        assert rouge_l("hello", "") == 0.0

    def test_chinese(self):
        assert rouge_l("你好世界", "你好世界") == 1.0
        r = rouge_l("你好世界", "你好")
        assert r > 0.0


class TestBleu:
    def test_bleu_1_identical(self):
        assert bleu_1("hello world", "hello world") > 0.9

    def test_bleu_4_exact(self):
        assert bleu_4("the quick brown fox", "the quick brown fox") > 0.9

    def test_bleu_partial(self):
        b = bleu_1("hello world foo", "hello world")
        assert b > 0.0

    def test_bleu_no_match(self):
        assert bleu_1("apple", "banana") < 0.1


class TestSemanticSimilarity:
    def test_identical(self):
        sim = semantic_similarity("hello world", "hello world")
        assert sim > 0.5  # Jaccard fallback

    def test_different(self):
        sim = semantic_similarity("apple", "banana")
        assert sim < 0.5

    def test_with_embedder(self):
        def mock_embedder(text):
            return [1.0] * 10  # dummy embedding
        sim = semantic_similarity("test", "test", embedder=mock_embedder)
        assert sim > 0.9


class TestComputeGenerationMetrics:
    def test_all_metrics_returned(self):
        m = compute_generation_metrics("hello world", "hello world")
        assert isinstance(m, GenerationMetrics)
        assert hasattr(m, "rouge_l")
        assert hasattr(m, "bleu_1")
        assert hasattr(m, "bleu_4")
        assert hasattr(m, "semantic_similarity")
        assert 0 <= m.rouge_l <= 1.0
        assert 0 <= m.bleu_1 <= 1.0
        assert 0 <= m.bleu_4 <= 1.0
        assert 0 <= m.semantic_similarity <= 1.0

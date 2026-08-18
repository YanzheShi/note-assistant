"""
评测数据集：问题 + 金标准答案 + 金标准来源。

格式：
    EvalQuestion(
        question="...",           # 用户问题
        golden_answer="...",      # 人工标注的标准答案
        relevant_files=["..."],   # 应该命中的文件（用于检索评估）
        relevant_chunk_ids=[""],  # 应该命中的 chunk ID（可选，用于细粒度评估）
    )

用法：
    # 内置小数据集（10 条，快速验证）
    from note_assistant.evaluation.eval_dataset import get_builtin_dataset
    
    dataset = get_builtin_dataset()
    
    # 或从 JSON 文件加载
    from note_assistant.evaluation.eval_dataset import load_eval_dataset
    
    dataset = load_eval_dataset("my_eval.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalTurn:
    """多轮剧本中的单轮（字段同单轮 EvalQuestion，但作为独立一轮存在）。

    多轮评测时，EvalQuestion.turns 是一串 EvalTurn，逐轮串联（naive 用 history、
    agent 用 session_id），每轮独立算检索/生成指标与 token 用量。
    """

    question: str
    golden_answer: str = ""
    relevant_files: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class EvalQuestion:
    """单条评测样本；可含多轮剧本（turns）。

    ``turns=None`` 退化为现有单轮（向后兼容，内置 10 条数据集照常工作）。
    """

    question: str
    golden_answer: str = ""
    relevant_files: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    turns: Optional[List[EvalTurn]] = None

    def is_multiturn(self) -> bool:
        """是否为多轮剧本（turns 非空）。"""
        return bool(self.turns)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuestion:
        turns = d.get("turns")
        if turns is not None:
            turns = [EvalTurn(**t) for t in turns]
        return cls(
            question=d["question"],
            golden_answer=d.get("golden_answer", ""),
            relevant_files=d.get("relevant_files", []),
            relevant_chunk_ids=d.get("relevant_chunk_ids", []),
            turns=turns,
        )


@dataclass
class EvalDataset:
    """评测数据集容器。"""
    name: str
    questions: List[EvalQuestion]

    @property
    def size(self) -> int:
        return len(self.questions)

    def save(self, path: str | Path) -> None:
        """保存为 JSON。"""
        p = Path(path)
        data = {
            "name": self.name,
            "questions": [q.to_dict() for q in self.questions],
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        """从 JSON 加载。"""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        questions = [EvalQuestion.from_dict(q) for q in data["questions"]]
        return cls(name=data["name"], questions=questions)

    def subset(self, n: int) -> EvalDataset:
        """取前 n 条。"""
        return EvalDataset(name=f"{self.name}_sample_{n}", questions=self.questions[:n])


# ──────────────────────────────────────────────────────────────
# 内置小数据集（10 条，覆盖不同类型问题）
# ──────────────────────────────────────────────────────────────

def get_builtin_dataset() -> EvalDataset:
    """
    返回内置评测集（10 条）。
    
    这些问题的答案可以从 Obsidian 笔记中找到，适合快速验证管线。
    用户可以根据自己的笔记内容修改或扩展。
    """
    questions = [
        EvalQuestion(
            question="什么是 RAG？",
            golden_answer="RAG 是 Retrieval-Augmented Generation 的缩写，即检索增强生成。它通过先从知识库检索相关文档，再将检索结果与用户问题一起交给 LLM 生成回答，解决了大语言模型知识截止和幻觉问题。",
            relevant_files=["08-RAG基础概念.md", "RAG常见方式详解.md", "12-Naive RAG实战.md"],
        ),
        EvalQuestion(
            question="BGE-M3 模型有什么特点？",
            golden_answer="BGE-M3 是一个多语言、多粒度、多功能的 embedding 模型。特点是支持多语言（中英文）、多粒度（单词/句子/文档）、多功能（密集/稀疏/多向量检索），embed_dim=1024。",
            relevant_files=["06-向量嵌入Embedding.md", "11-向量化与相似度计算.md"],
        ),
        EvalQuestion(
            question="BM25 和向量检索有什么区别？",
            golden_answer="BM25 是词频统计的稀疏检索，不依赖训练数据，擅长精确关键词匹配；向量检索基于语义 embedding，擅长语义相似但措辞不同的匹配。两者互补，混合检索（hybrid）结合二者优势。",
            relevant_files=["Day2-混合检索与Reranker.md", "11-向量化与相似度计算.md"],
        ),
        EvalQuestion(
            question="交叉编码器和双塔编码器的区别是什么？",
            golden_answer="双塔编码器（Bi-Encoder）分别编码 query 和 doc，预计算 doc embedding，速度快但忽略了 query-doc 交互；交叉编码器（Cross-Encoder）将 query 和 doc 拼接一起输入模型，能捕捉细粒度交互，精度高但速度慢。通常双塔做召回，交叉做精排。",
            relevant_files=["Day2-混合检索与Reranker.md", "大语言模型三大架构深度解析 & BERT原理全解.md"],
        ),
        EvalQuestion(
            question="什么是 chunk 切分？为什么需要？",
            golden_answer="Chunk 切分是将长文档切成小块的过程。目的是让每个 chunk 大小适中（如 800 token），既能包含足够语义信息，又不会超出模型上下文窗口。同时 chunk 级别的检索比全文检索更精确。",
            relevant_files=["10-文档分块策略.md", "切分策略分析与对比.md"],
        ),
        EvalQuestion(
            question="混合检索的 alpha 参数怎么调？",
            golden_answer="alpha 是 dense 权重，bm25_weight 是 sparse 权重，alpha = 1 - bm25_weight。默认 dense=0.7, sparse=0.3。调参方法：在评测集上网格搜索 alpha ∈ [0.3, 0.5, 0.7, 0.9]，看 recall@k 和 MRR 哪个最高。",
            relevant_files=["Day2-混合检索与Reranker.md"],
        ),
        EvalQuestion(
            question="Query Rewrite 的作用是什么？",
            golden_answer="Query Rewrite 对用户原始问题进行改写，包括：扩写（补充缩写/简称）、分句拆分、同义替换。目的是提高检索召回率——用户的问题往往表述不完整，改写后更容易命中相关文档。",
            relevant_files=["Day2-混合检索与Reranker.md", "LangChain RAG优化实践.md"],
        ),
        EvalQuestion(
            question="GraphRAG 和传统 RAG 的区别？",
            golden_answer="传统 RAG 是扁平检索，每个 chunk 独立，检索结果是孤立的。GraphRAG 利用文档间的显式关联（如 wikilink）构建知识图谱，检索命中后可以沿图扩展（BFS 1-hop），召回相关但未直接命中的文档，解决'关联问题'。",
            relevant_files=["Graph RAG.md", "Day3-双链图与增量索引.md"],
        ),
        EvalQuestion(
            question="增量索引怎么判断文件是否需要重索引？",
            golden_answer="两级检查：1）mtime 比较（O(1) 文件系统元数据），2）sha256 内容哈希（O(n) 但内容级保险）。mtime 变了或 sha256 变了都需要重索引。新文件（sync.db 无记录）也需要索引。",
            relevant_files=["Day3-双链图与增量索引.md"],
        ),
        EvalQuestion(
            question="为什么 RAG 系统需要评测？",
            golden_answer="RAG 系统涉及多个组件（检索、rerank、生成），每个组件的参数调整都可能影响最终效果。没有评测就无法量化改进。核心指标：检索侧（recall@k、MRR、nDCG）和生成侧（ROUGE、BLEU、语义相似度）。",
            relevant_files=["RAG 评估.md", "Day4-评测体系建设.md"],
        ),
    ]
    return EvalDataset(name="builtin_small", questions=questions)


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """从 JSON 文件加载评测集。"""
    return EvalDataset.load(path)

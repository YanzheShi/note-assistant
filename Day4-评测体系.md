# Day4 - 评测体系 (Evaluation Framework)

> 日期：2026-06-30
> 状态：✅ 完成

## 一、概述

RAG 系统涉及多个组件（检索、rerank、生成），每个组件的参数调整都可能影响最终效果。没有评测就无法量化改进。

Day4 构建了完整的评测体系，覆盖：
- **检索指标**：Recall@K、Precision@K、MRR、nDCG@K
- **生成指标**：ROUGE-L、BLEU-1/4、语义相似度
- **评测编排器**：批量跑评测集，输出结构化报告
- **内置评测集**：10 条示例问题，开箱即用

## 二、文件清单

| 文件 | 说明 |
|------|------|
| `src/note_assistant/evaluation/__init__.py` | 模块入口，导出所有公开 API |
| `src/note_assistant/evaluation/eval_dataset.py` | 评测数据集（EvalQuestion + EvalDataset） |
| `src/note_assistant/evaluation/retrieval_metrics.py` | 检索指标实现 |
| `src/note_assistant/evaluation/generation_metrics.py` | 生成指标实现 |
| `src/note_assistant/evaluation/evaluator.py` | 评测编排器（Evaluator） |
| `scripts/run_eval.py` | 命令行评测入口 |
| `tests/evaluation/test_evaluation.py` | 指标单元测试 |
| `tests/evaluation/test_evaluator.py` | 编排器单元测试 |

## 三、核心设计

### 3.1 评测数据集

```python
from note_assistant.evaluation import EvalDataset, EvalQuestion, get_builtin_dataset

# 内置 10 条
dataset = get_builtin_dataset()

# 或自定义
q = EvalQuestion(
    question="什么是 RAG？",
    golden_answer="RAG 是检索增强生成...",
    relevant_files=["RAG 概念.md"],
)
dataset = EvalDataset(name="my_eval", questions=[q])

# 保存/加载
dataset.save("my_eval.json")
dataset2 = EvalDataset.load("my_eval.json")
```

### 3.2 检索指标

```python
from note_assistant.evaluation import compute_retrieval_metrics

retrieved = ["a.md", "b.md", "c.md"]
relevant = {"a.md", "b.md"}
metrics = compute_retrieval_metrics(retrieved, relevant, k_values=[3, 5, 10])
# {"recall@3": 1.0, "precision@3": 0.67, "mrr": 1.0, "ndcg@3": 0.83, ...}
```

### 3.3 生成指标

```python
from note_assistant.evaluation import compute_generation_metrics

candidate = "RAG 是检索增强生成，通过检索知识库提升 LLM 回答准确性。"
reference = "RAG 全称 Retrieval-Augmented Generation，是一种结合检索和大模型的架构。"
metrics = compute_generation_metrics(candidate, reference)
# {"rouge_l": 0.45, "bleu_1": 0.38, "bleu_4": 0.12, "semantic_similarity": 0.82}
```

### 3.4 完整评测

```python
from note_assistant.evaluation import Evaluator, get_builtin_dataset

evaluator = Evaluator(rag_chain)  # rag_chain = RAGChain 实例
report = evaluator.run(get_builtin_dataset())

print(report.retrieval_metrics_avg)   # 平均检索指标
print(report.generation_metrics_avg)  # 平均生成指标
print(report.per_question)            # 逐条详情

report.save("eval_report.json")       # 保存报告
```

## 四、指标详解

### 4.1 检索指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **Recall@K** | \|Ret@K ∩ Rel\| / \|Rel\| | 相关文档中有多少被召回 |
| **Precision@K** | \|Ret@K ∩ Rel\| / K | 检索结果中有多少是相关的 |
| **MRR** | 1 / rank_of_first_relevant | 第一个相关文档的排名 |
| **nDCG@K** | DCG@K / IDCG@K | 考虑排名的加权相关度 |

### 4.2 生成指标

| 指标 | 说明 |
|------|------|
| **ROUGE-L** | 基于最长公共子序列的 F1，捕捉长程匹配 |
| **BLEU-1/4** | N-gram 精确度（带 brevity penalty） |
| **语义相似度** | Sentence embedding 余弦相似度（fallback: Jaccard） |

## 五、面试问答

### Q: 为什么 RAG 系统需要评测？

A: RAG 是多组件串联系统，每个组件的参数调整（alpha、top_k、chunk_size）都会影响最终效果。没有评测就无法知道哪个调整是正向的。评测提供量化基准，让优化有据可依。

### Q: 检索指标和生成指标的区别？

A: 检索指标关注"是否找到了正确的文档"（recall、precision、MRR），生成指标关注"回答是否准确"（ROUGE、BLEU、语义相似度）。两者互补：检索好但生成差 → prompt 问题；检索差但生成好 → LLM 有幻觉。

### Q: 为什么 ROUGE 和 BLEU 不够，还需要语义相似度？

A: ROUGE/BLEU 是词面匹配，无法捕捉语义等价。例如"RAG 是检索增强生成"和"RAG 全称 Retrieval-Augmented Generation"词面不同但语义相近。语义相似度用 embedding 余弦衡量语义接近程度。

### Q: nDCG 和 Recall 有什么区别？

A: Recall 只看"召回了多少"，不管排名；nDCG 考虑排名位置——排在第 1 的相关文档比排在第 10 的贡献更大。nDCG 更适合需要排序的场景（如搜索引擎）。

### Q: 评测集怎么构建？

A: 两种方法：1）人工标注：针对笔记内容编写问题+标准答案+相关文件；2）自动构建：从笔记中提取关键段落作为 golden_answer，随机选问题。内置 10 条是方法 1 的示例，用户可以扩展。

### Q: 如何判断评测结果好不好？

A: 没有绝对标准，要看基线对比。比如 alpha=0.7 时 recall@5=0.6，调到 0.5 后 recall@5=0.75，说明 0.5 更好。关键是**控制变量、横向对比**。

## 六、技术实现细节

### 6.1 ROUGE-L 实现

采用 LCS（最长公共子序列）算法，两行 DP 优化空间复杂度：

```python
# 核心：O(m*n) 时间，O(n) 空间
prev = [0] * (n + 1)
curr = [0] * (n + 1)
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            curr[j] = prev[j-1] + 1
        else:
            curr[j] = max(prev[j], curr[j-1])
```

### 6.2 BLEU 实现

带 brevity penalty 的 N-gram 精确度：
- 截断计数（clipped count）防止重复刷分
- BP = exp(1 - ref/cand) 当 cand < ref 时惩罚

### 6.3 语义相似度

优先使用 embedding 模型（如 BGE-M3）计算余弦相似度；若无 embedder，回退到 Jaccard 词重叠相似度。

## 七、评测报告示例

```
评测报告: builtin_small (10 条)
平均耗时: 2340 ms/条

--- 检索指标 ---
  mrr: 0.7200
  ndcg@3: 0.5800
  ndcg@5: 0.6100
  ndcg@10: 0.6300
  precision@3: 0.4000
  precision@5: 0.3200
  precision@10: 0.2000
  recall@3: 0.5200
  recall@5: 0.6800
  recall@10: 0.8500

--- 生成指标 ---
  bleu_1: 0.3500
  bleu_4: 0.0800
  rouge_l: 0.4200
  semantic_similarity: 0.7800
```

## 八、后续优化方向

1. **自动评测集生成**：从笔记中自动抽取问题（LLM 辅助）
2. **人工评分**：增加人工打分接口（1-5 分）
3. **A/B 对比**：支持两组指标对比（如 alpha=0.7 vs 0.5）
4. **可视化**：生成 HTML 报告（含图表）

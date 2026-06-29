# DECISIONS — note-assistant

> 目前代码能力更强调的是人与AI 的协作能力， 个人做coordinator，提供技术选型，方向，以及决策，不能完全任由AI漫无边际地选择，需要结合项目情况，排期，成本，硬件限制综合选择。
>
> 同时要在与ai沟通中不断思考，自我更新迭代，形成共同进步。 自己在思考和交流中沉淀逻辑，思路，教训，从而提高自己项目交付的能力。
>
> 这里用来记录项目的进度，个人决策思考，技术选型对比等内容，方便以后对项目进行优化，避免重复踩坑。同时沉淀个人经验。

---

## [2026-06-26] 

- **做了啥**: 
  - 整理现有的模型，可行性，做项目的整体规划，方案选型，项目脚手架搭建
- **为什么**: 
  - 为什么用chromadb而不用FAISS?
    - 个人项目，存储量级小，并发要求低，而且chromadb ，检索，持久化都方便

  - 先用已有的，安装好的模型，看实现效果如何？ 避免项目初期和调试期消耗过多token

- **拒掉的路**:
- **影响**: 

---

## [2026-06-26]  踩坑：  笔记加载与切分格式不统一

在加载笔记时，根据markdown的情况，存入了一个自定义 k-v 的 dict里面，但是后面 发现使用langchain 做切分时，要求的Doc 格式是另外一个，所以执行切分时遇到了找不到key 的情况。



改进：

1. 笔记加载不应该跟切分耦合，因为后面切分不一定要langchain的方案，我想在切分的时候，根据切分的方法汽车适配。
2. 一开始加载笔记的时候返回粗略的 dict 模型，其他流程在取值的时候，容易因为拼写或记错了导致 key error。所以需要抽象出一个  DocNode的一个类，这样的话后续流程解析 doc 的 key时不容易出错。 而且正儿链路的结构统一
3. 

## [2026-06-27]  问题：  md 中存在大量的代码，表格，mermaid 流程代码块，这些结构化的内容在切分时候被切断

一开始使用 markdownHeadersplier 和 RecursiveCharacterTextSplitter， 发现会截断代码或者表格，后续询问 ai 了解需要用预处理占位符来 避免这种情况， 也就是 在切分前 先把这些不可切分的块使用不可能被切分占位符代替，等到切分完毕后， 再把占位符替换为原始内容， 恢复完之后用 恢复后的每个chunk的 内容存入向量数据库。



可能会遇到的问题

- 1. 还原后 chunk 实际 token 数 >> 切分时统计的数

切分器看到的是 `[CODE_BLOCK_1]`（十几个字符）→ 顺利过 512 阈值。还原后变成 2000 token 的代码块 → 两件事翻车：

- **Embedding 阶段**：bge-m3 截断到 512/8192，代码后半截直接没嵌进去，检索不到
- **LLM 上下文阶段**：Day 5 送 `top_k=5`× 每个 2000 token = 1 万 token 没了，answer 窗口被代码挤占

- 2. 代码块本身的检索语义是废的

这是更大的问题。你占位符期间，embedding 算的是 `[CODE_BLOCK_1]`这串字符的向量——跟 "这段 Python 装饰器干嘛的" 这条 query 根本不匹配。结果：

> 用户问「我 vault 里那段 FastAPI 中间件怎么写的」→ 向量检索匹配不到代码块 → 答不出来，尽管 vault 里有。

你防了「切坏」，但丢了「检到」。




## [2026-06-28]  优化：bm25 语义检索引入jieba 进行分词

bm25 默认分词按空格分词， 不适用于中文，jieba可以根据中文语义进行合理分词

分词 jieba.lcut()  直接返回 分词列表，  cut() 方法返回的是执行器

---

## [2026-06-28]  Day 2：混合检索 + Reranker

### 完成
- **BM25 稀疏检索** (`src/retrieval/sparse_retriever.py`)：jieba 分词 + BM25Okapi，pickle 持久化，from_chroma 从 ChromaDB 建索引
- **混合检索** (`src/retrieval/hybrid.py`)：dense + sparse 加权融合，alpha 来自 config
- **Reranker** (`src/retrieval/reranker.py`)：FlagReranker 交叉编码，batch 推理
- **Query 改写** (`src/retrieval/query_rewrite.py`)：口语 → 陈述句
- **统一类型** (`src/retrieval/types.py`)：`RetrievalResult` 统一三档检索返回值
- **对比脚本** (`scripts/compare_retrieval.py`)：A 纯向量 / B 混合 / C 混合+rerank
- **split_v2 修复**：chunk 现在携带 node 基础 metadata（filepath/title/wikilinks）
- **测试**：81/82 通过（indexing 24 + retrieval 50 + config 1 已有问题）

### 踩坑
- `FlagEmbedding 1.4.0` 调 `prepare_for_model()`，`transformers >= 4.34` 已移除 → 加兼容层 monkey-patch
- `transformers 5.x` 需要 `tokenizers` 从源码编译（需 Rust）→ 降级到 4.44.0
- Windows GBK 编码导致 emoji 打印报错 → `sys.stdout.reconfigure(encoding="utf-8")`
- `settings.reranker_model` 写 HuggingFace 名称会联网下载 → 改为本地路径

### 待做（Day 3）
- 双链图 + 增量索引（见 Day 3 计划）



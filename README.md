
# Obsidian RAG —— 个人知识库智能问答系统

> 基于个人 Obsidian 笔记库构建的垂直领域 RAG 系统，支持自然语言问答、来源溯源、混合检索、双链关联扩展，并具备量化评测与公网部署能力。

## 一、项目整体描述

### 1.1 项目背景

Obsidian 是我的个人知识库，累计数百篇 Markdown 笔记，涵盖技术学习、算法推导、项目记录等内容。传统检索方式（Obsidian 内置搜索、文件名模糊匹配）无法处理"语义层面"的提问，例如：

- "我之前记的 FlashAttention 的优化点有哪些？"
    
- "LoRA 和 QLoRA 的区别，我笔记里是怎么写的？"
    
- "RAG 的 chunk 策略我对比过哪几种？"
    

这类问题需要**跨笔记、跨标题、结合上下文**才能回答，且必须**标注来源**（哪篇笔记、哪个标题下），否则无法验证正确性。

### 1.2 项目定位

面向**个人 Obsidian 笔记库**的垂直 RAG 问答系统，区别于通用知识库 RAG 的三个差异化点：

- **数据结构感知**：解析 Markdown 标题层级、YAML front matter（tags/aliases/created）、`[[wikilink]]` 双链
    
- **双链关联检索**：基于 `[[wikilink]]` 构建笔记关联图（NetworkX），检索时扩展一跳关联 chunk，GraphRAG 雏形
    
- **个人数据闭环**：数据不出本地（embedding + reranker 全本地），生成侧可切换本地/API
    

### 1.3 核心功能点

| 功能       | 说明                                                                        |
| -------- | ------------------------------------------------------------------------- |
| 自然语言问答   | 用户输入问题，系统返回答案 + 来源（文件路径、标题路径）                                             |
| 混合检索     | 稠密向量（bge-m 3）+ BM 25 稀疏，加权融合                                              |
| Rerank   | 本地 BGE-Reranker-v 2-m 3 对 TopK 重排序                                        |
| Query 改写 | LLM 将口语化问题改写为"笔记中可能出现的陈述句"                                                |
| 双链关联扩展   | 检索命中 chunk 后，自动扩展其 `[[wikilink]]` 一跳邻居                                    |
| 增量索引     | 新增/修改笔记后增量更新 ChromaDB，不需全量重跑                                              |
| 量化评测     | 100 条 QA 测试集 + Ragas（Faithfulness / Answer Relevancy / Context Precision） |
| Web UI   | Streamlit 前端，来源折叠展示 + 检索过程可视化                                             |
| 公网部署     | Docker + 腾讯云轻量 2 C 4 G，面试官可点开                                             |
|          |                                                                           |

---

## 二、技术栈说明

### 2.1 核心技术栈总览

| 层级        | 技术                                                              | 用途                                                          |
| --------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| 数据层       | Obsidian Vault（`.md`）                                           | 原始知识库                                                       |
| 解析层       | Python（`markdown`, `pyyaml`, `obsidiantools`）                   | 读取 `.md`、解析 front matter、提取 `[[wikilink]]`                  |
| 分块层       | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | 按 `#/##/###` 标题层级切分，长 chunk 递归再切                            |
| Embedding | **Ollama `bge-m3:latest` **（本地）                                 | 稠密向量（1024 维），Ollama 已装，仅用 dense 信号                          |
| 稀疏信号      | `rank-bm25`                                                     | 补 Ollama 不暴露的 bge-m 3 sparse 向量，BM 25 等价替代                  |
| 向量库       | **ChromaDB**​                                                   | 自动持久化（SQLite）+ metadata 过滤（filepath / tags / heading）       |
| Reranker  | **BAAI/bge-reranker-v 2-m 3**（本地，`FlagReranker`）                | 从 modelscope 下载，~1.1 GB，本地 `use_fp16=True` 跑                |
| 生成模型      | **DeepSeek V 4-Pro**（API）                                       | 国产 coding SOTA（SWE-bench 80.6%），¥3/¥6 每 MTok，调试便宜           |
| 生成备选      | Qwen 2.5-7 B-Instruct（本地）/ Qwen 3-Coder-Plus（长上下文场景）            | 本地兜底 / 长上下文 agent 场景                                        |
| 后端        | **FastAPI**​                                                    | 封装 `/ask` POST 接口，返回 `{answer, sources}`                    |
| 前端        | **Streamlit**​                                                  | 输入框 + 聊天历史 + 来源折叠 + 检索过程可视化                                 |
| 评测        | **Ragas**​                                                      | Faithfulness / Answer Relevancy / Context Precision         |
| 图结构       | NetworkX                                                        | 双链 `[[wikilink]]` → 有向图，一跳扩展                                |
| 开发环境      | WSL 2 + Docker Desktop                                          | Ollama 在 WSL 原生跑，Docker 通过 `host.d.docker.internal:11434` 连 |
| 部署        | Docker Compose + 腾讯云轻量 2 C 4 G                                  | 公网 URL，面试官可点                                                |
| AI 辅助     | Cursor + Claude Code (Sonnet)                                   | 代码生成 + plan mode 诊断                                         |

### 2.2 技术选型的关键决策

> 📌 **为什么 ChromaDB 而不是 FAISS？**​
> 
> 个人场景（几万 chunk）FAISS 和 Chroma 性能差距可忽略，但 Chroma 自动持久化 + `where` metadata 过滤（按 tags/filepath/heading 过滤）对 Obsidian 场景非常顺手，FAISS 要自己搞。

> 📌 **为什么 Ollama bge-m 3 而不用 bge-large-zh-v 1.5？**​
> 
> 已装 Ollama，bge-m 3 C-MTEB 中文 65.79 > v 1.5 的 64.53，且 8192 上下文更长。代价是 Ollama 只暴露 dense 向量，sparse 用 BM 25 补——等价替代，不亏。

> 📌 **为什么 FastAPI 而不是 LangServe 直接挂？**​
> 
> LangServe 够 V 1 用，但后续要加"来源溯源格式化 + 双链扩展 + 流式 yield"时，FastAPI 自己写更可控


## 三、项目难点
### 复杂文档处理

| **<br><br>内容类型<br><br>** | **<br><br>要不要向量化<br><br>** | **<br><br>做法<br><br>**    |
| ------------------------ | -------------------------- | ------------------------- |
| 普通正文文本                   | ✅                          | 正常切 chunk + embed         |
| Markdown 表格              | ✅ 但要结构化                    | 表级 summary + 行级切片，双路存     |
| Mermaid 流程图              | ✅ 但特殊                      | 源码保留 + LLM 生成文字描述一起 embed |
| 图片                       | ⚠️ 看情况                     | 路径存 metadata + 两种可选方案（见下） |
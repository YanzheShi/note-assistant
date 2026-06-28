好，这个思路对——splitter 是 Day 1 里最值得你自己走一遍决策链的模块，loader 那层面试官问的是"你怎么扛脏数据"，splitter 这层问的是"你的 chunk 策略为什么这么定"，后者更能拉开差距。

我给你搭一个下午能自己走完的框架：思考链路模板（带提问，不代答） + 预留多实现的 stub + 验证框架 + 提示清单。你填决策、填实现、填测试，最后出定稿。

一、splitter 思考链路模板（你填）

把下面每个 [Q] 当面试官的追问，自己先答口头 → 再答代码决策 → 最后 DECISIONS.md 落一条。

🧭 宏观：chunk 策略选型

[Q1] 你的 vault 是中文课程笔记（## 一、xxx 起手，大多没 h1），单篇长度 1-3KB。问"FlashAttention 的优化点"这种问题，召回的 chunk 需要带什么上下文才能答出来？

→ 推演：如果只按 800 char 硬切，会不会出现 "## FlashAttention-2 改进 那段被切成独立 chunk，但 ## FlashAttention 基础 那段没带过来，问'FA2 相比 FA1 改了啥'时 chunk 缺父上下文"？

[Q2] 那业界两种主流切法你选哪个，为什么？
• A. 单层 Recursive（800/150）直接切

• B. 两层：HeaderSplitter 按 # ## ### 先切保标题树 → Recursive 再细切

→ 要答出 "B 的代价是 header 那段如果很短（比如 ## 检索 只有一行 '# 检索'），HeaderSplitter 切出来的父 doc 可能 < chunk_size，child_splitter 不切直接原样过，导致 metadata 有 h2 但 content 只有 5 个字"——这个你 vault 会不会出现？出现了怎么处理（return_each_line=False 还是 True）？

[Q3] return_each_line=False vs True 选哪个？看 LC 文档：False = 同 header 下多行合并成一个 doc；True = 每行独立。你 vault 里 ## 检索 下面常跟一段正文，选错会怎样？

✂️ 微观：Recursive 参数

[Q4] separators 默认是 ["\n\n", "\n", " ", ""]，你 vault 中文为主，要不要加 "。"？加了会怎样，不加会怎样？举一个"不加导致一句被劈两半"的具体例子（从你 87 篇里找一篇真实段落推演）。

[Q5] keep_separator=True vs False，对你"Chroma 存 + LLM 读"这条链路有啥影响？chunk 末尾 。 丢了会让 LLM 读起来怪吗？

[Q6] chunk_size=800 是 char 不是 token。bge-m3 中文 1 token ≈ 1-2 汉字，800 char ≈ 400-800 token。top_k_rerank=5 时每个 chunk 喂给 reranker，5×800token 上下文够不够答你 vault 里的问题？要不要调到 1000？

🏷️ Metadata 缝合

[Q7] heading_path = "h1 > h2 > h3" 拼接时，你 vault ## 起手 h1 空，拼出来是 "> 一、什么是大模型 > 检索" 还是 "一、什么是大模型 > 检索"？前者丑但保"h1 空"这个信息，后者干净但丢了"这篇没 h1"的信号——你选哪个？理由跟 vault 形态绑定。

[Q8] wikilinks 整篇级缝进每个 chunk（loader 提的 [[A]] 整篇扫一次 → 每 chunk metadata 都有），还是 chunk 级（每 chunk 的 content 再扫一次 [[...]]）？
• 整篇级：O(1) 篇级扫，metadata 冗余（87 篇 × 平均 3 chunk = 261 个 chunk 都带同个 wikilinks 列表），但 Day 3 双链图直接能用

• chunk 级：O(N_chunk) 扫，metadata 精确（某个 chunk 只链了 A B，另一个只链了 C），但 Day 1 多写 10 行

→ 你 Day 1 选哪个？Day 3 要不要升级到 chunk 级？

[Q9] metadata 里除了 filepath/title/tags/wikilinks/heading_path/h1h2h3h4，还要不要 chunk_index（同篇第几个 chunk）？Day 4 评测 "Context Precision" 时，如果同一篇的 chunk 2 和 chunk 3 都被召回，LLM 生成时能不能按 chunk_index 重排让上下文连贯？

🔌 接口设计

[Q10] split_doc(doc: Dict, hs, cs) -> List[Dict] 这个 signature 合理吗？
• 入参 doc 是 loader dict（含 raw_md + fm + wikilinks）

• 入参 hs, cs 是外部 factory 给的（不在 split_doc 里 new，避免每篇 new 两次 splitter）

• 出参 List[Dict] with {"content": str, "metadata": dict}

→ 有没有更好的？比如出参直接 List[Document]（LC 原生），让 consumer（ingestor）少转一次？但那样 splitter 就绑 LC 了——你愿意吗？（参考上午 loader 把 LC 胶水挪 splitter 的决策）

二、预留多实现的 stub（你填实现）

src/note_assistant/indexing/splitter.py，先只放 stub + 工厂 + 验证入口，三个实现版本你下午填：
# src/note_assistant/indexing/splitter.py
"""
Splitter 模块 —— 下午自己填实现，对比三版

决策链见 DECISIONS.md:2026-06-XX-splitter-strategy
"""

from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from note_assistant.config import settings
from note_assistant.indexing.vault_loader import VaultLoader


# ================================================================
# 标题层级配置（Obsidian 常见 # ## ### ####）
# ================================================================
_HEADER_CONFIG = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
]


# ================================================================
# 工厂：两层 splitter（v2/v3 共用，参数可调）
# ================================================================
def make_splitters(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: List[str] | None = None,
    keep_sep: bool = True,
    return_each_line: bool = False,
) -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    """v2/v3 共用工厂，参数暴露给你调"""
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap
    seps = separators or ["\n\n", "\n", "。", " ", ""]

    header_sp = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_CONFIG,
        return_each_line=return_each_line,
    )
    child_sp = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=seps,
        keep_separator=keep_sep,
    )
    return header_sp, child_sp


# ================================================================
# v1: 单层 Recursive（基线，用来对比）
# ================================================================
def split_v1(doc: Dict[str, Any], sp: RecursiveCharacterTextSplitter) -> List[Dict[str, Any]]:
    """
    基线：纯 Recursive 800/150，不保标题层级
    返回 chunk 的 metadata 只有 loader 层的 filepath/title/tags/wikilinks，无 heading_path
    【你填实现】
    """
    raise NotImplementedError("v1: 纯 Recursive，raw_md → sp.split_text → 缝 metadata")


# ================================================================
# v2: 两层 Header + Recursive（生产路候选）
# ================================================================
def split_v2(
    doc: Dict[str, Any],
    header_sp: MarkdownHeaderTextSplitter,
    child_sp: RecursiveCharacterTextSplitter,
) -> List[Dict[str, Any]]:
    """
    生产路候选：HeaderSplitter 保 h1/h2/h3 → Recursive 细切
    缝 heading_path = "h1 > h2 > h3"（h1 空 skip，你 vault ## 起手）
    wikilinks 整篇级缝进每个 chunk
    【你填实现】（参考上午给的 split_doc 逻辑，自己重写一遍别 copy）
    """
    raise NotImplementedError("v2: 两层，heading_path + wikilinks 整篇级")


# ================================================================
# v3: v2 + chunk 级 wikilinks 重扫（可选，Day 3 预研）
# ================================================================
def split_v3(
    doc: Dict[str, Any],
    header_sp: MarkdownHeaderTextSplitter,
    child_sp: RecursiveCharacterTextSplitter,
) -> List[Dict[str, Any]]:
    """
    v2 基础上：每个 chunk 的 content 再扫一次 [[...]]，
    chunk["metadata"]["wikilinks"] = 本 chunk 出现的 wikilink（不是整篇级）
    【你填实现，Day 1 可选不做，stub 留着】
    """
    raise NotImplementedError("v3: v2 + chunk 级 wikilinks 重扫")


# ================================================================
# LC 胶水（从 loader 挪来，不变）
# ================================================================
def loader_dict_to_lc_doc(item: Dict[str, Any]) -> Document:
    return Document(
        page_content=item["raw_md"],
        metadata={
            "filepath": item["filepath"],
            "title": item["title"],
            "tags": item["tags"],
            "wikilinks": item["wikilinks"],
        },
    )


def load_vault_as_lc(vault_path: Path | None = None) -> List[Document]:
    loader = VaultLoader(vault_path)
    return [loader_dict_to_lc_doc(d) for d in loader.load_all()]


# ================================================================
# 验证入口：uv run python -m note_assistant.indexing.splitter
# ================================================================
if __name__ == "__main__":
    loader = VaultLoader()
    docs = loader.load_all()
    print(f"📁 loader: {len(docs)} 篇")

    # v1 基线
    _, sp_v1 = make_splitters()  # 只用 child
    # v2 生产路
    hs_v2, cs_v2 = make_splitters()
    # v3 预留
    # hs_v3, cs_v3 = make_splitters()

    # ===== 对比：前三篇每篇 ≈ chunk 数 + heading_path 样例 =====
    for label, split_fn, args in [
        ("v1-Recursive", split_v1, (sp_v1,)),
        ("v2-Header+Rec", split_v2, (hs_v2, cs_v2)),
    ]:
        total = 0
        for d in docs[:3]:
            chunks = split_fn(d, *args)
            total += len(chunks)
            # 首 chunk 抽样
            c0 = chunks[0]
            hp = c0["metadata"].get("heading_path", "N/A")
            print(f"  [{label}] {d['filepath']} → {len(chunks)} chunks | hp={hp}")
        print(f"  [{label}] 前三篇合计: {total} chunks")

    # 全库估算（v2）
    all_v2 = sum(len(split_v2(d, hs_v2, cs_v2)) for d in docs)
    print(f"\n📊 全库 v2: {len(docs)} 篇 → ~{all_v2} chunks")

    # TODO: 你下午补 —— 对比 v1 vs v2 的：
    #   1. 同篇 chunk 数差异（v2 因为 header 短段可能更少？）
    #   2. heading_path 覆盖率（v2 有多少 chunk 有 h2/h3）
    #   3. 找一个"v1 切坏的中文句子"例子


三、验证框架（你下午跑对比）

光"跑通"不够，要能量化 v1 vs v2 差异，DECISIONS 才有"为什么选 v2"的证据。

要对比的指标

指标 v1 v2 你 vault 预期

全库 chunk 总数 ? ? v2 可能略少（header 短段没被 child 再切）或略多（header 每段独立）

首篇 ## 一、xxx 起手的 heading_path N/A "一、xxx > ..." 还是 "> 一、xxx > ..."? 你定

中文句子被劈例 找 1-2 个 找 1-2 个 separators 加 。 后应该减少

h2 短段（如 ## 检索 只有一行） N/A 这种 chunk content 多短？ 决定 return_each_line 选 T/F

快速验证脚本（嵌在 __main__ 里，上面 stub 已留对比 loop）

你下午要补的 TODO（stub 最后那几行）：
# TODO 你补：
# 1. 找一篇笔记，手动数 "。" 被劈的例子（v1 下 separators 去掉 "。" 对比）
# 2. 找一篇 ## 起手没 h1 的，看 heading_path 拼出来啥样，决定 h1 空 skip 还是保留 ">"
# 3. 算 v1 vs v2 全库 chunk 数差百分之几


四、提示 & 框架信息（避坑）

LC 的坑，提前告诉你省得你下午踩

1. MarkdownHeaderTextSplitter 的 return_each_line=False：同 header 下多行合并成一个 Document。你 vault 里 ## 检索 下面常跟 3-5 行正文，合并后大概率 > 800 char，child_splitter 会再切——符合预期。如果 True，## 检索 这行本身就成独立 Document（content="" 或只剩标题文本），child 不切，浪费一个 chunk。→ 你 vault 形态下 False 更合理，自己验证下。

2. RecursiveCharacterTextSplitter 的 separators 优先级：是按列表顺序试的，匹配到第一个就切。所以 ["\n\n", "\n", "。", " ", ""] 的意思是：先找 \n\n 切，切不完找 \n，再找不到找 。，再找不到找空格，再找不到按 char 硬切。→ "。" 放 " " 前面，不然会先按空格切碎中文（中文空格少，但万一有英文混排 RAG 是... 会按空格切在 RAG 和 是 之间，还行不致命）。

3. chunk_size=800 是 char 不是 token：bge-m3 中文 tokenizer 大概 1 token ≈ 1.5 汉字（看具体词），800 char ≈ 530 token。top_k_rerank=5 → 5×530=2650 token 进 reranker，bge-reranker-v2-m3 支持 512/1024 窗口？查一下——哦 v2-m3 是 8192 窗口，5×530 没问题。但 chunk_size 如果调到 1500 就要留意 reranker 的 truncate。

4. keep_separator=True 的行为：separator 粘回前一个 chunk 的尾部（不是后一个的头部）。所以 "。" 切 → 前 chunk 末尾带 。，后 chunk 开头是下一句。读起来连贯。→ 你 vault 中文为主，这个必开。

5. Obsidian 笔记里会炸 splitter 的特例（Day 1 先不处理，DECISIONS 记"已知待处理"）：
   • code fence（`python ... `）里的内容被 Recursive 按 。\n 切碎 → Day 2 预处理 preprocessor.py 做 code fence 保护

   • mermaid / math block 同理

   • table（| col1 | col2 |）→ 一行切半

五、下午节奏建议（4-5h）

时段 内容

30 min 走 [Q1]-[Q10]，口头答完，DECISIONS.md 先落 5 条草稿

60 min 填 split_v1 + split_v2 实现（v3 可选）

30 min 跑对比：v1 vs v2 chunk 数、找中文劈句例、heading_path 样例

45 min 定稿（选 v2 参数：return_each_line=False / separators 含 。 / h1 空 skip / wikilinks 整篇级）→ 替换上午的 split_doc 为定稿 v2

30 min tests/indexing/test_splitter.py 补 3-4 个 case（heading_path / metadata 传播 / 无 h1 兜底 / v1 vs v2 差异锁一个）

30 min DECISIONS.md splitter 段收尾（5 条决策 + vault 形态绑定理由）

收完 splitter 定稿 → Day 1 剩下 embedder → Chroma → 命令行 demo 三步（明天上半场也行，看你自己节奏，Day 1 已经 loader + splitter + test 两轮了，下午如果 splitter 决策+实现+test 走完，Day 1 主线已经超原计划了）。

去吧，DECISIONS.md 的 splitter 段等你下午定稿后我帮你过一遍措辞（面试能直接背的那种）。[Q1]-[Q10] 有哪个你自己答着虚的，先喊我，不用全答完再动笔。
"""Agentic RAG：基于 langgraph 自写 StateGraph 的 ReAct 问答 agent。

模块职责：
    - tools.py   ：底层检索能力封装成 agent 工具（hybrid/graph_expand/vector/bm25/filtered/get_note/query_rewrite）
    - agent.py   ：自写 StateGraph（Router / agent / tools / reflect(Judge) / rewrite / generate / direct_chat）
    - runner.py  ：易用的异步入口（ainvoke / astream），提取答案/来源/轨迹，并接入语义缓存
    - cache.py   ：语义缓存（精确 + 近邻）
    - evaluation.py：轨迹级评测闭环骨架

原有 RAGChain 保留为传统模式与对比基线，本模块是其 agentic 升级版。
"""
from note_assistant.agent.runner import AgentRunResult, ainvoke, astream, reset_cache

__all__ = ["ainvoke", "astream", "AgentRunResult", "reset_cache"]

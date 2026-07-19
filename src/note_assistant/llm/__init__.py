"""统一 LLM 通道。

所有 LLM 调用（agent 决策、检索路由、查询改写、答案生成）统一走 DeepSeek
（OpenAI 兼容）通道，取代原先散落在各模块的 agnes/longcat/裸 httpx 调用。
"""
from note_assistant.llm.client import get_llm

__all__ = ["get_llm"]

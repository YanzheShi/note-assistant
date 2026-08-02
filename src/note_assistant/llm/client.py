"""统一 LLM 客户端。

所有 LLM 调用（agent 决策、检索路由、查询改写、答案生成）统一走 AGENT_*
（OpenAI 兼容）通道，取代原先散落在各模块的 agnes/longcat/裸 httpx 调用。

配置来源（config.py）：
    agent_api_key   —— 必填
    agent_base_url  —— 模型网关地址（OpenAI 兼容 /v1）
    agent_model     —— 模型名
"""
from functools import lru_cache
from typing import Optional

from langchain.chat_models import init_chat_model

from note_assistant.config import settings


@lru_cache(maxsize=None)
def get_llm(
    *,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
):
    """构造 DeepSeek ChatModel（OpenAI 兼容）。

    Args:
        temperature: 采样温度，低温度更稳定（改写/路由/决策）
        max_tokens: 最大生成长度，None 表示模型默认
        streaming: 是否开启流式（影响 astream 是否逐 token）

    Returns:
        langchain ChatModel 实例，支持 .invoke/.ainvoke/.astream/.bind_tools
    """
    kwargs: dict = dict(
        model=settings.agent_model,
        model_provider="openai",
        api_key=settings.agent_api_key,
        openai_api_base=settings.agent_base_url,
        temperature=temperature,
        streaming=streaming,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)


@lru_cache(maxsize=None)
def get_vlm(
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
):
    """构造视觉模型 ChatModel（OpenAI 兼容，走 VLM_* 配置）。

    与 `get_llm()` 通道完全独立：纯文本 LLM（agent/生成）走 AGENT_*，
    多模态理解走 VLM_*。两者模型/网关/密钥互不串台。

    Returns:
        langchain ChatModel 实例，支持图文混合 message（image_url content part）。

    Raises:
        RuntimeError: VLM_* 未配置（vlm_api_key / vlm_base_url / vlm_model 任一为空）。
    """
    if not (settings.vlm_api_key and settings.vlm_base_url and settings.vlm_model):
        raise RuntimeError(
            "VLM 未配置：请在 .env 设置 VLM_API_KEY / VLM_BASE_URL / VLM_MODEL，"
            "或关闭 image_understand_enabled 以跳过多模态理解。"
        )
    kwargs: dict = dict(
        model=settings.vlm_model,
        model_provider="openai",
        api_key=settings.vlm_api_key,
        openai_api_base=settings.vlm_base_url,
        temperature=temperature,
        streaming=streaming,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)

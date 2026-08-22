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
    # 零侵入 token 计量：挂全局 handler（评测 set_meter 才采集，默认零副作用）
    from note_assistant.llm.usage import get_token_handler

    return init_chat_model(**kwargs).with_config(callbacks=[get_token_handler()])


def _build_openai_llm(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float = 0.2,
) -> "object":
    """构造一个 OpenAI 兼容 ChatModel（凝练 / 摘要专属通道复用，不进 lru_cache）。

    Args:
        model: 模型名（必填）
        api_key: API key；为空则用主通道 AGENT_API_KEY
        base_url: 网关地址；为空则用主通道 AGENT_BASE_URL
        temperature: 凝练/摘要追求确定性，默认 0.2

    Returns:
        langchain ChatModel 实例，已挂 token 计量 handler
    """
    kwargs: dict = dict(
        model=model,
        model_provider="openai",
        api_key=api_key or settings.agent_api_key,
        openai_api_base=base_url or settings.agent_base_url,
        temperature=temperature,
    )
    from note_assistant.llm.usage import get_token_handler

    return init_chat_model(**kwargs).with_config(callbacks=[get_token_handler()])


@lru_cache(maxsize=None)
def get_condense_llm():
    """凝练（消指代改写）专属 LLM。

    若 ``AGENT_CONDENSE_MODEL`` 非空 → 用专属模型（独立网关/密钥可单独配，留空回落主通道）；
    否则复用主文本通道 ``get_llm()``（与改造前逐字节等价）。

    Returns:
        langchain ChatModel 实例
    """
    if settings.agent_condense_model:
        return _build_openai_llm(
            settings.agent_condense_model,
            settings.agent_condense_api_key,
            settings.agent_condense_base_url,
            temperature=0.2,
        )
    return get_llm()


@lru_cache(maxsize=None)
def get_summarize_llm():
    """长程摘要（滚动压缩）专属 LLM。

    解析逻辑同 ``get_condense_llm()``，仅作用于 ``AGENT_SUMMARIZE_*`` 配置。
    ``AGENT_SUMMARIZE_MODEL`` 非空 → 专属模型；否则复用主通道。

    Returns:
        langchain ChatModel 实例
    """
    if settings.agent_summarize_model:
        return _build_openai_llm(
            settings.agent_summarize_model,
            settings.agent_summarize_api_key,
            settings.agent_summarize_base_url,
            temperature=0.2,
        )
    return get_llm()


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
    # 零侵入 token 计量：挂全局 handler（评测 set_meter 才采集，默认零副作用）
    from note_assistant.llm.usage import get_token_handler

    return init_chat_model(**kwargs).with_config(callbacks=[get_token_handler()])

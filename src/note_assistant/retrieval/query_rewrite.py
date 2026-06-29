# src/note_assistant/retrieval/query_rewrite.py
"""
查询改写：口语化问题 → 陈述句（匹配笔记表述）
"""

from langchain.chat_models import init_chat_model

from note_assistant.config import settings


class QueryRewriter:
    """查询改写器：将口语化问题改写为知识库中可能出现的陈述句。"""

    SYSTEM_PROMPT = (
        "你是一个查询改写助手。用户会问口语化问题，"
        "你需要改写成知识库中可能出现的陈述句表述。"
        "只输出改写后的问题，不要解释，不要加引号。"
    )

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选，传入自定义 LLM；默认用 DeepSeek-Chat
        """
        self.llm = llm or init_chat_model(
            model=settings.longcat_model,
            model_provider="openai",
            api_key=settings.longcat_api_key,
            openai_api_base=settings.longcat_base_url,
            temperature=0.3,       # 稳定输出，不要创意
            max_tokens=100,        # 改写结果很短
        )

    def rewrite(self, question: str) -> str:
        """
        口语 → 陈述句。

        Args:
            question: 用户原始问题

        Returns:
            改写后的陈述句
        """
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("user", f"原问题：{question}"),
        ]
        result = self.llm.invoke(messages)
        return result.content.strip()

# src/note_assistant/generation/generator.py
"""
答案生成器：RAG 终步，流式输出
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from note_assistant.config import settings
from note_assistant.retrieval.types import RetrievalResult


class Generator:
    """答案生成器：检索结果 + 用户问题 → 最终回答"""

    SYSTEM_PROMPT = (
        "你是一个个人知识库助手，基于用户提供的笔记内容回答问题。\n"
        "规则：\n"
        "1. 只基于提供的上下文回答，不要编造\n"
        "2. 如果上下文不足以回答，诚实说明\n"
        "3. 回答要简洁、结构化，使用 Markdown\n"
        "4. 代码示例要完整可运行\n"
        "5. 引用来源时标注笔记标题"
    )

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选，传入自定义 LLM；默认用 DeepSeek-Chat
        """
        self.llm = llm or init_chat_model(
            model=settings.agnes_model,
            model_provider="openai",
            api_key=settings.agnes_api_key,
            openai_api_base=settings.agnes_base_url,
            temperature=0.6,       # 比 rewrite 高，允许一定创造力
            max_tokens=2048,       # 回答可以长一些
        )

        self.stream_llm = llm or init_chat_model(
            model=settings.longcat_model,
            model_provider="openai",
            api_key=settings.longcat_api_key,
            openai_api_base=settings.longcat_base_url,
            temperature=0.6,       # 比 rewrite 高，允许一定创造力
            max_tokens=2048,       # 回答可以长一些
            streaming=True
        )

    def build_prompt(self, question: str, context: list[dict]) -> ChatPromptTemplate:
        context_text = self._format_context(context)
        # 用 partial 把 system + context 固定，只留 question 为变量
        return ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "## 参考笔记\n{context_text}\n\n## 问题\n{question}"),
        ]).partial(context_text=context_text)

    def generate(self, question: str, context: list[dict]) -> str:
        """
        非流式生成完整答案。

        Args:
            question: 用户问题（可能已被 QueryRewriter 改写）
            context: 检索结果列表，每项包含 page_content 和 metadata

        Returns:
            完整回答文本
        """
        agent = self.build_prompt(question, context) | self.llm | StrOutputParser()
        answer = agent.invoke({"question": question})
        return answer

    async def generate_stream(self, question: str, context: list[dict]):
        """
        流式生成答案，逐 token 返回。

        Args:
            question: 用户问题
            context: 检索结果列表

        Yields:
            每个 token 的文本片段
        """
        context_text = self._format_context(context)
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("user", f"## 参考笔记\n{context_text}\n\n## 问题\n{question}"),
        ]
        print("prompt: ", messages)

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    @staticmethod
    def _format_context(context: list[RetrievalResult]) -> str:
        """
        将检索结果格式化为 LLM 可读的上下文。

        每条笔记标注来源标题，方便 LLM 引用。
        """
        parts = []
        for i, item in enumerate(context, 1):
            title = item.metadata.get("title", "未知笔记")
            content = item.page_content
            parts.append(f"### [{i}] {title}\n{content}")
        return "\n\n".join(parts)

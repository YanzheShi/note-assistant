# src/note_assistant/generation/generator.py
"""
答案生成器：RAG 终步，流式输出。支持多轮对话历史注入。
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from note_assistant.config import settings
from note_assistant.pipeline.image_answer import render_image_block
from note_assistant.retrieval.types import RetrievalResult
from note_assistant.security.guardrails import (
    append_guardrail,
    wrap_history_tuples,
    wrap_retrieved_context,
    wrap_user_question,
)

# 保留的最近对话轮数（再往前可能 context window 撑爆且参考价值低）
MAX_HISTORY_TURNS = 10


class Generator:
    """答案生成器：检索结果 + 用户问题 + 历史 → 最终回答"""

    SYSTEM_PROMPT = (
        "你是一个智能助手。\n\n"
        "规则：\n"
        "1. 如果提供了参考笔记，基于笔记内容回答，不要编造\n"
        "2. 如果没有提供参考笔记（空 context），说明该问题超出知识库范围，请回答：\n"
        "   \"抱歉，我的知识库中没有关于该问题的相关内容。请尝试换一个问题。\"\n"
        "3. 诚实回答，不要编造笔记中不存在的内容\n"
        "4. 回答要简洁、结构化，使用 Markdown\n"
        "5. 引用来源时标注笔记标题\n"
        "6. 参考笔记中标注【图片】的条目来自笔记里的插图，其内容由视觉模型解析得到。\n"
        "   - 引用图片信息时，说明\"根据笔记中的架构图/流程图\"，不要说\"根据文档描述\"\n"
        "   - 如果图片信息对回答有帮助，在相应位置插入 [[IMG:asset_id]] 标记，系统会自动替换为图片\n"
        "   - 严禁描述图片解析结果中不存在的细节"
    )

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选，传入自定义 LLM；默认用 AGENT_* 配置的模型
        """
        self.llm = llm or init_chat_model(
            model=settings.agent_model,
            model_provider="openai",
            api_key=settings.agent_api_key,
            openai_api_base=settings.agent_base_url,
            temperature=0.6,  # 比 rewrite 高，允许一定创造力
            max_tokens=2048,  # 回答可以长一些
            # 使用流式输出保证首字响应时间
            streaming=True,
        )


    @staticmethod
    def _format_history(history: list[dict]) -> list[tuple[str, str]]:
        """
        将历史对话格式化为 LangChain 消息元组列表。

        只保留最近 MAX_HISTORY_TURNS 轮（避免撑爆 context window），
        且只保留 role 和 content 字段（过滤掉 sources/timing 等前端字段）。
        """
        truncated = history[-(MAX_HISTORY_TURNS * 2):]  # 每轮 2 条消息
        messages = []
        for msg in truncated:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                # LangChain 用 "human" 对应 "user"，"ai" 对应 "assistant"
                lc_role = "human" if role == "user" else "ai"
                messages.append((lc_role, content))
        return messages

    def build_prompt(self, question: str, context: list[dict], history: list[dict] | None = None) -> ChatPromptTemplate:
        """
        构建含历史上下文的 prompt。

        Args:
            question: 当前问题
            context: 检索结果（RetrievalResult 列表）
            history: 历史对话列表，每项 {"role": "user"|"assistant", "content": str}

        Returns:
            ChatPromptTemplate
        """
        context_text = self._format_context(context)
        messages = [("system", append_guardrail(self.SYSTEM_PROMPT))]
        if history:
            messages.extend(wrap_history_tuples(self._format_history(history)))
        messages.append(("human", "## 参考笔记\n{context_text}\n\n## 问题\n{question}"))
        return ChatPromptTemplate.from_messages(messages).partial(
            context_text=wrap_retrieved_context(context_text)
        )

    def generate(self, question: str, context: list[dict], history: list[dict] | None = None) -> str:
        """
        非流式生成完整答案，支持历史对话。

        Args:
            question: 用户问题（可能已被 QueryRewriter 改写）
            context: 检索结果列表，每项包含 page_content 和 metadata
            history: 历史对话列表

        Returns:
            完整回答文本
        """
        agent = self.build_prompt(question, context, history) | self.llm | StrOutputParser()
        answer = agent.invoke({"question": wrap_user_question(question)})
        return answer

    async def generate_stream(self, question: str, context: list[dict], history: list[dict] | None = None):
        """
        流式生成答案，逐 token 返回。支持历史对话。

        Args:
            question: 用户问题
            context: 检索结果列表
            history: 历史对话列表

        Yields:
            每个 token 的文本片段
        """
        context_text = self._format_context(context)
        messages = [("system", append_guardrail(self.SYSTEM_PROMPT))]
        if history:
            messages.extend(wrap_history_tuples(self._format_history(history)))
        messages.append((
            "human",
            f"## 参考笔记\n{wrap_retrieved_context(context_text)}\n\n"
            f"## 问题\n{wrap_user_question(question)}",
        ))

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    @staticmethod
    def _format_context(context: list[RetrievalResult]) -> str:
        """
        将检索结果格式化为 LLM 可读的上下文。

        每条笔记标注来源标题，方便 LLM 引用。
        L2：每条内容过注入扫描（默认 flag 只记日志；redact 时遮蔽命中跨度）。
        """
        from note_assistant.security.sanitize import sanitize_text

        parts = []
        for i, item in enumerate(context, 1):
            title = item.metadata.get("title", "未知笔记")
            block = render_image_block(item)
            if block is not None:
                # image chunk：结构化渲染 + [[IMG:asset_id]] 引用标记
                content, _ = sanitize_text(block, source=item.filepath)
                parts.append(f"### [{i}] {title}\n{content}")
                continue
            content, _ = sanitize_text(item.page_content, source=item.filepath)
            parts.append(f"### [{i}] {title}\n{content}")
        return "\n\n".join(parts)

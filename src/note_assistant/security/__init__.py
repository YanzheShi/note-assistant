# src/note_assistant/security/__init__.py
"""安全防御包（docs/prompt-injection-defense-design.md，L0–L4）。

- guardrails：L1 提示词硬化——安全护栏条款 + 不可信内容分隔符包裹
- sanitize：L2 确定性输入清洗——注入形状启发式检测（flag/redact）
- output_guard：L4 输出治理——远程图片中和 + system prompt 泄露指纹
"""
from note_assistant.security.guardrails import (
    SECURITY_GUARDRAIL,
    append_guardrail,
    wrap_history_messages,
    wrap_retrieved_context,
    wrap_tool_result,
    wrap_user_question,
)
from note_assistant.security.sanitize import detect_injection, sanitize_text

__all__ = [
    "SECURITY_GUARDRAIL",
    "append_guardrail",
    "wrap_retrieved_context",
    "wrap_user_question",
    "wrap_tool_result",
    "wrap_history_messages",
    "detect_injection",
    "sanitize_text",
]

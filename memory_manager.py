"""
Rule-based memory manager for Adabot v1.

This layer decides what is worth saving and how retrieved memories are formatted
for prompt injection. It intentionally avoids LLM-based memory judgment for now.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from memory_store import add_memory, search_memory


DEFAULT_USER_ID = "default_user"
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 4000

MEMORY_TRIGGERS = [
    "记住",
    "以后",
    "我的项目",
    "我现在的项目",
    "我正在做",
    "我的环境",
    "我的系统",
    "我用的是",
    "我喜欢",
    "我不喜欢",
    "报错",
    "路径",
    "模型接口",
    "OCR",
    "Skill",
    "Gradio",
]


def trim_chat_history(history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    messages = list(history or [])
    return messages[-MAX_HISTORY_TURNS * 2 :]


def append_turn_to_history(
    history: List[Dict[str, str]] | None,
    user_text: str,
    assistant_answer: str,
    image_count: int = 0,
) -> List[Dict[str, str]]:
    messages = trim_chat_history(history)
    user_content = (user_text or "").strip()
    if not user_content and image_count:
        user_content = f"用户上传了 {image_count} 张图片。"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_answer})
    return trim_chat_history(messages)


def format_chat_history(history: List[Dict[str, str]] | None) -> Tuple[str, int]:
    messages = trim_chat_history(history)
    if not messages:
        return "", 0

    lines: List[str] = []
    for message in messages:
        role = "用户" if message.get("role") == "user" else "助手"
        content = " ".join((message.get("content") or "").split())
        if not content:
            continue
        lines.append(f"{role}: {content}")

    text = "\n".join(lines)
    if len(text) > MAX_HISTORY_CHARS:
        text = text[-MAX_HISTORY_CHARS:]
    return text, len(messages) // 2


def retrieve_memory_context(user_id: str, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
    memories = search_memory(user_id=user_id, query=query, top_k=top_k)
    if not memories:
        return [], ""

    lines = []
    for item in memories:
        lines.append(f"- [{item.get('memory_type', 'general')}] {item.get('content', '')}")
    return memories, "\n".join(lines)


def _memory_type_for(text: str) -> str:
    if any(keyword in text for keyword in ["我的项目", "我现在的项目", "我正在做", "核心流程", "main.py"]):
        return "project"
    if any(keyword in text for keyword in ["我的环境", "我的系统", "我用的是", "路径", "模型接口", "OCR", "Skill", "Gradio"]):
        return "environment"
    if any(keyword in text for keyword in ["我喜欢", "我不喜欢", "以后"]):
        return "preference"
    if "报错" in text or "错误" in text or "失败" in text:
        return "error_history"
    if any(keyword in text for keyword in ["进度", "已经完成", "下一步", "正在做"]):
        return "task_progress"
    return "general"


def _importance_for(memory_type: str, text: str) -> float:
    if "记住" in text:
        return 0.9
    if memory_type in {"project", "preference"}:
        return 0.8
    if memory_type in {"environment", "error_history"}:
        return 0.7
    return 0.5


def _clean_memory_content(text: str) -> str:
    cleaned = " ".join((text or "").split())
    for prefix in ["记住，", "记住,", "请记住，", "请记住,"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned


def extract_and_save_memories(
    user_id: str,
    user_text: str,
    assistant_answer: str | None = None,
    source: str = "chat",
) -> List[Dict[str, Any]]:
    """Persist durable user/project facts from the current turn."""
    text = (user_text or "").strip()
    if len(text) < 8:
        return []
    if not any(trigger in text for trigger in MEMORY_TRIGGERS):
        return []

    memory_type = _memory_type_for(text)
    if memory_type == "general" and len(text) < 16:
        return []

    content = _clean_memory_content(text)
    if not content:
        return []

    saved = add_memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=_importance_for(memory_type, text),
        source=source,
    )
    return [saved]

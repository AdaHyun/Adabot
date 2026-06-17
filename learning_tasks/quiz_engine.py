from __future__ import annotations

from .profile_store import mark_review


def generate_quiz(subject: str, knowledge_path: str, count: int = 3, difficulty: str = "基础") -> str:
    count = max(1, min(int(count or 3), 10))
    title = knowledge_path or subject or "当前知识点"
    lines = [f"### {title} 小测 ({difficulty})"]
    for index in range(1, count + 1):
        lines.append(f"{index}. 请用自己的话说明 `{title}` 的核心概念，并给出一个典型例子或解题步骤。")
    return "\n".join(lines)


def grade_quiz(task_id: str, knowledge_node_id: str, answer: str) -> str:
    text = (answer or "").strip()
    correct = len(text) >= 30 and "不会" not in text and "不懂" not in text
    mark_review(task_id, knowledge_node_id or "general", "correct" if correct else "wrong", text[:500])
    if correct:
        return "批改结果：初步通过。已提高该知识点 mastery_score。"
    return "批改结果：答案偏短或暴露不理解。已记录为需要继续复习。"


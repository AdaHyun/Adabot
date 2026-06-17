from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from .task_store import connect_db, init_db, new_id, now_text


def add_to_mistake_book(
    task_id: str,
    question_id: str | None,
    event: dict[str, Any],
    original_question: str = "",
    correct_solution: str = "",
    mistake_reason: str = "",
) -> str:
    init_db()
    stamp = now_text()
    next_review = (datetime.now() + timedelta(days=1)).isoformat(timespec="seconds")
    item_id = new_id("mb")
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO mistake_book
            (id, task_id, question_id, subject, knowledge_node_id, knowledge_path,
             original_question, mistake_reason, correct_solution, review_status,
             next_review_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                task_id,
                question_id,
                event.get("subject", ""),
                event.get("knowledge_node_id", "general"),
                json.dumps(event.get("knowledge_path") or [], ensure_ascii=False),
                original_question,
                mistake_reason or event.get("weakness_signal", "用户标记为错题"),
                correct_solution,
                "pending",
                next_review,
                stamp,
                stamp,
            ),
        )
    return item_id


def list_mistakes(task_id: str, subject: str = "", keyword: str = "") -> list[dict[str, Any]]:
    init_db()
    sql = "SELECT * FROM mistake_book WHERE task_id = ?"
    params: list[Any] = [task_id]
    if subject:
        sql += " AND subject = ?"
        params.append(subject)
    if keyword:
        sql += " AND (original_question LIKE ? OR knowledge_path LIKE ? OR mistake_reason LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    sql += " ORDER BY created_at DESC"
    with connect_db() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def render_mistakes(task_id: str, subject: str = "", keyword: str = "") -> str:
    rows = list_mistakes(task_id, subject, keyword)
    if not rows:
        return "暂无错题记录。"
    lines = ["### 错题本"]
    for index, row in enumerate(rows, start=1):
        path = " > ".join(json.loads(row.get("knowledge_path") or "[]"))
        lines.append(
            f"{index}. **{row.get('subject') or '通用'} / {path or row.get('knowledge_node_id')}**\n"
            f"   - 问题: {row.get('original_question') or '(未保存原题文本)'}\n"
            f"   - 错因: {row.get('mistake_reason') or '-'}\n"
            f"   - 下次复习: {row.get('next_review_at') or '-'}"
        )
    return "\n".join(lines)


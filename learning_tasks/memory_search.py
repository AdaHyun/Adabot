from __future__ import annotations

import json
from typing import Any

from .profile_analyzer import analyze_learning_event
from .profile_store import update_question_with_event, update_knowledge_state
from .task_router import build_learning_context
from .task_schema import LearningTask
from .task_store import connect_db, get_task, init_db


def search_questions(task_id: str, keyword: str = "", knowledge_node_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    task = get_task(task_id)
    if task:
        repair_obvious_misclassifications(task)
    sql = "SELECT * FROM questions WHERE task_id = ?"
    params: list[Any] = [task_id]
    if keyword:
        sql += " AND (user_input LIKE ? OR assistant_answer LIKE ? OR knowledge_path LIKE ? OR subject LIKE ? OR primary_skill LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
    if knowledge_node_id:
        sql += " AND knowledge_node_id = ?"
        params.append(knowledge_node_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with connect_db() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def render_search(task_name: str, task_id: str, keyword: str = "") -> str:
    rows = search_questions(task_id, keyword)
    if not rows:
        return f"当前任务【{task_name}】下没有检索到相关历史问题。"
    lines = [f"### 当前任务【{task_name}】的历史问题"]
    for index, row in enumerate(rows, start=1):
        path = _loads_path(row.get("knowledge_path"))
        answer = _summary(row.get("assistant_answer") or "")
        lines.append(
            f"{index}. **{row.get('created_at') or row.get('timestamp') or '-'}**\n"
            f"   - 用户问题: {row.get('user_input') or '(图片问题)'}\n"
            f"   - 回答摘要: {answer or '-'}\n"
            f"   - task_id: `{row.get('task_id')}`\n"
            f"   - task_name: {task_name}\n"
            f"   - subject: {row.get('subject') or '-'}\n"
            f"   - knowledge_path: {' > '.join(path) if path else row.get('knowledge_node_id') or '未分类'}\n"
            f"   - skill: {row.get('primary_skill') or '未识别'}\n"
            f"   - 是否来自当前任务: {'是' if row.get('task_id') == task_id else '否'}"
        )
    return "\n".join(lines)


def repair_obvious_misclassifications(task: LearningTask) -> int:
    """Fix old rows such as mechanical questions classified as advanced math."""
    repaired = 0
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM questions
            WHERE task_id = ?
            ORDER BY created_at DESC
            LIMIT 300
            """,
            (task.id,),
        ).fetchall()
    for row in rows:
        question = row["user_input"] or ""
        answer = row["assistant_answer"] or ""
        if not _looks_mechanical(question + "\n" + answer):
            continue
        if row["subject"] and "机械" in row["subject"] and row["primary_skill"] == "vehicle_engineering":
            continue
        context = build_learning_context(task, question, "vehicle_engineering")
        event = analyze_learning_event(task, context, question, answer)
        event["primary_skill"] = "vehicle_engineering"
        update_question_with_event(row["id"], event)
        update_knowledge_state(task.id, event)
        with connect_db() as conn:
            conn.execute("UPDATE questions SET primary_skill = ? WHERE id = ?", ("vehicle_engineering", row["id"]))
        repaired += 1
    return repaired


def _looks_mechanical(text: str) -> bool:
    return any(word in (text or "") for word in ["机械结构", "机械原理", "机构", "自由度", "运动副", "构件", "平面机构"])


def _loads_path(value: str | None) -> list[str]:
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _summary(text: str, limit: int = 160) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")

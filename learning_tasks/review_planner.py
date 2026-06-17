from __future__ import annotations

import json

from .task_store import connect_db, init_db


def generate_review_plan(task_id: str, limit: int = 10) -> str:
    init_db()
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM knowledge_state
            WHERE task_id = ?
            ORDER BY
                CASE status WHEN 'weak' THEN 0 WHEN 'learning' THEN 1 WHEN 'seen' THEN 2 ELSE 3 END,
                mistake_count DESC,
                weak_count DESC,
                COALESCE(last_reviewed_at, '') ASC,
                COALESCE(last_seen_at, '') ASC
            LIMIT ?
            """,
            (task_id, limit),
        ).fetchall()
    if not rows:
        return "暂无学习记录。先完成几次提问后，我会根据薄弱点生成复习建议。"
    lines = ["### 今日建议复习"]
    for index, row in enumerate(rows, start=1):
        path = " > ".join(json.loads(row["knowledge_path"] or "[]"))
        reason = []
        if row["status"] == "weak":
            reason.append("薄弱")
        if int(row["mistake_count"] or 0) > 0:
            reason.append(f"错题 {row['mistake_count']} 次")
        if int(row["weak_count"] or 0) > 0:
            reason.append(f"不理解信号 {row['weak_count']} 次")
        if not reason:
            reason.append("已接触但未掌握")
        lines.append(f"{index}. {path or row['knowledge_node_id']}: {'，'.join(reason)}，建议复习定义、典型题和易错点。")
    lines.append("\n### 本周建议\n优先把今日列表中的薄弱点各做 2-3 道基础题，并在两天后复盘错题。")
    return "\n".join(lines)


from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from .task_store import connect_db, init_db, new_id, now_text


def save_question(
    task_id: str,
    user_input: str,
    assistant_answer: str,
    primary_skill: str,
    subject: str,
    knowledge_node_id: str = "",
    knowledge_path: list[str] | None = None,
    question_type: str = "",
    difficulty: str = "",
) -> str:
    init_db()
    question_id = new_id("q")
    stamp = now_text()
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO questions
            (id, task_id, timestamp, user_input, assistant_answer, primary_skill, subject,
             knowledge_node_id, knowledge_path, question_type, difficulty, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                task_id,
                stamp,
                user_input,
                assistant_answer,
                primary_skill,
                subject,
                knowledge_node_id,
                json.dumps(knowledge_path or [], ensure_ascii=False),
                question_type,
                difficulty,
                stamp,
            ),
        )
    return question_id


def update_question_with_event(question_id: str, event: dict[str, Any]) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE questions
            SET subject = ?, knowledge_node_id = ?, knowledge_path = ?, question_type = ?, difficulty = ?
            WHERE id = ?
            """,
            (
                event.get("subject", ""),
                event.get("knowledge_node_id", ""),
                json.dumps(event.get("knowledge_path") or [], ensure_ascii=False),
                event.get("question_type", ""),
                event.get("difficulty", ""),
                question_id,
            ),
        )


def save_knowledge_event(question_id: str, event: dict[str, Any]) -> str:
    event_id = new_id("ke")
    stamp = now_text()
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO knowledge_events
            (id, task_id, question_id, subject, knowledge_node_id, knowledge_path,
             question_type, difficulty, confidence, weakness_signal,
             should_add_to_mistake_book, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event.get("task_id"),
                question_id,
                event.get("subject"),
                event.get("knowledge_node_id"),
                json.dumps(event.get("knowledge_path") or [], ensure_ascii=False),
                event.get("question_type"),
                event.get("difficulty"),
                float(event.get("confidence") or 0),
                event.get("weakness_signal", ""),
                1 if event.get("should_add_to_mistake_book") else 0,
                stamp,
            ),
        )
    return event_id


def update_knowledge_state(task_id: str, event: dict[str, Any]) -> None:
    node_id = event.get("knowledge_node_id") or "general"
    stamp = now_text()
    next_review_at = _next_review_time(stamp, days=7)
    weak = bool(event.get("weakness_signal"))
    mistake = bool(event.get("should_add_to_mistake_book"))
    with connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_state WHERE task_id = ? AND knowledge_node_id = ?",
            (task_id, node_id),
        ).fetchone()
        if row is None:
            status = "weak" if (weak or mistake) else "seen"
            conn.execute(
                """
                INSERT INTO knowledge_state
                (id, task_id, subject, knowledge_node_id, knowledge_path, status,
                 seen_count, weak_count, mistake_count, mastery_score,
                 first_accessed_at, last_seen_at, next_review_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("ks"),
                    task_id,
                    event.get("subject"),
                    node_id,
                    json.dumps(event.get("knowledge_path") or [], ensure_ascii=False),
                    status,
                    1,
                    1 if weak else 0,
                    1 if mistake else 0,
                    0.0,
                    stamp,
                    stamp,
                    next_review_at,
                    stamp,
                ),
            )
            return
        seen_count = int(row["seen_count"] or 0) + 1
        weak_count = int(row["weak_count"] or 0) + (1 if weak else 0)
        mistake_count = int(row["mistake_count"] or 0) + (1 if mistake else 0)
        mastery_score = float(row["mastery_score"] or 0)
        if mistake or weak:
            status = "weak"
        elif seen_count >= 3 and mastery_score >= 0.6:
            status = "mastered"
        elif seen_count >= 2:
            status = "learning"
        else:
            status = "seen"
        conn.execute(
            """
            UPDATE knowledge_state
            SET subject = ?, knowledge_path = ?, status = ?, seen_count = ?,
                weak_count = ?, mistake_count = ?, last_seen_at = ?, next_review_at = ?, updated_at = ?
            WHERE task_id = ? AND knowledge_node_id = ?
            """,
            (
                event.get("subject"),
                json.dumps(event.get("knowledge_path") or [], ensure_ascii=False),
                status,
                seen_count,
                weak_count,
                mistake_count,
                stamp,
                next_review_at,
                stamp,
                task_id,
                node_id,
            ),
        )


def mark_review(task_id: str, knowledge_node_id: str, result: str, notes: str = "") -> None:
    stamp = now_text()
    delta = 0.15 if result == "correct" else -0.1
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO review_records (id, task_id, knowledge_node_id, review_type, result, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id("rr"), task_id, knowledge_node_id, "mini_quiz", result, notes, stamp),
        )
        row = conn.execute(
            "SELECT mastery_score, review_count FROM knowledge_state WHERE task_id = ? AND knowledge_node_id = ?",
            (task_id, knowledge_node_id),
        ).fetchone()
        if row:
            score = min(1.0, max(0.0, float(row["mastery_score"] or 0) + delta))
            status = "proficient" if score >= 0.85 else "mastered" if score >= 0.65 else "learning"
            if result != "correct":
                status = "weak"
            conn.execute(
                """
                UPDATE knowledge_state
                SET mastery_score = ?, review_count = ?, status = ?, last_reviewed_at = ?, next_review_at = ?, updated_at = ?
                WHERE task_id = ? AND knowledge_node_id = ?
                """,
                (score, int(row["review_count"] or 0) + 1, status, stamp, _next_review_time(stamp, days=14), stamp, task_id, knowledge_node_id),
            )


def load_recent_chat_messages(task_id: str, limit: int = 20) -> list[dict[str, str]]:
    init_db()
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT user_input, assistant_answer
            FROM questions
            WHERE task_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (task_id, max(1, limit)),
        ).fetchall()
    messages: list[dict[str, str]] = []
    for row in reversed(rows):
        if row["user_input"]:
            messages.append({"role": "user", "content": row["user_input"]})
        if row["assistant_answer"]:
            messages.append({"role": "assistant", "content": row["assistant_answer"]})
    return messages


def set_knowledge_status(task_id: str, knowledge_node_id: str, status: str) -> None:
    if status not in {"unvisited", "seen", "learning", "weak", "mastered", "proficient"}:
        raise ValueError(f"unsupported status: {status}")
    stamp = now_text()
    with connect_db() as conn:
        row = conn.execute(
            "SELECT id FROM knowledge_state WHERE task_id = ? AND knowledge_node_id = ?",
            (task_id, knowledge_node_id),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE knowledge_state SET status = ?, updated_at = ? WHERE task_id = ? AND knowledge_node_id = ?",
                (status, stamp, task_id, knowledge_node_id),
            )


def _next_review_time(stamp: str, days: int) -> str:
    try:
        base = datetime.fromisoformat(stamp)
    except ValueError:
        base = datetime.now()
    return (base + timedelta(days=days)).isoformat(timespec="seconds")

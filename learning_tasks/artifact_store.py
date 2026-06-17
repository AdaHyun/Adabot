from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .knowledge_graph_store import list_all_nodes, list_children, upsert_knowledge_path
from .mistake_store import list_mistakes
from .task_schema import LearningTask
from .task_store import connect_db


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "tasks"


def task_artifact_dir(task_id: str) -> Path:
    path = TASK_ARTIFACT_ROOT / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def sync_task_artifacts(task: LearningTask) -> None:
    task_artifact_dir(task.id)
    for subject in task.subjects or []:
        write_knowledge_graph(task, subject)
    write_learning_profile(task)
    write_history_index(task)


def write_knowledge_graph(task: LearningTask, subject: str) -> Path:
    states = _state_map(task.id)
    question_counts = _question_counts(task.id)
    nodes = []
    all_nodes = list_all_nodes(task, subject)
    children_by_node = {node["id"]: list_children(task, subject, node["id"]) for node in all_nodes}
    for node in all_nodes:
        state = states.get(node["id"]) or {}
        nodes.append(
            {
                "node_id": node["id"],
                "title": node["title"],
                "path": node["path"],
                "parent_id": node.get("parent_id") or "",
                "children": [child["id"] for child in children_by_node.get(node["id"], [])],
                "basic_framework": [child["title"] for child in children_by_node.get(node["id"], [])],
                "mastery_status": _status_label(state.get("status") or "unvisited"),
                "question_count": question_counts.get(node["id"], 0),
                "wrong_question_count": int(state.get("mistake_count") or 0),
                "last_touched_at": state.get("last_seen_at") or "",
                "summary": state.get("summary", ""),
                "source": node.get("source_type") or "db_graph",
            }
        )
    data = {
        "task_id": task.id,
        "task_name": task.task_name,
        "subject": subject,
        "updated_at": _now(),
        "nodes": nodes,
    }
    path = task_artifact_dir(task.id) / f"knowledge_graph_{_safe_name(subject)}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_learning_profile(task: LearningTask) -> Path:
    states = _state_map(task.id)
    subject_progress: dict[str, Any] = {}
    for subject in task.subjects or []:
        subject_states = [item for item in states.values() if item.get("subject") == subject]
        subject_progress[subject] = {
            "touched_nodes": sum(1 for item in subject_states if item.get("status") != "unvisited"),
            "weak_nodes": [item.get("knowledge_path") for item in subject_states if item.get("status") == "weak"],
            "mastered_nodes": [
                item.get("knowledge_path")
                for item in subject_states
                if item.get("status") in {"mastered", "proficient"}
            ],
            "missing_prerequisites": [],
        }
    data = {
        "task_id": task.id,
        "task_name": task.task_name,
        "updated_at": _now(),
        "subject_progress": subject_progress,
        "user_notes_summary": _read_json_list(task_artifact_dir(task.id) / "notes_summary.json"),
        "learning_gaps": [],
        "next_review_suggestions": [],
    }
    path = task_artifact_dir(task.id) / "learning_profile.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_history_index(task: LearningTask) -> Path:
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, user_input, assistant_answer, primary_skill, subject,
                   knowledge_node_id, knowledge_path
            FROM questions
            WHERE task_id = ?
            ORDER BY created_at DESC
            """,
            (task.id,),
        ).fetchall()
    mistake_ids = {item.get("question_id") for item in list_mistakes(task.id)}
    items = []
    for row in rows:
        path = _loads(row["knowledge_path"])
        items.append(
            {
                "question_id": row["id"],
                "created_at": row["created_at"],
                "user_question": row["user_input"],
                "answer_summary": _summary(row["assistant_answer"] or ""),
                "full_answer": row["assistant_answer"] or "",
                "subject": row["subject"] or "",
                "skill": row["primary_skill"] or "",
                "knowledge_path": path,
                "node_id": row["knowledge_node_id"] or "",
                "is_wrong_question": row["id"] in mistake_ids,
                "need_review": False,
                "task_id": task.id,
            }
        )
    data = {"task_id": task.id, "task_name": task.task_name, "updated_at": _now(), "items": items}
    path = task_artifact_dir(task.id) / "history_index.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def merge_notes_into_knowledge_graph(
    task: LearningTask,
    detected_subject: str,
    extracted_nodes: list[dict[str, Any]],
    source_file: str = "",
) -> None:
    subject = detected_subject or (task.subjects[0] if task.subjects else "??")
    for node in extracted_nodes:
        node_path = node.get("path") or [subject, node.get("title", "??????")]
        upsert_knowledge_path(task, node_path, source_type="user_note", source_ref=source_file)
    write_knowledge_graph(task, subject)


def stable_node_id(path: list[str]) -> str:
    text = "_".join(path)
    return "note_" + str(abs(hash(text)) % 10_000_000)


def _state_map(task_id: str) -> dict[str, dict[str, Any]]:
    with connect_db() as conn:
        rows = conn.execute("SELECT * FROM knowledge_state WHERE task_id = ?", (task_id,)).fetchall()
    return {row["knowledge_node_id"]: dict(row) for row in rows}


def _question_counts(task_id: str) -> dict[str, int]:
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT knowledge_node_id, COUNT(*) AS count FROM questions WHERE task_id = ? GROUP BY knowledge_node_id",
            (task_id,),
        ).fetchall()
    return {row["knowledge_node_id"]: int(row["count"]) for row in rows}


def _status_label(status: str) -> str:
    return {
        "unvisited": "未接触",
        "seen": "已接触",
        "learning": "需要复习",
        "weak": "需要复习",
        "mastered": "已掌握",
        "proficient": "已掌握",
    }.get(status or "unvisited", "未接触")


def _read_json_list(path: Path) -> list[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _loads(value: str | None) -> list[str]:
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _summary(text: str, limit: int = 180) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value or "subject")[:80]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

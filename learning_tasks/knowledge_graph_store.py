from __future__ import annotations

import hashlib
import json
from typing import Any

from .syllabus_manager import flatten_syllabus_nodes, get_task_subject_syllabus
from .task_schema import LearningTask
from .task_store import connect_db, init_db, now_text


ROOT_NODE_ID = "__root__"


def ensure_knowledge_graph_seeded(task: LearningTask, subject: str) -> None:
    """Seed the long-lived graph table from the configured syllabus once."""
    init_db()
    subject = subject or (task.subjects[0] if task.subjects else "通用")
    with connect_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM knowledge_nodes WHERE task_id = ? AND subject = ? LIMIT 1",
            (task.id, subject),
        ).fetchone()
    if row:
        return

    syllabus = get_task_subject_syllabus(task, subject)
    flat_nodes = flatten_syllabus_nodes(syllabus)
    if not flat_nodes:
        upsert_knowledge_path(task, [subject, "通用"], source_type="seed")
        return

    parent_by_path = _parent_id_by_path(flat_nodes)
    stamp = now_text()
    with connect_db() as conn:
        for index, node in enumerate(flat_nodes):
            path = node.get("path") or [subject, node.get("name", "未命名知识点")]
            node_id = str(node.get("id") or stable_node_id(task.id, subject, path))
            conn.execute(
                """
                INSERT OR IGNORE INTO knowledge_nodes
                (id, task_id, subject, parent_id, title, path_json, source_type, source_ref,
                 sort_order, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    node_id,
                    task.id,
                    subject,
                    parent_by_path.get(" > ".join(path), ""),
                    node.get("name") or path[-1],
                    json.dumps(path, ensure_ascii=False),
                    "syllabus",
                    syllabus.get("syllabus_id", ""),
                    index,
                    stamp,
                    stamp,
                ),
            )


def list_children(task: LearningTask, subject: str, parent_id: str | None = ROOT_NODE_ID) -> list[dict[str, Any]]:
    ensure_knowledge_graph_seeded(task, subject)
    parent = "" if parent_id in {"", None, ROOT_NODE_ID} else str(parent_id)
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM knowledge_nodes
            WHERE task_id = ? AND subject = ? AND COALESCE(parent_id, '') = ? AND is_active = 1
            ORDER BY sort_order, title, id
            """,
            (task.id, subject, parent),
        ).fetchall()
    return [_row_to_node(row) for row in rows]


def get_node(task: LearningTask, subject: str, node_id: str | None) -> dict[str, Any] | None:
    if not node_id:
        return None
    ensure_knowledge_graph_seeded(task, subject)
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM knowledge_nodes
            WHERE task_id = ? AND subject = ? AND id = ? AND is_active = 1
            """,
            (task.id, subject, node_id),
        ).fetchone()
    return _row_to_node(row) if row else None


def list_all_nodes(task: LearningTask, subject: str) -> list[dict[str, Any]]:
    ensure_knowledge_graph_seeded(task, subject)
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM knowledge_nodes
            WHERE task_id = ? AND subject = ? AND is_active = 1
            ORDER BY sort_order, title, id
            """,
            (task.id, subject),
        ).fetchall()
    return [_row_to_node(row) for row in rows]


def upsert_knowledge_path(
    task: LearningTask,
    path: list[str],
    source_type: str = "agent",
    source_ref: str = "",
) -> str:
    """Create missing nodes along a path and return the leaf node id."""
    clean_path = [str(item).strip() for item in path or [] if str(item).strip()]
    if not clean_path:
        clean_path = [(task.subjects or ["通用"])[0], "通用"]
    subject = clean_path[0]
    parent_id = ""
    leaf_id = ""
    stamp = now_text()
    start_depth = 2 if len(clean_path) > 1 else 1
    for depth in range(start_depth, len(clean_path) + 1):
        node_path = clean_path[:depth]
        title = node_path[-1]
        node_id = stable_node_id(task.id, subject, node_path)
        with connect_db() as conn:
            parent_path = clean_path[: depth - 1]
            if depth > start_depth:
                parent_row = conn.execute(
                    """
                    SELECT id FROM knowledge_nodes
                    WHERE task_id = ? AND subject = ? AND path_json = ?
                    """,
                    (task.id, subject, json.dumps(parent_path, ensure_ascii=False)),
                ).fetchone()
                parent_id = parent_row["id"] if parent_row else parent_id
            existing = conn.execute(
                """
                SELECT id FROM knowledge_nodes
                WHERE task_id = ? AND subject = ? AND path_json = ?
                """,
                (task.id, subject, json.dumps(node_path, ensure_ascii=False)),
            ).fetchone()
            if existing:
                leaf_id = existing["id"]
                parent_id = leaf_id
                continue
            conn.execute(
                """
                INSERT INTO knowledge_nodes
                (id, task_id, subject, parent_id, title, path_json, source_type, source_ref,
                 sort_order, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    node_id,
                    task.id,
                    subject,
                    parent_id,
                    title,
                    json.dumps(node_path, ensure_ascii=False),
                    source_type,
                    source_ref,
                    _next_sort_order(task.id, subject, parent_id),
                    stamp,
                    stamp,
                ),
            )
        leaf_id = node_id
        parent_id = node_id
    return leaf_id


def find_best_node_for_task(task: LearningTask, subject: str, text: str) -> tuple[str, list[str], str, float]:
    ensure_knowledge_graph_seeded(task, subject)
    lower = (text or "").lower()
    best_node: dict[str, Any] | None = None
    best_score = 0.0
    for node in list_all_nodes(task, subject):
        title = node.get("title", "")
        path = node.get("path") or []
        score = 0.0
        if title and title.lower() in lower:
            score += len(title) + 3
        for part in path:
            if part and str(part).lower() in lower:
                score += max(1, len(str(part)) / 2)
        if score > best_score:
            best_score = score
            best_node = node
    if best_node:
        return (
            best_node["id"],
            best_node.get("path") or [subject, best_node.get("title", "")],
            "db_graph",
            min(0.95, 0.45 + best_score / 20),
        )
    return "general", [subject or "通用", "通用"], "db_graph", 0.2


def stable_node_id(task_id: str, subject: str, path: list[str]) -> str:
    raw = " > ".join([task_id, subject, *path])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"kn_{digest}"


def _parent_id_by_path(flat_nodes: list[dict[str, Any]]) -> dict[str, str]:
    by_path = {" > ".join(node.get("path") or []): str(node.get("id") or "") for node in flat_nodes}
    result: dict[str, str] = {}
    for node in flat_nodes:
        path = node.get("path") or []
        key = " > ".join(path)
        parent_key = " > ".join(path[:-1])
        result[key] = by_path.get(parent_key, "") if len(path) > 1 else ""
    return result


def _next_sort_order(task_id: str, subject: str, parent_id: str) -> int:
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order
            FROM knowledge_nodes
            WHERE task_id = ? AND subject = ? AND COALESCE(parent_id, '') = ?
            """,
            (task_id, subject, parent_id or ""),
        ).fetchone()
    return int(row["next_order"] or 0)


def _row_to_node(row) -> dict[str, Any]:
    data = dict(row)
    try:
        path = json.loads(data.get("path_json") or "[]")
    except Exception:
        path = []
    data["path"] = path if isinstance(path, list) else []
    data["name"] = data.get("title") or data.get("id")
    return data

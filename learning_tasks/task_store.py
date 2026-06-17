from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .task_schema import LearningTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "learning_agent.db"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS learning_tasks (
                id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                role_type TEXT NOT NULL,
                goal_description TEXT,
                target_exam TEXT,
                target_date TEXT,
                subjects_json TEXT,
                syllabus_config_json TEXT,
                answer_style TEXT,
                enabled_skills_json TEXT,
                is_active INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                timestamp TEXT,
                user_input TEXT,
                assistant_answer TEXT,
                primary_skill TEXT,
                subject TEXT,
                knowledge_node_id TEXT,
                knowledge_path TEXT,
                question_type TEXT,
                difficulty TEXT,
                created_at TEXT,
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_events (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                subject TEXT,
                knowledge_node_id TEXT,
                knowledge_path TEXT,
                question_type TEXT,
                difficulty TEXT,
                confidence REAL,
                weakness_signal TEXT,
                should_add_to_mistake_book INTEGER DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id),
                FOREIGN KEY(question_id) REFERENCES questions(id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_state (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                subject TEXT,
                knowledge_node_id TEXT,
                knowledge_path TEXT,
                status TEXT,
                seen_count INTEGER DEFAULT 0,
                weak_count INTEGER DEFAULT 0,
                mistake_count INTEGER DEFAULT 0,
                review_count INTEGER DEFAULT 0,
                mastery_score REAL DEFAULT 0,
                last_seen_at TEXT,
                last_reviewed_at TEXT,
                updated_at TEXT,
                UNIQUE(task_id, knowledge_node_id),
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id)
            );
            CREATE TABLE IF NOT EXISTS mistake_book (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                question_id TEXT,
                subject TEXT,
                knowledge_node_id TEXT,
                knowledge_path TEXT,
                original_question TEXT,
                mistake_reason TEXT,
                correct_solution TEXT,
                review_status TEXT,
                next_review_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id)
            );
            CREATE TABLE IF NOT EXISTS review_records (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                knowledge_node_id TEXT,
                review_type TEXT,
                result TEXT,
                notes TEXT,
                created_at TEXT,
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id)
            );
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                parent_id TEXT,
                title TEXT NOT NULL,
                path_json TEXT,
                source_type TEXT,
                source_ref TEXT,
                sort_order INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY(task_id, subject, id),
                FOREIGN KEY(task_id) REFERENCES learning_tasks(id)
            );
            """
        )


def create_task(
    task_name: str,
    role_type: str = "custom",
    goal_description: str = "",
    target_exam: str = "",
    target_date: str = "",
    subjects: list[str] | None = None,
    syllabus_config: dict[str, Any] | None = None,
    answer_style: str = "",
    enabled_skills: list[str] | None = None,
    make_active: bool = False,
) -> LearningTask:
    init_db()
    stamp = now_text()
    task = LearningTask(
        id=new_id("task"),
        task_name=task_name.strip() or "默认学习任务",
        role_type=role_type or "custom",
        goal_description=goal_description or "",
        target_exam=target_exam or "",
        target_date=target_date or "",
        subjects=subjects or [],
        syllabus_config=syllabus_config or {},
        answer_style=answer_style or "",
        enabled_skills=enabled_skills or [],
        is_active=make_active,
        created_at=stamp,
        updated_at=stamp,
    )
    with connect_db() as conn:
        if make_active:
            conn.execute("UPDATE learning_tasks SET is_active = 0")
        conn.execute(
            """
            INSERT INTO learning_tasks
            (id, task_name, role_type, goal_description, target_exam, target_date,
             subjects_json, syllabus_config_json, answer_style, enabled_skills_json,
             is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            task.to_db_tuple(),
        )
    return task


def list_tasks(include_inactive: bool = True) -> list[LearningTask]:
    init_db()
    sql = "SELECT * FROM learning_tasks"
    if not include_inactive:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY is_active DESC, updated_at DESC, created_at DESC"
    with connect_db() as conn:
        return [LearningTask.from_row(row) for row in conn.execute(sql).fetchall()]


def get_task(task_id: str | None) -> LearningTask | None:
    if not task_id:
        return None
    init_db()
    with connect_db() as conn:
        row = conn.execute("SELECT * FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
    return LearningTask.from_row(row) if row else None


def set_active_task(task_id: str) -> LearningTask | None:
    init_db()
    stamp = now_text()
    with connect_db() as conn:
        exists = conn.execute("SELECT id FROM learning_tasks WHERE id = ?", (task_id,)).fetchone()
        if not exists:
            return None
        conn.execute("UPDATE learning_tasks SET is_active = 0")
        conn.execute(
            "UPDATE learning_tasks SET is_active = 1, updated_at = ? WHERE id = ?",
            (stamp, task_id),
        )
    return get_task(task_id)


def get_active_task() -> LearningTask | None:
    init_db()
    with connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM learning_tasks WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            row = conn.execute("SELECT * FROM learning_tasks ORDER BY created_at DESC LIMIT 1").fetchone()
    return LearningTask.from_row(row) if row else None


def update_task(task_id: str, **fields: Any) -> LearningTask | None:
    init_db()
    allowed = {
        "task_name",
        "role_type",
        "goal_description",
        "target_exam",
        "target_date",
        "answer_style",
    }
    db_fields: dict[str, Any] = {key: value for key, value in fields.items() if key in allowed}
    if "subjects" in fields:
        db_fields["subjects_json"] = json.dumps(fields["subjects"] or [], ensure_ascii=False)
    if "syllabus_config" in fields:
        db_fields["syllabus_config_json"] = json.dumps(fields["syllabus_config"] or {}, ensure_ascii=False)
    if "enabled_skills" in fields:
        db_fields["enabled_skills_json"] = json.dumps(fields["enabled_skills"] or [], ensure_ascii=False)
    if not db_fields:
        return get_task(task_id)
    db_fields["updated_at"] = now_text()
    assignments = ", ".join(f"{key} = ?" for key in db_fields)
    values = list(db_fields.values()) + [task_id]
    with connect_db() as conn:
        conn.execute(f"UPDATE learning_tasks SET {assignments} WHERE id = ?", values)
    return get_task(task_id)


def deactivate_task(task_id: str) -> None:
    init_db()
    with connect_db() as conn:
        conn.execute(
            "UPDATE learning_tasks SET is_active = 0, updated_at = ? WHERE id = ?",
            (now_text(), task_id),
        )


def get_or_create_default_task() -> LearningTask:
    active = get_active_task()
    if active:
        return active
    return create_task(
        task_name="默认学习任务",
        role_type="custom",
        goal_description="通用学习",
        subjects=["自定义专业示例"],
        answer_style="清晰、分步骤、适合复习",
        enabled_skills=["math", "english", "vehicle_engineering", "general"],
        make_active=True,
    )

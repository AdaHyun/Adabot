from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _dumps(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False)


@dataclass
class LearningTask:
    id: str
    task_name: str
    role_type: str = "custom"
    goal_description: str = ""
    target_exam: str = ""
    target_date: str = ""
    subjects: list[str] = field(default_factory=list)
    syllabus_config: dict[str, Any] = field(default_factory=dict)
    answer_style: str = ""
    enabled_skills: list[str] = field(default_factory=list)
    is_active: bool = False
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_row(cls, row: Any) -> "LearningTask":
        return cls(
            id=row["id"],
            task_name=row["task_name"],
            role_type=row["role_type"],
            goal_description=row["goal_description"] or "",
            target_exam=row["target_exam"] or "",
            target_date=row["target_date"] or "",
            subjects=_loads_list(row["subjects_json"]),
            syllabus_config=json.loads(row["syllabus_config_json"] or "{}"),
            answer_style=row["answer_style"] or "",
            enabled_skills=_loads_list(row["enabled_skills_json"]),
            is_active=bool(row["is_active"]),
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def to_db_tuple(self) -> tuple[Any, ...]:
        return (
            self.id,
            self.task_name,
            self.role_type,
            self.goal_description,
            self.target_exam,
            self.target_date,
            _dumps(self.subjects),
            json.dumps(self.syllabus_config or {}, ensure_ascii=False),
            self.answer_style,
            _dumps(self.enabled_skills),
            1 if self.is_active else 0,
            self.created_at,
            self.updated_at,
        )


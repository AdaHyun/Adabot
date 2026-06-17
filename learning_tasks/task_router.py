from __future__ import annotations

from .context_schema import LearningContext
from .knowledge_graph_store import find_best_node_for_task
from .profile_analyzer import infer_subject
from .syllabus_manager import load_role_config
from .task_schema import LearningTask


def build_learning_context(task: LearningTask, user_input: str, primary_skill: str) -> LearningContext:
    role = load_role_config(task.role_type)
    subject = infer_subject(task, user_input, primary_skill)
    node_id, path, syllabus_id, _confidence = find_best_node_for_task(task, subject, user_input)
    return LearningContext(
        task_id=task.id,
        task_name=task.task_name,
        role_type=task.role_type,
        goal_description=task.goal_description,
        subject=subject,
        primary_skill=primary_skill or "general",
        secondary_skills=["question_to_map", "learning_profile", "memory_search"],
        syllabus_id=syllabus_id,
        knowledge_node_id=node_id,
        knowledge_path=path,
        answer_style=task.answer_style or role.get("default_answer_style", ""),
    )

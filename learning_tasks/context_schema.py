from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LearningContext:
    task_id: str
    task_name: str
    role_type: str
    goal_description: str = ""
    subject: str = "通用"
    primary_skill: str = "general"
    secondary_skills: list[str] = field(default_factory=list)
    syllabus_id: str = ""
    knowledge_node_id: str = "general"
    knowledge_path: list[str] = field(default_factory=list)
    answer_style: str = ""
    profile_scope: str = "task"

    def to_prompt_text(self, role_name: str = "") -> str:
        path = " > ".join(self.knowledge_path) if self.knowledge_path else "暂未匹配"
        return (
            f"当前学习任务: {self.task_name}\n"
            f"当前角色: {role_name or self.role_type}\n"
            f"当前目标: {self.goal_description or '通用学习'}\n"
            f"当前科目: {self.subject}\n"
            f"当前主 Skill: {self.primary_skill}\n"
            f"当前知识点: {path}\n"
            f"回答风格: {self.answer_style or '清晰、分步骤、适合复习'}"
        )


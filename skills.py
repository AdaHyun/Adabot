"""
skills.py

Tool-style Skill registry.

The old keyword router has been removed. Intent routing should be handled by an
LLM router that chooses one or more registered tools and passes structured
arguments that match each tool's JSON schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


ToolFunc = Callable[..., Any]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc

    def to_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


SKILL_REGISTRY: Dict[str, SkillSpec] = {}


def register_skill(
    name: str,
    func: ToolFunc,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> None:
    if not name or not name.strip():
        raise ValueError("Skill 名称不能为空")
    if not callable(func):
        raise TypeError("Skill 必须是可调用函数")
    SKILL_REGISTRY[name] = SkillSpec(
        name=name,
        description=description.strip(),
        parameters=parameters or {"type": "object", "properties": {}, "additionalProperties": False},
        func=func,
    )


def list_skills() -> List[str]:
    return sorted(SKILL_REGISTRY.keys())


def tool_specs() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool schemas for an LLM router."""
    return [SKILL_REGISTRY[name].to_tool_schema() for name in list_skills()]


def call_skill(name: str, **kwargs: Any) -> Any:
    spec = SKILL_REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Unknown skill: {name}")
    return spec.func(**kwargs)


def select_skills(text: str) -> List[str]:
    """
    Backward-compatible fallback.

    Keyword routing is intentionally disabled. Until the LLM router is wired in,
    the chat prompt receives only the general tutoring policy.
    """
    return ["general"] if "general" in SKILL_REGISTRY else []


def run_selected_skills(text: str) -> Tuple[List[str], str]:
    """
    Compatibility adapter for the current main.py prompt flow.

    The new architecture expects structured tool calls. This adapter keeps the
    existing Gradio app working by executing the general prompt-policy tool with
    a structured argument.
    """
    selected_names = select_skills(text)
    outputs: list[str] = []
    for name in selected_names:
        try:
            result = call_skill(name, user_text=text)
            if isinstance(result, dict):
                prompt = result.get("prompt") or result.get("message") or ""
                outputs.append(str(prompt or result))
            else:
                outputs.append(str(result))
        except Exception as exc:
            outputs.append(f"[Skill {name} 执行失败]: {exc}")
    return selected_names, "\n\n".join(item for item in outputs if item)


def skill_general_tutor(user_text: str = "", **_: Any) -> dict[str, Any]:
    return {
        "prompt": (
            "[通用学习助手]\n"
            "请作为考研学习助手回答。优先给出直接答案，再补充必要解释、步骤和复习建议。\n"
            "如果问题来自 OCR，注意可能存在识别错误，请结合上下文合理判断。"
        ),
        "input_preview": user_text[:200],
    }


def skill_solve_math_problem(
    problem_text: str = "",
    topic_path: str = "",
    constraints: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    return {
        "prompt": (
            "[数学解题工具]\n"
            "请优先识别题型，例如极限、导数、积分、级数、线性代数或概率统计。"
            "回答时先写核心思路，再分步骤推导；涉及公式时保持符号清晰。"
        ),
        "problem_text": problem_text,
        "topic_path": topic_path,
        "constraints": constraints or {},
    }


def skill_explain_vehicle_engineering(
    question: str = "",
    topic_path: str = "",
    **_: Any,
) -> dict[str, Any]:
    return {
        "prompt": (
            "[车辆工程工具]\n"
            "请从结构组成、工作原理、关键参数、常见失效和设计取舍角度回答。"
            "如果适合考研复习，请补充高频考点和易混概念。"
        ),
        "question": question,
        "topic_path": topic_path,
    }


def skill_polish_english(
    text: str = "",
    task_type: str = "general",
    target_level: str = "",
    **_: Any,
) -> dict[str, Any]:
    return {
        "prompt": (
            "[英语学习工具]\n"
            "阅读题要定位关键词、解释同义替换和答案依据；写作题要给出结构、论点和可替换表达。"
        ),
        "text": text,
        "task_type": task_type,
        "target_level": target_level,
    }


def skill_store_mistake(
    image_path: str = "",
    user_text: str = "",
    mistake_reason: str = "",
    current_graph_context: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    return {
        "action": "store_mistake",
        "image_path": image_path,
        "user_text": user_text,
        "mistake_reason": mistake_reason,
        "current_graph_context": current_graph_context or {},
    }


def skill_generate_quiz(topic_path: str = "", count: int = 3, difficulty: str = "基础", **_: Any) -> list[dict[str, Any]]:
    count = max(1, min(int(count or 3), 10))
    return [
        {
            "type": "short_answer",
            "topic_path": topic_path,
            "difficulty": difficulty,
            "question": f"请围绕「{topic_path or '当前知识点'}」生成第 {index} 道练习题。",
        }
        for index in range(1, count + 1)
    ]


STRING_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"user_text": {"type": "string", "description": "用户原始输入文本"}},
    "required": ["user_text"],
    "additionalProperties": True,
}


register_skill(
    "general",
    skill_general_tutor,
    "通用学习辅导策略，用于没有更具体工具时的兜底回答。",
    STRING_INPUT_SCHEMA,
)
register_skill(
    "math",
    skill_solve_math_problem,
    "解决数学题，适用于高数、线代、概率论等题目。",
    {
        "type": "object",
        "properties": {
            "problem_text": {"type": "string"},
            "topic_path": {"type": "string"},
            "constraints": {"type": "object"},
        },
        "required": ["problem_text"],
        "additionalProperties": True,
    },
)
register_skill(
    "vehicle_engineering",
    skill_explain_vehicle_engineering,
    "解释车辆工程、机械原理、机构结构等专业问题。",
    {
        "type": "object",
        "properties": {"question": {"type": "string"}, "topic_path": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": True,
    },
)
register_skill(
    "english",
    skill_polish_english,
    "处理英语阅读、写作、翻译、语法和表达优化。",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "task_type": {"type": "string"},
            "target_level": {"type": "string"},
        },
        "required": ["text"],
        "additionalProperties": True,
    },
)
register_skill(
    "mistake_book",
    skill_store_mistake,
    "把用户的错题文本和可选截图整理为错题记录。",
    {
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "user_text": {"type": "string"},
            "mistake_reason": {"type": "string"},
            "current_graph_context": {"type": "object"},
        },
        "required": ["user_text"],
        "additionalProperties": True,
    },
)
register_skill(
    "mini_quiz",
    skill_generate_quiz,
    "围绕指定知识点生成小测验题目。",
    {
        "type": "object",
        "properties": {
            "topic_path": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "difficulty": {"type": "string"},
        },
        "required": ["topic_path"],
        "additionalProperties": True,
    },
)


def _placeholder_tool(**kwargs: Any) -> dict[str, Any]:
    return {"status": "placeholder", "arguments": kwargs}


for _name, _description in {
    "paper_reading": "论文精读与文献分析工具。",
    "project_debugger": "项目调试与报错定位工具。",
    "crawler_assistant": "爬虫任务规划与数据抽取工具。",
    "interview_coach": "面试表达与追问准备工具。",
    "socratic_tutor": "苏格拉底式引导学习工具。",
    "learning_profile": "学习画像查询与图谱上下文工具。",
    "review_planner": "复习计划生成工具。",
    "memory_search": "历史记忆检索工具。",
}.items():
    register_skill(_name, _placeholder_tool, _description, {"type": "object", "properties": {}, "additionalProperties": True})

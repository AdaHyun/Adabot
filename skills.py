"""
skills.py

Skill 注册与调用模块。

每个 Skill 都是一个普通函数：
- 输入：用户问题文本。
- 输出：给模型看的辅助策略文本。

后续要动态注册新 Skill，只需要调用 register_skill。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple


# Skill 函数类型。
SkillFunc = Callable[[str], str]

# Skill 注册表。
SKILL_REGISTRY: Dict[str, SkillFunc] = {}

# 关键词表，用于简单意图识别。
SKILL_KEYWORDS: Dict[str, List[str]] = {
    "math": [
        "高数",
        "数学",
        "微积分",
        "极限",
        "导数",
        "积分",
        "级数",
        "线代",
        "概率",
        "证明",
        "方程",
        "函数",
    ],
    "vehicle_engineering": [
        "车辆",
        "汽车",
        "机械",
        "机械原理",
        "机械结构",
        "机构",
        "平面机构",
        "机构结构",
        "结构分析",
        "自由度",
        "构件",
        "运动副",
        "发动机",
        "底盘",
        "变速器",
        "悬架",
        "制动",
        "转向",
        "新能源车",
        "电机",
        "动力学",
    ],
    "english": [
        "雅思",
        "ielts",
        "reading",
        "writing",
        "essay",
        "作文",
        "阅读",
        "翻译",
        "grammar",
        "vocabulary",
        "英文",
        "英语",
    ],
}


def register_skill(name: str, func: SkillFunc, keywords: List[str] | None = None) -> None:
    """
    动态注册 Skill。

    示例：
    def new_skill(text):
        return f"[新技能]: {text}"

    register_skill("skill_name", new_skill, keywords=["关键词"])
    """
    # 校验名称。
    if not name or not name.strip():
        raise ValueError("Skill 名称不能为空")

    # 校验函数。
    if not callable(func):
        raise TypeError("Skill 必须是可调用函数")

    # 注册函数。
    SKILL_REGISTRY[name] = func

    # 如果提供关键词，则注册关键词。
    if keywords:
        SKILL_KEYWORDS[name] = keywords


def list_skills() -> List[str]:
    """返回当前已注册 Skill 名称。"""
    return sorted(SKILL_REGISTRY.keys())


def select_skills(text: str) -> List[str]:
    """根据关键词选择 Skill。"""
    lower_text = (text or "").lower()
    matched_skills: List[str] = []

    # 只要命中一个关键词，就选择对应 Skill。
    for skill_name, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower_text:
                matched_skills.append(skill_name)
                break

    # 无命中时使用通用 Skill。
    if not matched_skills and "general" in SKILL_REGISTRY:
        matched_skills.append("general")

    return matched_skills


def run_selected_skills(text: str) -> Tuple[List[str], str]:
    """执行匹配到的 Skill，并合并输出。"""
    selected_names = select_skills(text)
    outputs: List[str] = []

    # 单个 Skill 失败不影响整体流程。
    for name in selected_names:
        skill = SKILL_REGISTRY.get(name)
        if skill is None:
            continue
        try:
            outputs.append(skill(text))
        except Exception as exc:
            outputs.append(f"[Skill {name} 执行失败]: {exc}")

    return selected_names, "\n\n".join(outputs)


def math_skill(text: str) -> str:
    """高数和数学题 Skill。"""
    return (
        "[高数 Skill]\n"
        "请优先识别题型，例如极限、导数、积分、级数、微分方程、线性代数或概率统计。\n"
        "回答时先写核心思路，再分步骤推导；涉及公式时保持符号清晰。\n"
        "如果题目条件不足，请明确指出缺失条件，并给出可继续求解的方向。"
    )


def vehicle_engineering_skill(text: str) -> str:
    """车辆工程 Skill。"""
    return (
        "[车辆工程 Skill]\n"
        "请从车辆工程角度回答，优先覆盖结构组成、工作原理、关键参数、常见失效或设计取舍。\n"
        "如果问题涉及计算，请列出已知量、公式、单位换算和结果解释。\n"
        "如果适合考研复习，请补充高频考点和容易混淆的概念。"
    )


def english_skill(text: str) -> str:
    """英语和雅思 Skill。"""
    return (
        "[英语 Skill]\n"
        "阅读题要定位关键词、解释同义替换和答案依据；写作题要给出结构、论点和可替换表达。\n"
        "如果用户要求改作文，请指出语法、词汇、逻辑和任务回应问题，并给出更自然的英文表达。\n"
        "必要时提供中文解释，帮助考研或雅思备考理解。"
    )


def general_skill(text: str) -> str:
    """通用 Skill。"""
    return (
        "[通用 Skill]\n"
        "请作为考研学习助手回答。优先给出直接答案，再补充必要解释、步骤和复习建议。\n"
        "如果问题来自 OCR，注意可能存在识别错误，请结合上下文合理判断。"
    )


# 模块加载时注册默认 Skill。
register_skill("math", math_skill, SKILL_KEYWORDS["math"])
register_skill("vehicle_engineering", vehicle_engineering_skill, SKILL_KEYWORDS["vehicle_engineering"])
register_skill("english", english_skill, SKILL_KEYWORDS["english"])
register_skill("general", general_skill)


def paper_reading_skill(text: str) -> str:
    return (
        "[论文精读 Skill]\n"
        "请围绕论文问题、核心贡献、方法结构、实验设置、指标结果、局限性和可复现步骤来回答。"
    )


def project_debugger_skill(text: str) -> str:
    return (
        "[项目调试 Skill]\n"
        "请先定位报错位置和根因，再给出最小可执行修复步骤；涉及命令时说明运行目录和前置条件。"
    )


def crawler_assistant_skill(text: str) -> str:
    return (
        "[爬虫助手 Skill]\n"
        "请从目标字段、请求解析、断点续爬、去重、导出和异常处理角度给出实现建议。"
    )


def interview_coach_skill(text: str) -> str:
    return (
        "[面试教练 Skill]\n"
        "请把答案整理成面试可表达版本：先一句话结论，再分点说明，最后补充可能追问。"
    )


def socratic_tutor_skill(text: str) -> str:
    return (
        "[苏格拉底导师 Skill]\n"
        "请优先用问题引导用户发现关键条件，并在必要时给出提示和阶段性总结。"
    )


def background_skill(text: str) -> str:
    return "[后台辅助 Skill]\n该能力主要用于学习画像、错题本、复习计划或历史记忆检索。"


register_skill("paper_reading", paper_reading_skill, ["论文", "paper", "精读", "method", "benchmark", "实验", "复现"])
register_skill("project_debugger", project_debugger_skill, ["报错", "error", "traceback", "conda", "pip", "git", "docker", "vscode"])
register_skill("crawler_assistant", crawler_assistant_skill, ["爬虫", "抓取", "解析", "JSONL", "checkpoint", "字段", "Excel"])
register_skill("interview_coach", interview_coach_skill, ["面试", "怎么说", "口语版", "自我介绍", "项目介绍", "追问"])
register_skill("socratic_tutor", socratic_tutor_skill, ["引导我", "启发", "苏格拉底", "别直接给答案"])
register_skill("learning_profile", background_skill, ["学习画像", "知识图谱", "学习进度", "我学到哪", "还有什么没学", "复习建议"])
register_skill("mistake_book", background_skill, ["错题", "我错了", "不会", "不懂", "加入错题本", "再讲一遍"])
register_skill("review_planner", background_skill, ["复习计划", "今天学什么", "一周计划", "备考计划", "复习安排"])
register_skill("mini_quiz", background_skill, ["测试我", "出题", "小测", "检验", "批改", "练习题"])
register_skill("memory_search", background_skill, ["我之前问过", "历史问题", "以前的问题", "复习以前", "查一下我问过"])

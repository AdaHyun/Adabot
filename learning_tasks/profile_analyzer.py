from __future__ import annotations

import re
from typing import Any

from .context_schema import LearningContext
from .knowledge_graph_store import find_best_node_for_task, list_all_nodes, upsert_knowledge_path
from .task_schema import LearningTask


WEAK_WORDS = ["不懂", "不会", "为什么", "看不懂", "错了", "再讲一遍", "没明白", "不会做"]
MISTAKE_WORDS = ["这题我错了", "我做错了", "加入错题本", "错题", "我错了"]
MECHANICAL_WORDS = ["机械", "机械原理", "机械结构", "机构", "平面机构", "结构分析", "自由度", "构件", "运动副"]


def infer_subject(task: LearningTask, text: str, primary_skill: str) -> str:
    combined = text or ""
    subjects = task.subjects or ["通用"]
    if any(word in combined for word in MECHANICAL_WORDS):
        for subject in subjects:
            if subject in {"机械原理", "车辆工程", "汽车理论"} or "机械" in subject:
                return subject
    rules = {
        "math": ["数学", "高等数学", "线性代数", "概率论"],
        "english": ["英语", "Listening", "Reading", "Writing", "Speaking"],
        "vehicle_engineering": ["机械原理", "车辆工程", "汽车理论"],
        "paper_reading": ["论文阅读", "文献综述"],
        "project_debugger": ["项目调试"],
    }
    for subject in subjects:
        if subject and subject.lower() in combined.lower():
            return subject
    for candidate in rules.get(primary_skill, []):
        if candidate in subjects:
            return candidate
    return subjects[0] if subjects else "通用"


def classify_question(text: str) -> tuple[str, str]:
    lower = (text or "").lower()
    if any(word in text for word in ["证明", "推导"]):
        return "证明推导", "进阶"
    if any(word in text for word in ["计算", "怎么做", "求"]):
        return "计算解题", "基础"
    if any(word in lower for word in ["paper", "method", "benchmark"]) or "论文" in text:
        return "论文理解", "进阶"
    return "概念理解", "基础"


def analyze_learning_event(
    task: LearningTask,
    context: LearningContext,
    user_input: str,
    assistant_answer: str,
) -> dict[str, Any]:
    text = "\n".join([user_input or "", assistant_answer or ""])
    subject = _subject_from_model_path(task, assistant_answer) or infer_subject(task, text, context.primary_skill)
    node_id, path, syllabus_id, confidence = find_best_node_for_task(task, subject, text)
    model_path = _extract_model_knowledge_path(assistant_answer)
    if model_path:
        model_subject = model_path[0]
        if model_subject in (task.subjects or []):
            subject = model_subject
            mapped = _match_path_to_syllabus(task, subject, model_path)
            if mapped:
                node_id, path, syllabus_id, confidence = mapped
            else:
                node_id = upsert_knowledge_path(task, model_path, source_type="agent")
                path = model_path
                syllabus_id = "db_graph"
                confidence = max(confidence, 0.72)
    question_type, difficulty = classify_question(user_input)
    weak = any(word in user_input for word in WEAK_WORDS)
    mistake = any(word in user_input for word in MISTAKE_WORDS)
    return {
        "task_id": task.id,
        "role_type": task.role_type,
        "primary_skill": context.primary_skill,
        "subject": subject,
        "syllabus_id": syllabus_id,
        "knowledge_node_id": node_id,
        "knowledge_path": path,
        "question_type": question_type,
        "difficulty": difficulty,
        "weakness_signal": "用户表达了不理解或做错" if weak or mistake else "",
        "should_add_to_mistake_book": bool(mistake),
        "confidence": confidence,
    }


def _extract_model_knowledge_path(answer: str) -> list[str]:
    text = answer or ""
    patterns = [
        r"所属知识点[：:]\s*([^\n。]+)",
        r"知识点[：:]\s*([^\n。]+)",
        r"属于[：:]\s*([^\n。]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(1).strip()
        raw = re.sub(r"^[【\[]|[】\]]$", "", raw)
        parts = [part.strip(" -　\t") for part in re.split(r">\s*|＞|/|→", raw) if part.strip()]
        if parts:
            return parts
    return []


def _subject_from_model_path(task: LearningTask, answer: str) -> str:
    parts = _extract_model_knowledge_path(answer)
    if parts and parts[0] in (task.subjects or []):
        return parts[0]
    return ""


def _match_path_to_syllabus(
    task: LearningTask,
    subject: str,
    model_path: list[str],
) -> tuple[str, list[str], str, float] | None:
    names = set(model_path[1:] or model_path)
    best = None
    best_score = 0
    for node in list_all_nodes(task, subject):
        path = node.get("path") or []
        score = sum(1 for name in names if name in path or name == node.get("title") or name == node.get("name"))
        if score > best_score:
            best_score = score
            best = (node["id"], path, "db_graph", min(0.98, 0.65 + score * 0.12))
    return best

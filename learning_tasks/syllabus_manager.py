from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from .task_schema import LearningTask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    try:
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_role_config(role_type: str) -> dict[str, Any]:
    role = (role_type or "custom").strip() or "custom"
    data = _yaml_load(CONFIG_DIR / "roles" / f"{role}.yaml")
    if data:
        return data
    return {
        "role_id": role,
        "role_name": role,
        "default_answer_style": "清晰、分步骤、适合复习",
        "default_subjects": ["通用"],
        "default_skills": ["math", "english", "vehicle_engineering", "general"],
    }


def load_syllabus(role_type: str, syllabus_name: str) -> dict[str, Any]:
    role = role_type or "custom"
    candidates = [
        CONFIG_DIR / "syllabi" / role / f"{syllabus_name}.yaml",
        CONFIG_DIR / "syllabi" / role / f"{_slug_subject(syllabus_name)}.yaml",
    ]
    for path in candidates:
        data = _yaml_load(path)
        if data:
            return data
    return {}


def load_syllabus_by_relative_path(relative_path: str) -> dict[str, Any]:
    if not relative_path:
        return {}
    safe_path = Path(relative_path)
    if safe_path.is_absolute() or ".." in safe_path.parts:
        return {}
    return _yaml_load(CONFIG_DIR / "syllabi" / safe_path)


def _slug_subject(subject: str) -> str:
    mapping = {
        "高等数学": "advanced_math",
        "线性代数": "linear_algebra",
        "概率论": "probability",
        "英语": "english",
        "考研英语": "english",
        "机械原理": "mechanical_principles",
        "数学": "math",
        "物理": "physics",
        "Listening": "listening",
        "Reading": "reading",
        "Writing": "writing",
        "Speaking": "speaking",
        "论文阅读": "paper_reading",
        "实验复现": "experiment_reproduction",
        "研究方法": "research_method",
        "学术写作": "academic_writing",
        "自定义专业示例": "example_custom_major",
    }
    return mapping.get(subject, subject.lower().replace(" ", "_"))


def get_subject_syllabus(role_type: str, subject: str) -> dict[str, Any]:
    return load_syllabus(role_type, subject)


def get_task_subject_syllabus(task: LearningTask, subject: str) -> dict[str, Any]:
    subject_map = (task.syllabus_config or {}).get("subjects") or {}
    relative_path = subject_map.get(subject)
    if relative_path:
        data = load_syllabus_by_relative_path(relative_path)
        if data:
            return data
    return get_subject_syllabus(task.role_type, subject)


def flatten_syllabus_nodes(syllabus: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]], path: list[str]) -> None:
        for node in nodes or []:
            node_id = str(node.get("id") or "")
            name = str(node.get("name") or node_id)
            item = {
                "id": node_id,
                "name": name,
                "path": path + [name],
                "children": node.get("children") or [],
            }
            flat.append(item)
            walk(item["children"], item["path"])

    walk(syllabus.get("nodes") or [], [syllabus.get("subject") or "通用"])
    return flat


def find_node_by_id(syllabus: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for item in flatten_syllabus_nodes(syllabus):
        if item["id"] == node_id:
            return item
    return None


def find_best_node(role_type: str, subject: str, text: str) -> tuple[str, list[str], str, float]:
    syllabus = get_subject_syllabus(role_type, subject)
    return find_best_node_in_syllabus(syllabus, subject, text)


def find_best_node_for_task(task: LearningTask, subject: str, text: str) -> tuple[str, list[str], str, float]:
    syllabus = get_task_subject_syllabus(task, subject)
    return find_best_node_in_syllabus(syllabus, subject, text)


def find_best_node_in_syllabus(syllabus: dict[str, Any], subject: str, text: str) -> tuple[str, list[str], str, float]:
    if not syllabus:
        return "general", [subject or "通用", "通用"], "", 0.2
    lower = (text or "").lower()
    best: tuple[str, list[str], str, float] = ("general", [syllabus.get("subject", subject), "通用"], syllabus.get("syllabus_id", ""), 0.2)
    best_score = 0.0
    for node in flatten_syllabus_nodes(syllabus):
        name = node["name"]
        score = 0.0
        if name and name.lower() in lower:
            score += len(name) + 3
        score += _alias_score(lower, name)
        for part in node["path"]:
            if part and part.lower() in lower:
                score += max(1, len(part) / 2)
        if score > best_score:
            best_score = score
            best = (node["id"], node["path"], syllabus.get("syllabus_id", ""), min(0.95, 0.45 + score / 20))
    return best


def _alias_score(lower_text: str, node_name: str) -> float:
    aliases = [
        (["机械结构", "机构结构", "结构分析"], ["平面机构的结构分析", "机构结构分析"]),
        (["自由度", "自由度计算"], ["基本概念与自由度计算"]),
        (["运动副"], ["运动副"]),
        (["构件"], ["机构组成", "基本概念与自由度计算"]),
    ]
    for input_words, node_words in aliases:
        if any(word in lower_text for word in input_words) and any(word in node_name for word in node_words):
            return 8.0
    return 0.0


def parse_outline_text(subjects: list[str], outline_text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse a simple indented outline into per-subject syllabus nodes."""
    result: dict[str, list[dict[str, Any]]] = {subject: [] for subject in subjects}
    current_subject = subjects[0] if subjects else "自定义科目"
    stack: list[tuple[int, dict[str, Any]]] = []
    counters: dict[str, int] = {}
    for raw_line in (outline_text or "").splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        name = re.sub(r"^[-*+\d.、\s]+", "", raw_line.strip()).strip()
        if not name:
            continue
        if name in result:
            current_subject = name
            stack = []
            continue
        counters[current_subject] = counters.get(current_subject, 0) + 1
        node = {
            "id": f"custom_{_safe_slug(current_subject)}_{counters[current_subject]:03d}",
            "name": name,
        }
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            stack[-1][1].setdefault("children", []).append(node)
        else:
            result.setdefault(current_subject, []).append(node)
        stack.append((indent, node))
    for subject in subjects:
        if not result.get(subject):
            result[subject] = [{"id": f"custom_{_safe_slug(subject)}_001", "name": subject}]
    return result


def save_custom_task_syllabi(task_id: str, subjects: list[str], outline_text: str) -> dict[str, Any]:
    import yaml

    base_dir = CONFIG_DIR / "syllabi" / "custom_tasks" / task_id
    base_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_outline_text(subjects, outline_text)
    subject_map: dict[str, str] = {}
    for subject, nodes in parsed.items():
        filename = f"{_safe_slug(subject)}.yaml"
        data = {
            "subject": subject,
            "syllabus_id": f"{task_id}_{_safe_slug(subject)}",
            "nodes": nodes,
        }
        path = base_dir / filename
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        subject_map[subject] = str(Path("custom_tasks") / task_id / filename).replace("\\", "/")
    return {"subjects": subject_map}


def _safe_slug(text: str) -> str:
    parts = re.findall(r"[a-zA-Z0-9]+", text or "")
    if parts:
        return "_".join(parts).lower()[:40]
    return f"s{abs(hash(text or 'subject')) % 100000}"

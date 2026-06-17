from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .artifact_store import merge_notes_into_knowledge_graph, task_artifact_dir, write_learning_profile
from .syllabus_manager import get_task_subject_syllabus
from .task_schema import LearningTask
from utils import extract_text_from_images
from utils import call_qwen_model


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv"}


def process_user_notes(task: LearningTask, files: Any, pasted_text: str = "") -> dict[str, Any]:
    base = task_artifact_dir(task.id) / "notes"
    raw_dir = base / "raw"
    parsed_dir = base / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    combined_texts = []
    for file_path in _normalize_files(files):
        try:
            source = Path(file_path)
            target = raw_dir / f"{_stamp()}_{source.name}"
            shutil.copy2(source, target)
            text = extract_note_text(target)
            parsed_path = parsed_dir / f"{target.stem}.txt"
            parsed_path.write_text(text, encoding="utf-8")
            combined_texts.append((target.name, text))
            results.append({"file": source.name, "status": "ok", "chars": len(text), "parsed_path": str(parsed_path)})
        except Exception as exc:
            results.append({"file": str(file_path), "status": "failed", "error": str(exc)})

    if pasted_text.strip():
        name = f"{_stamp()}_pasted_note.txt"
        parsed_path = parsed_dir / name
        parsed_path.write_text(pasted_text, encoding="utf-8")
        combined_texts.append((name, pasted_text))
        results.append({"file": "pasted_text", "status": "ok", "chars": len(pasted_text), "parsed_path": str(parsed_path)})

    analyses = []
    for source_name, text in combined_texts:
        analysis = analyze_note_text(task, text, source_name)
        analyses.append(analysis)
        merge_notes_into_knowledge_graph(
            task=task,
            detected_subject=analysis.get("detected_subject", ""),
            extracted_nodes=analysis.get("extracted_knowledge_nodes", []),
            source_file=source_name,
        )

    _append_notes_summary(task, analyses)
    write_learning_profile(task)
    return {"task_id": task.id, "files": results, "analysis": analyses}


def extract_note_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in IMAGE_SUFFIXES:
        text, error = extract_text_from_images([str(path)])
        if error and not text:
            raise RuntimeError(error)
        return text
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def analyze_note_text(task: LearningTask, text: str, source_name: str = "") -> dict[str, Any]:
    llm_result = _analyze_note_text_with_llm(task, text, source_name)
    if llm_result:
        return llm_result
    subject = _detect_subject(task, text)
    extracted_nodes = _extract_nodes(task, subject, text)
    weak_hits = [word for word in ["不会", "不懂", "难点", "易错", "需要复习", "没掌握"] if word in text]
    suggestions = ["先复习本笔记中标记的重点概念，再回到学习画像中查看相关历史问题。"]
    if weak_hits:
        suggestions.insert(0, "笔记中出现不会/难点信号，建议优先复习这些节点。")
    return {
        "detected_subject": subject,
        "source_file": source_name,
        "extracted_knowledge_nodes": extracted_nodes,
        "user_mastery_updates": [
            {
                "path": node["path"],
                "mastery_status": node.get("mastery_signal", "已接触"),
                "reason": node.get("evidence", ""),
            }
            for node in extracted_nodes
        ],
        "missing_prerequisites": _missing_prerequisites(text),
        "review_suggestions": suggestions,
    }


def _analyze_note_text_with_llm(task: LearningTask, text: str, source_name: str) -> dict[str, Any] | None:
    if not os.getenv("QWEN_API_KEY", "").strip() or not text.strip():
        return None
    clipped = text[:6000]
    prompt = f"""请分析用户学习笔记，并只输出 JSON，不要 Markdown。
当前任务: {task.task_name}
任务角色: {task.role_type}
任务科目: {', '.join(task.subjects)}
来源文件: {source_name}

输出格式:
{{
  "detected_subject": "机械原理",
  "extracted_knowledge_nodes": [
    {{
      "title": "机构自由度计算",
      "path": ["机械原理", "平面机构的结构分析", "自由度计算"],
      "summary": "",
      "mastery_signal": "需要复习",
      "evidence": ""
    }}
  ],
  "user_mastery_updates": [],
  "missing_prerequisites": [],
  "review_suggestions": []
}}

笔记正文:
{clipped}
"""
    raw = call_qwen_model(prompt=prompt, images=None, temperature=0.1, max_tokens=2048)
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start : end + 1])
        if isinstance(data, dict) and "extracted_knowledge_nodes" in data:
            return data
    except Exception:
        return None
    return None


def _extract_nodes(task: LearningTask, subject: str, text: str) -> list[dict[str, Any]]:
    syllabus = get_task_subject_syllabus(task, subject)
    nodes = []
    for raw in (syllabus.get("nodes") or []):
        _collect_matching_nodes(subject, raw, text, [subject], nodes)
    if not nodes:
        for line in text.splitlines():
            clean = line.strip(" -\t")
            if 2 <= len(clean) <= 30 and not clean.endswith(("。", "，")):
                nodes.append(
                    {
                        "title": clean,
                        "path": [subject, clean],
                        "summary": clean,
                        "mastery_signal": "已接触",
                        "evidence": "来自用户笔记标题或条目",
                    }
                )
                if len(nodes) >= 12:
                    break
    return nodes[:30]


def _collect_matching_nodes(subject: str, node: dict[str, Any], text: str, parent_path: list[str], output: list[dict[str, Any]]) -> None:
    name = str(node.get("name") or "")
    path = parent_path + [name]
    if name and name in text:
        mastery = "需要复习" if any(word in text for word in [f"不会{name}", f"{name}不会", "不懂", "难点", "易错"]) else "已接触"
        output.append(
            {
                "title": name,
                "path": path,
                "summary": f"用户笔记提到了{name}",
                "mastery_signal": mastery,
                "evidence": f"笔记文本命中：{name}",
            }
        )
    for child in node.get("children") or []:
        _collect_matching_nodes(subject, child, text, path, output)


def _detect_subject(task: LearningTask, text: str) -> str:
    for subject in task.subjects or []:
        if subject and subject in text:
            return subject
    if any(word in text for word in ["机械", "机构", "自由度", "运动副", "构件"]):
        for subject in task.subjects or []:
            if "机械" in subject:
                return subject
    return (task.subjects or ["未分类"])[0]


def _missing_prerequisites(text: str) -> list[str]:
    items = []
    for word in ["运动副分类", "局部自由度", "虚约束", "构件", "运动链"]:
        if word not in text:
            items.append(word)
    return items[:5]


def _extract_docx(path: Path) -> str:
    texts = []
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            texts.append(node.text)
    return "\n".join(texts)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            raise RuntimeError("PDF text extraction needs pypdf or PyPDF2 installed") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _append_notes_summary(task: LearningTask, analyses: list[dict[str, Any]]) -> None:
    path = task_artifact_dir(task.id) / "notes_summary.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []
    existing.extend({"created_at": _stamp(), **item} for item in analyses)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_files(files: Any) -> list[str]:
    if files is None:
        return []
    raw = files if isinstance(files, list) else [files]
    result = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, dict):
            path = item.get("path") or item.get("name")
        else:
            path = getattr(item, "path", None) or getattr(item, "name", None) or str(item)
        if path:
            result.append(path)
    return result


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

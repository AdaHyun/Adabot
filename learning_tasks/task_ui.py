from __future__ import annotations

from pathlib import Path
import json
import time
import traceback

import gradio as gr

from .memory_search import repair_obvious_misclassifications, render_search
from .mistake_store import add_to_mistake_book, render_mistakes
from .knowledge_graph_store import get_node, list_all_nodes, list_children, upsert_knowledge_path
from .profile_renderer import render_profile_visual
from .profile_store import load_recent_chat_messages
from .notes_processor import process_user_notes
from .quiz_engine import generate_quiz, grade_quiz
from .review_planner import generate_review_plan
from .syllabus_manager import load_role_config, save_custom_task_syllabi
from .task_store import create_task, get_active_task, get_or_create_default_task, get_task, list_tasks, set_active_task, update_task
from .task_store import connect_db


def debug_log(name: str, msg: str) -> None:
    print(f"[DEBUG][{name}] {msg}", flush=True)


ROLE_LABEL_TO_ID = {
    "考研": "kaoyan",
    "高考": "gaokao",
    "雅思": "ielts",
    "本科生": "undergraduate",
    "研究生": "postgraduate",
    "自定义": "custom",
}


def task_choices() -> list[str]:
    tasks = list_tasks()
    if not tasks:
        tasks = [get_or_create_default_task()]
    return [f"{_display_task_name(task.task_name, task.id)} ({task.id})" for task in tasks]


def parse_task_id(choice: str | None) -> str:
    if choice and "(" in choice and choice.endswith(")"):
        return choice.rsplit("(", 1)[-1][:-1]
    task = get_active_task() or get_or_create_default_task()
    return task.id


def active_task_label() -> str:
    task = get_active_task() or get_or_create_default_task()
    return f"当前激活任务：{_display_task_name(task.task_name, task.id)} / {task.role_type} / {task.id}"


def refresh_task_dropdown():
    import gradio as gr

    choices = task_choices()
    active = get_active_task() or get_or_create_default_task()
    value = next((item for item in choices if item.endswith(f"({active.id})")), choices[0] if choices else None)
    return gr.update(choices=choices, value=value), active_task_label()


def switch_task(choice: str):
    task_id = parse_task_id(choice)
    task = set_active_task(task_id) or get_or_create_default_task()
    return active_task_label(), load_recent_chat_messages(task.id)


def _split_subjects(text: str) -> list[str]:
    raw = (text or "").replace("，", ",").replace("、", ",").split(",")
    return [item.strip() for item in raw if item.strip()]


def create_task_from_form(
    task_name: str,
    role_label: str,
    goal_description: str,
    target_exam: str,
    target_date: str,
    subjects_text: str,
    answer_style: str,
    enable_profile: bool,
    enable_mistake: bool,
    enable_review: bool,
    custom_outline: str = "",
):
    import gradio as gr

    role_type = ROLE_LABEL_TO_ID.get(role_label, "custom")
    role_config = load_role_config(role_type)
    subjects = _split_subjects(subjects_text) or list(role_config.get("default_subjects") or ["通用"])
    skills = list(role_config.get("default_skills") or [])
    if enable_profile and "learning_profile" not in skills:
        skills.append("learning_profile")
    if enable_mistake and "mistake_book" not in skills:
        skills.append("mistake_book")
    if enable_review and "review_planner" not in skills:
        skills.append("review_planner")
    first = len(list_tasks()) == 0
    make_custom_syllabus = role_type == "custom" and bool((custom_outline or "").strip())
    task = create_task(
        task_name=_clean_task_name(task_name) or "新学习任务",
        role_type=role_type,
        goal_description=goal_description,
        target_exam=target_exam,
        target_date=target_date,
        subjects=subjects,
        answer_style=answer_style or role_config.get("default_answer_style", ""),
        enabled_skills=skills,
        make_active=first or True,
    )
    if make_custom_syllabus:
        syllabus_config = save_custom_task_syllabi(task.id, subjects, custom_outline)
        task = update_task(task.id, syllabus_config=syllabus_config) or task
    if first:
        set_active_task(task.id)
    choices = task_choices()
    value = next((item for item in choices if item.endswith(f"({task.id})")), choices[0])
    return gr.update(choices=choices, value=value), f"已创建任务：{_display_task_name(task.task_name, task.id)}\n\n{active_task_label()}", render_task_list(), load_recent_chat_messages(task.id)


def render_task_list() -> str:
    tasks = list_tasks()
    if not tasks:
        return "暂无任务。"
    lines = ["### 已有任务"]
    for task in tasks:
        marker = "active" if task.is_active else "inactive"
        lines.append(
            f"- [{marker}] **{_display_task_name(task.task_name, task.id)}** / {task.role_type} / `{task.id}`\n"
            f"  - 目标: {task.goal_description or '-'}\n"
            f"  - 科目: {', '.join(task.subjects) or '-'}"
        )
    return "\n".join(lines)


def subject_choices_for_task(choice: str):
    import gradio as gr

    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    subjects = task.subjects or _fallback_subjects(task.role_type)
    subjects = [str(item) for item in subjects if str(item).strip()] or ["通用"]
    return gr.update(choices=subjects, value=subjects[0])


def render_profile_ui(choice: str, subject: str, view_label: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    repair_obvious_misclassifications(task)
    return render_profile_visual(task, subject or (task.subjects[0] if task.subjects else "通用"), view_label or "知识网络")


ROOT_NODE_ID = "__root__"


NODE_TABLE_HEADERS = ["知识点", "状态", "提问数", "类型", "ID"]


def _table_update(rows: list[list[str]]):
    import gradio as gr

    return gr.update(value=rows)


def initialize_knowledge_drilldown(choice: str, subject: str):
    import gradio as gr

    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    subject = subject or (task.subjects[0] if task.subjects else "通用")
    debug_log("initialize_knowledge_drilldown", "START")
    t0 = time.time()
    rows = _current_level_rows(task, subject, ROOT_NODE_ID)
    debug_log("initialize_knowledge_drilldown", f"END elapsed={time.time() - t0:.2f}s rows={len(rows)}")
    return (
        ROOT_NODE_ID,
        [],
        "",
        subject,
        rows,
        _breadcrumb_html(task, subject, ROOT_NODE_ID),
        _table_update(rows),
        _level_overview_html(task, subject, ROOT_NODE_ID),
        gr.update(interactive=False),
    )


def on_knowledge_node_click(
    current_rows: list[list[str]] | None,
    current_node_id: str,
    nav_stack: list[str] | None,
    choice: str,
    subject: str,
    evt: gr.SelectData,
):
    debug_log("on_knowledge_node_click", "START")
    t0 = time.time()

    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    subject = subject or (task.subjects[0] if task.subjects else "通用")
    current_node_id = current_node_id or ROOT_NODE_ID
    stack = list(nav_stack or [])
    node_id = _node_id_from_table_event(current_rows, evt)

    if not node_id:
        debug_log("on_knowledge_node_click", f"END elapsed={time.time() - t0:.2f}s empty")
        rows = list(current_rows or [])
        return (
            current_node_id,
            stack,
            "",
            rows,
            _table_update(rows),
            _breadcrumb_html(task, subject, current_node_id),
            _level_overview_html(task, subject, current_node_id),
            gr.update(interactive=bool(stack)),
        )

    node = get_node(task, subject, node_id)

    if not node:
        debug_log("on_knowledge_node_click", f"END elapsed={time.time() - t0:.2f}s missing node={node_id}")
        rows = list(current_rows or [])
        return (
            current_node_id,
            stack,
            "",
            rows,
            _table_update(rows),
            _breadcrumb_html(task, subject, current_node_id),
            "<div class='node-detail-empty'>未找到该知识点。</div>",
            gr.update(interactive=bool(stack)),
        )

    children = list_children(task, subject, node_id)

    if children:
        new_stack = list(stack)
        new_stack.append(current_node_id)
        rows = _current_level_rows(task, subject, node_id)

        debug_log(
            "on_knowledge_node_click",
            f"END elapsed={time.time() - t0:.2f}s drilldown node={node_id} rows={len(rows)}",
        )

        return (
            node_id,
            new_stack,
            "",
            rows,
            _table_update(rows),
            _breadcrumb_html(task, subject, node_id),
            _level_overview_html(task, subject, node_id),
            gr.update(interactive=True),
        )

    debug_log("on_knowledge_node_click", f"END elapsed={time.time() - t0:.2f}s detail node={node_id}")
    rows = list(current_rows or [])

    return (
        current_node_id,
        stack,
        node_id,
        rows,
        _table_update(rows),
        _breadcrumb_html(task, subject, current_node_id),
        render_knowledge_node_detail(task, subject, node_id),
        gr.update(interactive=bool(stack)),
    )

def on_back_to_parent(current_node_id: str, nav_stack: list[str] | None, choice: str, subject: str, selected_node_id: str | None):
    import gradio as gr

    debug_log("on_back_to_parent", "START")
    t0 = time.time()
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    subject = subject or (task.subjects[0] if task.subjects else "通用")
    stack = list(nav_stack or [])
    if not stack:
        current = ROOT_NODE_ID
    else:
        current = stack.pop()
    rows = _current_level_rows(task, subject, current)
    detail = (
        render_knowledge_node_detail(task, subject, selected_node_id)
        if selected_node_id
        else _level_overview_html(task, subject, current)
    )
    debug_log("on_back_to_parent", f"END elapsed={time.time() - t0:.2f}s current={current} rows={len(rows)}")
    return (
        current,
        stack,
        selected_node_id or "",
        rows,
        _table_update(rows),
        _breadcrumb_html(task, subject, current),
        detail,
        gr.update(interactive=bool(stack)),
    )

def render_knowledge_node_detail(task: object, subject: str, node_id: str | None) -> str:
    debug_log("render_knowledge_node_detail", "START")
    t0 = time.time()
    if not node_id:
        return "<div class='node-detail-empty'>请选择一个知识点。</div>"
    if not hasattr(task, "id"):
        task = get_task(str(task)) or get_or_create_default_task()
    node = get_node(task, subject, node_id)
    if not node:
        return "<div class='node-detail-empty'>未找到该知识点。</div>"
    state = _state_for(task.id, node_id)
    questions = _questions_for_node(task.id, node_id, node.get("path") or [])
    mistakes = _mistakes_for_node(task.id, node_id)
    framework = _ensure_basic_framework(task, subject, node)
    typical = _typical_questions_for(node.get("title", ""))
    status = _status_label((state or {}).get("status") or "unvisited")
    path = node.get("path") or [subject, node.get("title", "")]
    debug_log("render_knowledge_node_detail", f"END elapsed={time.time() - t0:.2f}s node={node_id}")
    return f"""
<div class="node-detail-panel">
  <h2>{_esc(node.get('title'))}</h2>
  <p class="muted">路径：{_esc(' > '.join(path))}</p>
  <h3>1. 基础知识框架</h3>
  <ul>{''.join(f'<li>{_esc(item)}</li>' for item in framework)}</ul>
  <h3>2. 典型题型</h3>
  <ul>{''.join(f'<li>{_esc(item)}</li>' for item in typical)}</ul>
  <h3>3. 当前掌握状态</h3>
  <div class="stat-grid">
    <span>状态<br><b>{_esc(status)}</b></span>
    <span>seen<br><b>{(state or {}).get('seen_count', 0)}</b></span>
    <span>weak<br><b>{(state or {}).get('weak_count', 0)}</b></span>
    <span>mistake<br><b>{(state or {}).get('mistake_count', 0)}</b></span>
    <span>mastery<br><b>{float((state or {}).get('mastery_score') or 0):.2f}</b></span>
  </div>
  <h3>4. 你问过的问题</h3>
  {_history_cards(questions)}
  <h3>5. 错题</h3>
  {_mistake_cards(mistakes)}
  <h3>6. 题集</h3>
  <p class="muted">这个知识点暂时还没有题集。</p>
  <h3>7. 复习建议</h3>
  <p>{_esc(_review_suggestion(status, node.get('title', ''), bool(questions)))}</p>
</div>
"""

def profile_node_choices(choice: str, subject: str):
    import gradio as gr

    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    subject = subject or (task.subjects[0] if task.subjects else "")
    choices = []
    for node in list_all_nodes(task, subject):
        label = f"{' > '.join(node.get('path') or [])} ({node['id']})"
        choices.append(label)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value), render_node_detail_ui(choice, subject, value)


def render_node_detail_ui(choice: str, subject: str, node_choice: str | None) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    node_id = _parse_node_id(node_choice)
    return render_knowledge_node_detail(task, subject, node_id)

def render_mistakes_ui(choice: str, subject: str, keyword: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    return render_mistakes(task.id, subject or "", keyword or "")


def manual_add_mistake_ui(choice: str, subject: str, question_text: str, reason: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    event = {
        "subject": subject or (task.subjects[0] if task.subjects else "通用"),
        "knowledge_node_id": "manual",
        "knowledge_path": [subject or "通用", "手动添加"],
        "weakness_signal": reason or "手动加入错题本",
    }
    add_to_mistake_book(task.id, None, event, original_question=question_text, mistake_reason=reason)
    return render_mistakes(task.id, subject or "", "")


def render_review_plan_ui(choice: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    return generate_review_plan(task.id)


def search_history_ui(choice: str, keyword: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    return render_search(task.task_name, task.id, keyword or "")


def process_notes_ui(choice: str, files, pasted_text: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    result = process_user_notes(task, files, pasted_text or "")
    ok = [item for item in result.get("files", []) if item.get("status") == "ok"]
    failed = [item for item in result.get("files", []) if item.get("status") != "ok"]
    analyses = result.get("analysis", [])
    lines = [
        "### 笔记解析结果",
        f"- 任务: {task.task_name} / `{task.id}`",
        f"- 成功文件: {len(ok)}",
        f"- 失败文件: {len(failed)}",
        f"- 分析批次: {len(analyses)}",
        "",
    ]
    for item in ok:
        lines.append(f"- OK: {item.get('file')}，提取 {item.get('chars')} 字，保存到 `{item.get('parsed_path')}`")
    for item in failed:
        lines.append(f"- FAILED: {item.get('file')}，原因: {item.get('error')}")
    for analysis in analyses:
        lines.append(f"\n#### 科目: {analysis.get('detected_subject')}")
        lines.append("识别知识点:")
        for node in analysis.get("extracted_knowledge_nodes", [])[:12]:
            lines.append(f"- {' > '.join(node.get('path') or [])}: {node.get('mastery_signal', '已接触')}")
        if analysis.get("missing_prerequisites"):
            lines.append("缺失前置知识: " + "、".join(analysis["missing_prerequisites"]))
        if analysis.get("review_suggestions"):
            lines.append("复习建议: " + "；".join(analysis["review_suggestions"]))
    return "\n".join(lines)

def generate_quiz_ui(choice: str, subject: str, knowledge: str, count: int, difficulty: str) -> str:
    _task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    return generate_quiz(subject, knowledge, count, difficulty)


def grade_quiz_ui(choice: str, knowledge_node_id: str, answer: str) -> str:
    task = get_task(parse_task_id(choice)) or get_or_create_default_task()
    return grade_quiz(task.id, knowledge_node_id or "general", answer)


def ensure_profile_export_dir() -> None:
    Path(__file__).resolve().parents[1].joinpath("data", "profile_exports").mkdir(parents=True, exist_ok=True)


def _clean_task_name(name: str | None) -> str:
    text = (name or "").strip()
    if not text or set(text) <= {"?"}:
        return ""
    return text


def _display_task_name(name: str | None, task_id: str) -> str:
    clean = _clean_task_name(name)
    return clean or f"未命名任务 {task_id[-6:]}"


def _fallback_subjects(role_type: str) -> list[str]:
    role_config = load_role_config(role_type)
    return list(role_config.get("default_subjects") or ["通用"])


def _current_level_rows(task, subject: str, current_node_id: str) -> list[list[str]]:
    nodes = _children_for(task, subject, current_node_id)
    rows = []
    for node in nodes:
        child_count = len(list_children(task, subject, node.get("id", "")))
        state = _state_for(task.id, node.get("id", ""))
        status = _status_label((state or {}).get("status") or "unvisited")
        q_count = _question_count_for_node(task.id, node.get("id", ""), _path_for_raw_node(task, subject, node.get("id", "")))
        node_type = f"{child_count} 个子目录" if child_count else "叶子知识点"
        rows.append([node.get("title") or node.get("name") or "", status, str(q_count), node_type, node.get("id", "")])
    return rows or [["无可用知识点", "", "", "", ""]]


def _node_id_from_table_event(current_rows: list[list[str]] | None, evt) -> str:
    rows = list(current_rows or [])
    index = getattr(evt, "index", None)
    if isinstance(index, (list, tuple)) and index:
        row_index = index[0]
    else:
        row_index = index
    try:
        row = rows[int(row_index)]
    except Exception:
        return ""
    if not isinstance(row, (list, tuple)) or len(row) < 5:
        return ""
    return str(row[4] or "")


def _children_for(task, subject: str, current_node_id: str) -> list[dict]:
    if current_node_id in {"", ROOT_NODE_ID, None}:
        return list_children(task, subject, ROOT_NODE_ID)
    return list_children(task, subject, current_node_id)


def _raw_node_by_id(task, subject: str, node_id: str) -> dict | None:
    return get_node(task, subject, node_id)


def _path_for_raw_node(task, subject: str, node_id: str) -> list[str]:
    node = get_node(task, subject, node_id)
    return (node or {}).get("path") or [subject]


# def _breadcrumb_html(task, subject: str, current_node_id: str) -> str:
#     path = [subject] if current_node_id in {"", ROOT_NODE_ID, None} else _path_for_raw_node(task, subject, current_node_id)
#     return (
#         "<div class='knowledge-breadcrumb'>"
#         "<span>当前路径：</span>"
#         f"{path_html}"
#         "</div>"
#     )

def _breadcrumb_html(task, subject: str, current_node_id: str) -> str:
    path = [subject] if current_node_id in {"", ROOT_NODE_ID, None} else _path_for_raw_node(task, subject, current_node_id)
    path_html = " <span class='crumb-sep'>›</span> ".join(
        f"<b>{_esc(item)}</b>" for item in path
    )
    return (
        "<div class='knowledge-breadcrumb'>"
        "<span>当前路径：</span>"
        f"{path_html}"
        "</div>"
    )


def _level_overview_html(task, subject: str, current_node_id: str) -> str:
    debug_log("render_current_level_nodes", "START")
    t0 = time.time()
    path = [subject] if current_node_id in {"", ROOT_NODE_ID, None} else _path_for_raw_node(task, subject, current_node_id)
    children = _children_for(task, subject, current_node_id)
    if current_node_id in {"", ROOT_NODE_ID, None}:
        title = subject
        framework = [node.get("title", "") for node in children]
    else:
        raw = _raw_node_by_id(task, subject, current_node_id) or {}
        title = raw.get("title", path[-1] if path else subject)
        framework = [node.get("title", "") for node in children] or _basic_framework_for(title)
    debug_log("render_current_level_nodes", f"END elapsed={time.time() - t0:.2f}s current={current_node_id} children={len(children)}")
    return f"""
<div class="node-detail-panel">
  <h2>{_esc(title)}</h2>
  <p class="muted">路径：{_esc(' > '.join(path))}</p>
  <h3>章节概览</h3>
  <ul>{''.join(f'<li>{_esc(item)}</li>' for item in framework)}</ul>
  <p class="muted">左侧选择有子目录的节点会继续下钻；选择叶子知识点会在这里显示详情。</p>
</div>
"""

def _ensure_basic_framework(task, subject: str, node: dict) -> list[str]:
    children = [child.get("title", "") for child in list_children(task, subject, node.get("id", "")) if child.get("title")]
    if children:
        return children
    title = node.get("title", "")
    return _basic_framework_for(title)


def _typical_questions_for(title: str) -> list[str]:
    if "矩阵加法" in title:
        return ["判断两个矩阵能否相加", "计算矩阵加法", "利用矩阵加法性质化简", "与数乘、矩阵乘法混合运算"]
    if "矩阵乘法" in title:
        return ["判断矩阵乘法是否有定义", "计算矩阵乘积", "说明矩阵乘法通常不可交换", "结合线性变换理解乘法顺序"]
    if "自由度" in title:
        return ["计算平面机构自由度", "识别复合铰链", "判断局部自由度", "判断虚约束"]
    return ["概念判断题", "基础计算题", "性质应用题", "综合运用题"]

def _question_count_for_node(task_id: str, node_id: str, path: list[str]) -> int:
    return len(_questions_for_node(task_id, node_id, path))


def _mistake_cards(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>这个知识点暂时没有错题。</p>"
    items = []
    for row in rows:
        items.append(
            f"<li>{_esc(row.get('created_at') or '')} - {_esc(row.get('mistake_reason') or '错题')} - 下次复习：{_esc(row.get('next_review_at') or '-')}</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"

def _parse_node_id(node_choice: str | None) -> str:
    if node_choice and "(" in node_choice and node_choice.endswith(")"):
        return node_choice.rsplit("(", 1)[-1][:-1]
    return ""


def _state_for(task_id: str, node_id: str) -> dict | None:
    with connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_state WHERE task_id = ? AND knowledge_node_id = ?",
            (task_id, node_id),
        ).fetchone()
    return dict(row) if row else None


def _questions_for_node(task_id: str, node_id: str, path: list[str]) -> list[dict]:
    path_prefix = json.dumps(path, ensure_ascii=False)[:-1]
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM questions
            WHERE task_id = ?
              AND (knowledge_node_id = ? OR knowledge_path LIKE ?)
            ORDER BY created_at DESC
            LIMIT 30
            """,
            (task_id, node_id, f"{path_prefix}%"),
        ).fetchall()
    return [dict(row) for row in rows]


def _mistakes_for_node(task_id: str, node_id: str) -> list[dict]:
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM mistake_book WHERE task_id = ? AND knowledge_node_id = ? ORDER BY created_at DESC",
            (task_id, node_id),
        ).fetchall()
    return [dict(row) for row in rows]


def _history_cards(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>这个知识点下还没有提问记录。</p>"
    cards = []
    for row in rows:
        path = " > ".join(_loads(row.get("knowledge_path")))
        answer = _summary(row.get("assistant_answer") or "")
        cards.append(
            f"""
<details class="qa-card">
  <summary>{_esc(row.get('created_at') or row.get('timestamp') or '-')} - {_esc(row.get('user_input') or '(图片问题)')}</summary>
  <p><b>回答摘要：</b>{_esc(answer)}</p>
  <p><b>Skill：</b>{_esc(row.get('primary_skill') or '-')}</p>
  <p><b>知识点：</b>{_esc(path or row.get('knowledge_node_id') or '未分类')}</p>
  <pre>{_esc(row.get('assistant_answer') or '')}</pre>
</details>
"""
        )
    return "\n".join(cards)

def _basic_framework_for(title: str) -> list[str]:
    if "矩阵加法" in title:
        return ["矩阵加法的定义", "同型矩阵才能相加", "对应元素相加", "矩阵加法满足交换律和结合律", "零矩阵与负矩阵"]
    if "矩阵减法" in title:
        return ["矩阵减法的定义", "同型矩阵才能相减", "对应元素相减", "减法可转化为加负矩阵"]
    if "数乘矩阵" in title:
        return ["数乘矩阵的定义", "每个元素同时乘以同一个数", "数乘与矩阵加法的分配律"]
    if "矩阵乘法" in title:
        return ["矩阵乘法的定义", "左矩阵列数等于右矩阵行数", "行乘列规则", "矩阵乘法通常不交换", "单位矩阵与零矩阵"]
    if "矩阵转置" in title:
        return ["转置的定义", "行列互换", "转置的运算性质", "乘积转置公式"]
    if "机构" in title or "机械" in title:
        return ["机构的基本组成", "构件", "运动副", "运动链", "机架", "原动件", "从动件"]
    if "自由度" in title:
        return ["活动构件数", "低副和高副", "局部自由度", "虚约束", "自由度公式"]
    return ["核心定义", "基本性质", "典型例题", "易错点", "复习题"]

def _review_suggestion(status: str, title: str, has_questions: bool) -> str:
    if status in {"未接触", "unvisited"}:
        return f"建议先学习《{title}》的定义、组成和典型例题，再提一个基础概念问题。"
    if "复习" in status or status == "薄弱":
        return f"建议回看历史问题，总结《{title}》的易错点，并做 2-3 道基础题。"
    if not has_questions:
        return "目前没有历史提问，可先补充一个概念理解题，帮助系统建立画像。"
    return "建议间隔复习，并用小测验检查是否真正掌握。"


def _status_label(status: str) -> str:
    return {
        "unvisited": "未接触",
        "seen": "已接触",
        "learning": "需要复习",
        "weak": "需要复习",
        "mastered": "已掌握",
        "proficient": "已掌握",
    }.get(status or "unvisited", "未接触")

def _loads(value: str | None) -> list[str]:
    try:
        data = json.loads(value or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _summary(text: str, limit: int = 220) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _esc(value) -> str:
    import html

    return html.escape(str(value or ""))




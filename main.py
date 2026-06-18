"""
main.py

Adabot 项目入口：
1. 启动 Gradio Web UI。
2. 支持文本输入和最多 3 张图片输入。
3. 图片支持一张一张添加，也支持一次批量选择多张。
4. 主图片输入框保留上传、粘贴、拍摄三个入口。
5. OCR 作为手动开启的辅助能力保留。
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Iterator, Tuple

import gradio as gr

from memory_manager import (
    DEFAULT_USER_ID,
    append_turn_to_history,
    extract_and_save_memories,
    format_chat_history,
    retrieve_memory_context,
    trim_chat_history,
)
from skills import list_skills, run_selected_skills
from learning_tasks.mistake_store import add_to_mistake_book
from learning_tasks.artifact_store import sync_task_artifacts
from learning_tasks.profile_analyzer import analyze_learning_event
from learning_tasks.profile_store import (
    load_recent_chat_messages,
    save_knowledge_event,
    save_question,
    update_knowledge_state,
    update_question_with_event,
)
from learning_tasks.syllabus_manager import load_role_config
from learning_tasks.task_router import build_learning_context
from learning_tasks.task_store import get_or_create_default_task, get_task, init_db, set_active_task
from learning_tasks.task_ui import (
    active_task_label,
    create_task_from_form,
    export_mindmap_markdown,
    export_mindmap_opml,
    generate_quiz_ui,
    grade_quiz_ui,
    manual_add_mistake_ui,
    mark_profile_loaded,
    initialize_knowledge_drilldown,
    on_knowledge_node_double_click,
    on_enter_child_node,
    on_back_to_parent,
    on_knowledge_node_click,
    process_notes_ui,
    profile_node_choices,
    refresh_task_dropdown,
    render_node_detail_ui,
    render_mistakes_ui,
    render_mistakes_ui_filtered,
    render_profile_chart,
    render_profile_chart_if_loaded,
    render_profile_ui,
    render_review_plan_ui,
    render_selected_node_question_set,
    render_task_list,
    reset_profile_loaded,
    search_history_ui,
    subject_choices_for_task,
    subject_choices_for_task_pair,
    switch_task,
    task_choices,
)
from utils import (
    build_gallery_items,
    build_prompt,
    call_qwen_model,
    call_qwen_model_stream,
    configure_utf8_runtime,
    ensure_project_dirs,
    extract_text_from_images,
    normalize_markdown_math,
    normalize_uploaded_files,
    save_image_to_artifacts,
    write_log,
)


LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

DEBUG_DISABLE_PROFILE_EVENTS = False
# DEBUG_DISABLE_PROFILE_EVENTS = True

KNOWLEDGE_DBLCLICK_JS = """
() => {
  if (window.__knowledgeDblClickBridgeInstalled) return;
  window.__knowledgeDblClickBridgeInstalled = true;

  function findInPath(event, predicate) {
    var path = event.composedPath ? event.composedPath() : [];
    for (var i = 0; i < path.length; i++) {
      var item = path[i];
      if (item && item.nodeType === 1 && predicate(item)) return item;
    }
    return null;
  }

  function setNativeValue(el, value) {
    var proto = Object.getPrototypeOf(el);
    var descriptor = proto && Object.getOwnPropertyDescriptor(proto, "value");
    if (descriptor && descriptor.set) descriptor.set.call(el, value);
    else el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function clickBridgeButton() {
    var wrap = document.querySelector("#profile-dblclick-btn");
    var btn = wrap && (wrap.querySelector("button") || wrap);
    if (!btn) return false;
    btn.click();
    return true;
  }

  document.addEventListener("dblclick", function(event) {
    var target = event.target;
    var tableRoot = target.closest && target.closest("#knowledge-node-table");
    if (!tableRoot) {
      tableRoot = findInPath(event, function(el) {
        return el.id === "knowledge-node-table" || (el.classList && el.classList.contains("knowledge-node-table"));
      });
    }
    if (!tableRoot) return;

    var row = target.closest && target.closest("tbody tr, tr, [role='row']");
    if (!row || !tableRoot.contains(row)) {
      row = findInPath(event, function(el) {
        return tableRoot.contains(el) && (el.matches("tbody tr, tr") || el.getAttribute("role") === "row");
      });
    }
    if (!row) return;

    var rows = Array.prototype.slice.call(tableRoot.querySelectorAll("tbody tr"));
    if (!rows.length) {
      rows = Array.prototype.slice.call(tableRoot.querySelectorAll("[role='row']")).filter(function(item) {
        return item.querySelector("[role='gridcell'], td");
      });
    }
    var rowIndex = rows.indexOf(row);
    if (rowIndex < 0) return;

    var inputWrap = document.querySelector("#profile-dblclick-row");
    var input = inputWrap && inputWrap.querySelector("input, textarea");
    if (!input) return;
    setNativeValue(input, String(rowIndex));
    setTimeout(clickBridgeButton, 50);
  }, true);
}
"""

def find_available_port(start_port: int, max_attempts: int = 20) -> int:
    """Find an available local TCP port, starting from the preferred port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise OSError(f"Cannot find empty port in range: {start_port}-{start_port + max_attempts - 1}")


def toggle_ocr_output(enable_ocr: bool):
    """根据 OCR 开关控制 OCR 输出框是否显示。"""
    return gr.update(visible=enable_ocr)


def add_single_image(image: Any, current_images: list[Any] | None):
    """Add one image from the main image input."""
    images = normalize_uploaded_files(current_images)
    if image is not None:
        images.append(save_image_to_artifacts(image))

    images = images[:3]
    return images, build_gallery_items(images), gr.update(value=None), None


def add_batch_images(files: Any, current_images: list[Any] | None):
    """Add multiple images from the batch upload input."""
    images = normalize_uploaded_files(current_images)
    images.extend(normalize_uploaded_files(files))
    images = images[:3]
    return images, build_gallery_items(images), None


def clear_images():
    """Clear selected images."""
    return [], [], None


def clear_composer_after_send():
    """Reset the composer after one send finishes."""
    return "", [], [], None


def select_image(evt: gr.SelectData):
    """Record the selected gallery index."""
    return evt.index


def delete_selected_image(current_images: list[Any] | None, selected_index: int | None):
    """Delete the selected image from the gallery."""
    images = normalize_uploaded_files(current_images)
    if selected_index is not None and 0 <= selected_index < len(images):
        images.pop(selected_index)
    return images, build_gallery_items(images), None


def mark_generation_stopped(
    current_messages: list[dict[str, str]] | None,
    current_debug_info: str | None,
):
    """Update the UI state after stopping generation."""
    messages = list(current_messages or [])
    stop_notice = "\n\n> 模型已停止思考。"

    if messages and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content") or ""
        if "模型已停止思考" not in content:
            messages[-1] = {**messages[-1], "content": f"{content}{stop_notice}".strip()}
    else:
        messages.append({"role": "assistant", "content": "模型已停止思考。"})

    debug_info = current_debug_info or ""
    if "状态：模型流式输出中" in debug_info:
        debug_info = debug_info.replace("状态：模型流式输出中", "状态：模型已停止")
    elif "状态：模型已停止" not in debug_info:
        debug_info = f"{debug_info}\n状态：模型已停止".strip()

    return messages, debug_info


def run_agent_sync_legacy(user_text: str, image_state: list[Any] | None, enable_ocr: bool) -> Tuple[str, str, str]:
    """Legacy synchronous Agent flow."""
    ensure_project_dirs()

    clean_user_text = (user_text or "").strip()
    uploaded_images = normalize_uploaded_files(image_state)
    uploaded_image_count = len(uploaded_images)
    images = uploaded_images[:3]
    image_count = len(images)
    has_images = image_count > 0

    ocr_text = ""
    ocr_error = ""
    if enable_ocr and has_images:
        ocr_text, ocr_error = extract_text_from_images(images)
        if not ocr_text and not ocr_error:
            ocr_error = "OCR 已启用，但未识别到文字。"
    elif enable_ocr:
        ocr_error = "OCR 已启用，但未上传图片。"

    combined_text = "\n".join(part for part in [clean_user_text, ocr_text] if part).strip()
    if not combined_text and not has_images:
        return "请输入问题，或添加 1-3 张包含题目的图片。", "", "未调用 Skill；未写入日志。"

    skill_input = combined_text or f"用户上传了 {image_count} 张题目图片，请直接阅读图片并回答。"
    selected_skills, skill_context = run_selected_skills(skill_input)
    prompt = build_prompt(
        user_text=clean_user_text,
        ocr_text=ocr_text,
        skill_context=skill_context,
        image_count=image_count,
    )
    answer = call_qwen_model(prompt=prompt, images=images if has_images else None)
    answer = normalize_markdown_math(answer)
    ocr_display = f"{ocr_text}\n\n{ocr_error}".strip() if ocr_error else ocr_text
    log_path = write_log(
        {
            "user_text": clean_user_text,
            "uploaded_image_count": uploaded_image_count,
            "image_count": image_count,
            "ignored_image_count": max(uploaded_image_count - image_count, 0),
            "ocr_enabled": enable_ocr,
            "ocr_text": ocr_text,
            "ocr_error": ocr_error,
            "selected_skills": selected_skills,
            "skill_context": skill_context,
            "prompt": prompt,
            "model_answer": answer,
        }
    )
    debug_info = (
        f"图片直传模型：{'是' if has_images else '否'}\n"
        f"已添加图片数：{uploaded_image_count}\n"
        f"实际使用图片数：{image_count}\n"
        f"忽略图片数：{max(uploaded_image_count - image_count, 0)}\n"
        f"OCR 启用：{'是' if enable_ocr else '否'}\n"
        f"已调用 Skill：{', '.join(selected_skills) if selected_skills else '无'}\n"
        f"日志文件：{log_path}"
    )
    return answer, ocr_display, debug_info

def run_agent(
    task_choice: str | None,
    user_text: str,
    image_state: list[Any] | None,
    enable_ocr: bool,
    chat_history_state: list[dict[str, str]] | None,
) -> Iterator[Tuple[list[dict[str, str]], str, str, list[dict[str, str]]]]:
    """Agent main flow: log early, then stream model output to the page."""
    ensure_project_dirs()

    clean_user_text = (user_text or "").strip()
    chat_history = list(chat_history_state or [])[-40:]
    uploaded_images = normalize_uploaded_files(image_state)
    uploaded_image_count = len(uploaded_images)
    init_db()
    task = None
    if task_choice:
        task_id = str(task_choice)
        if "(" in task_id and task_id.endswith(")"):
            task_id = task_id.rsplit("(", 1)[-1][:-1]
        task = get_task(task_id)
        if task:
            set_active_task(task.id)
    task = task or get_or_create_default_task()

    images = uploaded_images[:3]
    image_count = len(images)
    has_images = image_count > 0

    user_content = clean_user_text or f"用户上传了 {image_count} 张图片。"
    if image_count:
        user_content = f"{user_content}\n\n[已上传 {image_count} 张图片]"
    messages = chat_history + [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]

    if not clean_user_text and not has_images:
        messages[-1]["content"] = "请输入问题，或添加 1-3 张包含题目的图片。"
        yield messages, "", "未调用 Skill；未写入日志。", chat_history
        return

    ocr_text = ""
    ocr_error = ""
    if enable_ocr and has_images:
        messages[-1]["content"] = "正在识别图片文字..."
        yield messages, "", "OCR 处理中，稍后会开始调用模型。", chat_history

        ocr_text, ocr_error = extract_text_from_images(images)
        if not ocr_text and not ocr_error:
            ocr_error = "OCR 已启用，但未识别到文字。"
    elif enable_ocr:
        ocr_error = "OCR 已启用，但未上传图片。"

    combined_text = "\n".join(part for part in [clean_user_text, ocr_text] if part).strip()
    skill_input = combined_text or f"用户上传了 {image_count} 张题目图片，请直接阅读图片并回答。"
    selected_skills, skill_context = run_selected_skills(skill_input)
    primary_skill = selected_skills[0] if selected_skills else "general"
    learning_context = build_learning_context(task, skill_input, primary_skill)
    role_config = load_role_config(task.role_type)
    learning_context_text = learning_context.to_prompt_text(role_config.get("role_name", task.role_type))
    memory_query = "\n".join(part for part in [clean_user_text, ocr_text, skill_context] if part).strip()
    retrieved_memories, long_term_memory_text = retrieve_memory_context(DEFAULT_USER_ID, memory_query)
    chat_history_text, history_turns = format_chat_history(chat_history)

    prompt = build_prompt(
        user_text=clean_user_text,
        ocr_text=ocr_text,
        skill_context=skill_context,
        image_count=image_count,
        chat_history_text=chat_history_text,
        long_term_memory_text=long_term_memory_text,
        learning_context_text=learning_context_text,
    )

    ocr_display = ocr_text
    if ocr_error:
        ocr_display = f"{ocr_display}\n\n{ocr_error}".strip()

    base_record = {
        "user_text": clean_user_text,
        "uploaded_image_count": uploaded_image_count,
        "image_count": image_count,
        "ignored_image_count": max(uploaded_image_count - image_count, 0),
        "ocr_enabled": enable_ocr,
        "ocr_text": ocr_text,
        "ocr_error": ocr_error,
        "selected_skills": selected_skills,
        "skill_context": skill_context,
        "learning_task": {
            "task_id": task.id,
            "task_name": task.task_name,
            "role_type": task.role_type,
            "subject": learning_context.subject,
            "primary_skill": learning_context.primary_skill,
            "knowledge_node_id": learning_context.knowledge_node_id,
            "knowledge_path": learning_context.knowledge_path,
        },
        "history_used": history_turns > 0,
        "history_turns": history_turns,
        "memory": {
            "retrieved": [
                {
                    "id": item.get("id"),
                    "content": item.get("content"),
                    "memory_type": item.get("memory_type"),
                    "score": item.get("score"),
                }
                for item in retrieved_memories
            ],
            "saved": [],
        },
        "prompt": prompt,
    }

    request_log_path = write_log({"event": "request_received", **base_record})
    debug_info = (
        f"图片直传模型：{'是' if has_images else '否'}\n"
        f"已添加图片数：{uploaded_image_count}\n"
        f"实际使用图片数：{image_count}\n"
        f"忽略图片数：{max(uploaded_image_count - image_count, 0)}\n"
        f"OCR 启用：{'是' if enable_ocr else '否'}\n"
        f"已调用 Skill：{', '.join(selected_skills) if selected_skills else '无'}\n"
        f"学习任务：{task.task_name} / {task.role_type}\n"
        f"画像预判：{learning_context.subject} / {' > '.join(learning_context.knowledge_path) if learning_context.knowledge_path else learning_context.knowledge_node_id}\n"
        f"短期历史轮数：{history_turns}\n"
        f"长期记忆命中：{len(retrieved_memories)}\n"
        f"日志文件：{request_log_path}\n"
        "状态：模型流式输出中"
    )
    messages[-1]["content"] = "正在调用模型..."
    yield messages, ocr_display, debug_info, chat_history

    answer_parts: list[str] = []
    last_yield_time = time.monotonic()
    last_yield_length = 0
    min_yield_interval = 0.5
    min_yield_chars = 80
    try:
        for delta in call_qwen_model_stream(prompt=prompt, images=images if has_images else None):
            answer_parts.append(delta)
            raw_answer = "".join(answer_parts)
            now = time.monotonic()
            enough_time = now - last_yield_time >= min_yield_interval
            enough_text = len(raw_answer) - last_yield_length >= min_yield_chars
            if enough_time or enough_text:
                messages[-1]["content"] = raw_answer
                yield messages, ocr_display, debug_info, chat_history
                last_yield_time = now
                last_yield_length = len(raw_answer)

        final_answer = normalize_markdown_math("".join(answer_parts))
        messages[-1]["content"] = final_answer
        saved_memories = extract_and_save_memories(
            user_id=DEFAULT_USER_ID,
            user_text=clean_user_text,
            assistant_answer=final_answer,
            source="chat",
        )
        updated_history = (chat_history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": final_answer},
        ])[-40:]
        completion_record = {
            **base_record,
            "memory": {
                **base_record["memory"],
                "saved": [
                    {
                        "id": item.get("id"),
                        "content": item.get("content"),
                        "memory_type": item.get("memory_type"),
                    }
                    for item in saved_memories
                ],
            },
            "model_answer": final_answer,
        }
        completion_log_path = write_log(
            {
                "event": "model_completed",
                **completion_record,
            }
        )
        profile_note = ""
        try:
            question_id = save_question(
                task_id=task.id,
                user_input=clean_user_text or skill_input,
                assistant_answer=final_answer,
                primary_skill=learning_context.primary_skill,
                subject=learning_context.subject,
                knowledge_node_id=learning_context.knowledge_node_id,
                knowledge_path=learning_context.knowledge_path,
            )
            event = analyze_learning_event(
                task=task,
                context=learning_context,
                user_input=clean_user_text or skill_input,
                assistant_answer=final_answer,
            )
            update_question_with_event(question_id, event)
            save_knowledge_event(question_id, event)
            update_knowledge_state(task.id, event)
            if event.get("should_add_to_mistake_book"):
                add_to_mistake_book(
                    task.id,
                    question_id,
                    event,
                    original_question=clean_user_text or skill_input,
                    correct_solution=final_answer,
                )
            sync_task_artifacts(task)
            profile_note = (
                f"\n画像记录：{event.get('subject') or '-'} / "
                f"{' > '.join(event.get('knowledge_path') or []) or event.get('knowledge_node_id') or '-'}"
            )
        except Exception as profile_exc:
            write_log(
                {
                    "event": "profile_update_failed",
                    "task_id": task.id,
                    "error": str(profile_exc),
                }
            )
            profile_note = f"\n画像记录失败：{profile_exc}"
        final_debug_info = debug_info.replace("状态：模型流式输出中", "状态：模型调用完成")
        final_debug_info = f"{final_debug_info}\n新增长期记忆：{len(saved_memories)}{profile_note}\n完成日志：{completion_log_path}"
        yield messages, ocr_display, final_debug_info, updated_history
    except Exception as exc:
        partial_answer = normalize_markdown_math("".join(answer_parts))
        error_log_path = write_log(
            {
                "event": "model_failed",
                **base_record,
                "partial_model_answer": partial_answer,
                "error": str(exc),
            }
        )
        messages[-1]["content"] = partial_answer or f"模型调用失败：{exc}"
        yield messages, ocr_display, f"{debug_info}\n状态：模型调用失败\n错误日志：{error_log_path}", chat_history


def build_ui() -> gr.Blocks:
    """Backward-compatible wrapper for the current UI."""
    return build_ui_v2()

def build_ui_v2() -> gr.Blocks:
    """Build the task-aware Gradio UI."""
    init_db()
    active_task = get_or_create_default_task()
    initial_chat_history = load_recent_chat_messages(active_task.id)
    choices = task_choices()
    initial_subjects = active_task.subjects or ["通用"]
    initial_subject_choices = [(item, item) for item in initial_subjects]
    css_path = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
    css_text = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_text = f.read()
    with gr.Blocks(title="Adabot") as demo:
        if css_text:
            gr.HTML(f"<style>{css_text}</style>")
        image_state = gr.State([])
        selected_image_index = gr.State(None)
        chat_history_state = gr.State(initial_chat_history)
        current_node_id_state = gr.State("__root__")
        nav_stack_state = gr.State([])
        selected_node_id_state = gr.State("")
        profile_node_rows_state = gr.State([])
        profile_loaded_state = gr.State(False)
        current_subject_state = gr.State((active_task.subjects or ["通用"])[0])

        gr.Markdown("# Adabot")
        gr.Markdown(f"当前已注册 Skill：`{', '.join(list_skills())}`")
        with gr.Row():
            task_dropdown = gr.Dropdown(
                label="当前学习任务",
                choices=choices,
                value=active_task.id if choices else None,
                scale=4,
            )
            refresh_tasks_btn = gr.Button("刷新", size="sm", scale=1)
        active_task_md = gr.Markdown(active_task_label())
        with gr.Row():
            ui_health_btn = gr.Button("UI 健康检查", size="sm")
            ui_health_output = gr.Textbox(label="UI 健康检查结果", value="", interactive=False)

        with gr.Tabs():
            with gr.Tab("聊天"):
                answer = gr.Chatbot(
                    label="",
                    height=620,
                    buttons=["copy", "copy_all"],
                    latex_delimiters=LATEX_DELIMITERS,
                    render_markdown=True,
                    line_breaks=True,
                    elem_classes=["chat-stream"],
                    value=initial_chat_history,
                )
                with gr.Row():
                    user_text = gr.Textbox(
                        label="",
                        placeholder="输入你的问题，或上传题目图片。例如：矩阵乘法为什么是行乘列？这题我错了，加入错题本。",
                        lines=5,
                        elem_classes=["composer-text"],
                    )
                    with gr.Column():
                        image_input = gr.Image(
                            label="",
                            show_label=False,
                            type="pil",
                            sources=["upload", "clipboard", "webcam"],
                            height=190,
                            elem_classes=["composer-image"],
                        )
                        image_preview = gr.Gallery(
                            label="",
                            show_label=False,
                            type="filepath",
                            columns=3,
                            rows=1,
                            height=120,
                            object_fit="contain",
                            allow_preview=True,
                            preview=True,
                            elem_classes=["image-preview-strip"],
                        )
                        with gr.Row():
                            batch_upload = gr.UploadButton(
                                "批量上传",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                size="sm",
                            )
                            delete_btn = gr.Button("删除选中图片", size="sm")
                            clear_btn = gr.Button("清空图片", size="sm")

                with gr.Row(elem_classes=["composer-actions"]):
                    enable_ocr = gr.Checkbox(label="OCR", value=False)
                    submit_btn = gr.Button("发送", variant="primary")
                    stop_btn = gr.Button("停止生成", variant="stop")
                ocr_output = gr.Textbox(label="OCR 状态 / 识别结果", lines=6, visible=False)
                debug_output = gr.Textbox(label="调试信息", lines=6)

            with gr.Tab("任务管理"):
                task_list_md = gr.Markdown(render_task_list())
                with gr.Row():
                    new_task_name = gr.Textbox(label="任务名称", value="2027考研机械")
                    new_role = gr.Dropdown(
                        label="角色类型",
                        choices=["考研", "高考", "雅思", "本科生", "研究生", "自定义"],
                        value="考研",
                    )
                goal_text = gr.Textbox(label="学习目标描述", value="备考考研数学一、英语一、机械原理", lines=2)
                with gr.Row():
                    target_exam = gr.Textbox(label="目标考试", value="2027考研")
                    target_date = gr.Textbox(label="目标日期", value="2027-12-20")
                subjects_text = gr.Textbox(label="科目列表，逗号分隔", value="高等数学,线性代数,概率论,英语,机械原理")
                custom_outline = gr.Textbox(
                    label="自定义知识框架（custom 任务可填写，按缩进或 - 表示层级）",
                    placeholder="公共卫生基础\n  - 公共卫生定义\n  - 核心职能\n流行病学\n  - 发病率\n  - 队列研究\n医学大模型\n  - 数据集\n  - 评测基准",
                    lines=8,
                )
                answer_style_box = gr.Textbox(label="回答风格", value="", lines=2)
                with gr.Row():
                    enable_profile = gr.Checkbox(label="启用学习画像", value=True)
                    enable_mistake = gr.Checkbox(label="启用错题本", value=True)
                    enable_review = gr.Checkbox(label="启用复习计划", value=True)
                create_task_btn = gr.Button("新建任务", variant="primary")
                create_task_status = gr.Markdown()

            with gr.Tab("学习画像"):
                with gr.Row():
                    profile_subject = gr.Dropdown(label="科目", choices=initial_subject_choices, value=initial_subjects[0])
                    profile_view = gr.Radio(
                        choices=["树状图", "思维导图", "知识网络图谱"],
                        label="视图切换",
                        value="树状图",
                    )
                    profile_refresh_btn = gr.Button("刷新画像")
                    profile_export_btn = gr.DownloadButton("导出思维导图 (Markdown)")
                    profile_export_opml_btn = gr.DownloadButton("导出思维导图 (OPML/XMind)")
                profile_chart = gr.HTML("<div class='node-detail-empty'>请点击“刷新画像”加载图谱。</div>")
                with gr.Row():
                    with gr.Column(scale=2):
                        back_parent_btn = gr.Button("<", elem_id="profile-hidden-back", elem_classes=["hidden-bridge"], interactive=False, min_width=36, scale=0)
                        enter_child_btn = gr.Button("进入", elem_id="profile-hidden-enter", elem_classes=["hidden-bridge"], interactive=False, min_width=54, scale=0)
                        profile_dblclick_row = gr.Textbox(value="", elem_id="profile-dblclick-row", elem_classes=["hidden-bridge"], show_label=False, container=False)
                        profile_dblclick_btn = gr.Button("双击下钻", elem_id="profile-dblclick-btn", elem_classes=["hidden-bridge"])
                        profile_breadcrumb = gr.Markdown("### 当前科目：")
                        profile_node_selector = gr.Dataframe(
                            label="",
                            headers=["知识点", "状态", "累计提问"],
                            datatype=["str", "str", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                            elem_id="knowledge-node-table",
                            elem_classes=["knowledge-node-table"],
                        )
                    with gr.Column(scale=1):
                        question_time_filter = gr.Dropdown(
                            label="题集时间筛选",
                            choices=["全部时间", "今天需复习", "最近3天出错", "超过1周未看"],
                            value="全部时间",
                        )
                        profile_node_detail = gr.HTML("<div class='node-detail-empty'>请选择一个知识点。</div>")
                gr.Markdown("### 上传我的笔记 / 知识框架")
                with gr.Row():
                    notes_files = gr.File(
                        label="上传笔记文件",
                        file_count="multiple",
                        file_types=["image", ".pdf", ".docx", ".txt", ".md"],
                    )
                    notes_text = gr.Textbox(
                        label="直接粘贴笔记",
                        placeholder="可以粘贴电子版知识框架、学习心得、错题总结...",
                        lines=6,
                    )
                notes_process_btn = gr.Button("解析并更新学习画像")
                notes_result = gr.Markdown()

            with gr.Tab("错题本"):
                with gr.Row():
                    mistake_subject = gr.Textbox(label="科目过滤", placeholder="可留空")
                    mistake_keyword = gr.Textbox(label="关键词 / 知识点过滤", placeholder="例如：矩阵乘法")
                    mistake_time_filter = gr.Dropdown(
                        label="时间筛选",
                        choices=["全部时间", "今天需复习", "最近3天出错", "超过1周未看"],
                        value="全部时间",
                    )
                    mistake_refresh_btn = gr.Button("刷新错题本")
                mistake_output = gr.Markdown()
                gr.Markdown("### 手动添加错题")
                with gr.Group():
                    manual_image = gr.Image(label="上传题目截图（可选）", type="filepath")
                    manual_question = gr.Textbox(label="原题/问题补充", lines=3)
                    manual_reason = gr.Textbox(label="错误原因分析", lines=2)
                    manual_add_btn = gr.Button("手动加入错题本")

            with gr.Tab("复习计划"):
                review_refresh_btn = gr.Button("生成复习建议")
                review_output = gr.Markdown()

            with gr.Tab("历史记忆检索"):
                history_keyword = gr.Textbox(label="关键词", placeholder="例如：矩阵乘法")
                history_search_btn = gr.Button("检索当前任务")
                history_output = gr.Markdown()

            with gr.Tab("小测验"):
                with gr.Row():
                    quiz_subject = gr.Dropdown(label="科目", choices=initial_subject_choices, value=initial_subjects[0])
                    quiz_knowledge = gr.Textbox(label="知识点路径 / 名称", value="线性代数 > 矩阵 > 矩阵乘法")
                with gr.Row():
                    quiz_count = gr.Number(label="题目数量", value=3, precision=0)
                    quiz_difficulty = gr.Dropdown(label="难度", choices=["基础", "进阶", "综合"], value="基础")
                    quiz_btn = gr.Button("生成小测")
                quiz_output = gr.Markdown()
                quiz_node_id = gr.Textbox(label="知识点 ID", value="general")
                quiz_answer = gr.Textbox(label="提交答案", lines=5)
                quiz_grade_btn = gr.Button("批改并更新掌握状态")
                quiz_grade_output = gr.Markdown()

        ui_health_btn.click(fn=lambda: "OK: UI callback works", inputs=[], outputs=[ui_health_output])
        refresh_tasks_btn.click(fn=refresh_task_dropdown, inputs=[], outputs=[task_dropdown, active_task_md])
        task_switch_event = task_dropdown.change(
            fn=switch_task, inputs=[task_dropdown], 
            outputs=[active_task_md, chat_history_state]).then(
            fn=subject_choices_for_task_pair,
            inputs=[task_dropdown],
            outputs=[profile_subject, quiz_subject],
        ).then(
            fn=lambda history: history,
            inputs=[chat_history_state],
            outputs=[answer],
        ).then(
            fn=reset_profile_loaded,
            inputs=[],
            outputs=[profile_loaded_state, profile_chart],
        )
        # if not DEBUG_DISABLE_PROFILE_EVENTS:
        #     task_switch_event.then(
        #         fn=initialize_knowledge_drilldown,
        #         inputs=[task_dropdown, profile_subject],
        #         outputs=[
        #             current_node_id_state,
        #             nav_stack_state,
        #             selected_node_id_state,
        #             current_subject_state,
        #             profile_breadcrumb,
        #             profile_node_selector,
        #             profile_node_detail,
        #             back_parent_btn,
        #         ],
        #     )
        create_task_event = create_task_btn.click(
            fn=create_task_from_form,
            inputs=[
                new_task_name,
                new_role,
                goal_text,
                target_exam,
                target_date,
                subjects_text,
                answer_style_box,
                enable_profile,
                enable_mistake,
                enable_review,
                custom_outline,
            ],
            outputs=[task_dropdown, create_task_status, task_list_md, chat_history_state],
        ).then(fn=refresh_task_dropdown, inputs=[], outputs=[task_dropdown, active_task_md]).then(
            fn=subject_choices_for_task_pair,
            inputs=[task_dropdown],
            outputs=[profile_subject, quiz_subject],
        ).then(
            fn=lambda history: history,
            inputs=[chat_history_state],
            outputs=[answer],
        ).then(
            fn=reset_profile_loaded,
            inputs=[],
            outputs=[profile_loaded_state, profile_chart],
        )
        # if not DEBUG_DISABLE_PROFILE_EVENTS:
        #     create_task_event.then(
        #         fn=initialize_knowledge_drilldown,
        #         inputs=[task_dropdown, profile_subject],
        #         outputs=[
        #             current_node_id_state,
        #             nav_stack_state,
        #             selected_node_id_state,
        #             current_subject_state,
        #             profile_breadcrumb,
        #             profile_node_selector,
        #             profile_node_detail,
        #             back_parent_btn,
        #         ],
        #     )
        if not DEBUG_DISABLE_PROFILE_EVENTS:
            profile_refresh_event = profile_refresh_btn.click(
                fn=initialize_knowledge_drilldown,
                inputs=[task_dropdown, profile_subject],
                outputs=[
                    current_node_id_state,
                    nav_stack_state,
                    selected_node_id_state,
                    current_subject_state,
                    profile_node_rows_state,
                    profile_breadcrumb,
                    profile_node_selector,
                    profile_node_detail,
                    back_parent_btn,
                    enter_child_btn,
                ],
            )
            profile_refresh_event.then(
                fn=render_profile_chart,
                inputs=[task_dropdown, profile_subject, profile_view],
                outputs=[profile_chart],
            ).then(
                fn=mark_profile_loaded,
                inputs=[],
                outputs=[profile_loaded_state],
            )
            profile_view.change(
                fn=render_profile_chart_if_loaded,
                inputs=[profile_loaded_state, task_dropdown, profile_subject, profile_view],
                outputs=[profile_chart],
            )
            profile_node_selector.select(
                fn=on_knowledge_node_click,
                inputs=[profile_node_rows_state, current_node_id_state, nav_stack_state, task_dropdown, profile_subject],
                outputs=[
                    selected_node_id_state,
                    profile_breadcrumb,
                    profile_node_detail,
                    enter_child_btn,
                ],
            )
            enter_child_btn.click(
                fn=on_enter_child_node,
                inputs=[selected_node_id_state, current_node_id_state, nav_stack_state, task_dropdown, profile_subject],
                outputs=[
                    current_node_id_state,
                    nav_stack_state,
                    selected_node_id_state,
                    profile_node_rows_state,
                    profile_node_selector,
                    profile_breadcrumb,
                    profile_node_detail,
                    back_parent_btn,
                    enter_child_btn,
                ],
            )
            profile_dblclick_btn.click(
                fn=on_knowledge_node_double_click,
                inputs=[profile_dblclick_row, profile_node_rows_state, current_node_id_state, nav_stack_state, task_dropdown, profile_subject],
                outputs=[
                    current_node_id_state,
                    nav_stack_state,
                    selected_node_id_state,
                    profile_node_rows_state,
                    profile_node_selector,
                    profile_breadcrumb,
                    profile_node_detail,
                    back_parent_btn,
                    enter_child_btn,
                ],
            )
            back_parent_btn.click(
                fn=on_back_to_parent,
                inputs=[current_node_id_state, nav_stack_state, task_dropdown, profile_subject],
                outputs=[
                    current_node_id_state,
                    nav_stack_state,
                    selected_node_id_state,
                    profile_node_rows_state,
                    profile_node_selector,
                    profile_breadcrumb,
                    profile_node_detail,
                    back_parent_btn,
                    enter_child_btn,
                ],
            )
            notes_process_btn.click(
                fn=process_notes_ui,
                inputs=[task_dropdown, notes_files, notes_text],
                outputs=[notes_result],
            )
            profile_export_btn.click(
                fn=export_mindmap_markdown,
                inputs=[task_dropdown, profile_subject],
                outputs=[profile_export_btn],
            )
            profile_export_opml_btn.click(
                fn=export_mindmap_opml,
                inputs=[task_dropdown, profile_subject],
                outputs=[profile_export_opml_btn],
            )
            question_time_filter.change(
                fn=render_selected_node_question_set,
                inputs=[task_dropdown, profile_subject, selected_node_id_state, question_time_filter],
                outputs=[profile_node_detail],
            )

        mistake_refresh_btn.click(fn=render_mistakes_ui_filtered, inputs=[task_dropdown, mistake_subject, mistake_keyword, mistake_time_filter], outputs=[mistake_output])
        manual_add_btn.click(fn=manual_add_mistake_ui, inputs=[task_dropdown, mistake_subject, manual_image, manual_question, manual_reason], outputs=[mistake_output])
        review_refresh_btn.click(fn=render_review_plan_ui, inputs=[task_dropdown], outputs=[review_output])
        history_search_btn.click(fn=search_history_ui, inputs=[task_dropdown, history_keyword], outputs=[history_output])
        quiz_btn.click(fn=generate_quiz_ui, inputs=[task_dropdown, quiz_subject, quiz_knowledge, quiz_count, quiz_difficulty], outputs=[quiz_output])
        quiz_grade_btn.click(fn=grade_quiz_ui, inputs=[task_dropdown, quiz_node_id, quiz_answer], outputs=[quiz_grade_output])

        image_input.change(fn=add_single_image, inputs=[image_input, image_state], outputs=[image_state, image_preview, image_input, selected_image_index])
        batch_upload.upload(fn=add_batch_images, inputs=[batch_upload, image_state], outputs=[image_state, image_preview, selected_image_index])
        image_preview.select(fn=select_image, inputs=[], outputs=[selected_image_index])
        delete_btn.click(fn=delete_selected_image, inputs=[image_state, selected_image_index], outputs=[image_state, image_preview, selected_image_index])
        clear_btn.click(fn=clear_images, inputs=[], outputs=[image_state, image_preview, selected_image_index])
        enable_ocr.change(fn=toggle_ocr_output, inputs=[enable_ocr], outputs=[ocr_output])

        submit_event = submit_btn.click(
            fn=run_agent,
            inputs=[task_dropdown, user_text, image_state, enable_ocr, chat_history_state],
            outputs=[answer, ocr_output, debug_output, chat_history_state],
        )
        submit_done = submit_event.then(
            fn=clear_composer_after_send,
            inputs=[],
            outputs=[user_text, image_state, image_preview, selected_image_index],
        )
        enter_event = user_text.submit(
            fn=run_agent,
            inputs=[task_dropdown, user_text, image_state, enable_ocr, chat_history_state],
            outputs=[answer, ocr_output, debug_output, chat_history_state],
        )
        enter_done = enter_event.then(
            fn=clear_composer_after_send,
            inputs=[],
            outputs=[user_text, image_state, image_preview, selected_image_index],
        )
        stop_btn.click(
            fn=mark_generation_stopped,
            inputs=[answer, debug_output],
            outputs=[answer, debug_output],
            cancels=[submit_event, enter_event],
        )
        demo.load(fn=None, inputs=[], outputs=[], js=KNOWLEDGE_DBLCLICK_JS)
    return demo


if __name__ == "__main__":
    configure_utf8_runtime()
    ensure_project_dirs()
    init_db()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    server_port = find_available_port(preferred_port)
    app = build_ui_v2()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        debug=True,
        show_error=True,
        # share=True
        share=False
    )

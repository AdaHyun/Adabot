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
    """
    从主图片框添加单张图片。

    主图片框负责上传、粘贴、拍摄；添加后保存到 artifacts，并刷新缩略图列表。
    """
    images = normalize_uploaded_files(current_images)
    if image is not None:
        images.append(save_image_to_artifacts(image))

    images = images[:3]
    return images, build_gallery_items(images), gr.update(value=None), None


def add_batch_images(files: Any, current_images: list[Any] | None):
    """
    从批量上传入口添加多张图片。

    该入口用于一次选择多张；仍然只保留前 3 张。
    """
    images = normalize_uploaded_files(current_images)
    images.extend(normalize_uploaded_files(files))
    images = images[:3]
    return images, build_gallery_items(images), None


def clear_images():
    """清空当前已添加图片。"""
    return [], [], None


def select_image(evt: gr.SelectData):
    """记录 Gallery 中当前选中的图片序号。"""
    return evt.index


def delete_selected_image(current_images: list[Any] | None, selected_index: int | None):
    """删除 Gallery 中选中的单张图片。"""
    images = normalize_uploaded_files(current_images)
    if selected_index is not None and 0 <= selected_index < len(images):
        images.pop(selected_index)
    return images, build_gallery_items(images), None


def mark_generation_stopped(
    current_messages: list[dict[str, str]] | None,
    current_debug_info: str | None,
):
    """停止生成后同步更新界面，避免状态一直停留在“模型流式输出中”。"""
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
    """
    Agent 主流程。

    参数：
    - user_text：用户输入的文本说明。
    - image_state：当前已添加的图片路径列表。
    - enable_ocr：是否启用 OCR 辅助识别。
    """
    ensure_project_dirs()

    clean_user_text = (user_text or "").strip()
    uploaded_images = normalize_uploaded_files(image_state)
    uploaded_image_count = len(uploaded_images)

    # 最多使用前 3 张图片；如果未来有更多来源传入，也只取前 3 张。
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
        return "请输入 Text Description，或添加 1-3 张包含题目的图片。", "", "未调用 Skill；未写入日志。"

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

    ocr_display = ocr_text
    if ocr_error:
        ocr_display = f"{ocr_display}\n\n{ocr_error}".strip()

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
    user_text: str,
    image_state: list[Any] | None,
    enable_ocr: bool,
    chat_history_state: list[dict[str, str]] | None,
) -> Iterator[Tuple[list[dict[str, str]], str, str, list[dict[str, str]]]]:
    """Agent main flow: log early, then stream model output to the page."""
    ensure_project_dirs()

    clean_user_text = (user_text or "").strip()
    chat_history = trim_chat_history(chat_history_state)
    uploaded_images = normalize_uploaded_files(image_state)
    uploaded_image_count = len(uploaded_images)

    images = uploaded_images[:3]
    image_count = len(images)
    has_images = image_count > 0

    messages = [
        {
            "role": "user",
            "content": clean_user_text or f"用户上传了 {image_count} 张图片。"
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]

    if not clean_user_text and not has_images:
        messages[-1]["content"] = "请输入 Text Description，或添加 1-3 张包含题目的图片。"
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
        updated_history = append_turn_to_history(chat_history, clean_user_text, final_answer, image_count=image_count)
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
        final_debug_info = debug_info.replace("状态：模型流式输出中", "状态：模型调用完成")
        final_debug_info = f"{final_debug_info}\n新增长期记忆：{len(saved_memories)}\n完成日志：{completion_log_path}"
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
    """构建 Gradio 页面。"""
    with gr.Blocks(title="Adabot") as demo:
        image_state = gr.State([])
        selected_image_index = gr.State(None)
        chat_history_state = gr.State([])

        gr.Markdown("# Adabot")
        gr.Markdown(f"当前已注册 Skill：`{', '.join(list_skills())}`")

        with gr.Row():
            user_text = gr.Textbox(
                label="Text Description",
                placeholder="例如：告诉我图片中第一个积分怎么算；分析这道车辆工程题；帮我修改 IELTS Task 2 作文...",
                lines=8,
            )

            with gr.Column():
                image_input = gr.Image(
                    label="",
                    show_label=False,
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    height=260,
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

        enable_ocr = gr.Checkbox(
            label="启用 OCR 辅助识别",
            value=False,
        )

        with gr.Row():
            submit_btn = gr.Button("launch", variant="primary")
            stop_btn = gr.Button("停止生成", variant="stop")

        # Gradio 6.x 的 Chatbot 默认接收 messages 格式，不再支持 type="messages"。
        # 复制按钮也改为通过 buttons 配置，避免旧参数在 6.x 中启动失败。
        answer = gr.Chatbot(
            label="Agent 回答",
            height=520,
            buttons=["copy", "copy_all"],
            # Chatbot 默认也会渲染 Markdown，但公式分隔符需要显式传入。
            # 否则模型输出的 $...$、$$...$$、\(...\)、\[...\] 容易显示成原始文本。
            latex_delimiters=LATEX_DELIMITERS,
            render_markdown=True,
            line_breaks=True,
        )
        ocr_output = gr.Textbox(label="OCR 状态 / 识别结果", lines=6, visible=False)
        debug_output = gr.Textbox(label="调试信息", lines=5)

        image_input.change(
            fn=add_single_image,
            inputs=[image_input, image_state],
            outputs=[image_state, image_preview, image_input, selected_image_index],
        )

        batch_upload.upload(
            fn=add_batch_images,
            inputs=[batch_upload, image_state],
            outputs=[image_state, image_preview, selected_image_index],
        )

        image_preview.select(
            fn=select_image,
            inputs=[],
            outputs=[selected_image_index],
        )

        delete_btn.click(
            fn=delete_selected_image,
            inputs=[image_state, selected_image_index],
            outputs=[image_state, image_preview, selected_image_index],
        )

        clear_btn.click(
            fn=clear_images,
            inputs=[],
            outputs=[image_state, image_preview, selected_image_index],
        )

        enable_ocr.change(
            fn=toggle_ocr_output,
            inputs=[enable_ocr],
            outputs=[ocr_output],
        )

        submit_event = submit_btn.click(
            fn=run_agent,
            inputs=[user_text, image_state, enable_ocr, chat_history_state],
            outputs=[answer, ocr_output, debug_output, chat_history_state],
        )

        enter_event = user_text.submit(
            fn=run_agent,
            inputs=[user_text, image_state, enable_ocr, chat_history_state],
            outputs=[answer, ocr_output, debug_output, chat_history_state],
        )

        # Gradio 会取消 still running 的流式事件；这里同时更新 UI 状态。
        stop_btn.click(
            fn=mark_generation_stopped,
            inputs=[answer, debug_output],
            outputs=[answer, debug_output],
            cancels=[submit_event, enter_event],
        )

    return demo


if __name__ == "__main__":
    configure_utf8_runtime()
    ensure_project_dirs()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    server_port = find_available_port(preferred_port)
    app = build_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        # share=True
        share=False
    )

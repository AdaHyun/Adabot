"""
utils.py

Adabot 的工具模块：
1. 图片输入规范化，支持 1-3 张图片。
2. 图片转 base64 data URL，供多模态模型直接读取。
3. OCR 图片文字识别，可对多张图片逐张识别。
4. OpenAI 兼容接口模型调用。
5. JSONL 日志记录。
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from openai import OpenAI
from PIL import Image


# 项目根目录，保证从任意工作目录启动时都能找到 logs 和 artifacts。
PROJECT_ROOT = Path(__file__).resolve().parent

# 日志目录，用于保存每次请求的完整链路。
LOG_DIR = PROJECT_ROOT / "logs"

# 临时文件目录，后续可放上传文件、缓存等。
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# 上传图片暂存目录，用于跨 Gradio 事件保存一张张添加的图片。
UPLOAD_DIR = ARTIFACT_DIR / "uploads"

# OCR 引擎全局缓存，避免每次上传图片都重新加载模型。
_OCR_ENGINE = None
# 本地 PaddleOCR 模型目录
OCR_MODEL_ROOT = PROJECT_ROOT / "models" / "paddleocr"

OCR_DET_MODEL_DIR = OCR_MODEL_ROOT / "det" /  "ch" / "ch_PP-OCRv4_det_infer"
OCR_REC_MODEL_DIR = OCR_MODEL_ROOT / "rec" /  "ch" / "ch_PP-OCRv4_rec_infer"
OCR_CLS_MODEL_DIR = OCR_MODEL_ROOT / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"


def ensure_project_dirs() -> None:
    """确保项目运行所需目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_image_to_artifacts(image: Any) -> str:
    """将单张图片保存到 artifacts/uploads，并返回文件路径。"""
    ensure_project_dirs()
    pil_image = normalize_image(image)
    file_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    pil_image.save(file_path, format="JPEG", quality=95)
    return str(file_path)


def build_gallery_items(images: List[Any]) -> List[Any]:
    """为 Gradio Gallery 构建预览数据，只保留前 3 张。"""
    return normalize_uploaded_files(images)[:3]


def normalize_markdown_math(text: str) -> str:
    """
    规范化模型输出中的常见 LaTeX 写法，提升 Gradio Markdown 渲染成功率。

    处理示例：
    - $2^x=3, \\log_4 ...$ 这类普通行内公式保持不变。
    - 修复模型偶尔漏写结束 $ 的情况。
    - 将裸露的 \\(...\\) 和 \\[...\\] 保持为 Gradio 可识别格式。
    """
    if not text:
        return ""

    normalized = text

    # 修复中文标点后紧贴的美元符号异常空格。
    normalized = normalized.replace("＄", "$")

    # 常见模型会把整段写成 “已知 $...$，求 $...$。”，这是可渲染的，保留即可。
    # 如果某一行只有奇数个 $，通常说明漏了结尾 $，在行尾补齐。
    fixed_lines: List[str] = []
    for line in normalized.splitlines():
        if line.count("$") % 2 == 1:
            line = f"{line}$"
        fixed_lines.append(line)
    normalized = "\n".join(fixed_lines)

    # 防止列表里 “$log” 被当成普通文本，统一常见 log 写法。
    normalized = re.sub(r"(?<!\\)log_", r"\\log_", normalized)

    return normalized


def normalize_uploaded_files(files: Any) -> List[Any]:
    """
    将 Gradio 上传结果统一整理为列表。

    Gradio File 在不同版本里可能返回：
    - None
    - 单个路径字符串
    - 路径字符串列表
    - 带 name/path 字段的对象或字典
    """
    if files is None:
        return []

    if isinstance(files, (list, tuple)):
        raw_files = list(files)
    else:
        raw_files = [files]

    normalized: List[Any] = []
    for item in raw_files:
        if item is None:
            continue

        # Gallery 可能返回 (image, caption) 形式，只取第一项图片本体。
        if isinstance(item, tuple) and item:
            item = item[0]

        # Gradio 某些版本会返回 dict，文件路径通常在 path 或 name 字段。
        if isinstance(item, dict):
            path = item.get("path") or item.get("name")
            if path:
                normalized.append(path)
            continue

        # Gradio 某些版本会返回带 path/name 属性的文件对象。
        path_attr = getattr(item, "path", None) or getattr(item, "name", None)
        if path_attr:
            normalized.append(path_attr)
            continue

        normalized.append(item)

    return normalized


def normalize_image(image: Any) -> Image.Image:
    """
    将单张图片统一转换为 RGB PIL Image。

    输入可能是 PIL.Image、numpy.ndarray、本地路径、Gradio 文件对象或 dict。
    """
    if image is None:
        raise ValueError("图片为空")

    if isinstance(image, dict):
        image = image.get("path") or image.get("name")

    path_attr = getattr(image, "path", None) or getattr(image, "name", None)
    if path_attr:
        image = path_attr

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")

    raise TypeError(f"不支持的图片类型：{type(image)}")


def image_to_data_url(image: Any, max_side: int = 1600) -> str:
    """
    将单张图片转换为 OpenAI 兼容接口可接收的 data URL。

    图片过大时会按比例缩小，避免请求体过大。
    """
    pil_image = normalize_image(image)

    width, height = pil_image.size
    longest_side = max(width, height)
    if longest_side > max_side:
        scale = max_side / longest_side
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_ocr_engine():
    """
    获取 PaddleOCR 引擎。

    当前项目优先把图片直传给多模态模型，OCR 只作为可手动开启的辅助能力。
    """
    global _OCR_ENGINE

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(f"PaddleOCR 无法导入：{exc}") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"PaddleOCR 依赖加载失败：{exc}") from exc

    missing_dirs = [
        path for path in [OCR_DET_MODEL_DIR, OCR_REC_MODEL_DIR, OCR_CLS_MODEL_DIR]
        if not path.exists()
    ]

    if missing_dirs:
        missing_text = "\n".join(str(path) for path in missing_dirs)
        raise RuntimeError(f"PaddleOCR 本地模型目录不存在，请检查模型复制位置：\n{missing_text}")

    try:
        _OCR_ENGINE = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            det_model_dir=str(OCR_DET_MODEL_DIR),
            rec_model_dir=str(OCR_REC_MODEL_DIR),
            cls_model_dir=str(OCR_CLS_MODEL_DIR),
            show_log=True,
        )
    except Exception:
        _OCR_ENGINE = PaddleOCR(
            lang="ch",
            det_model_dir=str(OCR_DET_MODEL_DIR),
            rec_model_dir=str(OCR_REC_MODEL_DIR),
            cls_model_dir=str(OCR_CLS_MODEL_DIR),
            show_log=True,
        )

    return _OCR_ENGINE


def _parse_ocr_result(result: Any) -> str:
    """兼容解析 PaddleOCR 2.x 和 3.x 的返回结果。"""
    lines: List[str] = []

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "rec_texts" in item:
                lines.extend(str(text).strip() for text in item["rec_texts"] if text)
                continue

            rec_texts = getattr(item, "rec_texts", None)
            if rec_texts:
                lines.extend(str(text).strip() for text in rec_texts if text)
                continue

            if isinstance(item, list):
                for line in item:
                    try:
                        text = line[1][0]
                        if text:
                            lines.append(str(text).strip())
                    except (IndexError, TypeError):
                        continue

    return "\n".join(lines)


def extract_text_from_image(image: Any) -> str:
    """从单张图片中提取 OCR 文本。"""
    pil_image = normalize_image(image)
    image_array = np.array(pil_image)
    ocr_engine = get_ocr_engine()

    if hasattr(ocr_engine, "ocr"):
        try:
            result = ocr_engine.ocr(image_array, cls=True)
        except TypeError:
            result = ocr_engine.ocr(image_array)
    elif hasattr(ocr_engine, "predict"):
        result = ocr_engine.predict(image_array)
    else:
        raise RuntimeError("当前 PaddleOCR 引擎没有可用的 ocr 或 predict 方法")

    return _parse_ocr_result(result)


def extract_text_from_image_with_vision_model(image: Any) -> str:
    """Use the configured multimodal model as OCR fallback when local PaddleOCR is unavailable."""
    if not os.getenv("QWEN_API_KEY", "").strip():
        raise RuntimeError("未配置 QWEN_API_KEY，无法使用多模态模型 OCR fallback")

    prompt = (
        "请只提取这张图片中的可见文字、公式和符号，尽量保持原始顺序和换行。"
        "不要解题，不要解释，不要补充图片中没有的内容。"
        "如果图片里没有可读文字，只输出：未识别到文字。"
    )
    result = call_qwen_model(
        prompt=prompt,
        images=[image],
        temperature=0.1,
        max_tokens=1024,
    ).strip()

    if result == "未识别到文字。":
        return ""
    return result


def extract_text_from_images(images: List[Any]) -> tuple[str, str]:
    """
    对多张图片逐张 OCR。

    返回：
    - ocr_text：成功识别出的文本，按图片编号合并。
    - ocr_error：失败或空结果的状态说明。
    """
    text_blocks: List[str] = []
    status_blocks: List[str] = []

    for index, image in enumerate(images, start=1):
        try:
            text = extract_text_from_image(image)
            backend = "PaddleOCR"
        except Exception as exc:
            try:
                text = extract_text_from_image_with_vision_model(image)
                backend = "多模态模型 OCR fallback"
                if text:
                    status_blocks.append(f"图片 {index}：PaddleOCR 失败，已改用多模态模型识别。原始错误：{exc}")
            except Exception as fallback_exc:
                status_blocks.append(
                    f"图片 {index}：OCR 识别失败。PaddleOCR 错误：{exc}；多模态 fallback 错误：{fallback_exc}"
                )
                continue

        if text:
            text_blocks.append(f"【图片 {index}】({backend})\n{text}")
        else:
            status_blocks.append(f"图片 {index}：OCR 未识别到文字。")

    return "\n\n".join(text_blocks), "\n".join(status_blocks)


def build_prompt(user_text: str, ocr_text: str, skill_context: str, image_count: int = 0) -> str:
    """
    构建最终发送给模型的 prompt。

    image_count > 0 时，模型会同时收到图片内容，因此 prompt 明确要求优先看图。
    """
    if image_count:
        image_instruction = f"用户上传了 {image_count} 张图片。请优先直接阅读图片内容；OCR 只作为辅助参考。"
    else:
        image_instruction = "用户没有上传图片，请仅根据文本和 OCR 文本回答。"

    return f"""你是 Adabot，一个面向考研与英语考试的多模态学习 Agent。

【输入说明】
{image_instruction}

【回答要求】
1. 使用中文为主回答；如果用户明确要求英文或英语写作，请给出英文内容并附中文解释。
2. 遇到数学题要分步骤推导，公式清晰，最后给出结论。
3. 遇到车辆工程题要解释概念、结构、原理、参数或工程意义。
4. 遇到雅思阅读/写作题要给出定位、思路、表达优化或范文结构。
5. 如果 OCR 文本和图片不一致，请以图片内容为准，并说明 OCR 可能有误。
6. 不编造题目缺失条件；条件不足时先说明，再给出可行解法。

【用户文本输入】
{user_text or "无"}

【OCR 辅助文本】
{ocr_text or "未启用或未识别到文本"}

【已调用 Skill 提供的辅助策略】
{skill_context or "无"}

请基于以上信息给出清晰、可执行、适合备考复习的回答。 /no_think
"""


def call_qwen_model(
    prompt: str,
    images: List[Any] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    调用硅基流动或其他 OpenAI 兼容模型服务。

    如果传入图片列表，则使用 OpenAI 兼容的多模态 content 格式，
    把文本和每张图片一起传给模型。
    """
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B").strip()

    if not api_key:
        return (
            "模型服务尚未配置。请在启动 python main.py 的同一个 PowerShell 中设置：\n\n"
            "$env:QWEN_API_KEY='你的硅基流动 API Key'\n"
            "$env:QWEN_BASE_URL='https://api.siliconflow.cn/v1'\n"
            "$env:QWEN_MODEL='硅基流动控制台里的多模态模型 ID'\n"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    image_list = images or []

    if not image_list:
        user_content: str | List[Dict[str, Any]] = prompt
    else:
        user_content = [{"type": "text", "text": prompt}]
        for image in image_list:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(image),
                    },
                }
            )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的考研与英语考试学习助手，回答要准确、分步骤、便于复习。",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )

        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content:
            return content

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            return f"模型只返回了思考内容，未返回最终答案。以下是服务端返回内容：\n\n{reasoning_content}"

        return f"模型调用成功，但返回内容为空。原始响应：\n{response.model_dump_json(indent=2, ensure_ascii=False)}"
    except Exception as exc:
        return f"调用模型失败：{exc}"

def call_qwen_model_stream(
    prompt: str,
    images: List[Any] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Stream model output through the OpenAI-compatible chat completions API."""
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B").strip()

    if not api_key:
        yield (
            "模型服务尚未配置。请在启动 python main.py 的同一个 PowerShell 中设置：\n\n"
            "$env:QWEN_API_KEY='你的硅基流动 API Key'\n"
            "$env:QWEN_BASE_URL='https://api.siliconflow.cn/v1'\n"
            "$env:QWEN_MODEL='硅基流动控制台里的多模态模型 ID'\n"
        )
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    image_list = images or []

    if not image_list:
        user_content: str | List[Dict[str, Any]] = prompt
    else:
        user_content = [{"type": "text", "text": prompt}]
        for image in image_list:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_data_url(image),
                    },
                }
            )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的考研与英语考试学习助手，回答要准确、分步骤、便于复习。",
                },
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
    except Exception as exc:
        yield f"调用模型失败：{exc}"


def write_log(record: Dict[str, Any]) -> Path:
    """将一次请求写入 JSONL 日志。"""
    ensure_project_dirs()

    record_with_time = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **record,
    }
    log_path = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record_with_time, ensure_ascii=False) + "\n")

    return log_path

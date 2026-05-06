"""对现有工作区文件做局部字符串替换的工具（相对 WorkDir）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal.schema.message import ToolDefinition
from internal.tools.read_file import _ResolvedPathStrictlyUnderRoot


def FuzzyReplace(original_content: str, old_text: str, new_text: str) -> str:
    """四级容错降级替换：L1 精确 → L2 换行归一化 → L3 Trim → L4 逐行去缩进。"""
    # L1: 精确匹配
    count = original_content.count(old_text)
    if count == 1:
        return original_content.replace(old_text, new_text, 1)
    if count > 1:
        raise RuntimeError(
            f"old_text 匹配到了 {count} 处，请提供更多的上下文代码以确保唯一性"
        )

    # L2: 换行符归一化（统一将 \\r\\n 转换为 \\n）
    normalized_content = original_content.replace("\r\n", "\n")
    normalized_old = old_text.replace("\r\n", "\n")

    count = normalized_content.count(normalized_old)
    if count == 1:
        return normalized_content.replace(normalized_old, new_text, 1)

    # L3: Trim Space 匹配（忽略首尾的空白与空行）
    trimmed_old = normalized_old.strip()
    if trimmed_old != "":
        count = normalized_content.count(trimmed_old)
        if count == 1:
            return normalized_content.replace(trimmed_old, new_text, 1)

    # L4: 逐行去缩进匹配
    return LineByLineReplace(normalized_content, normalized_old, new_text)


def LineByLineReplace(content: str, old_text: str, new_text: str) -> str:
    """按行切割，对每行去空白后做滑动窗口匹配；唯一命中则替换为 new_text（整段插入）。"""
    content_lines = content.split("\n")
    old_lines = old_text.strip().split("\n")

    if len(old_lines) == 0 or len(content_lines) < len(old_lines):
        raise RuntimeError("找不到该代码片段")

    for i in range(len(old_lines)):
        old_lines[i] = old_lines[i].strip()

    match_count = 0
    match_start_index = -1
    match_end_index = -1

    last_i = len(content_lines) - len(old_lines)
    for i in range(last_i + 1):
        is_match = True
        for j in range(len(old_lines)):
            if content_lines[i + j].strip() != old_lines[j]:
                is_match = False
                break
        if is_match:
            match_count += 1
            match_start_index = i
            match_end_index = i + len(old_lines)

    if match_count == 0:
        raise RuntimeError(
            "在文件中未找到 old_text，请大模型先调用 read_file 仔细确认文件内容和缩进"
        )
    if match_count > 1:
        raise RuntimeError(
            f"模糊匹配到了 {match_count} 处相似代码，请提供更多上下行代码以精确定位"
        )

    new_content_lines: list[str] = []
    new_content_lines.extend(content_lines[:match_start_index])
    new_content_lines.append(new_text)
    new_content_lines.extend(content_lines[match_end_index:])
    return "\n".join(new_content_lines)


class EditFileTool:
    """在限定目录下读取文件，经模糊替换算法将 ``old_text`` 换成 ``new_text`` 后写回。"""

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir

    def Name(self) -> str:
        return "edit_file"

    def Definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.Name(),
            description=(
                "对现有文件进行局部的字符串替换。这比重写整个文件更安全、更快速。"
                "请提供足够的 old_text 上下文以确保匹配的唯一性。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要修改的文件路径",
                    },
                    "old_text": {
                        "type": "string",
                        "description": (
                            "文件中原有的文本。必须包含足够的上下文（建议上下各多包含几行），"
                            "以确保在文件中的唯一性。"
                        ),
                    },
                    "new_text": {
                        "type": "string",
                        "description": "要替换成的新文本",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    def Execute(self, ctx: Any, raw_args: str) -> str:
        del ctx
        try:
            data = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"参数解析失败: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("参数解析失败: 根必须为 JSON 对象")

        path_raw = data.get("path")
        if not isinstance(path_raw, str) or path_raw.strip() == "":
            raise ValueError("参数解析失败: path 缺失或为空")

        if Path(path_raw).is_absolute():
            raise ValueError("path 必须为相对工作区的路径")

        old_text = data.get("old_text")
        if not isinstance(old_text, str):
            raise ValueError("参数解析失败: old_text 缺失或非字符串")
        if old_text == "":
            raise ValueError("参数解析失败: old_text 不能为空，否则无法唯一匹配")

        if "new_text" not in data:
            raise ValueError("参数解析失败: new_text 缺失")
        new_text = data.get("new_text")
        if not isinstance(new_text, str):
            raise ValueError("参数解析失败: new_text 必须为字符串")

        # 等价于 Go 的 filepath.Join(workDir, path)；此处仍解析并校验不越出工作区
        full_path = _ResolvedPathStrictlyUnderRoot(self._work_dir, path_raw)

        # 1. 读取原文件内容（与 os.ReadFile 等价：失败则将原因交给调用方 / 大模型）
        try:
            original = full_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"读取文件失败，请确认路径是否正确: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"读取文件失败，请确认路径是否正确: {exc}"
            ) from exc

        # 2. 多级模糊替换；具体匹配错误原样抛出，便于模型自纠
        new_content = FuzzyReplace(original, old_text, new_text)

        # 3. 写回磁盘（权限 0644）
        try:
            full_path.write_text(new_content, encoding="utf-8", newline="")
        except OSError as exc:
            raise RuntimeError(f"写回文件失败: {exc}") from exc

        try:
            full_path.chmod(0o644)
        except OSError:
            pass

        return f"✅ 成功修改文件: {path_raw}"


def NewEditFileTool(work_dir: str) -> EditFileTool:
    return EditFileTool(work_dir=work_dir)

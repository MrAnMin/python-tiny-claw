"""创建或覆盖写入工作区内文件的工具（相对 WorkDir）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal.schema.message import ToolDefinition
from internal.tools.read_file import _ResolvedPathStrictlyUnderRoot


class WriteFileTool:
    """在工作区内创建或覆盖文件；仅允许在注入的 ``work_dir`` 及其子目录下写入。"""

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir

    def Name(self) -> str:
        return "write_file"

    def Definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.Name(),
            description=(
                "创建或覆盖写入一个文件。如果目录不存在会自动创建。"
                "请提供相对于工作区的相对路径。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要写入的文件路径，如 src/main.go",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容",
                    },
                },
                "required": ["path", "content"],
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

        content_raw = data.get("content")
        if not isinstance(content_raw, str):
            raise ValueError("参数解析失败: content 缺失或非字符串")

        full_path = _ResolvedPathStrictlyUnderRoot(self._work_dir, path_raw)

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"创建父目录失败: {exc}") from exc

        try:
            full_path.write_text(content_raw, encoding="utf-8", newline="")
        except OSError as exc:
            raise RuntimeError(f"写入文件失败: {exc}") from exc

        try:
            full_path.chmod(0o644)
        except OSError:
            pass

        return f"成功将内容写入到文件: {path_raw}"


def NewWriteFileTool(work_dir: str) -> WriteFileTool:
    return WriteFileTool(work_dir=work_dir)

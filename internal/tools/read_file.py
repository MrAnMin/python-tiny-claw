"""读取本地工作区内文件内容的工具（相对 WorkDir）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from internal.schema.message import ToolDefinition

_MAX_LEN_BYTES = 8000


class ReadFileTool:
    """读取本地文件内容的工具；仅允许在注入的 ``work_dir`` 及其子目录下访问。"""

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir

    def Name(self) -> str:
        return "read_file"

    def Definition(self) -> ToolDefinition:
        """向大模型描述工具用途与参数格式（JSON Schema）。"""
        return ToolDefinition(
            name=self.Name(),
            description="读取指定路径的文件内容。请提供相对工作区的路径。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径，如 cmd/claw/main.go",
                    },
                },
                "required": ["path"],
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

        full_path = _ResolvedPathStrictlyUnderRoot(self._work_dir, path_raw)

        try:
            f = full_path.open("rb")
        except OSError as exc:
            raise RuntimeError(f"打开文件失败: {exc}") from exc

        try:
            content_bytes = f.read()
        except OSError as exc:
            raise RuntimeError(f"读取文件内容失败: {exc}") from exc
        finally:
            f.close()

        if len(content_bytes) > _MAX_LEN_BYTES:
            chunk = content_bytes[:_MAX_LEN_BYTES].decode("utf-8", errors="replace")
            return (
                f"{chunk}\n\n"
                f"...[由于内容过长，已被系统截断至前 {_MAX_LEN_BYTES} 字节]..."
            )

        return content_bytes.decode("utf-8", errors="replace")


def NewReadFileTool(work_dir: str) -> ReadFileTool:
    return ReadFileTool(work_dir=work_dir)


def _ResolvedPathStrictlyUnderRoot(work_dir: str, rel: str) -> Path:
    """拼接 ``work_dir`` 与相对路径，并校验解析后不越出工作区（防 ../../../ 等）。 """
    root = Path(work_dir).expanduser().resolve()
    # 禁止绝对路径已在外层判断；Join 后 resolve
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径越出工作区: {rel!r}") from exc
    return candidate

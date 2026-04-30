"""在当前工作区执行 bash 命令的工具。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from internal.schema.message import ToolDefinition

_TIMEOUT_SEC = 30
_MAX_OUTPUT_BYTES = 8000


class BashTool:
    """在绑定的工作区目录下执行 ``bash -c``，合并 stdout/stderr，带超时与输出截断。"""

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir

    def Name(self) -> str:
        return "bash"

    def Definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.Name(),
            description=(
                "在当前工作区执行任意的 bash 命令。支持链式命令(如 &&)。"
                "返回标准输出(stdout)和标准错误(stderr)。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 bash 命令，例如: ls -la 或 go test ./...",
                    },
                },
                "required": ["command"],
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

        command = data.get("command")
        if not isinstance(command, str) or command.strip() == "":
            raise ValueError("参数解析失败: command 缺失或为空")

        work_path = Path(self._work_dir).expanduser()
        try:
            cwd = str(work_path.resolve())
        except OSError as exc:
            raise RuntimeError(f"解析工作目录失败: {exc}") from exc

        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=_TIMEOUT_SEC,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            output_str = exc.output if isinstance(exc.output, str) else ""
            if output_str is None:
                output_str = ""
            return (
                output_str
                + "\n[警告: 命令执行超时(30s)，已被系统强制终止。"
                "如果是启动常驻服务，请尝试将其转入后台。]"
            )
        except OSError as exc:
            return f"执行报错: {exc}\n输出:\n"

        output_str = proc.stdout if proc.stdout is not None else ""

        if proc.returncode != 0:
            return f"执行报错: 退出状态 {proc.returncode}\n输出:\n{output_str}"

        if output_str == "":
            return "命令执行成功，无终端输出。"

        encoded = output_str.encode("utf-8", errors="replace")
        if len(encoded) > _MAX_OUTPUT_BYTES:
            truncated = encoded[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            return (
                f"{truncated}\n\n"
                f"...[终端输出过长，已截断至前 {_MAX_OUTPUT_BYTES} 字节]..."
            )

        return output_str


def NewBashTool(work_dir: str) -> BashTool:
    return BashTool(work_dir=work_dir)

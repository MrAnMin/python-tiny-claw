"""工具的注册与分发执行契约。"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from internal.schema.message import ToolCall, ToolDefinition, ToolResult


@runtime_checkable
class Registry(Protocol):
    """定义了工具的注册与分发执行接口。"""

    def GetAvailableTools(self) -> List[ToolDefinition]:
        """返回当前系统挂载的所有可用工具的 Schema。"""
        ...

    def Execute(self, ctx: Any, call: ToolCall) -> ToolResult:
        """实际执行模型请求的工具，并返回结果。

        ``ctx`` 与 Go 的 ``context.Context`` 对应；若无需要可传 ``None``。
        执行失败时可将错误信息写入 ``ToolResult``（例如 ``is_error=True``），
        或由实现抛出异常，视项目约定而定。
        """
        ...

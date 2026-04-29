"""工具的注册与分发执行契约与默认实现。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol, runtime_checkable

from internal.schema.message import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseTool(Protocol):
    """所有具体工具必须实现的通用接口（对应 Go 的 ``BaseTool``）。"""

    def Name(self) -> str:
        """全局唯一名称 (大模型通过这个名字调用)。"""
        ...

    def Definition(self) -> ToolDefinition:
        """用于提交给大模型的工具元信息和参数 JSON Schema。"""
        ...

    def Execute(self, ctx: Any, raw_args: str) -> str:
        """接收大模型吐出的 JSON 参数字符串，执行具体业务逻辑。

        ``raw_args`` 对应 Go 的 ``json.RawMessage``（此处为 UTF-8 文本），反序列化由各个具体工具内部自行处理。

        对应 Go 的 ``(string, error)``: 成功时返回字符串；失败时抛出异常而非返回 ``error``。
        """
        ...


@runtime_checkable
class Registry(Protocol):
    """定义工具的注册与分发接口。"""

    def Register(self, tool: BaseTool) -> None:
        """挂载一个新的工具到系统中。"""
        ...

    def GetAvailableTools(self) -> List[ToolDefinition]:
        """返回当前系统挂载的所有工具的 Schema，供 Main Loop 交给 Provider。"""
        ...

    def Execute(self, ctx: Any, call: ToolCall) -> ToolResult:
        """路由并执行模型请求的工具调用。

        ``ctx`` 与 Go 的 ``context.Context`` 对应；若无需要可传 ``None``。
        """
        ...


class RegistryImpl:
    """Registry 的默认实现；以工具的 Name 为 Key 做 O(1) 路由查找。"""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def Register(self, tool: BaseTool) -> None:
        name = tool.Name()
        if name in self._tools:
            logger.warning("工具 '%s' 已经被注册，将被覆盖。", name)
        self._tools[name] = tool
        logger.info("[Registry] 成功挂载工具: %s", name)

    def GetAvailableTools(self) -> List[ToolDefinition]:
        return [t.Definition() for t in self._tools.values()]

    def Execute(self, ctx: Any, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            err_msg = f"Error: 系统中不存在名为 '{call.name}' 的工具。"
            return ToolResult(
                tool_call_id=call.id,
                output=err_msg,
                is_error=True,
            )

        try:
            output = tool.Execute(ctx, call.arguments)
        except Exception as exc:
            err_msg = f"Error executing {call.name}: {exc}"
            return ToolResult(
                tool_call_id=call.id,
                output=err_msg,
                is_error=True,
            )

        return ToolResult(
            tool_call_id=call.id,
            output=output,
            is_error=False,
        )


def NewRegistry() -> RegistryImpl:
    """构造函数，等价于 Go 的 ``NewRegistry()``。"""
    return RegistryImpl()

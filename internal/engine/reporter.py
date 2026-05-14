"""引擎向外界输出信息的契约：可对接 CLI、飞书、钉钉、WebUI 等不同展现层。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Agent 引擎对外通知事件的接口（对应 Go 的 ``Reporter``）。

    实现方可将各类事件转发到终端、IM、Web 等通道，而引擎核心不依赖具体展现技术。
    """

    def OnThinking(self, ctx: Any) -> None:
        """模型进入慢思考（Reasoning）阶段时调用。"""
        ...

    def OnToolCall(self, ctx: Any, tool_name: str, args: str) -> None:
        """模型决定发起（可并行的）工具调用时调用。"""
        ...

    def OnToolResult(
        self,
        ctx: Any,
        tool_name: str,
        result: str,
        is_error: bool,
    ) -> None:
        """某次工具在底层执行结束并产生输出时调用。"""
        ...

    def OnMessage(self, ctx: Any, content: str) -> None:
        """模型输出面向用户的最终纯文本回答（任务阶段性完成）时调用。"""
        ...

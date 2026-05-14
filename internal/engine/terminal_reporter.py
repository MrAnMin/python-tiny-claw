"""终端上的 ``Reporter``：把引擎事件打印到 stdout（对应 Go 版 ``TerminalReporter``）。"""

from __future__ import annotations

from typing import Any

_ARGS_PREVIEW_MAX_LEN = 150


class TerminalReporter:
    """在终端直观展示 Agent 思考、工具调用与回复。"""

    def OnThinking(self, ctx: Any) -> None:
        del ctx
        print("\n[🤔 思考中] 模型正在推理...")

    def OnToolCall(self, ctx: Any, tool_name: str, args: str) -> None:
        del ctx
        print(f"[🛠️ 调用工具] {tool_name}")
        display_args = args.replace("\n", "\\n").replace("\r", "\\r")
        if len(display_args) > _ARGS_PREVIEW_MAX_LEN:
            display_args = display_args[:_ARGS_PREVIEW_MAX_LEN] + "... (已截断)"
        print(f"   参数: {display_args}")

    def OnToolResult(
        self,
        ctx: Any,
        tool_name: str,
        result: str,
        is_error: bool,
    ) -> None:
        del ctx
        if is_error:
            print(f"[❌ 执行失败] {tool_name}")
            if result != "":
                print(f"   错误: {result}")
        else:
            print(f"[✅ 执行成功] {tool_name}")

    def OnMessage(self, ctx: Any, content: str) -> None:
        del ctx
        if content == "":
            return
        print(f"\n🤖 Agent 回复:\n{content}\n")


def NewTerminalReporter() -> TerminalReporter:
    return TerminalReporter()

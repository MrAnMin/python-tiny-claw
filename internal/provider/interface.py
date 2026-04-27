"""与大模型通信的 Provider 契约。"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable

from internal.schema.message import Message, ToolDefinition

import logging

logger = logging.getLogger(__name__)

@runtime_checkable
# Protocol 是 Python 3.8+ 引入的结构化类型，用于定义接口（类型约束），类似于 Go 的 interface。
class LLMProvider(Protocol):
    """定义了与大模型通信的统一契约。"""

    def Generate(
        self,
        ctx: Any,  # 上下文对象，用于支持取消、超时等控制。若无需要可传 None。
        messages: List[Message],  # 当前对话历史（上下文），按照顺序排列。
        # 当前系统可用的工具列表，供模型决策调用；传 None 表示剥夺工具（对应 Go 的 nil）。
        available_tools: Optional[List[ToolDefinition]],
    ) -> Message:
        """接收当前的上下文历史、可用工具列表，并发起一次大模型推理。

        对应 Go 的 ``(*schema.Message, error)``：成功时返回 ``Message``；
        失败时由实现抛出异常，而不是返回 error。
        ``ctx`` 与 Go 的 ``context.Context`` 对应，用于取消/超时等；若无需要可传 ``None``。
        """
        logger.info("[Provider] 接收当前的上下文历史、可用工具列表，并发起一次大模型推理。")
        print("[Provider] 接收当前的上下文历史、可用工具列表，并发起一次大模型推理。")

        
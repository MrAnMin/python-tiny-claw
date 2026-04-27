"""消息与工具相关的 schema（与大模型沟通的基石）。"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """消息的角色。"""
    System = "system"  # 系统提示词：确立 Agent 的性格与红线
    User = "user"  # 用户输入 / 工具执行的返回结果 (Observation)
    Assistant = "assistant"  # 模型的输出：包含推理(Reasoning)或工具调用(ToolCall)


class ToolCall(BaseModel):
    """模型请求调用某个具体的工具。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="工具调用的唯一 ID")
    name: str = Field(..., description='想要调用的工具名称 (例如 "bash")')
    # 与 Go 的 json.RawMessage 类似：延迟解析，由具体工具负责 json.loads
    arguments: str = Field(..., description="JSON 参数字符串")


class Message(BaseModel):
    """
    上下文中传递的单条消息。
    Pydantic 的规则是：没有默认值的字段就是必填
    """
    
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(..., description="存放纯文本内容")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="如果模型决定调用工具，此字段将被填充 (支持并行调用多个工具)",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="如果这是对某个工具调用的响应，此字段必须填写，以告知模型上下文的关联性",
    )


class ToolResult(BaseModel):
    """工具在本地执行完毕后返回的物理结果。"""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    output: str = Field(..., description="工具执行的控制台输出或报错堆栈")
    is_error: bool = Field(..., description="标记是否失败，供后续的驾驭工程进行错误自愈")


class ToolDefinition(BaseModel):
    """描述大模型可调用的工具元信息 (供模型理解工具有什么用)。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: Any = Field(..., description="对应 JSON Schema")


if __name__ == "__main__":
    t1 = Message(
        content="Hello, world!",
        tool_calls = [
            ToolCall(id="123", name="bash", arguments='{"command": "ls -la"}'),
            ToolCall(id="456", name="python", arguments='{"command": "print("Hello, world!")"}'),
        ],
        tool_call_id="123"
    )
    print(t1)
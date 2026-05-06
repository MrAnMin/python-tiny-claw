"""消息与工具相关的 schema（与大模型沟通的基石）。"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Role 是一个基于 Python 枚举（Enum）实现的类，用来规范化消息在多轮对话中的“角色”类型，通常用于 Agent 框架（如 RPA/大语言模型等场景）里标明每条消息的身份归属。
# 
# 用法说明：
# - Role.System: 表示“系统消息”，用于描述 AI 角色性格、设定边界等场景；
# - Role.User: 表示“用户消息”，既可以是用户输入，也可以是工具执行的观测结果反馈给模型；
# - Role.Assistant: 表示“助手（模型）消息”，即模型推理、回复用户，或提出工具调用的行为。
#
# 枚举（Enum）是 Python 构建有穷常量集的标准手段。其好处包括：
#   1. 类型安全，只有预先定义的取值可用，防止拼写错误；
#   2. 可读性强，含义一目了然；
#   3. 通过 `str` 派生，可以直接作为字符串参与序列化与入参传递。
#
# 例子：
#   msg.role == Role.User
#   msg.role.value   # 得到 "user"
#   print(Role.System)  # 输出 Role.System
#   print(Role.System.value)  # 输出 "system"

class Role(str, Enum):
    """定义消息在对话流程中的角色类型（用于类型安全与规范化）。"""
    System = "system"     # 系统提示词：确立 Agent 的性格与红线
    User = "user"         # 用户输入 / 工具执行的返回结果 (Observation)
    Assistant = "assistant"  # 模型的输出：包含推理(Reasoning)或工具调用(ToolCall)


class ToolCall(BaseModel):
    """模型请求调用某个具体的工具。"""

    # model_config 是 Pydantic v2 推荐用于数据模型配置的方式，类似于 v1 的 class Config。
    # 这里通过 ConfigDict(extra="forbid") 指定模型只允许定义的字段，反序列化时如果有多余字段会报错，
    # 能提升类型安全与健壮性。
    model_config = ConfigDict(extra="forbid")

    # 这里初始化了 ToolCall 的 id 字段，表示本次工具调用的唯一标识（例如 uuid4 生成）。
    # 它通过 Pydantic 的 Field 声明，不可为空（...），并带有详细的中文描述便于自动文档与 IDE 提示。
    id: str = Field(
        ...,
        description="工具调用的唯一 ID"
    )  # 工具调用在一次对话中的唯一标识，用于后续结果关联

    name: str = Field(
        ...,
        description='想要调用的工具名称 (例如 "bash")'
    )  # 调用的工具名称，需和已注册工具名完全一致

    arguments: str = Field(
        ...,
        description="JSON 参数字符串"
    )  # 工具调用的参数，以 JSON 字符串形式传递（具体结构由目标工具自行解析）；类似 Go 的 json.RawMessage，推迟到工具层解码


class Message(BaseModel):
    """
    Message 是用于多轮对话中传递单条消息的核心数据结构。

    本类作为基础通信单元，在人机对话流程 (如 AI Assistant, RPA, 工具链 Agent) 体系中用于
    统一本地消息与大模型消息格式，支持内容、工具调用、与工具响应的编号关联等。

    字段说明：
    - role: 指定消息在对话中的角色身份（如 system/user/assistant），依托枚举 Role 保证类型安全和可读性。
    - content: 存放该消息的纯文本内容。
    - tool_calls: 若模型需调用工具，该字段存储希望并行调用的工具及其参数（如 API 调用请求），否则为 None。
    - tool_call_id: 若该消息是针对某次工具调用的响应（如工具执行结果），此字段存储工具调用时分配的唯一 ID，用于上下文关联，若非工具响应则为 None。

    Pydantic 的规则补充说明：
    - 没有默认值的字段为必填（如 role, content）。
    - Optional/带 default=None 的字段可以缺省（如 tool_calls, tool_call_id）。
    - model_config.extra="forbid" 可确保反序列化时禁止多余/未定义字段输入，提高健壮性与类型安全。

    用法示例:
        Message(
            role=Role.Assistant,
            content="以下是根据你的请求执行的结果：",
            tool_calls=[
                ToolCall(id="123", name="read_file", arguments='{"path": "README.md"}')
            ],
            tool_call_id=None
        )
    """
    model_config = ConfigDict(extra="forbid")

    role: Role  # 消息角色（system/user/assistant）
    content: str = Field(
        ..., 
        description="存放纯文本内容"
    )
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
        role=Role.Assistant,
        content="Hello, world!",
        tool_calls = [
            ToolCall(id="123", name="bash", arguments='{"command": "ls -la"}'),
            ToolCall(id="456", name="python", arguments='{"command": "print("Hello, world!")"}'),
        ],
        tool_call_id="123"
    )
    print(t1.model_dump_json())
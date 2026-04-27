"""基于 OpenAI 兼容 API 的 Provider（含智谱 BigModel 兼容端点）。"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from openai import OpenAI

from internal.schema.message import Message, Role, ToolCall, ToolDefinition


def NewZhipuOpenAIProvider(model: str) -> OpenAIProvider:
    """构造函数：基于官方 OpenAI Python SDK，指向智谱兼容端点。"""
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        raise RuntimeError("请设置 ZHIPU_API_KEY 环境变量")

    base_url = "https://open.bigmodel.cn/api/paas/v4/"
    client = OpenAI(api_key=api_key, base_url=base_url)
    return OpenAIProvider(client=client, model=model)


class OpenAIProvider:
    """使用 OpenAI Chat Completions 协议与后端通信。"""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def Generate(
        self,
        ctx: Any,
        messages: List[Message],
        available_tools: Optional[List[ToolDefinition]],
    ) -> Message:
        del ctx  # 与 Go context.Context 对应；同步 SDK 暂未统一接入，可后续扩展

        openai_msgs = [_InternalMessageToOpenAIParam(m) for m in messages]

        openai_tools: List[dict[str, Any]] = []
        if available_tools:
            openai_tools = _ToolDefinitionsToOpenAIParams(available_tools)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_msgs,
        }
        # 【慢思考机制支撑】仅当存在工具定义时才挂载 Tools
        if openai_tools:
            kwargs["tools"] = openai_tools

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"OpenAI/Zhipu API 请求失败: {exc}") from exc

        if not resp.choices:
            raise RuntimeError("API 返回了空的 Choices")

        choice = resp.choices[0].message
        return _OpenAIChoiceToMessage(choice)


def _InternalMessageToOpenAIParam(msg: Message) -> dict[str, Any]:
    """将内部 Message 转为 Chat Completions 的 message 字典。"""
    if msg.role == Role.System:
        return {"role": "system", "content": msg.content}

    if msg.role == Role.User:
        if msg.tool_call_id:
            # 工具观察结果：OpenAI 协议使用 role=tool
            return {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            }
        return {"role": "user", "content": msg.content}

    if msg.role == Role.Assistant:
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
        tool_calls = msg.tool_calls or []
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return out

    raise ValueError(f"unsupported role: {msg.role!r}")


def _NormalizeJsonSchemaObject(schema: Any) -> dict[str, Any]:
    """与 Go 一致：优先 map，否则 JSON 往返以保证类型为 dict。"""
    if isinstance(schema, dict):
        return schema
    return json.loads(json.dumps(schema))


def _ToolDefinitionsToOpenAIParams(
    tool_defs: List[ToolDefinition],
) -> List[dict[str, Any]]:
    tools: List[dict[str, Any]] = []
    for td in tool_defs:
        params = _NormalizeJsonSchemaObject(td.input_schema)
        desc = td.description or None
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": desc,
                    "parameters": params,
                },
            }
        )
    return tools


def _OpenAIChoiceToMessage(choice: Any) -> Message:
    """将 API 返回的 message 对象转为内部 schema.Message。"""
    content = choice.content or ""
    tool_calls_out: List[ToolCall] = []

    raw_tcs = getattr(choice, "tool_calls", None) or []
    for tc in raw_tcs:
        if getattr(tc, "type", None) != "function":
            continue
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        fn_name = getattr(fn, "name", None)
        fn_args = getattr(fn, "arguments", None)
        tc_id = getattr(tc, "id", None)
        if not tc_id or not fn_name:
            continue
        if isinstance(fn_args, str):
            args_str = fn_args
        elif fn_args is None:
            args_str = "{}"
        else:
            args_str = json.dumps(fn_args)
        tool_calls_out.append(
            ToolCall(id=tc_id, name=fn_name, arguments=args_str),
        )

    return Message(
        role=Role.Assistant,
        content=content,
        tool_calls=tool_calls_out if tool_calls_out else None,
    )

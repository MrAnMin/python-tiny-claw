"""基于 Anthropic Messages API 的 Provider（含智谱 BigModel 兼容端点）。"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from anthropic import Anthropic

from internal.schema.message import Message, Role, ToolCall, ToolDefinition


def NewZhipuClaudeProvider(model: str) -> ClaudeProvider:
    """构造函数：基于 Anthropic Python SDK，指向智谱兼容端点。"""
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        raise RuntimeError("请设置 ZHIPU_API_KEY 环境变量")

    base_url = "https://open.bigmodel.cn/api/paas/v4/"
    client = Anthropic(api_key=api_key, base_url=base_url)
    return ClaudeProvider(client=client, model=model)


class ClaudeProvider:
    """使用 Anthropic Messages 协议与后端通信。"""

    def __init__(self, client: Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def Generate(
        self,
        ctx: Any,
        messages: List[Message],
        available_tools: Optional[List[ToolDefinition]],
    ) -> Message:
        del ctx  # 与 Go context.Context 对应；同步 SDK 可后续接入超时/取消

        system_prompt = ""
        anthropic_msgs: List[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.System:
                system_prompt = msg.content
            elif msg.role == Role.User:
                if msg.tool_call_id:
                    anthropic_msgs.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.tool_call_id,
                                    "content": msg.content,
                                    "is_error": False,
                                }
                            ],
                        }
                    )
                else:
                    anthropic_msgs.append(
                        {
                            "role": "user",
                            "content": msg.content,
                        }
                    )
            elif msg.role == Role.Assistant:
                blocks: List[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls or []:
                    try:
                        input_map = json.loads(tc.arguments) if tc.arguments else {}
                        if not isinstance(input_map, dict):
                            input_map = {}
                    except json.JSONDecodeError:
                        input_map = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": input_map,
                        }
                    )
                if blocks:
                    anthropic_msgs.append({"role": "assistant", "content": blocks})

        anthropic_tools: List[dict[str, Any]] = []
        if available_tools:
            for td in available_tools:
                anthropic_tools.append(_ToolDefinitionToAnthropicTool(td))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": anthropic_msgs,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Claude/Zhipu API 请求失败: {exc}") from exc

        return _AnthropicResponseToMessage(resp)


def _ToolDefinitionToAnthropicTool(td: ToolDefinition) -> dict[str, Any]:
    """与 Go 一致：从 InputSchema 中提取 properties / required，填入 ToolInputSchema。"""
    raw = td.input_schema
    if not isinstance(raw, dict):
        raw = json.loads(json.dumps(raw))

    properties: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(raw.get("properties"), dict):
        properties = dict(raw["properties"])
    r = raw.get("required")
    if isinstance(r, list):
        required = [str(x) for x in r]

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return {
        "name": td.name,
        "description": td.description or "",
        "input_schema": input_schema,
    }


def _AnthropicResponseToMessage(resp: Any) -> Message:
    """将 Messages API 响应转为内部 schema.Message。"""
    content_parts: List[str] = []
    tool_calls_out: List[ToolCall] = []

    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            content_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tc_id = getattr(block, "id", None)
            name = getattr(block, "name", None)
            inp = getattr(block, "input", None)
            if not tc_id or not name:
                continue
            args_str = json.dumps(inp) if inp is not None else "{}"
            tool_calls_out.append(
                ToolCall(id=tc_id, name=name, arguments=args_str),
            )

    return Message(
        role=Role.Assistant,
        content="".join(content_parts),
        tool_calls=tool_calls_out if tool_calls_out else None,
    )

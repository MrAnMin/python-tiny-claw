"""飞书机器人封装：事件回调 + 将引擎输出推到会话（对应 Go 版 ``feishu`` 包）。"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional, Tuple

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
)
from lark_oapi.api.im.v1.model.p2_im_message_receive_v1 import (
    P2ImMessageReceiveV1,
)
from lark_oapi.api.im.v1.model.p2_im_message_message_read_v1 import (
    P2ImMessageMessageReadV1,
)

from internal.engine.loop import AgentEngine
from internal.engine.reporter import Reporter

logger = logging.getLogger(__name__)


def _ParseTextFromContent(content_str: str) -> str:
    """从飞书文本消息的 content JSON 中取出纯文本（优先 json.loads，兜底 Go 同款前缀剥离）。"""
    s = content_str.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            t = obj.get("text")
            if isinstance(t, str):
                return t
    except json.JSONDecodeError:
        pass
    prefix, suffix = '{"text":"', '"}'
    if s.startswith(prefix) and s.endswith(suffix):
        return s[len(prefix) : -len(suffix)]
    return s


def _ChatIdAndTextFromEvent(data: P2ImMessageReceiveV1) -> Tuple[Optional[str], str]:
    evt = getattr(data, "event", None)
    msg = getattr(evt, "message", None) if evt is not None else None
    if msg is None:
        return None, ""
    content_raw = getattr(msg, "content", None)
    chat_id_attr = getattr(msg, "chat_id", None)
    if content_raw is None or chat_id_attr is None:
        return None, ""
    content_str = content_raw if isinstance(content_raw, str) else str(content_raw)
    chat_id = chat_id_attr if isinstance(chat_id_attr, str) else str(chat_id_attr)
    return chat_id, _ParseTextFromContent(content_str)


class FeishuReporter:
    """把引擎事件格式化为飞书文本消息；遵守 ``Reporter`` 协议。"""

    def __init__(self, client: lark.Client, chat_id: str) -> None:
        self._client = client
        self._chat_id = chat_id

    def SendMsg(self, text: str) -> None:
        payload = json.dumps({"text": text}, ensure_ascii=False)
        request: CreateMessageRequest = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(self._chat_id)
                .msg_type("text")
                .content(payload)
                .build()
            )
            .build()
        )
        try:
            resp = self._client.im.v1.message.create(request)
        except Exception as exc:
            logger.warning("[Feishu] message.create 调用异常: %s", exc)
            return
        if not resp.success():
            logger.warning(
                "[Feishu] message.create 失败 code=%s msg=%s log_id=%s",
                resp.code,
                resp.msg,
                resp.get_log_id(),
            )

    def OnThinking(self, ctx: Any) -> None:
        del ctx
        self.SendMsg("🤔 模型正在慢思考 (Thinking)...")

    def OnToolCall(self, ctx: Any, tool_name: str, args: str) -> None:
        del ctx
        self.SendMsg(f"🛠️ **正在执行工具**：`{tool_name}`\n参数：`{args}`")

    def OnToolResult(
        self,
        ctx: Any,
        tool_name: str,
        result: str,
        is_error: bool,
    ) -> None:
        del ctx
        if is_error:
            self.SendMsg(f"⚠️ **执行报错** ({tool_name})：\n{result}")
        else:
            self.SendMsg(f"✅ **执行成功** ({tool_name})")

    def OnMessage(self, ctx: Any, content: str) -> None:
        del ctx
        self.SendMsg(content)


class FeishuBot:
    """飞书应用配置 + 持有 ``AgentEngine``，提供可挂到 HTTP 的事件调度器。"""

    def __init__(self, eng: AgentEngine) -> None:
        app_id = os.environ.get("FEISHU_APP_ID", "").strip()
        app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
        if not app_id or not app_secret:
            raise RuntimeError("请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")

        self._engine = eng
        self._app_id = app_id
        self._app_secret = app_secret
        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

    def GetClient(self) -> lark.Client:
        return self._client

    def GetEventDispatcher(self) -> lark.EventDispatcherHandler:
        """构建与 Go 版 ``GetEventDispatcher`` 等价的回调处理器，供 Flask/FastAPI POST 路由使用。"""
        encrypt_key = os.environ.get("FEISHU_ENCRYPT_KEY", "")
        verify_token = os.environ.get("FEISHU_VERIFY_TOKEN", "")

        def OnP2ImMessageReceiveV1(data: P2ImMessageReceiveV1) -> None:
            chat_id, text = _ChatIdAndTextFromEvent(data)
            if not chat_id:
                logger.warning("[Feishu] 无法解析 chat_id，丢弃事件")
                return
            logger.info("[Feishu] 收到会话 %s 消息: %s", chat_id, text)
            threading.Thread(
                target=self._HandleAgentRun,
                args=(chat_id, text),
                daemon=True,
                name=f"feishu-agent-{chat_id[:8]}",
            ).start()

        def OnP2ImMessageReadV1(_data: P2ImMessageMessageReadV1) -> None:
            return

        builder = (
            lark.EventDispatcherHandler.builder(
                encrypt_key,
                verify_token,
                lark.LogLevel.INFO,
            )
            .register_p2_im_message_receive_v1(OnP2ImMessageReceiveV1)
            .register_p2_im_message_message_read_v1(OnP2ImMessageReadV1)
        )
        handler = builder.build()
        return handler

    def _HandleAgentRun(self, chat_id: str, prompt: str) -> None:
        reporter = FeishuReporter(self._client, chat_id)
        try:
            self._engine.Run(None, prompt, reporter)
        except Exception as exc:
            reporter.SendMsg(f"❌ Agent 运行崩溃: {exc}")


def NewFeishuBot(eng: AgentEngine) -> FeishuBot:
    return FeishuBot(eng)

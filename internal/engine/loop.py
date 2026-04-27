"""Agent 引擎：微型 OS 的核心驱动。"""

from __future__ import annotations

import logging
from typing import Any, List

from internal.provider.interface import LLMProvider
from internal.schema.message import Message, Role
from internal.tools.registry import Registry

logger = logging.getLogger(__name__)


class AgentEngine:
    """微型 OS 的核心驱动。"""

    def __init__(
        self,
        provider: LLMProvider,
        registry: Registry,
        work_dir: str,
        enable_thinking: bool = False,
    ) -> None:
        # 初始化 AgentEngine 实例，注入大模型 provider、工具注册表和工作目录
        self._provider = provider  # LLMProvider 的实例，用于与大模型通信
        self._registry = registry  # Registry 的实例，管理所有可用工具
        # 工作区：借鉴 OpenClaw 的理念，Agent 必须有一个明确的物理边界
        self.work_dir = work_dir  # 指定 Agent 的物理工作区路径
        # 慢思考模式开关（对应 Go 的 EnableThinking）
        self.enable_thinking = enable_thinking

    def Run(self, ctx: Any, user_prompt: str) -> None:
        """启动 Agent 的生命周期。失败时抛出异常（对应 Go 的 error）。"""
        logger.info("[Engine] 引擎启动，锁定工作区: %s", self.work_dir)
        logger.info("[Engine] 慢思考模式 (Thinking Phase): %s", self.enable_thinking)

        # 1. 初始化会话的 Context (上下文内存)
        # 在真实的场景中，这里会由动态 Prompt 组装器加载 AGENTS.md。目前我们先硬编码。
        context_history: List[Message] = [
            Message(
                role=Role.System,
                content=(
                    "You are go-tiny-claw, an expert coding assistant. "
                    "You have full access to tools in the workspace."
                ),
            ),
            Message(role=Role.User, content=user_prompt),
        ]

        logger.info("[Engine] 初始化会话的 Context: %s", context_history)

        turn_count = 0

        # 2. The Main Loop: 心跳开始 (标准的 ReAct 循环)
        while True:
            turn_count += 1
            logger.info("========== [Turn %d] 开始 ==========", turn_count)

            # 获取当前系统已挂载的所有可用工具的定义（ToolDefinition 列表），供模型推理和决策时选择调用
            available_tools = self._registry.GetAvailableTools()
            logger.info("[Engine] 当前系统已挂载的所有可用工具的定义: %s", available_tools)

            # ================================================================
            # Phase 1: 慢思考阶段 (Thinking) - 剥夺工具，强制规划
            # ================================================================
            if self.enable_thinking:
                logger.info("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

                # 核心机制：传入的 available_tools 为 None！
                # 大模型看不到任何 JSON Schema，被迫只能输出纯文本的思考过程。
                try:
                    think_resp = self._provider.Generate(
                        ctx, context_history, None
                    )
                except Exception as exc:
                    raise RuntimeError("Thinking 阶段生成失败") from exc

                # 如果模型输出了思考过程，作为 Assistant 消息追加到上下文中
                if think_resp.content:
                    print(f"🧠 [内部思考 Trace]: {think_resp.content}")
                    context_history.append(think_resp)

            # ================================================================
            # Phase 2: 行动阶段 (Action) - 恢复工具，顺着规划执行
            # ================================================================
            logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

            # 此时的 context_history 中已经包含了上一阶段模型自己的 Thinking Trace。
            # 模型会顺着自己的逻辑，结合恢复的 available_tools 发起精准的工具调用。
            try:
                action_resp = self._provider.Generate(
                    ctx, context_history, available_tools
                )
                # logger.info("[Engine] 大模型生成回复消息: %s", action_resp)
            except Exception as exc:
                raise RuntimeError("Action 阶段生成失败") from exc

            context_history.append(action_resp)

            if action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            # ================================================================
            # 退出与执行逻辑
            # ================================================================
            tool_calls = action_resp.tool_calls or []

            # 如果模型没有请求任何工具调用，说明它认为任务已经完成，跳出循环。
            if len(tool_calls) == 0:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info("[Engine] 模型请求调用 %d 个工具...", len(tool_calls))

            for tool_call in tool_calls:
                logger.info(
                    "  -> 🛠️ 执行工具: %s, 参数: %s",
                    tool_call.name,
                    tool_call.arguments,
                )

                result = self._registry.Execute(ctx, tool_call)

                if result.is_error:
                    logger.info("  -> ❌ 工具执行报错: %s", result.output)
                else:
                    out_bytes = len(result.output.encode("utf-8"))
                    logger.info("  -> ✅ 工具执行成功 (返回 %d 字节)", out_bytes)

                # 将工具执行的观察结果 (Observation) 封装为 User Message 追加到上下文中
                # ToolCallID 必须携带，以维系大模型推理链条
                observation_msg = Message(
                    role=Role.User,
                    content=result.output,
                    tool_call_id=tool_call.id,
                )
                context_history.append(observation_msg)


def NewAgentEngine(
    provider: LLMProvider,
    registry: Registry,
    work_dir: str,
    enable_thinking: bool = False,
) -> AgentEngine:
    return AgentEngine(
        provider=provider,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=enable_thinking,
    )

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
        self._provider = provider
        self._registry = registry  # Registry 的实例，管理所有可用工具
        # 工作区：借鉴 OpenClaw 的理念，Agent 必须有一个明确的物理边界
        self.work_dir = work_dir
        # 慢思考模式开关
        self.enable_thinking = enable_thinking

    def Run(self, ctx: Any, user_prompt: str) -> None:
        """
        启动 Agent 的生命周期。失败时抛出异常（对应 Go 的 error）。
        Args:
            ctx: 运行上下文对象（可用于超时/取消，暂未用）
            user_prompt: 用户输入的提示词
        """
        logger.info("[Engine] 引擎启动，锁定工作区: %s", self.work_dir)
        logger.info("[Engine] 慢思考模式 (Thinking Phase): %s", self.enable_thinking)

        # 1. 初始化会话的 Context (上下文内存、历史消息列表)
        # 在真实的场景中，这里会由动态 Prompt 组装器加载 AGENTS.md。当前我们先硬编码一个系统提示词。
        context_history: List[Message] = [
            Message(
                role=Role.System,
                content=(
                    "You are python-tiny-claw, an expert coding assistant. "
                    "You have full access to tools in the workspace."
                ),
            ),
            Message(role=Role.User, content=user_prompt),
        ]

        logger.info("[Engine] 初始化会话的 Context: %s", context_history)

        turn_count = 0  # 轮数计数器

        # 2. Main Loop: Agent 的 ReAct 主循环
        while True:
            turn_count += 1
            logger.info("========== [Turn %d] 开始 ==========", turn_count)

            # logger.info("[Engine] 本次的消息为:")
            # for msg in context_history:
            #     logger.info("[Engine] 消息内容: %s", msg.model_dump_json())

            # 获取当前系统挂载的所有可用工具（供模型选择是否调用）
            available_tools = self._registry.GetAvailableTools()
            logger.info("[Engine] 当前系统已挂载的所有可用工具的定义: %s", available_tools)

            # ================================================================
            # Phase 1: 慢思考阶段 (Thinking) - 剥夺工具，强制规划
            # ================================================================
            if self.enable_thinking:
                logger.info("[Engine][Phase 1] 剥夺工具访问权，强制进入慢思考与规划阶段...")

                # 关键机制：将 available_tools 置为 None，大模型只能输出纯文本的思考过程
                try:
                    think_resp = self._provider.Generate(
                        ctx, context_history, None
                    )
                except Exception as exc:
                    # 慢思考（思维链生成）阶段失败，向上抛出异常
                    raise RuntimeError("Thinking 阶段生成失败") from exc

                # 如果模型在慢思考阶段有内容输出，则追加到对话上下文
                if think_resp.content:
                    print(f"🧠 [内部思考 Trace]: {think_resp.content}")
                    context_history.append(think_resp)

            # ================================================================
            # Phase 2: 行动阶段 (Action) - 恢复工具挂载，等待模型行动
            # ================================================================
            logger.info("[Engine][Phase 2] 恢复工具挂载，等待模型采取行动...")

            # 上下文中已包含了模型的思考输出，继续推理，允许模型调用工具
            try:
                action_resp = self._provider.Generate(
                    ctx, context_history, available_tools
                )
                logger.info("[Engine] 大模型生成回复消息: %s", action_resp)
            except Exception as exc:
                # 执行（行动）阶段失败，直接中止
                raise RuntimeError("Action 阶段生成失败") from exc

            # 将模型生成的 Action 回复追加到上下文
            context_history.append(action_resp)

            # 如果有纯文本内容输出，直接展示给用户
            if action_resp.content:
                print(f"🤖 [对外回复]: {action_resp.content}")

            # ================================================================
            # 工具调用与执行逻辑（Observation）
            # ================================================================
            tool_calls = action_resp.tool_calls or []

            # 若模型没有请求工具调用，表明任务已完成，跳出循环
            if len(tool_calls) == 0:
                logger.info("[Engine] 模型未请求调用工具，任务宣告完成。")
                break

            logger.info("[Engine] 模型请求调用 %d 个工具...", len(tool_calls))

            # 针对每个工具调用请求，依次执行并记录结果
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

                # 将工具执行的观察结果 (Observation) 封装为 User Message 并追加到上下文
                # tool_call_id 必须携带，以维系大模型推理链
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
    """
    构造一个新的 AgentEngine 实例。

    参数说明:
    - provider: 实现 LLMProvider 协议的对象，用于与大模型进行对话。
    - registry: 工具注册与分发中心。
    - work_dir: 当前工作区的物理目录路径（工作区安全边界）。
    - enable_thinking: 是否开启「慢思考」模式，默认为 False。

    返回:
    - 已初始化完成的 AgentEngine 实例
    """
    return AgentEngine(
        provider=provider,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=enable_thinking,
    )

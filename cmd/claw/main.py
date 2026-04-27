"""命令入口：真实智谱 Provider + 伪造 Registry，驱动引擎（含慢思考）。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List

# 便于直接 ``python cmd/claw/main.py`` 时解析 ``internal.*``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from internal.engine.loop import NewAgentEngine
from internal.provider.interface import LLMProvider
from internal.schema.message import ToolCall, ToolDefinition, ToolResult
from internal.tools.registry import Registry


# ==========================================
# 伪造的工具注册表 (用于测试 Provider 的工具提取能力)
# ==========================================
class MockRegistry:
    def GetAvailableTools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_weather",
                description="获取指定城市的当前天气情况。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ]

    def Execute(self, _ctx: Any, call: ToolCall) -> ToolResult:
        logging.info("  -> [Mock 工具执行] 获取 %s 的天气中...", call.name)
        return ToolResult(
            tool_call_id=call.id,
            output="API 返回：今天是晴天，气温 25 度。",
            is_error=False,
        )


def Main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.environ.get("ZHIPU_API_KEY", ""):
        logging.error("请先导出 ZHIPU_API_KEY 环境变量")
        sys.exit(1)

    # 在确认已配置密钥后再加载 OpenAI SDK，避免仅检查环境变量时依赖未安装包
    from internal.provider.openai import NewZhipuOpenAIProvider

    work_dir = os.getcwd()

    # 1. 初始化真实的 Provider（指向智谱 GLM-4.5）
    # 可改为 internal.provider.claude.NewZhipuClaudeProvider，效果视网关兼容性而定
    llm_provider: LLMProvider = NewZhipuOpenAIProvider("glm-4.5-air")

    registry: Registry = MockRegistry()

    eng = NewAgentEngine(
        provider=llm_provider,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=True,
    )

    prompt = "我想去北京跑步，帮我查查天气适合吗？"

    try:
        eng.Run(None, prompt)
    except Exception as exc:
        logging.error("引擎运行崩溃: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    Main()

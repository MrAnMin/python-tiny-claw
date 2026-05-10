"""命令入口：真实智谱 Provider + 真实 Registry（挂载 read_file）。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 便于直接 ``python cmd/claw/main.py`` 时解析 ``internal.*``
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from internal.engine.loop import NewAgentEngine


def Main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.environ.get("ZHIPU_API_KEY", ""):
        logging.error("请先导出 ZHIPU_API_KEY 环境变量")
        sys.exit(1)

    # 在确认已配置密钥后再加载 OpenAI SDK
    from internal.provider.interface import LLMProvider
    from internal.provider.openai import NewZhipuOpenAIProvider
    from internal.tools.read_file import NewReadFileTool
    from internal.tools.registry import NewRegistry
    from internal.tools.bash import NewBashTool
    from internal.tools.write_file import NewWriteFileTool
    from internal.tools.edit_file import NewEditFileTool

    # 1. 工作区物理边界
    work_dir = os.getcwd()

    # 2. 真实 Provider（智谱 GLM，OpenAI 适配器）
    llm_provider: LLMProvider = NewZhipuOpenAIProvider("glm-4.5-air")

    # 3. 真实 Registry
    registry = NewRegistry()

    # 4. 挂载 ReadFile 工具
    registry.Register(NewReadFileTool(work_dir))
    registry.Register(NewBashTool(work_dir))
    registry.Register(NewWriteFileTool(work_dir))
    registry.Register(NewEditFileTool(work_dir))

    # 5. 核心引擎（任务简单，关闭慢思考以加快速度）
    eng = NewAgentEngine(
        provider=llm_provider,
        registry=registry,
        work_dir=work_dir,
        enable_thinking=True,
    )

    prompt = (
        "我当前目录下有 a.txt, b.txt, c.txt 三个文件。"
        "为了节省时间，请你同时一次性读取这三个文件，并将它们的内容综合起来，告诉我它们分别记录了什么领域的信息。"
        "注意，如果发现有工具可以实现，请优先使用工具实现，而不是自己编写代码实现"
    )
    try:
        eng.Run(None, prompt)
    except Exception as exc:
        logging.error("引擎运行崩溃: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    Main()

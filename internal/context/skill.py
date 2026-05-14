"""从工作区 ``.claw/skills`` 加载 SKILL.md（含可选 YAML Frontmatter）并拼成可注入上下文的文本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """从 ``SKILL.md`` 解析出的技能结构（对应 Go 的 ``Skill``）。"""

    name: str
    description: str
    body: str


class SkillLoader:
    """扫描 ``{work_dir}/.claw/skills``，解析其中的 ``SKILL.md``（对应 Go 的 ``SkillLoader``）。"""

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir

    def LoadAll(self) -> str:
        """遍历技能目录并返回一段 Markdown；无目录、失败或总长过短时返回空串。"""
        skill_base_dir = Path(self._work_dir).expanduser().resolve() / ".claw" / "skills"
        if not skill_base_dir.is_dir():
            return ""

        chunks: list[str] = []
        chunks.append("\n### 可用专业技能 (Agent Skills)\n")
        chunks.append(
            "以下是你拥有的标准化外挂技能，请在符合 description 描述的场景下严格遵循其正文指令：\n\n"
        )

        try:
            # 等价于 filepath.WalkDir：仅处理名为 SKILL.md 的文件（大小写敏感，与 Go 一致）
            for path in sorted(skill_base_dir.rglob("*")):
                if not path.is_file() or path.name != "SKILL.md":
                    continue
                try:
                    raw = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                skill = ParseSkillMd(raw)
                chunks.append(f"#### 技能名称: {skill.name}\n")
                chunks.append(f"**触发条件**: {skill.description}\n\n")
                chunks.append("**执行指南**:\n")
                chunks.append(skill.body)
                chunks.append("\n\n---\n")
        except OSError:
            return ""

        out = "".join(chunks)
        if len(out) < 100:
            return ""
        return out


def NewSkillLoader(work_dir: str) -> SkillLoader:
    return SkillLoader(work_dir=work_dir)


def ParseSkillMd(content: str) -> Skill:
    """极简解析带 ``---`` YAML Frontmatter 的 Markdown（对应 Go 的 ``parseSkillMD``）。"""
    skill = Skill(
        name="Unknown Skill",
        description="No description provided.",
        body=content,
    )
    if not (content.startswith("---\n") or content.startswith("---\r\n")):
        return skill

    # 与 strings.SplitN(content, "---", 3) 一致：最多两段分隔符 → 三部分
    parts = content.split("---", 2)
    if len(parts) != 3:
        return skill

    frontmatter = parts[1]
    body_text = parts[2].strip()

    name = skill.name
    description = skill.description
    for line in frontmatter.replace("\r\n", "\n").split("\n"):
        line_trim = line.strip()
        if line_trim.startswith("name:"):
            name = line_trim[len("name:") :].strip()
        elif line_trim.startswith("description:"):
            description = line_trim[len("description:") :].strip()

    return Skill(name=name, description=description, body=body_text)

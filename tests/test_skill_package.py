from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "kaggle-solution-retriever"


class SkillPackageTests(unittest.TestCase):
    def test_required_metadata_and_explicit_invocation_policy(self):
        skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
        self.assertRegex(skill_text, r"\A---\n")
        self.assertRegex(skill_text, r"(?m)^name: kaggle-solution-retriever$")
        self.assertRegex(skill_text, r"(?m)^description: \S")

        metadata = (SKILL_DIR / "agents" / "openai.yaml").read_text(encoding="utf-8")
        self.assertRegex(
            metadata,
            re.compile(r"(?ms)^policy:\s*\n\s+allow_implicit_invocation:\s*false\s*$"),
        )

    def test_runtime_files_and_progressive_disclosure_layout(self):
        self.assertFalse((SKILL_DIR / "README.md").exists())
        for relative in (
            "scripts/build_knowledge_base.py",
            "scripts/catalog.py",
            "scripts/knowledge_base.py",
            "scripts/prepare_research_manifest.py",
            "references/catalog-schema.md",
            "references/operations.md",
            "references/research-workflow.md",
            "references/task-profile-schema.md",
        ):
            self.assertTrue((SKILL_DIR / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()

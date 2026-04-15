from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_kuchiki_touko_character_assets_exist_and_are_consistent() -> None:
    slug = "kuchiki-touko"
    character_dir = REPO_ROOT / "characters" / slug

    assert character_dir.exists(), f"Missing character directory: {character_dir}"

    meta_path = character_dir / "meta.json"
    story_path = character_dir / "story.md"
    persona_path = character_dir / "persona.md"
    skill_path = character_dir / "SKILL.md"

    for path in (meta_path, story_path, persona_path, skill_path):
        assert path.exists(), f"Missing required file: {path}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    story = story_path.read_text(encoding="utf-8")
    persona = persona_path.read_text(encoding="utf-8")
    skill = skill_path.read_text(encoding="utf-8")

    assert meta["slug"] == slug
    assert meta["name"] == "朽木冬子"
    assert meta["profile"]["source_work"]
    assert meta["knowledge_sources"], "knowledge_sources should not be empty"

    assert "# 朽木冬子" in skill
    assert "## PART A：角色设定" in skill
    assert "## PART B：人物性格" in skill
    assert meta["profile"]["source_work"] in skill
    assert story.strip() in skill
    assert persona.strip() in skill

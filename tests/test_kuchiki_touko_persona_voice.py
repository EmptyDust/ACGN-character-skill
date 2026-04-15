from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_kuchiki_touko_persona_and_skill_include_extracted_voice_markers() -> None:
    persona = (REPO_ROOT / "characters" / "kuchiki-touko" / "persona.md").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "characters" / "kuchiki-touko" / "SKILL.md").read_text(encoding="utf-8")

    required_markers = [
        "侦探先生",
        "时坂先生",
        "叫我冬子就好",
        "真正的我",
        "轻俏",
        "猫",
    ]

    for marker in required_markers:
        assert marker in persona
        assert marker in skill

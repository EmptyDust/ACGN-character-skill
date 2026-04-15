from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_has_uv_packaging_metadata_and_docs() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    readme_path = REPO_ROOT / "README.md"

    assert pyproject_path.exists(), "Missing pyproject.toml for uv-managed environment"

    readme = readme_path.read_text(encoding="utf-8")
    assert "uv venv" in readme
    assert "source .venv/bin/activate" in readme
    assert "uv pip install -r requirements.txt" in readme

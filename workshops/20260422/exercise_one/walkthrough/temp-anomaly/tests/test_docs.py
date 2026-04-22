from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_exists():
    assert (ROOT / "README.md").is_file()


def test_agents_exists():
    assert (ROOT / "AGENTS.md").is_file()

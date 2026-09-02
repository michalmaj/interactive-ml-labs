"""Repository hygiene checks for accidentally tracked local artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

TRACKED_ARTIFACT_PATTERNS = (
    ".DS_Store",
    ".coverage",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".venv/",
    "__pycache__/",
    "coverage.xml",
)


TRACKED_ARTIFACT_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
    ".tmp",
    ".bak",
    "~",
)


TRACKED_ARTIFACT_FILENAMES = {"git", "switch"}

REQUIRED_ISSUE_TEMPLATES = (
    "bug_report.yml",
    "confusing_lesson.yml",
    "missing_topic.yml",
    "classroom_feedback.yml",
    "config.yml",
)


def test_no_local_artifacts_are_tracked() -> None:
    """Tracked files should not include local caches or accidental shell artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked_files = result.stdout.splitlines()
    offenders = [path for path in tracked_files if _is_local_artifact(path)]

    assert offenders == []


def test_release_readiness_files_exist() -> None:
    """The student alpha release should include license and feedback entry points."""
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert (repo_root / "docs/github_repository_setup.md").exists()

    issue_template_dir = repo_root / ".github/ISSUE_TEMPLATE"
    for template_name in REQUIRED_ISSUE_TEMPLATES:
        assert (issue_template_dir / template_name).exists()


def test_root_readme_has_first_run_commands() -> None:
    """The public README should show new users how to install and run the app."""
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "## Requirements" in readme
    assert "Python 3.12 or newer" in readme
    assert "uv sync" in readme
    assert "uv run --package interactive-ml-labs-app interactive-ml-labs" in readme


def test_pygame_tests_have_headless_default() -> None:
    """Workspace tests should default to a headless Pygame video driver."""
    repo_root = Path(__file__).resolve().parents[1]
    conftest = (repo_root / "conftest.py").read_text(encoding="utf-8")
    ci = (repo_root / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'os.environ.setdefault("SDL_VIDEODRIVER", "dummy")' in conftest
    assert "SDL_VIDEODRIVER: dummy" in ci
    assert "timeout-minutes:" in ci


def test_release_versioning_is_documented() -> None:
    """The alpha release should have a changelog entry and versioning decision."""
    repo_root = Path(__file__).resolve().parents[1]
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    versioning = (repo_root / "docs/versioning.md").read_text(encoding="utf-8")
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "## v0.0.9a - Student-Facing Alpha" in changelog
    assert 'version = "0.1.0"' in versioning
    assert "v0.0.9a" in versioning
    assert "docs/versioning.md" in readme


def _is_local_artifact(path: str) -> bool:
    """Return whether a tracked path looks like a generated local artifact."""
    filename = Path(path).name
    return (
        filename in TRACKED_ARTIFACT_FILENAMES
        or any(pattern in path for pattern in TRACKED_ARTIFACT_PATTERNS)
        or path.endswith(TRACKED_ARTIFACT_SUFFIXES)
    )

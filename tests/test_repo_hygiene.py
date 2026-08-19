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


def _is_local_artifact(path: str) -> bool:
    """Return whether a tracked path looks like a generated local artifact."""
    filename = Path(path).name
    return (
        filename in TRACKED_ARTIFACT_FILENAMES
        or any(pattern in path for pattern in TRACKED_ARTIFACT_PATTERNS)
        or path.endswith(TRACKED_ARTIFACT_SUFFIXES)
    )

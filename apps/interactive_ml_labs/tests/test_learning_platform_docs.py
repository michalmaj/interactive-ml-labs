"""Documentation checks for the guided learning platform direction."""

from pathlib import Path

from interactive_ml_labs import LEARNING_PATH_MANIFESTS

REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNED_LEVEL_3_PATH_ID = "representation_to_model_behavior"
PLANNED_LEVEL_3_PATH_TITLE_EN = "From representation to model behavior"
PLANNED_LEVEL_3_PATH_TITLE_PL = "Od reprezentacji do zachowania modelu"
PLANNED_LEVEL_3_DEMOS = (
    "Explained Variance Lab",
    "t-SNE / UMAP Exploration Lab",
    "Calibration Lab",
    "Model Monitoring Drift Lab",
    "Time Series Forecasting Lab",
)


def test_level_three_guided_path_is_documented_after_registration() -> None:
    """The Level 3 path should be documented and registered in the app."""
    learning_platform = (REPO_ROOT / "docs/learning_platform.md").read_text()
    roadmap = (REPO_ROOT / "docs/roadmap.md").read_text()
    levels = " ".join((REPO_ROOT / "docs/levels.md").read_text().split())
    readme = (REPO_ROOT / "README.md").read_text()

    assert "## Fifth Learning Path" in learning_platform
    assert f"**{PLANNED_LEVEL_3_PATH_TITLE_EN}**" in learning_platform
    assert f"Polish title: **{PLANNED_LEVEL_3_PATH_TITLE_PL}**" in learning_platform
    assert "not a full MLOps course" in learning_platform
    assert "shared Calibration and Monitoring lessons" in roadmap
    assert PLANNED_LEVEL_3_PATH_TITLE_EN in readme
    assert PLANNED_LEVEL_3_PATH_TITLE_PL in levels

    for demo_title in PLANNED_LEVEL_3_DEMOS:
        assert demo_title in learning_platform

    registered_path_ids = {path.id for path in LEARNING_PATH_MANIFESTS}
    assert PLANNED_LEVEL_3_PATH_ID in registered_path_ids

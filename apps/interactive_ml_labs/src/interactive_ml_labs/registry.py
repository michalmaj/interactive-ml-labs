"""Demo manifest registry for the unified shell."""

from __future__ import annotations

from collections.abc import Callable

from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LevelManifest, LocalizedText
from interactive_ml_labs.placeholder_scene import PlaceholderDemoScene
from interactive_ml_labs.scene import Scene

LEVEL_MANIFESTS: tuple[LevelManifest, ...] = (
    LevelManifest(
        number=1,
        title=LocalizedText(en="Level 1 - Fundamentals", pl="Poziom 1 - Fundamenty"),
        summary=LocalizedText(
            en="Core machine learning intuition and foundational algorithms.",
            pl="Pierwsze intuicje ML i algorytmy, do których często się wraca.",
        ),
    ),
    LevelManifest(
        number=2,
        title=LocalizedText(en="Level 2 - Practical ML", pl="Poziom 2 - Praktyczne ML"),
        summary=LocalizedText(
            en="Model evaluation, robustness, ensembles, and practical trade-offs.",
            pl="Ewaluacja modeli, odporność, ensemble i kompromisy z praktyki.",
        ),
    ),
    LevelManifest(
        number=3,
        title=LocalizedText(en="Level 3 - Advanced / Showcase", pl="Poziom 3 - Zaawansowane"),
        summary=LocalizedText(
            en="Advanced, specialized, and visually rich machine learning demos.",
            pl="Bardziej zaawansowane tematy i efektowne wizualnie eksperymenty ML.",
        ),
    ),
)

LEVEL_BY_NUMBER: dict[int, LevelManifest] = {
    manifest.number: manifest for manifest in LEVEL_MANIFESTS
}
LEVEL_NAMES: dict[int, LocalizedText] = {
    number: manifest.title for number, manifest in LEVEL_BY_NUMBER.items()
}


def _placeholder_demo(
    *,
    demo_id: str,
    level: int,
    title_en: str,
    title_pl: str,
    summary_en: str,
    summary_pl: str,
    tags: tuple[str, ...],
    objectives: tuple[LocalizedText, ...] | None = None,
    controls: tuple[ControlBinding, ...] | None = None,
    create_scene: Callable[[object], Scene] | None = None,
    difficulty: LocalizedText | None = None,
) -> DemoManifest:
    """Build a placeholder manifest for a demo."""
    return DemoManifest(
        id=demo_id,
        level=level,
        title=LocalizedText(en=title_en, pl=title_pl),
        summary=LocalizedText(en=summary_en, pl=summary_pl),
        objectives=objectives
        or (
            LocalizedText(
                en="Explore the main intuition behind the algorithm.",
                pl="Zrozum, co jest najważniejsze w działaniu algorytmu.",
            ),
            LocalizedText(
                en="Observe how parameter changes affect model behavior.",
                pl="Sprawdź, jak parametry zmieniają zachowanie modelu.",
            ),
        ),
        controls=controls
        or (
            ControlBinding(
                key="Enter",
                action=LocalizedText(en="start placeholder scene", pl="uruchom ekran demo"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(en="open pause menu or go back", pl="otwórz pauzę albo wróć"),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="toggle help overlay", pl="pokaż lub ukryj pomoc"),
            ),
        ),
        create_scene=create_scene
        or (
            lambda context, demo_id=demo_id: PlaceholderDemoScene(
                context,
                DEMO_BY_ID[demo_id],
            )
        ),
        difficulty=difficulty or LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=tags,
    )


DEMO_MANIFESTS: tuple[DemoManifest, ...] = (
    _placeholder_demo(
        demo_id="gradient_descent_playground",
        level=1,
        title_en="Gradient Descent Playground",
        title_pl="Gradient Descent Playground",
        summary_en="Optimization, loss, and learning rate intuition.",
        summary_pl="Optymalizacja, loss i intuicja stojąca za learning rate.",
        objectives=(
            LocalizedText(
                en="Watch gradient descent reduce loss step by step.",
                pl="Zobacz, jak gradient descent krok po kroku zmniejsza loss.",
            ),
            LocalizedText(
                en="Change learning rate and noise to see when training becomes unstable.",
                pl="Zmieniaj learning rate i noise, żeby zobaczyć, kiedy trening traci stabilność.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="start or pause automatic steps", pl="uruchom albo zatrzymaj kroki"
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="perform one gradient descent step", pl="wykonaj jeden krok gradient descent"
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change learning rate", pl="zmień learning rate"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current run", pl="zresetuj aktualny przebieg"),
            ),
        ),
        create_scene=create_gradient_descent_scene,
        tags=("regression", "optimization"),
    ),
    _placeholder_demo(
        demo_id="knn_vote_map",
        level=1,
        title_en="k-NN Vote Map",
        title_pl="k-NN Vote Map",
        summary_en="Distance-based classification and neighborhood voting.",
        summary_pl="Klasyfikacja przez odległość i głosowanie najbliższych sąsiadów.",
        tags=("classification", "distance"),
    ),
    _placeholder_demo(
        demo_id="logistic_regression_boundary_lab",
        level=1,
        title_en="Logistic Regression Boundary Lab",
        title_pl="Logistic Regression Boundary Lab",
        summary_en="Probabilities, thresholds, and decision boundaries.",
        summary_pl="Prawdopodobieństwa, progi i decision boundary w praktyce.",
        tags=("classification", "probability"),
    ),
    _placeholder_demo(
        demo_id="decision_tree_splitter",
        level=1,
        title_en="Decision Tree Splitter",
        title_pl="Decision Tree Splitter",
        summary_en="Splits, impurity, and interpretable rules.",
        summary_pl="Splity, impurity i reguły, które da się wyjaśnić człowiekowi.",
        tags=("classification", "trees"),
    ),
    _placeholder_demo(
        demo_id="random_forest_bagging_lab",
        level=2,
        title_en="Random Forest Bagging Lab",
        title_pl="Random Forest Bagging Lab",
        summary_en="Bootstrap sampling, voting, variance, and stability.",
        summary_pl="Bootstrap sampling, voting, wariancja i stabilność predykcji.",
        tags=("ensemble", "classification"),
    ),
    _placeholder_demo(
        demo_id="boosting_mistake_lab",
        level=2,
        title_en="Boosting Mistake Lab",
        title_pl="Boosting Mistake Lab",
        summary_en=(
            "Watch boosting combine weak learners, update sample weights, "
            "and track train/test accuracy across rounds."
        ),
        summary_pl=(
            "Zobacz, jak boosting łączy weak learners, zmienia wagi próbek "
            "i śledzi train/test accuracy po kolejnych rundach."
        ),
        objectives=(
            LocalizedText(
                en="See how each weak learner focuses on mistakes from previous rounds.",
                pl="Zobacz, jak kolejne weak learners skupiają się na wcześniejszych błędach.",
            ),
            LocalizedText(
                en="Compare staged train/test accuracy and the generalization gap.",
                pl="Porównuj staged train/test accuracy i generalization gap.",
            ),
            LocalizedText(
                en="Use confidence view and presets to discuss overfitting and robustness.",
                pl="Używaj confidence view i presetów do rozmowy o overfittingu i odporności.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(
                    en="change selected boosting stage",
                    pl="zmień wybraną rundę boostingu",
                ),
            ),
            ControlBinding(
                key="+ / -",
                action=LocalizedText(
                    en="change total round count",
                    pl="zmień liczbę rund",
                ),
            ),
            ControlBinding(
                key="1-4 / P",
                action=LocalizedText(
                    en="switch preset scenario",
                    pl="przełącz preset",
                ),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="toggle confidence view",
                    pl="włącz albo wyłącz confidence view",
                ),
            ),
            ControlBinding(
                key="E",
                action=LocalizedText(
                    en="export selected-stage decision boundary",
                    pl="wyeksportuj decision boundary dla wybranej rundy",
                ),
            ),
        ),
        create_scene=create_boosting_mistake_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("ensemble", "classification", "boosting"),
    ),
)

DEMO_BY_ID: dict[str, DemoManifest] = {manifest.id: manifest for manifest in DEMO_MANIFESTS}


def levels_from_manifests(manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS) -> tuple[int, ...]:
    """Return sorted level numbers present in the registry."""
    return tuple(sorted({manifest.level for manifest in manifests}))


def demos_for_level(
    level: int,
    manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS,
) -> tuple[DemoManifest, ...]:
    """Return demos belonging to one level."""
    return tuple(manifest for manifest in manifests if manifest.level == level)


def validate_demo_registry(
    *,
    level_manifests: tuple[LevelManifest, ...] = LEVEL_MANIFESTS,
    demo_manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS,
) -> None:
    """Validate registry consistency.

    Raises:
        ValueError: if the registry contains duplicate or incomplete metadata.
    """
    level_numbers = [level.number for level in level_manifests]
    duplicate_levels = _duplicates(level_numbers)
    if duplicate_levels:
        raise ValueError(f"Duplicate level numbers: {duplicate_levels}")

    known_levels = set(level_numbers)
    demo_ids = [demo.id for demo in demo_manifests]
    duplicate_demo_ids = _duplicates(demo_ids)
    if duplicate_demo_ids:
        raise ValueError(f"Duplicate demo ids: {duplicate_demo_ids}")

    for demo in demo_manifests:
        if demo.level not in known_levels:
            raise ValueError(f"Demo {demo.id!r} references unknown level {demo.level}")
        if not demo.id:
            raise ValueError("Demo id must not be empty")
        if not demo.title.en or not demo.title.pl:
            raise ValueError(f"Demo {demo.id!r} must have localized titles")
        if not demo.summary.en or not demo.summary.pl:
            raise ValueError(f"Demo {demo.id!r} must have localized summaries")
        if not demo.objectives:
            raise ValueError(f"Demo {demo.id!r} must define objectives")
        if not demo.controls:
            raise ValueError(f"Demo {demo.id!r} must define controls")
        if demo.create_scene is None:
            raise ValueError(f"Demo {demo.id!r} must define a scene factory")


def _duplicates(values: list[int] | list[str]) -> tuple[int | str, ...]:
    """Return duplicate values in first-seen order."""
    seen: set[int | str] = set()
    duplicates: list[int | str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return tuple(duplicates)


validate_demo_registry()

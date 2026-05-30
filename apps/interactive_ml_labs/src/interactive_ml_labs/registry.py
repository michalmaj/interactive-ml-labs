"""Demo manifest registry for the unified shell."""

from __future__ import annotations

from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LevelManifest, LocalizedText
from interactive_ml_labs.placeholder_scene import PlaceholderDemoScene

LEVEL_MANIFESTS: tuple[LevelManifest, ...] = (
    LevelManifest(
        number=1,
        title=LocalizedText(en="Level 1 - Fundamentals", pl="Poziom 1 - Fundamenty"),
        summary=LocalizedText(
            en="Core machine learning intuition and foundational algorithms.",
            pl="Podstawowa intuicja machine learningu i fundamentalne algorytmy.",
        ),
    ),
    LevelManifest(
        number=2,
        title=LocalizedText(en="Level 2 - Practical ML", pl="Poziom 2 - Praktyczne ML"),
        summary=LocalizedText(
            en="Model evaluation, robustness, ensembles, and practical trade-offs.",
            pl="Ewaluacja modeli, odpornosc, ensemble i praktyczne kompromisy.",
        ),
    ),
    LevelManifest(
        number=3,
        title=LocalizedText(en="Level 3 - Advanced / Showcase", pl="Poziom 3 - Zaawansowane"),
        summary=LocalizedText(
            en="Advanced, specialized, and visually rich machine learning demos.",
            pl="Zaawansowane, specjalistyczne i wizualnie bogate dema ML.",
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
) -> DemoManifest:
    """Build a placeholder manifest for a demo."""
    return DemoManifest(
        id=demo_id,
        level=level,
        title=LocalizedText(en=title_en, pl=title_pl),
        summary=LocalizedText(en=summary_en, pl=summary_pl),
        objectives=(
            LocalizedText(
                en="Explore the main intuition behind the algorithm.",
                pl="Zbadaj glowna intuicje stojaca za algorytmem.",
            ),
            LocalizedText(
                en="Observe how parameter changes affect model behavior.",
                pl="Zobacz, jak zmiany parametrow wplywaja na zachowanie modelu.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Enter",
                action=LocalizedText(en="start placeholder scene", pl="uruchom ekran placeholder"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(en="open pause menu or go back", pl="otworz pauze albo wroc"),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="toggle help overlay", pl="pokaz lub ukryj pomoc"),
            ),
        ),
        create_scene=lambda context, demo_id=demo_id: PlaceholderDemoScene(
            context,
            DEMO_BY_ID[demo_id],
        ),
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzajacy"),
        tags=tags,
    )


DEMO_MANIFESTS: tuple[DemoManifest, ...] = (
    _placeholder_demo(
        demo_id="gradient_descent_playground",
        level=1,
        title_en="Gradient Descent Playground",
        title_pl="Plac zabaw spadku gradientowego",
        summary_en="Optimization, loss, and learning rate intuition.",
        summary_pl="Intuicja optymalizacji, funkcji straty i learning rate.",
        tags=("regression", "optimization"),
    ),
    _placeholder_demo(
        demo_id="knn_vote_map",
        level=1,
        title_en="k-NN Vote Map",
        title_pl="Mapa glosowania k-NN",
        summary_en="Distance-based classification and neighborhood voting.",
        summary_pl="Klasyfikacja oparta o odleglosc i glosowanie sasiadow.",
        tags=("classification", "distance"),
    ),
    _placeholder_demo(
        demo_id="logistic_regression_boundary_lab",
        level=1,
        title_en="Logistic Regression Boundary Lab",
        title_pl="Laboratorium granicy regresji logistycznej",
        summary_en="Probabilities, thresholds, and decision boundaries.",
        summary_pl="Prawdopodobienstwa, progi i granice decyzyjne.",
        tags=("classification", "probability"),
    ),
    _placeholder_demo(
        demo_id="decision_tree_splitter",
        level=1,
        title_en="Decision Tree Splitter",
        title_pl="Dzielnik drzewa decyzyjnego",
        summary_en="Splits, impurity, and interpretable rules.",
        summary_pl="Podzialy, nieczystosc i interpretowalne reguly.",
        tags=("classification", "trees"),
    ),
    _placeholder_demo(
        demo_id="random_forest_bagging_lab",
        level=2,
        title_en="Random Forest Bagging Lab",
        title_pl="Laboratorium baggingu lasu losowego",
        summary_en="Bootstrap sampling, voting, variance, and stability.",
        summary_pl="Bootstrap, glosowanie, wariancja i stabilnosc.",
        tags=("ensemble", "classification"),
    ),
    _placeholder_demo(
        demo_id="boosting_mistake_lab",
        level=2,
        title_en="Boosting Mistake Lab",
        title_pl="Laboratorium bledow boostingu",
        summary_en="Sequential weak learners that focus on previous mistakes.",
        summary_pl="Sekwencyjni slabi uczniowie skupieni na poprzednich bledach.",
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

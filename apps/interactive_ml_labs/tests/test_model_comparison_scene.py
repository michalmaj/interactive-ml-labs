"""Tests for the native Model Comparison Lab skeleton scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.model_comparison_scene import (
    DATASETS,
    MODELS,
    ModelComparisonLabScene,
    create_model_comparison_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_model_comparison_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Model Comparison Lab should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())

        assert isinstance(scene, ModelComparisonLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_model_comparison_scene_handles_events_without_navigation(monkeypatch) -> None:
    """Model controls should update the scene without requesting navigation."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        update_command = scene.update(0.16)

        assert command.kind == SceneCommandKind.NONE
        assert update_command.kind == SceneCommandKind.NONE
        assert scene.selected_model_index == 1
        assert scene.selected_model is MODELS[1]
    finally:
        pygame.quit()


def test_model_comparison_scene_switches_between_three_models(monkeypatch) -> None:
    """Number keys should focus the available classifier previews."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        assert scene.selected_model.name == "Decision Tree"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
        assert scene.selected_model.name == "Logistic Regression"
    finally:
        pygame.quit()


def test_model_comparison_scene_toggles_boundary_visibility(monkeypatch) -> None:
    """A should show or hide inactive boundaries and R should reset the preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        assert scene.show_all_boundaries is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        assert scene.selected_dataset_index == 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.selected_model_index == 0
        assert scene.selected_dataset_index == 0
        assert scene.show_all_boundaries is True
    finally:
        pygame.quit()


def test_model_comparison_scene_cycles_dataset_presets(monkeypatch) -> None:
    """D should switch between comparison dataset presets."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())
        first_points = scene.points

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))

        assert scene.selected_dataset_index == 1
        assert scene.selected_dataset is DATASETS[1]
        assert scene.points != first_points

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))

        assert scene.selected_dataset_index == 0
    finally:
        pygame.quit()


def test_model_comparison_scene_reports_model_accuracies(monkeypatch) -> None:
    """Visible-point scores should change with model and dataset assumptions."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())
        curved_scores = tuple(scene._accuracy_for_model(index) for index in range(len(MODELS)))

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        linear_scores = tuple(scene._accuracy_for_model(index) for index in range(len(MODELS)))

        assert all(0.0 <= score <= 1.0 for score in curved_scores)
        assert all(0.0 <= score <= 1.0 for score in linear_scores)
        assert curved_scores != linear_scores
        assert scene._predict_model(0, scene.points[0], 0) in {0, 1}
    finally:
        pygame.quit()


def test_model_comparison_scene_localizes_model_copy(monkeypatch) -> None:
    """Model details should use the global language from AppContext."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_model_comparison_lab_scene(context)

        assert scene.selected_model.assumption_for_language("pl") == (
            "liniowy score i jeden globalny podział"
        )
        assert "decision boundary" in scene.selected_model.boundary_for_language("pl")
    finally:
        pygame.quit()


def test_model_comparison_side_panel_details_fit_inside_panel(monkeypatch) -> None:
    """Scoreboard details should fit inside the right-side panel in both languages."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        for language in ("en", "pl"):
            context = AppContext()
            context.settings.language = language
            scene = create_model_comparison_lab_scene(context)
            side_panel_rect = pygame.Rect(818, 132, 390, 474)
            scoreboard_rect = scene._side_panel_scoreboard_rect(side_panel_rect)

            assert scoreboard_rect.bottom <= side_panel_rect.bottom

            for model_index in range(len(MODELS)):
                scene.selected_model_index = model_index
                assert scene._scoreboard_content_bottom_y(scoreboard_rect) <= scoreboard_rect.bottom
    finally:
        pygame.quit()


def test_model_comparison_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The Model Comparison preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_comparison_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

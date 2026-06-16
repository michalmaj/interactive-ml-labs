"""Smoke tests for app-only Level 2 native scene layouts."""

from collections.abc import Callable

import pygame
from interactive_ml_labs.class_imbalance_scene import create_class_imbalance_lab_scene
from interactive_ml_labs.data_leakage_scene import create_data_leakage_lab_scene
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.feature_scaling_scene import create_feature_scaling_lab_scene
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.split_lab_scene import create_train_validation_test_lab_scene
from interactive_ml_labs.tuning_scene import create_hyperparameter_tuning_lab_scene

SceneFactory = Callable[[AppContext], object]

LEVEL_TWO_NATIVE_SCENES: tuple[SceneFactory, ...] = (
    create_data_leakage_lab_scene,
    create_train_validation_test_lab_scene,
    create_feature_scaling_lab_scene,
    create_hyperparameter_tuning_lab_scene,
    create_class_imbalance_lab_scene,
)


def test_level_two_native_scenes_render_in_english_and_polish(monkeypatch) -> None:
    """App-only Level 2 layouts should render in both supported shell languages."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        for language in ("en", "pl"):
            context = AppContext()
            context.settings.language = language
            for create_scene in LEVEL_TWO_NATIVE_SCENES:
                surface = pygame.Surface(DEFAULT_RESOLUTION)
                scene = create_scene(context)

                scene.render(surface)

                assert surface.get_bounding_rect().width > 0
                assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

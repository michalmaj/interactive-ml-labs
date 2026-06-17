"""Tests for the native K-Means Intro Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.kmeans_intro_scene import (
    MAX_K,
    MIN_K,
    PRESETS,
    KMeansIntroLabScene,
    KMeansStep,
    create_kmeans_intro_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_kmeans_intro_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """K-Means Intro should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_kmeans_intro_lab_scene(AppContext())

        assert isinstance(scene, KMeansIntroLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_kmeans_intro_scene_steps_assignment_and_centroid_update(monkeypatch) -> None:
    """Space should expose assignment and centroid update as separate phases."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_kmeans_intro_lab_scene(AppContext())
        starting_centroids = tuple(scene.centroids)

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert command.kind == SceneCommandKind.NONE
        assert scene.step_to_run == KMeansStep.UPDATE
        assert all(assignment >= 0 for assignment in scene.assignments)
        assert scene.inertia > 0
        assert len(scene.inertia_history) == 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert scene.step_to_run == KMeansStep.ASSIGN
        assert scene.iteration == 1
        assert tuple(scene.centroids) != starting_centroids
        assert len(scene.inertia_history) == 2
    finally:
        pygame.quit()


def test_kmeans_intro_scene_changes_k_presets_and_reset(monkeypatch) -> None:
    """Keyboard controls should switch datasets, change k, and reset the lab."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_kmeans_intro_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.k == 4
        assert len(scene.centroids) == 4

        for _ in range(10):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

        assert scene.k == MIN_K
        assert len(scene.centroids) == MIN_K

        for _ in range(10):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert scene.k == MAX_K
        assert len(scene.centroids) == MAX_K

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.preset is PRESETS[1]
        assert scene.step_to_run == KMeansStep.ASSIGN
        assert all(assignment < 0 for assignment in scene.assignments)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.auto_run is True
        assert scene.show_links is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.k == 3
        assert scene.auto_run is False
        assert scene.show_links is True
    finally:
        pygame.quit()


def test_kmeans_intro_scene_auto_run_advances_steps(monkeypatch) -> None:
    """Auto-run should advance K-Means over time."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_kmeans_intro_lab_scene(AppContext())
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))

        scene.update(2.0)

        assert scene.inertia_history
        assert scene.iteration >= 1
    finally:
        pygame.quit()


def test_kmeans_intro_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should use natural copy while keeping K-Means terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_kmeans_intro_lab_scene(context)

        assert scene._short_step_label() == "przypisz punkty"
        assert "centroidu" in scene._active_takeaway()

        scene.step()

        assert scene._short_step_label() == "przesuń centroidy"
        assert "tymczasowe" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_kmeans_intro_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The K-Means intro scene should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_kmeans_intro_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

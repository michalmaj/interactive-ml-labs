"""Tests for the native Clustering Lab scene."""

import pygame
from interactive_ml_labs.clustering_scene import (
    MAX_K,
    MIN_K,
    ClusteringLabScene,
    create_clustering_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_clustering_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Clustering Lab should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())

        assert isinstance(scene, ClusteringLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_clustering_scene_switches_dataset_presets(monkeypatch) -> None:
    """Number keys should switch between planned dataset presets."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        first_points = tuple(scene.points)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset_index == 2
        assert scene.preset.key == "moons"
        assert tuple(scene.points) != first_points
        assert scene.iteration == 0
    finally:
        pygame.quit()


def test_clustering_scene_changes_k_with_keyboard(monkeypatch) -> None:
    """Minus and equals should update k within scene limits."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())

        for _ in range(8):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.k == MAX_K

        for _ in range(8):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert scene.k == MIN_K
        assert len(scene.centroids) == MIN_K
    finally:
        pygame.quit()


def test_clustering_scene_step_updates_centroids_and_inertia(monkeypatch) -> None:
    """One K-Means step should move at least one centroid and track inertia."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        initial_centroids = tuple(scene.centroids)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert scene.iteration == 1
        assert tuple(scene.centroids) != initial_centroids
        assert len(scene.assignments) == len(scene.points)
        assert scene.inertia > 0.0
    finally:
        pygame.quit()


def test_clustering_scene_auto_run_steps_over_time(monkeypatch) -> None:
    """Auto-run should advance K-Means iterations during update."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        command = scene.update(0.60)

        assert command.kind == SceneCommandKind.NONE
        assert scene.auto_run is True
        assert scene.iteration == 1
    finally:
        pygame.quit()


def test_clustering_scene_new_sample_regenerates_points(monkeypatch) -> None:
    """N should regenerate the current dataset sample."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        first_points = tuple(scene.points)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        assert tuple(scene.points) != first_points
        assert scene.iteration == 0
    finally:
        pygame.quit()


def test_clustering_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The scene should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

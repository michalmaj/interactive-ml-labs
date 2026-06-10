"""Tests for the native t-SNE / UMAP Exploration Lab skeleton scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.tsne_umap_scene import (
    DEFAULT_NEIGHBOR_INDEX,
    NEIGHBOR_VALUES,
    PRESETS,
    TSNEUMAPExplorationScene,
    create_tsne_umap_exploration_scene,
)


def test_tsne_umap_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """t-SNE / UMAP scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())

        assert isinstance(scene, TSNEUMAPExplorationScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_tsne_umap_scene_handles_events_without_navigation(monkeypatch) -> None:
    """Scene controls should update state without requesting navigation."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        update_command = scene.update(0.16)

        assert command.kind == SceneCommandKind.NONE
        assert update_command.kind == SceneCommandKind.NONE
        assert scene.algorithm == "UMAP"
    finally:
        pygame.quit()


def test_tsne_umap_scene_switches_dataset_presets(monkeypatch) -> None:
    """Number keys should switch deterministic data presets."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())
        first_points = scene._active_embedding()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset_index == 2
        assert scene.preset is PRESETS[2]
        assert scene._active_embedding() != first_points
    finally:
        pygame.quit()


def test_tsne_umap_scene_tunes_neighbors_and_seed(monkeypatch) -> None:
    """Minus/equal and S should change deterministic embedding parameters."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())
        first_points = scene._active_embedding()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.neighbor_index == DEFAULT_NEIGHBOR_INDEX + 1
        assert scene.neighbors == NEIGHBOR_VALUES[DEFAULT_NEIGHBOR_INDEX + 1]
        assert scene._active_embedding() != first_points

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
        assert scene.seed_index == 1

        for _ in range(10):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert scene.neighbor_index == 0
    finally:
        pygame.quit()


def test_tsne_umap_scene_toggles_links_and_resets(monkeypatch) -> None:
    """L should toggle links and R should restore defaults."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.show_links is False
        assert scene.algorithm == "UMAP"
        assert scene.preset_index == 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.show_links is True
        assert scene.algorithm == "t-SNE"
        assert scene.preset_index == 0
        assert scene.neighbor_index == DEFAULT_NEIGHBOR_INDEX
        assert scene.seed_index == 0
    finally:
        pygame.quit()


def test_tsne_umap_scene_reports_embedding_scores(monkeypatch) -> None:
    """Local trust and global spread should stay in valid percentage ranges."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())
        trust = scene._local_trust_score()
        spread = scene._global_spread_score()

        assert 0.0 <= trust <= 1.0
        assert 0.0 <= spread <= 1.0
        assert scene._nearest_neighbor_pairs(scene._active_embedding())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert scene._global_spread_score() != spread
    finally:
        pygame.quit()


def test_tsne_umap_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The t-SNE / UMAP preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

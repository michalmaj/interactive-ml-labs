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
    """L/O should toggle helpers and R should restore defaults."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.show_links is False
        assert scene.show_raw_layout is False
        assert scene.algorithm == "UMAP"
        assert scene.preset_index == 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.show_links is True
        assert scene.show_raw_layout is True
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


def test_tsne_umap_scene_formats_reading_cues(monkeypatch) -> None:
    """Metric labels and reading cues should explain how to interpret the view."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())

        assert scene._format_score(0.82) == "82% · strong"
        assert scene._format_score(0.62) == "62% · mixed"
        assert scene._format_score(0.24) == "24% · weak"
        assert "Reading cue:" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert "Reading cue:" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_tsne_umap_comparison_panel_has_three_non_overlapping_plots(monkeypatch) -> None:
    """Comparison panel should reserve space for raw, t-SNE, and UMAP mini plots."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_tsne_umap_exploration_scene(AppContext())
        panel_rect = pygame.Rect(638, 132, 286, 474)
        raw_rect, tsne_rect, umap_rect = scene._comparison_plot_rects(panel_rect)
        caption_rect = scene._comparison_caption_rect(panel_rect)

        assert raw_rect.bottom < tsne_rect.top
        assert tsne_rect.bottom < umap_rect.top
        assert umap_rect.bottom + 32 <= caption_rect.top
        assert caption_rect.bottom < panel_rect.bottom

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o))
        assert scene.show_raw_layout is False
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

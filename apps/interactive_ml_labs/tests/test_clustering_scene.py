"""Tests for the native Clustering Lab scene."""

import pygame
from interactive_ml_labs.clustering_scene import (
    MAX_K,
    MIN_K,
    PANEL_RECT,
    PLOT_RECT,
    AlgorithmMode,
    ClusteringLabScene,
    KMeansPhase,
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
        initial_history = tuple(scene.inertia_history)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert scene.iteration == 1
        assert scene.kmeans_phase == KMeansPhase.ASSIGN
        assert tuple(scene.centroids) != initial_centroids
        assert len(scene.assignments) == len(scene.points)
        assert scene.inertia > 0.0
        assert len(scene.inertia_history) == len(initial_history) + 1
        assert scene.inertia_history[-1] == scene.inertia
    finally:
        pygame.quit()


def test_clustering_scene_assignment_phase_updates_without_new_iteration(monkeypatch) -> None:
    """K-Means should expose assignment and centroid-update as separate phases."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        scene.step()
        inertia_after_update = scene.inertia

        scene.step()

        assert scene.iteration == 1
        assert scene.kmeans_phase == KMeansPhase.UPDATE
        assert scene.inertia_history[-1] == scene.inertia
        assert scene.inertia <= inertia_after_update
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


def test_clustering_scene_toggles_centroid_links(monkeypatch) -> None:
    """C should show or hide point-to-centroid helper lines."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.show_links is True

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.show_links is False
    finally:
        pygame.quit()


def test_clustering_scene_toggles_dbscan_mode(monkeypatch) -> None:
    """M should switch between K-Means and DBSCAN."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert scene.algorithm_mode == AlgorithmMode.DBSCAN
        assert len(scene.dbscan_labels) == len(scene.points)
        assert scene.dbscan_cluster_count >= 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert scene.algorithm_mode == AlgorithmMode.KMEANS
        assert scene.kmeans_phase == KMeansPhase.UPDATE
    finally:
        pygame.quit()


def test_clustering_scene_dbscan_eps_changes_with_keyboard(monkeypatch) -> None:
    """Minus and equals should tune eps in DBSCAN mode."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        initial_eps = scene.dbscan_eps

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.dbscan_eps > initial_eps

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert scene.dbscan_eps == initial_eps
        assert len(scene.dbscan_labels) == len(scene.points)
    finally:
        pygame.quit()


def test_clustering_scene_dbscan_ignores_auto_run(monkeypatch) -> None:
    """DBSCAN mode should not advance K-Means auto-run iterations."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.auto_run = True

        scene.update(1.20)

        assert scene.algorithm_mode == AlgorithmMode.DBSCAN
        assert scene.iteration == 0
    finally:
        pygame.quit()


def test_clustering_scene_status_rows_follow_algorithm_mode(monkeypatch) -> None:
    """The side panel should show only mode-relevant status values."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_clustering_lab_scene(context)

        kmeans_labels = {label for label, _value in scene._status_rows()}
        assert "k" in kmeans_labels
        assert "Inertia" in kmeans_labels
        assert "Auto-run" in kmeans_labels
        assert "eps" not in kmeans_labels

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        dbscan_labels = {label for label, _value in scene._status_rows()}

        assert "eps" in dbscan_labels
        assert "min points" in dbscan_labels
        assert "Klastry" in dbscan_labels
        assert "Noise" in dbscan_labels
        assert "k" not in dbscan_labels
        assert "Inertia" not in dbscan_labels
        assert "Auto-run" not in dbscan_labels
    finally:
        pygame.quit()


def test_clustering_scene_observation_hint_follows_algorithm_mode(monkeypatch) -> None:
    """Observation copy should explain the active clustering algorithm."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        assert "K-Means" in scene._observation_hint()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert "DBSCAN" in scene._observation_hint()
    finally:
        pygame.quit()


def test_clustering_scene_drags_nearest_point(monkeypatch) -> None:
    """Mouse drag should move a picked point and refresh assignments."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        plot_rect = pygame.Rect(PLOT_RECT)
        first_point_position = scene._to_screen(scene.points[0], plot_rect)
        target_position = (plot_rect.centerx, plot_rect.centery)
        scene.auto_run = True
        initial_history_length = len(scene.inertia_history)

        scene.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=first_point_position),
        )
        scene.handle_event(
            pygame.event.Event(
                pygame.MOUSEMOTION, pos=target_position, rel=(0, 0), buttons=(1, 0, 0)
            ),
        )
        scene.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=target_position),
        )

        assert scene.auto_run is False
        assert scene.dragged_point_index is None
        assert abs(scene.points[0].x) < 0.01
        assert abs(scene.points[0].y) < 0.01
        assert len(scene.assignments) == len(scene.points)
        assert len(scene.inertia_history) == initial_history_length
        assert scene.inertia_history[-1] == scene.inertia
    finally:
        pygame.quit()


def test_clustering_scene_resets_inertia_history_with_centroids(monkeypatch) -> None:
    """Changing k should restart the inertia history for the new centroid setup."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_clustering_lab_scene(AppContext())
        scene.step()
        assert len(scene.inertia_history) == 2

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert scene.iteration == 0
        assert scene.kmeans_phase == KMeansPhase.UPDATE
        assert len(scene.inertia_history) == 1
        assert scene.inertia_history[0] == scene.inertia
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


def test_clustering_scene_controls_fit_inside_panel(monkeypatch) -> None:
    """The right-side panel should contain the full controls block."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_clustering_lab_scene(context)
        panel_rect = pygame.Rect(PANEL_RECT)

        for preset_index in range(4):
            scene.preset_index = preset_index
            for mode_key in (None, pygame.K_m):
                if mode_key is not None:
                    scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=mode_key))
                controls_y = scene._panel_controls_start_y(panel_rect)

                assert scene._controls_bottom_y(controls_y) <= panel_rect.bottom - 16
    finally:
        pygame.quit()

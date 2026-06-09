"""Tests for the native Calibration Lab skeleton scene."""

import pygame
from interactive_ml_labs.calibration_scene import (
    PRESETS,
    CalibrationLabScene,
    create_calibration_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_calibration_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Calibration Lab should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())

        assert isinstance(scene, CalibrationLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_calibration_scene_handles_events_without_navigation(monkeypatch) -> None:
    """Calibration controls should update state without requesting navigation."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        update_command = scene.update(0.16)

        assert command.kind == SceneCommandKind.NONE
        assert update_command.kind == SceneCommandKind.NONE
        assert scene.preset_index == 1
        assert scene.preset is PRESETS[1]
    finally:
        pygame.quit()


def test_calibration_scene_switches_between_presets(monkeypatch) -> None:
    """Number keys should focus the available calibration presets."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        assert scene.preset.name_en == "Better calibrated"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
        assert scene.preset.name_en == "Overconfident"
    finally:
        pygame.quit()


def test_calibration_scene_toggles_error_bars_and_resets(monkeypatch) -> None:
    """E should toggle error bars and R should restore the first preset."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.show_error_bars is False
        assert scene.preset_index == 2

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.show_error_bars is True
        assert scene.preset_index == 0
    finally:
        pygame.quit()


def test_calibration_scene_reports_valid_bin_metrics(monkeypatch) -> None:
    """Calibration bins, Brier score, and ECE should stay in valid probability ranges."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())
        bins = scene._calibration_bins()
        brier = scene._brier_score()
        ece = scene._expected_calibration_error()

        assert len(bins) == 5
        assert sum(int(calibration_bin["count"]) for calibration_bin in bins) == len(
            scene.preset.samples
        )
        assert all(0.0 <= float(calibration_bin["confidence"]) <= 1.0 for calibration_bin in bins)
        assert all(0.0 <= float(calibration_bin["accuracy"]) <= 1.0 for calibration_bin in bins)
        assert 0.0 <= brier <= 1.0
        assert 0.0 <= ece <= 1.0

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene._brier_score() != brier
        assert scene._expected_calibration_error() != ece
    finally:
        pygame.quit()


def test_calibration_scene_localizes_preset_copy(monkeypatch) -> None:
    """Preset text should use the global language from AppContext."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_calibration_lab_scene(context)

        assert scene.preset.name_for_language("pl") == "Overconfident"
        assert "Score" in scene.preset.summary_for_language("pl")
    finally:
        pygame.quit()


def test_calibration_score_summary_does_not_overlap_plot(monkeypatch) -> None:
    """Score summary copy should fit below the plot in both languages."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        for language in ("en", "pl"):
            context = AppContext()
            context.settings.language = language
            scene = create_calibration_lab_scene(context)
            score_panel_rect = pygame.Rect(620, 132, 300, 474)
            plot_rect = scene._score_distribution_plot_rect(score_panel_rect)
            summary_top_y = scene._score_summary_top_y(score_panel_rect)

            assert summary_top_y >= plot_rect.bottom + 18

            for preset_index in range(len(PRESETS)):
                scene.preset_index = preset_index
                summary_bottom_y = scene._wrapped_text_bottom_y(
                    scene.preset.summary_for_language(language),
                    score_panel_rect.width - 48,
                    scene._font_small,
                    top_y=summary_top_y,
                    line_height=18,
                )

                assert summary_bottom_y <= score_panel_rect.bottom - 18
    finally:
        pygame.quit()


def test_calibration_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The Calibration preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_calibration_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

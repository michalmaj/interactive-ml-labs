"""Tests for the guided learning path screen renderer."""

import pygame
from interactive_ml_labs.screens.learning_path_screen import (
    LearningPathDetails,
    LearningPathMetric,
    LearningPathRenderer,
    LearningPathScreenColors,
    LearningPathScreenFonts,
)
from interactive_ml_labs.settings import AppSettings


def test_learning_path_renderer_returns_menu_and_scroll_geometry(monkeypatch) -> None:
    """Learning path renderer should expose hitboxes and details viewport geometry."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((1280, 720))
        font = pygame.font.Font(None, 24)
        renderer = LearningPathRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(resolution=(1280, 720)),
            selected_index=0,
            menu_labels=["[ ] Path A", "[x] Path B"],
            details=LearningPathDetails(
                title="How models learn from error",
                summary="A short guided route.",
                lesson_count_label="4 lessons",
                progress_metrics=[
                    LearningPathMetric(
                        label="Lessons: 1/4 completed",
                        completed_count=1,
                        total_count=4,
                    ),
                ],
                status_label="In progress",
                next_action_label="Next action: start",
                badge_labels=["[x] Residual Reader"],
                course_map_heading="Course map",
                lesson_labels=["1. Read residuals - Completed; tasks 2/2"],
            ),
            scroll_offset=0,
            max_scroll=0,
            fonts=LearningPathScreenFonts(
                title=pygame.font.Font(None, 44),
                heading=pygame.font.Font(None, 30),
                body=font,
                small=font,
            ),
            colors=LearningPathScreenColors(
                text=(255, 255, 255),
                muted_text=(180, 180, 180),
                accent=(120, 220, 160),
                panel=(30, 30, 30),
                selected_panel=(50, 80, 100),
                border=(90, 90, 90),
                progress_track=(50, 50, 50),
            ),
            footer_y=660,
        )

        assert [item.label for item in result.menu_items] == ["[ ] Path A", "[x] Path B"]
        assert result.viewport.width > 0
        assert result.viewport.height > 0
        assert result.content_end > result.viewport.top
    finally:
        pygame.quit()

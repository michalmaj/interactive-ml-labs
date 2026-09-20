"""Tests for the course map screen renderer."""

import pygame
from interactive_ml_labs.screens.course_map_screen import (
    CourseMapMetric,
    CourseMapRenderer,
    CourseMapScreenColors,
    CourseMapScreenFonts,
    CourseMapStepDetails,
)
from interactive_ml_labs.settings import AppSettings


def test_course_map_renderer_returns_menu_and_scroll_geometry(monkeypatch) -> None:
    """Course map renderer should expose hitboxes and details viewport geometry."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((1280, 720))
        font = pygame.font.Font(None, 24)
        renderer = CourseMapRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(resolution=(1280, 720)),
            selected_index=0,
            menu_labels=["Step 1", "All guided paths"],
            details=CourseMapStepDetails(
                step_label="Step 1",
                title="How models learn from error",
                rationale="A short rationale.",
                progress_metrics=[
                    CourseMapMetric(
                        label="Lessons: 0/2 completed",
                        completed_count=0,
                        total_count=2,
                    ),
                ],
                next_action_label="Next action: start",
                next_reason_label="Then compare another model.",
                practice_heading="You will practice",
                practice_items=["Gradient descent"],
            ),
            scroll_offset=0,
            max_scroll=0,
            fonts=CourseMapScreenFonts(
                title=pygame.font.Font(None, 44),
                heading=pygame.font.Font(None, 30),
                body=font,
                small=font,
            ),
            colors=CourseMapScreenColors(
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

        assert [item.label for item in result.menu_items] == ["Step 1", "All guided paths"]
        assert result.viewport.width > 0
        assert result.viewport.height > 0
        assert result.content_end > result.viewport.top
    finally:
        pygame.quit()

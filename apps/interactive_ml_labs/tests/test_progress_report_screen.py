"""Tests for the shell progress report renderer."""

import pygame
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.screens.progress_report_screen import (
    ProgressReportColors,
    ProgressReportDetails,
    ProgressReportFonts,
    ProgressReportMetric,
    ProgressReportPath,
    ProgressReportRenderer,
)
from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import BadgeItem


def _fonts() -> ProgressReportFonts:
    return ProgressReportFonts(
        title=make_ui_font(44, bold=True),
        heading=make_ui_font(30, bold=True),
        body=make_ui_font(22),
        small=make_ui_font(18),
    )


def _colors() -> ProgressReportColors:
    return ProgressReportColors(
        background=(22, 25, 29),
        text=(235, 238, 241),
        muted_text=(166, 173, 181),
        accent=(113, 204, 152),
        panel=(38, 43, 49),
        selected_panel=(55, 97, 126),
        border=(72, 79, 88),
        progress_track=(55, 61, 69),
        unlocked_fill=(226, 176, 83),
        unlocked_outline=(250, 218, 139),
        locked_fill=(61, 68, 76),
        locked_outline=(109, 118, 128),
    )


def _details() -> ProgressReportDetails:
    return ProgressReportDetails(
        title="Progress report",
        subtitle="A compact progress overview.",
        metrics=[
            ProgressReportMetric(
                label="Lessons completed: 2/8",
                completed_count=2,
                total_count=8,
            ),
        ],
        explain_heading="What you can already explain",
        explain_lines=["After a long lesson title: explain the visible signal."],
        paths_heading="Guided paths",
        paths=[
            ProgressReportPath(
                title=(
                    "A deliberately long guided path title that should wrap without "
                    "colliding with the metric row below"
                ),
                status_label="In progress",
                lesson_label="Lessons: 2/4",
                task_label="Tasks: 3/8",
                badge_label="Badges: 1/4",
                completed_count=2,
                total_count=4,
            ),
        ],
        badges_heading="Badges",
        badge_summary="Unlocked badges: 1/2",
        badges=[
            BadgeItem(label="First Badge", unlocked=True),
            BadgeItem(label="Second Badge", unlocked=False),
        ],
    )


def test_progress_report_renderer_returns_scroll_geometry(monkeypatch) -> None:
    """Progress report renderer should expose viewport and Back hitbox geometry."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((1280, 720))
        renderer = ProgressReportRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(resolution=(1280, 720)),
            details=_details(),
            scroll_offset=0,
            max_scroll=0,
            content_bottom=636,
            footer_y=670,
            fonts=_fonts(),
            colors=_colors(),
        )

        assert result.viewport.x == 80
        assert result.viewport.width == 1040
        assert result.viewport.bottom == 560
        assert result.content_end > result.viewport.y
        assert [item.label for item in result.menu_items] == ["Back"]
        assert result.menu_items[0].rect == pygame.Rect(80, 574, 260, 54)
    finally:
        pygame.quit()


def test_progress_report_renderer_localizes_back_label(monkeypatch) -> None:
    """Progress report renderer should localize shell controls."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((1280, 720))
        renderer = ProgressReportRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(language="pl", resolution=(1280, 720)),
            details=_details(),
            scroll_offset=0,
            max_scroll=0,
            content_bottom=636,
            footer_y=670,
            fonts=_fonts(),
            colors=_colors(),
        )

        assert [item.label for item in result.menu_items] == ["Wróć"]
    finally:
        pygame.quit()

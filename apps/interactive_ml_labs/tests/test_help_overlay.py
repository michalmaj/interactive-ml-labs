"""Tests for the shell help overlay renderer."""

import pygame
from interactive_ml_labs import demos_for_level
from interactive_ml_labs.screens.help_overlay import (
    HelpOverlayColors,
    HelpOverlayDetails,
    HelpOverlayFonts,
    HelpOverlayRenderer,
)
from interactive_ml_labs.settings import AppSettings


def _fonts() -> HelpOverlayFonts:
    """Return stable fonts for help overlay tests."""
    return HelpOverlayFonts(
        heading=pygame.font.Font(None, 30),
        body=pygame.font.Font(None, 24),
        small=pygame.font.Font(None, 20),
    )


def _colors() -> HelpOverlayColors:
    """Return visible colors for help overlay tests."""
    return HelpOverlayColors(
        text=(255, 255, 255),
        muted_text=(180, 180, 180),
        accent=(120, 220, 160),
        background=(12, 14, 17),
    )


def _details_for_demo(demo_id: str) -> HelpOverlayDetails:
    """Build renderer details for one demo in Polish."""
    demo = next(
        demo for level in (1, 2, 3) for demo in demos_for_level(level) if demo.id == demo_id
    )
    return HelpOverlayDetails(
        title="Pomoc - " + demo.title.pl,
        summary=demo.summary.pl,
        objectives=[objective.pl for objective in demo.objectives],
        controls=[f"{control.key}: {control.action.pl}" for control in demo.controls],
        fallback_body="",
        goals_heading="Cele",
        controls_heading="Sterowanie",
    )


def test_help_overlay_renderer_uses_columns_for_demo_controls(monkeypatch) -> None:
    """Help overlay should split goals and controls into columns on wide screens."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    renderer = HelpOverlayRenderer()
    wrapped_items: list[tuple[str, tuple[int, int]]] = []

    def capture_wrapped(
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        _ = surface, width, color
        wrapped_items.append((text, position))
        return position[1] + font.get_linesize()

    try:
        renderer._draw_wrapped = capture_wrapped  # type: ignore[method-assign]
        renderer.render(
            pygame.Surface((1280, 720)),
            settings=AppSettings(resolution=(1280, 720), language="pl"),
            details=_details_for_demo("random_forest_bagging_lab"),
            fonts=_fonts(),
            colors=_colors(),
        )

        goal_positions = [
            position
            for text, position in wrapped_items
            if text.startswith("- Porównuj single tree baseline")
        ]
        control_positions = [
            position
            for text, position in wrapped_items
            if text.startswith(("- Up / Down:", "- W / S:", "- B / V:"))
        ]

        assert goal_positions
        assert control_positions
        assert all(position[0] < 500 for position in goal_positions)
        assert all(position[0] > 600 for position in control_positions)
        assert max(position[1] for position in control_positions) < 620
    finally:
        pygame.quit()


def test_help_overlay_renderer_keeps_copy_inside_overlay(monkeypatch) -> None:
    """Help overlay renderer should keep level copy inside the dialog body."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    renderer = HelpOverlayRenderer()
    settings = AppSettings(resolution=(1280, 720), language="pl")
    surface = pygame.Surface(settings.resolution)

    try:
        for level in (1, 2, 3):
            for demo in demos_for_level(level):
                details = HelpOverlayDetails(
                    title="Pomoc - " + demo.title.pl,
                    summary=demo.summary.pl,
                    objectives=[objective.pl for objective in demo.objectives],
                    controls=[f"{control.key}: {control.action.pl}" for control in demo.controls],
                    fallback_body="",
                    goals_heading="Cele",
                    controls_heading="Sterowanie",
                )
                result = renderer.render(
                    surface,
                    settings=settings,
                    details=details,
                    fonts=_fonts(),
                    colors=_colors(),
                )

                assert result.content_bottom <= result.rect.bottom - 32
    finally:
        pygame.quit()

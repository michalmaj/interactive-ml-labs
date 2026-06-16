"""Native SVM Margin Lab scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.readout_panel import (
    ReadoutPanelColors,
    ReadoutPanelFonts,
    draw_readout_panel,
)
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BACKGROUND: Final[tuple[int, int, int]] = (20, 23, 27)
PANEL: Final[tuple[int, int, int]] = (34, 39, 45)
PLOT_BG: Final[tuple[int, int, int]] = (16, 19, 23)
GRID: Final[tuple[int, int, int]] = (48, 55, 63)
TEXT: Final[tuple[int, int, int]] = (236, 239, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (166, 174, 184)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
SECONDARY: Final[tuple[int, int, int]] = (248, 183, 96)
GOOD: Final[tuple[int, int, int]] = (147, 218, 155)
WARNING: Final[tuple[int, int, int]] = (246, 132, 134)

ANGLE_STEP: Final[float] = 5.0
OFFSET_STEP: Final[float] = 0.2


@dataclass(frozen=True, slots=True)
class MarginPoint:
    """Point in a two-class margin dataset."""

    x: float
    y: float
    label: int


@dataclass(frozen=True, slots=True)
class MarginPreset:
    """Static SVM margin scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    points: tuple[MarginPoint, ...]
    start_angle: float
    start_offset: float
    best_angle: float
    best_offset: float

    def name_for_language(self, language: str) -> str:
        """Return localized preset name."""
        if language == "pl":
            return self.name_pl
        return self.name_en

    def summary_for_language(self, language: str) -> str:
        """Return localized summary."""
        if language == "pl":
            return self.summary_pl
        return self.summary_en


PRESETS: Final[tuple[MarginPreset, ...]] = (
    MarginPreset(
        name_en="Wide gap",
        name_pl="Szeroki margines",
        summary_en="A clean gap makes it easy to see the largest safe margin.",
        summary_pl="Czysta przerwa ułatwia zobaczenie największego bezpiecznego margin.",
        points=(
            MarginPoint(-2.7, -1.9, -1),
            MarginPoint(-2.0, -0.8, -1),
            MarginPoint(-1.2, -2.0, -1),
            MarginPoint(-0.7, -0.6, -1),
            MarginPoint(0.8, 0.5, 1),
            MarginPoint(1.5, 1.7, 1),
            MarginPoint(2.1, 0.8, 1),
            MarginPoint(2.8, 2.1, 1),
        ),
        start_angle=135.0,
        start_offset=-0.8,
        best_angle=40.0,
        best_offset=0.0,
    ),
    MarginPreset(
        name_en="Tilted classes",
        name_pl="Skośne klasy",
        summary_en="The boundary must rotate before the margin becomes balanced.",
        summary_pl="Boundary musi się obrócić, zanim margin będzie zbalansowany.",
        points=(
            MarginPoint(-2.5, 1.1, -1),
            MarginPoint(-1.9, 0.0, -1),
            MarginPoint(-1.2, -1.1, -1),
            MarginPoint(-0.2, -1.8, -1),
            MarginPoint(0.0, 1.7, 1),
            MarginPoint(0.9, 0.8, 1),
            MarginPoint(1.6, -0.2, 1),
            MarginPoint(2.4, -1.1, 1),
        ),
        start_angle=90.0,
        start_offset=0.2,
        best_angle=40.0,
        best_offset=0.1,
    ),
    MarginPreset(
        name_en="Close support vectors",
        name_pl="Bliskie support vectors",
        summary_en="A few close points decide the margin even when most points are far away.",
        summary_pl="Kilka bliskich punktów decyduje o margin, nawet gdy reszta jest daleko.",
        points=(
            MarginPoint(-2.9, -1.7, -1),
            MarginPoint(-2.0, -2.4, -1),
            MarginPoint(-1.0, -0.4, -1),
            MarginPoint(-0.5, -1.1, -1),
            MarginPoint(0.4, 0.8, 1),
            MarginPoint(1.1, 0.3, 1),
            MarginPoint(2.1, 2.0, 1),
            MarginPoint(2.8, 1.2, 1),
        ),
        start_angle=145.0,
        start_offset=0.5,
        best_angle=40.0,
        best_offset=0.1,
    ),
)


class SVMMarginLabScene:
    """Interactive slice for SVM margin intuition."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic margin scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.angle = PRESETS[0].start_angle
        self.offset = PRESETS[0].start_offset

    @property
    def preset(self) -> MarginPreset:
        """Return the active margin preset."""
        return PRESETS[self.preset_index]

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        if isinstance(event, pygame.event.Event) and event.type == pygame.KEYDOWN:
            self._handle_keydown(event.key)
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the margin lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_margin_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self._reset_boundary()
        elif key == pygame.K_LEFT:
            self.angle -= ANGLE_STEP
        elif key == pygame.K_RIGHT:
            self.angle += ANGLE_STEP
        elif key == pygame.K_DOWN:
            self.offset -= OFFSET_STEP
        elif key == pygame.K_UP:
            self.offset += OFFSET_STEP
        elif key == pygame.K_f:
            self.angle = self.preset.best_angle
            self.offset = self.preset.best_offset
        elif key == pygame.K_r:
            self.preset_index = 0
            self._reset_boundary()

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "SVM Margin Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Rotate a boundary and watch support vectors define the margin.",
                "Obracaj boundary i zobacz, jak support vectors definiują margin.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_margin_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Boundary and margin", "Boundary i margin"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface, self._boundary_label(), (rect.x + 24, rect.y + 54), self._font_small, SECONDARY
        )
        plot_rect = pygame.Rect(rect.x + 84, rect.y + 104, 532, 270)
        self._draw_feature_space(surface, plot_rect)
        self._draw_margin_strip(surface, pygame.Rect(rect.x + 84, rect.y + 405, 532, 30))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 48),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("angle", "angle"), f"{self.angle:.0f} deg"),
            (self._label("offset", "offset"), f"{self.offset:+.1f}"),
            (self._label("accuracy", "accuracy"), self._accuracy_label()),
            (self._label("margin", "margin"), f"{self._margin():.2f}"),
            (self._label("support vectors", "support vectors"), self._support_labels()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Margin readout", "Odczyt margin"),
            rows=rows,
            options=options,
            takeaway=self._active_takeaway(),
            fonts=ReadoutPanelFonts(heading=self._font_heading, small=self._font_small),
            colors=ReadoutPanelColors(
                panel=PANEL,
                text=TEXT,
                muted_text=MUTED_TEXT,
                accent=ACCENT,
                secondary=SECONDARY,
            ),
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: separate classes while making the closest safe gap as wide as possible.",
                "Cel: rozdziel klasy i zrób jak najszerszy bezpieczny gap.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_feature_space(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

        margin = self._margin()
        self._draw_boundary_line(surface, rect, self.angle, self.offset - margin, MUTED_TEXT, 1)
        self._draw_boundary_line(surface, rect, self.angle, self.offset + margin, MUTED_TEXT, 1)
        self._draw_boundary_line(surface, rect, self.angle, self.offset, ACCENT, 3)

        support_vectors = set(self._support_vectors())
        for index, point in enumerate(self.preset.points, start=1):
            predicted = self._predict(point)
            color = GOOD if point.label > 0 else SECONDARY
            if predicted != point.label:
                color = WARNING
            position = self._to_screen(rect, point.x, point.y)
            radius = 8 if point in support_vectors else 6
            pygame.draw.circle(surface, color, position, radius)
            if point in support_vectors:
                pygame.draw.circle(surface, TEXT, position, radius + 3, width=1)
            self._draw_text(
                surface, str(index), (position[0] + 9, position[1] - 9), self._font_small, color
            )

    def _draw_boundary_line(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        angle: float,
        offset: float,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        nx, ny = self._normal(angle)
        points: list[tuple[float, float]] = []
        for x_value in (-3.2, 3.2):
            if abs(ny) > 0.001:
                points.append((x_value, ((-offset - (nx * x_value)) / ny)))
        for y_value in (-3.2, 3.2):
            if abs(nx) > 0.001:
                points.append((((-offset - (ny * y_value)) / nx), y_value))
        visible = [point for point in points if -3.5 <= point[0] <= 3.5 and -3.5 <= point[1] <= 3.5]
        if len(visible) >= 2:
            pygame.draw.line(
                surface,
                color,
                self._to_screen(rect, visible[0][0], visible[0][1]),
                self._to_screen(rect, visible[1][0], visible[1][1]),
                width,
            )

    def _draw_margin_strip(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        best_margin = self._best_margin()
        current_width = (
            0 if best_margin == 0 else round((self._margin() / best_margin) * rect.width)
        )
        current_width = max(0, min(rect.width, current_width))
        pygame.draw.rect(surface, GRID, rect, border_radius=5)
        pygame.draw.rect(
            surface, GOOD, pygame.Rect(rect.x, rect.y, current_width, rect.height), border_radius=5
        )
        self._draw_text(
            surface,
            self._label("current margin vs best", "current margin vs best"),
            (rect.x, rect.y - 22),
            self._font_small,
            MUTED_TEXT,
        )

    def _normal(self, angle: float) -> tuple[float, float]:
        value = radians(angle)
        return (cos(value), sin(value))

    def _signed_distance(self, point: MarginPoint) -> float:
        return self._signed_distance_for(point, self.angle, self.offset)

    def _signed_distance_for(self, point: MarginPoint, angle: float, offset: float) -> float:
        nx, ny = self._normal(angle)
        return (nx * point.x) + (ny * point.y) + offset

    def _predict(self, point: MarginPoint) -> int:
        return 1 if self._signed_distance(point) >= 0 else -1

    def _accuracy(self) -> float:
        correct = sum(1 for point in self.preset.points if self._predict(point) == point.label)
        return correct / len(self.preset.points)

    def _accuracy_label(self) -> str:
        return f"{self._accuracy():.0%}"

    def _margin(self) -> float:
        if self._accuracy() < 1.0:
            return 0.0
        return min(abs(self._signed_distance(point)) for point in self.preset.points)

    def _best_margin(self) -> float:
        return self._margin_for(self.preset.best_angle, self.preset.best_offset)

    def _margin_for(self, angle: float, offset: float) -> float:
        predictions = tuple(
            1 if self._signed_distance_for(point, angle, offset) >= 0 else -1
            for point in self.preset.points
        )
        if any(
            predicted != point.label
            for predicted, point in zip(predictions, self.preset.points, strict=True)
        ):
            return 0.0
        return min(
            abs(self._signed_distance_for(point, angle, offset)) for point in self.preset.points
        )

    def _support_vectors(self) -> tuple[MarginPoint, ...]:
        distances = tuple(abs(self._signed_distance(point)) for point in self.preset.points)
        minimum = min(distances)
        return tuple(
            point
            for point, distance in zip(self.preset.points, distances, strict=True)
            if distance <= minimum + 0.12
        )

    def _support_labels(self) -> str:
        labels = tuple(
            str(index)
            for index, point in enumerate(self.preset.points, start=1)
            if point in self._support_vectors()
        )
        return ", ".join(labels)

    def _diagnosis_key(self) -> str:
        if self._accuracy() < 1.0:
            return "errors"
        if self._margin() >= self._best_margin() - 0.08:
            return "wide"
        return "narrow"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "errors":
            return self._label("classification errors", "błędy klasyfikacji")
        if diagnosis == "wide":
            return self._label("wide margin", "szeroki margin")
        return self._label("narrow margin", "wąski margin")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "errors":
            return self._label(
                "First separate the classes; margin only matters after the split is correct.",
                "Najpierw rozdziel klasy; margin ma sens dopiero przy poprawnym podziale.",
            )
        if diagnosis == "wide":
            return self._label(
                "Support vectors sit closest to the boundary and define the margin.",
                "Support vectors leżą najbliżej boundary i definiują margin.",
            )
        return self._label(
            "The classes are separated, but the closest safe gap is still small.",
            "Klasy są rozdzielone, ale najbliższy bezpieczny gap nadal jest mały.",
        )

    def _boundary_label(self) -> str:
        return f"angle={self.angle:.0f} deg, offset={self.offset:+.1f}"

    def _reset_boundary(self) -> None:
        self.angle = self.preset.start_angle
        self.offset = self.preset.start_offset

    def _to_screen(self, rect: pygame.Rect, x_value: float, y_value: float) -> tuple[int, int]:
        x = rect.x + round((x_value + 3.2) / 6.4 * rect.width)
        y = rect.bottom - round((y_value + 3.2) / 6.4 * rect.height)
        return (x, y)

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        surface.blit(font.render(text, True, color), position)

    def _draw_wrapped(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        *,
        line_height: int,
    ) -> None:
        x, y = position
        current = ""
        for word in text.split():
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
                continue
            if current:
                self._draw_text(surface, current, (x, y), font, color)
                y += line_height
            current = word
        if current:
            self._draw_text(surface, current, (x, y), font, color)

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl
        return en


def create_svm_margin_lab_scene(context: AppContext) -> SVMMarginLabScene:
    """Create the unified shell SVM Margin Lab scene."""
    return SVMMarginLabScene(context)

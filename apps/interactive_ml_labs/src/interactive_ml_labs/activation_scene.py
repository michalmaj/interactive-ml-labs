"""Native Activation Functions Lab scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, tanh
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

INPUT_STEP: Final[float] = 0.25
INPUT_MIN: Final[float] = -4.0
INPUT_MAX: Final[float] = 4.0


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Activation function metadata."""

    name: str
    summary_en: str
    summary_pl: str

    def summary_for_language(self, language: str) -> str:
        """Return localized summary."""
        if language == "pl":
            return self.summary_pl
        return self.summary_en


ACTIVATIONS: Final[tuple[ActivationSpec, ...]] = (
    ActivationSpec(
        name="sigmoid",
        summary_en="Sigmoid squeezes values into 0..1, but large inputs saturate.",
        summary_pl="Sigmoid ściska wartości do 0..1, ale duże wejścia się saturują.",
    ),
    ActivationSpec(
        name="tanh",
        summary_en="tanh centers output around zero and saturates near -1 and 1.",
        summary_pl="tanh centruje output wokół zera i saturuje blisko -1 oraz 1.",
    ),
    ActivationSpec(
        name="ReLU",
        summary_en="ReLU keeps positive inputs and clips negative inputs to zero.",
        summary_pl="ReLU przepuszcza dodatnie wejścia i ucina ujemne do zera.",
    ),
)


class ActivationFunctionsLabScene:
    """Interactive slice for understanding neural network activation functions."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic activation function scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.activation_index = 0
        self.input_value = 0.0

    @property
    def activation(self) -> ActivationSpec:
        """Return the active activation spec."""
        return ACTIVATIONS[self.activation_index]

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
        """Draw the activation lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_curve_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.activation_index = key - pygame.K_1
        elif key == pygame.K_LEFT:
            self.input_value = max(INPUT_MIN, self.input_value - INPUT_STEP)
        elif key == pygame.K_RIGHT:
            self.input_value = min(INPUT_MAX, self.input_value + INPUT_STEP)
        elif key == pygame.K_0:
            self.input_value = 0.0
        elif key == pygame.K_r:
            self.activation_index = 0
            self.input_value = 0.0

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Activation Functions Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Shape a neuron output before building a whole neural network.",
                "Zobacz kształt outputu neuronu, zanim zbudujesz całą neural network.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_curve_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Activation curve", "Krzywa activation"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._activation_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        plot_rect = pygame.Rect(rect.x + 70, rect.y + 104, 560, 270)
        self._draw_activation_curve(surface, plot_rect)
        self._draw_input_slider(surface, pygame.Rect(rect.x + 92, rect.y + 410, 516, 28))
        self._draw_wrapped(
            surface,
            self.activation.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 44),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        rows = (
            (self._label("activation", "activation"), self.activation.name),
            (self._label("input x", "input x"), f"{self.input_value:+.2f}"),
            (self._label("output", "output"), f"{self._output():+.2f}"),
            (self._label("local gradient", "local gradient"), f"{self._gradient():.2f}"),
            (self._label("range", "zakres"), self._range_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {activation.name}", index == self.activation_index)
            for index, activation in enumerate(ACTIVATIONS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Neuron readout", "Odczyt neuronu"),
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
                "Goal: see how activations change signal and gradient flow.",
                "Cel: zobacz, jak activations zmieniają sygnał i przepływ gradientu.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_activation_curve(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)
        zero_y = self._to_screen(rect, 0.0, 0.0)[1]
        zero_x = self._to_screen(rect, 0.0, 0.0)[0]
        pygame.draw.line(surface, MUTED_TEXT, (rect.left, zero_y), (rect.right, zero_y), 1)
        pygame.draw.line(surface, MUTED_TEXT, (zero_x, rect.top), (zero_x, rect.bottom), 1)

        points = [
            self._to_screen(rect, x / 20, self._activation_value(x / 20)) for x in range(-80, 81)
        ]
        pygame.draw.lines(surface, ACCENT, False, points, 3)
        input_position = self._to_screen(rect, self.input_value, self._output())
        pygame.draw.line(
            surface, WARNING, (input_position[0], rect.top), (input_position[0], rect.bottom), 2
        )
        pygame.draw.circle(surface, SECONDARY, input_position, 7)
        self._draw_text(
            surface, "x", (input_position[0] + 8, rect.bottom - 22), self._font_small, WARNING
        )

    def _draw_input_slider(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(
            surface, GRID, pygame.Rect(rect.x, rect.centery - 3, rect.width, 6), border_radius=3
        )
        ratio = (self.input_value - INPUT_MIN) / (INPUT_MAX - INPUT_MIN)
        handle_x = rect.x + round(ratio * rect.width)
        pygame.draw.circle(surface, SECONDARY, (handle_x, rect.centery), 9)
        self._draw_text(
            surface, f"x = {self.input_value:+.2f}", (rect.x, rect.y - 24), self._font_small, TEXT
        )

    def _activation_value(self, value: float) -> float:
        if self.activation.name == "sigmoid":
            return 1 / (1 + exp(-value))
        if self.activation.name == "tanh":
            return tanh(value)
        return max(0.0, value)

    def _output(self) -> float:
        return self._activation_value(self.input_value)

    def _gradient(self) -> float:
        output = self._output()
        if self.activation.name == "sigmoid":
            return output * (1 - output)
        if self.activation.name == "tanh":
            return 1 - (output * output)
        return 1.0 if self.input_value > 0 else 0.0

    def _range_label(self) -> str:
        if self.activation.name == "sigmoid":
            return "0..1"
        if self.activation.name == "tanh":
            return "-1..1"
        return "0..inf"

    def _diagnosis_label(self) -> str:
        if self._gradient() < 0.08:
            return self._label("saturated", "saturacja")
        if self.activation.name == "ReLU" and self.input_value <= 0:
            return self._label("inactive", "nieaktywny")
        return self._label("gradient flows", "gradient płynie")

    def _active_takeaway(self) -> str:
        if self._gradient() < 0.08:
            return self._label(
                "A tiny local gradient means learning will move slowly here.",
                "Mały local gradient oznacza, że uczenie będzie tu poruszać się wolno.",
            )
        if self.activation.name == "ReLU" and self.input_value <= 0:
            return self._label(
                "Negative ReLU input outputs zero and blocks local gradient.",
                "Ujemny input w ReLU daje zero i blokuje local gradient.",
            )
        return self._label(
            "Here the activation passes useful signal and gradient.",
            "Tutaj activation przepuszcza użyteczny sygnał i gradient.",
        )

    def _activation_label(self) -> str:
        return f"{self.activation.name}: x={self.input_value:+.2f}, y={self._output():+.2f}"

    def _to_screen(self, rect: pygame.Rect, x_value: float, y_value: float) -> tuple[int, int]:
        x = rect.x + round((x_value - INPUT_MIN) / (INPUT_MAX - INPUT_MIN) * rect.width)
        y_min = -1.2
        y_max = 4.2 if self.activation.name == "ReLU" else 1.2
        y = rect.bottom - round((y_value - y_min) / (y_max - y_min) * rect.height)
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


def create_activation_functions_lab_scene(context: AppContext) -> ActivationFunctionsLabScene:
    """Create the unified shell Activation Functions Lab scene."""
    return ActivationFunctionsLabScene(context)

"""Native Neural Network Playground scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, tanh
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

WEIGHT_STEP: Final[float] = 0.2
BIAS_STEP: Final[float] = 0.2
WEIGHT_MIN: Final[float] = 0.2
WEIGHT_MAX: Final[float] = 2.2
BIAS_MIN: Final[float] = -1.2
BIAS_MAX: Final[float] = 1.2

ACTIVATIONS: Final[tuple[str, ...]] = ("sigmoid", "tanh", "ReLU")


@dataclass(frozen=True, slots=True)
class NetworkInputPreset:
    """One input/target example for the mini neural network."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    x1: float
    x2: float
    target: int

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


@dataclass(frozen=True, slots=True)
class NetworkForwardPass:
    """Forward-pass values for the mini network."""

    hidden_raw: tuple[float, float]
    hidden_output: tuple[float, float]
    output_logit: float
    probability: float
    loss: float


PRESETS: Final[tuple[NetworkInputPreset, ...]] = (
    NetworkInputPreset(
        name_en="both features low",
        name_pl="obie cechy niskie",
        summary_en="The network should keep probability low for this negative example.",
        summary_pl="Sieć powinna utrzymać niskie probability dla tego negatywnego przykładu.",
        x1=-0.8,
        x2=-0.6,
        target=0,
    ),
    NetworkInputPreset(
        name_en="mixed signal",
        name_pl="mieszany sygnał",
        summary_en="One feature pushes up while the other feature pushes down.",
        summary_pl="Jedna cecha pcha wynik w górę, druga pcha go w dół.",
        x1=0.9,
        x2=-0.5,
        target=1,
    ),
    NetworkInputPreset(
        name_en="both features high",
        name_pl="obie cechy wysokie",
        summary_en="Both inputs carry positive evidence, but activation still shapes it.",
        summary_pl="Oba inputy niosą pozytywny sygnał, ale activation nadal go kształtuje.",
        x1=0.7,
        x2=0.9,
        target=1,
    ),
)


class NeuralNetworkPlaygroundScene:
    """Interactive slice for a tiny neural network forward pass."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic neural network playground."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.activation_index = 0
        self.weight_scale = 1.0
        self.hidden_bias = 0.0

    @property
    def preset(self) -> NetworkInputPreset:
        """Return the active input preset."""
        return PRESETS[self.preset_index]

    @property
    def activation_name(self) -> str:
        """Return the active activation function name."""
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
        """Draw the neural network playground."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_network_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key == pygame.K_a:
            self.activation_index = (self.activation_index + 1) % len(ACTIVATIONS)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.weight_scale = min(WEIGHT_MAX, self.weight_scale + WEIGHT_STEP)
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.weight_scale = max(WEIGHT_MIN, self.weight_scale - WEIGHT_STEP)
        elif key == pygame.K_UP:
            self.hidden_bias = min(BIAS_MAX, self.hidden_bias + BIAS_STEP)
        elif key == pygame.K_DOWN:
            self.hidden_bias = max(BIAS_MIN, self.hidden_bias - BIAS_STEP)
        elif key == pygame.K_r:
            self.preset_index = 0
            self.activation_index = 0
            self.weight_scale = 1.0
            self.hidden_bias = 0.0

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Neural Network Playground", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Follow one forward pass through a tiny hidden layer.",
                "Prześledź jeden forward pass przez małą warstwę hidden.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_network_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Mini network", "Mini network"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._network_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        network_rect = pygame.Rect(rect.x + 54, rect.y + 104, 592, 260)
        self._draw_network(surface, network_rect)
        self._draw_probability_bar(surface, pygame.Rect(rect.x + 94, rect.y + 404, 512, 28))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 44),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        forward = self._forward()
        rows = (
            (self._label("example", "przykład"), self.preset.name_for_language(self._language)),
            (self._label("activation", "activation"), self.activation_name),
            (self._label("x1 / x2", "x1 / x2"), f"{self.preset.x1:+.1f} / {self.preset.x2:+.1f}"),
            (self._label("hidden bias", "hidden bias"), f"{self.hidden_bias:+.1f}"),
            (self._label("weight scale", "weight scale"), f"{self.weight_scale:.1f}x"),
            (self._label("probability", "probability"), f"{forward.probability:.0%}"),
            (self._label("target", "target"), str(self.preset.target)),
            (self._label("loss", "loss"), f"{forward.loss:.2f}"),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Forward pass", "Forward pass"),
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
                "Goal: see how inputs, weights, bias, activation, and loss connect.",
                "Cel: połącz inputs, weights, bias, activation i loss w jeden forward pass.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_network(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        forward = self._forward()
        input_nodes = ((rect.x + 72, rect.y + 78), (rect.x + 72, rect.y + 182))
        hidden_nodes = ((rect.x + 294, rect.y + 70), (rect.x + 294, rect.y + 190))
        output_node = (rect.x + 514, rect.y + 130)

        weights = self._weights()
        for input_index, input_node in enumerate(input_nodes):
            for hidden_index, hidden_node in enumerate(hidden_nodes):
                weight = weights[input_index][hidden_index]
                self._draw_weight_line(surface, input_node, hidden_node, weight)
        self._draw_weight_line(surface, hidden_nodes[0], output_node, 1.0)
        self._draw_weight_line(surface, hidden_nodes[1], output_node, -0.8)

        self._draw_node(surface, input_nodes[0], f"x1\n{self.preset.x1:+.1f}", ACCENT)
        self._draw_node(surface, input_nodes[1], f"x2\n{self.preset.x2:+.1f}", ACCENT)
        self._draw_node(surface, hidden_nodes[0], f"h1\n{forward.hidden_output[0]:+.2f}", SECONDARY)
        self._draw_node(surface, hidden_nodes[1], f"h2\n{forward.hidden_output[1]:+.2f}", SECONDARY)
        self._draw_node(surface, output_node, f"y\n{forward.probability:.0%}", GOOD)

    def _draw_weight_line(
        self,
        surface: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        weight: float,
    ) -> None:
        color = GOOD if weight >= 0 else WARNING
        width = max(1, min(5, round(abs(weight) * 3)))
        pygame.draw.line(surface, color, start, end, width)

    def _draw_node(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.circle(surface, color, center, 30)
        pygame.draw.circle(surface, TEXT, center, 30, width=1)
        lines = label.split("\n")
        for index, line in enumerate(lines):
            rendered = self._font_small.render(line, True, BACKGROUND)
            surface.blit(
                rendered, (center[0] - rendered.get_width() // 2, center[1] - 17 + index * 17)
            )

    def _draw_probability_bar(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        probability = self._forward().probability
        pygame.draw.rect(surface, GRID, rect, border_radius=5)
        fill_width = round(probability * rect.width)
        pygame.draw.rect(
            surface, GOOD, pygame.Rect(rect.x, rect.y, fill_width, rect.height), border_radius=5
        )
        self._draw_text(surface, "0", (rect.x - 20, rect.y + 4), self._font_small, MUTED_TEXT)
        self._draw_text(surface, "1", (rect.right + 8, rect.y + 4), self._font_small, MUTED_TEXT)
        self._draw_text(
            surface,
            f"p(target=1) = {probability:.0%}",
            (rect.x, rect.y - 24),
            self._font_small,
            TEXT,
        )

    def _weights(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (1.2 * self.weight_scale, -0.4 * self.weight_scale),
            (-0.7 * self.weight_scale, 1.1 * self.weight_scale),
        )

    def _forward(self) -> NetworkForwardPass:
        weights = self._weights()
        h1_raw = (
            (self.preset.x1 * weights[0][0]) + (self.preset.x2 * weights[1][0]) + self.hidden_bias
        )
        h2_raw = (
            (self.preset.x1 * weights[0][1]) + (self.preset.x2 * weights[1][1]) - self.hidden_bias
        )
        h1 = self._activation_value(h1_raw)
        h2 = self._activation_value(h2_raw)
        output_logit = (1.0 * h1) + (-0.8 * h2) + 0.2
        probability = 1 / (1 + exp(-output_logit))
        target_probability = probability if self.preset.target == 1 else 1 - probability
        loss = -_safe_log(target_probability)
        return NetworkForwardPass(
            hidden_raw=(h1_raw, h2_raw),
            hidden_output=(h1, h2),
            output_logit=output_logit,
            probability=probability,
            loss=loss,
        )

    def _activation_value(self, value: float) -> float:
        if self.activation_name == "sigmoid":
            return 1 / (1 + exp(-value))
        if self.activation_name == "tanh":
            return tanh(value)
        return max(0.0, value)

    def _diagnosis_label(self) -> str:
        probability = self._forward().probability
        predicted = 1 if probability >= 0.5 else 0
        if predicted == self.preset.target:
            return self._label("correct side", "dobra strona")
        return self._label("wrong side", "zła strona")

    def _active_takeaway(self) -> str:
        forward = self._forward()
        if forward.loss < 0.35:
            return self._label(
                "The forward pass is confident in the target direction.",
                "Forward pass jest pewny w kierunku targetu.",
            )
        if self.activation_name == "ReLU" and min(forward.hidden_output) == 0:
            return self._label(
                "ReLU can turn a hidden unit off when its raw input is negative.",
                "ReLU może wyłączyć hidden unit, gdy jego raw input jest ujemny.",
            )
        return self._label(
            "Change activation or weights and watch probability and loss move.",
            "Zmień activation albo weights i obserwuj zmianę probability oraz loss.",
        )

    def _network_label(self) -> str:
        return (
            f"{self.activation_name}, weight scale={self.weight_scale:.1f}x, "
            f"hidden bias={self.hidden_bias:+.1f}"
        )

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


def _safe_log(value: float) -> float:
    return log(max(0.0001, min(0.9999, value)))


def create_neural_network_playground_scene(context: AppContext) -> NeuralNetworkPlaygroundScene:
    """Create the unified shell Neural Network Playground scene."""
    return NeuralNetworkPlaygroundScene(context)

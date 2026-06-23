"""Native PCA Lab skeleton scene for the unified shell."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.ui_helpers import draw_panel, draw_text, draw_wrapped_text

BACKGROUND: Final[tuple[int, int, int]] = (21, 24, 28)
PANEL: Final[tuple[int, int, int]] = (34, 39, 45)
PLOT_BG: Final[tuple[int, int, int]] = (17, 20, 24)
GRID: Final[tuple[int, int, int]] = (48, 54, 61)
TEXT: Final[tuple[int, int, int]] = (235, 238, 241)
MUTED_TEXT: Final[tuple[int, int, int]] = (165, 172, 181)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
SECONDARY: Final[tuple[int, int, int]] = (246, 181, 111)
POINT: Final[tuple[int, int, int]] = (146, 217, 150)
RESIDUAL: Final[tuple[int, int, int]] = (92, 103, 116)
DEFAULT_PROJECTION_ANGLE_DEGREES: Final[int] = 31
ANGLE_STEP_DEGREES: Final[int] = 5
MIN_NOISE_LEVEL: Final[int] = 0
MAX_NOISE_LEVEL: Final[int] = 5


@dataclass(frozen=True, slots=True)
class PCADataPreset:
    """Dataset preset for the PCA preview."""

    name_en: str
    name_pl: str
    key: str

    def for_language(self, language: str) -> str:
        """Return a localized preset name."""
        if language == "pl":
            return self.name_pl

        return self.name_en


class PCAProjectionMode(StrEnum):
    """Projection control mode used by the PCA scene."""

    MANUAL = "manual"
    FIT = "fit"


DATA_PRESETS: Final[tuple[PCADataPreset, ...]] = (
    PCADataPreset("Linear cloud", "Linear cloud", "linear"),
    PCADataPreset("Noisy cloud", "Noisy cloud", "noisy"),
    PCADataPreset("Two bands", "Two bands", "bands"),
)


class PCALabScene:
    """Interactive first slice for the PCA / dimensionality reduction lab."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic PCA preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.noise_level = 1
        self.sample_seed = 11
        self._points = self._build_preview_points()
        self.projection_angle_degrees = DEFAULT_PROJECTION_ANGLE_DEGREES
        self.projection_mode = PCAProjectionMode.MANUAL
        self.show_residuals = True

    @property
    def preset(self) -> PCADataPreset:
        """Return the active dataset preset."""
        return DATA_PRESETS[self.preset_index]

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
        """Draw the PCA preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_original_space(surface, pygame.Rect(52, 138, 520, 420))
        self._draw_projection(surface, pygame.Rect(620, 138, 300, 420))
        self._draw_explained_variance(surface, pygame.Rect(960, 138, 260, 420))
        self._draw_footer(surface)

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "PCA Lab", (52, 42), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Change the data, rotate the projection, then fit PCA.",
                "Zmieniaj dane, obracaj projekcję, a potem dopasuj PCA.",
            ),
            (52, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_original_space(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            f"{self._label('Data', 'Dane')}: {self.preset.for_language(self._language)}",
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        plot_rect = pygame.Rect(rect.x + 34, rect.y + 72, rect.width - 68, rect.height - 112)
        self._draw_grid(surface, plot_rect)
        fitted_direction = self._direction_for_angle(self._fitted_pca_angle_degrees())
        fitted_start = self._to_screen(
            (-fitted_direction[0] * 0.88, -fitted_direction[1] * 0.88),
            plot_rect,
        )
        fitted_end = self._to_screen(
            (fitted_direction[0] * 0.88, fitted_direction[1] * 0.88),
            plot_rect,
        )
        pygame.draw.line(surface, SECONDARY, fitted_start, fitted_end, 2)
        direction = self._projection_direction()
        axis_start = self._to_screen((-direction[0] * 0.88, -direction[1] * 0.88), plot_rect)
        axis_end = self._to_screen((direction[0] * 0.88, direction[1] * 0.88), plot_rect)
        pygame.draw.line(surface, ACCENT, axis_start, axis_end, 4)
        if self.show_residuals:
            for point in self._points:
                projected = self._project_point_to_active_axis(point)
                pygame.draw.line(
                    surface,
                    RESIDUAL,
                    self._to_screen(point, plot_rect),
                    self._to_screen(projected, plot_rect),
                    1,
                )
        for point in self._points:
            pygame.draw.circle(surface, POINT, self._to_screen(point, plot_rect), 5)
        self._draw_text(
            surface,
            self._label("projection direction", "kierunek projekcji"),
            (plot_rect.x + 14, plot_rect.bottom + 16),
            self._font_small,
            ACCENT,
        )
        self._draw_text(
            surface,
            self._label("PCA best", "PCA best"),
            (plot_rect.x + 174, plot_rect.bottom + 16),
            self._font_small,
            SECONDARY,
        )
        self._draw_text(
            surface,
            f"{self._active_projection_angle_degrees()}°",
            (plot_rect.right - 48, plot_rect.bottom + 16),
            self._font_small,
            MUTED_TEXT,
        )

    def _draw_projection(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("1D projection", "Projekcja 1D"),
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        track = pygame.Rect(rect.x + 38, rect.y + 230, rect.width - 76, 8)
        pygame.draw.rect(surface, GRID, track, border_radius=4)
        sorted_scores = sorted(self._projection_score(point) for point in self._centered_points())
        minimum = sorted_scores[0]
        maximum = sorted_scores[-1]
        span = max(maximum - minimum, 0.001)
        for score in sorted_scores:
            x = track.left + round((score - minimum) * track.width / span)
            pygame.draw.circle(surface, POINT, (x, track.centery), 6)
        self._draw_wrapped(
            surface,
            self._label(
                "The best projection keeps as much spread as possible on this line.",
                "Najlepsza projekcja zachowuje na tej linii jak największy rozrzut.",
            ),
            (rect.x + 22, rect.y + 290),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=20,
        )

    def _draw_explained_variance(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            "Explained variance",
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        kept_variance = self._explained_variance_ratio()
        bars = (
            (kept_variance, ACCENT, self._label("kept", "zostaje")),
            (1.0 - kept_variance, SECONDARY, self._label("lost", "tracimy")),
        )
        y = rect.y + 90
        for value, color, label in bars:
            self._draw_text(surface, label, (rect.x + 22, y + 4), self._font_small, TEXT)
            bar_rect = pygame.Rect(rect.x + 76, y, round((rect.width - 120) * value), 24)
            pygame.draw.rect(surface, color, bar_rect, border_radius=5)
            self._draw_text(
                surface,
                f"{round(value * 100)}%",
                (rect.right - 58, y + 4),
                self._font_small,
                MUTED_TEXT,
            )
            y += 52
        status_y = self._explained_variance_status_start_y(rect)
        for label, value in self._status_rows():
            self._draw_text(
                surface,
                f"{label}: {value}",
                (rect.x + 22, status_y),
                self._font_small,
                TEXT,
            )
            status_y += 18
        help_y = self._explained_variance_help_y(rect)
        self._draw_wrapped(
            surface,
            self._label(
                "C toggles residuals. F fits PCA. Left/Right rotates.",
                "C przełącza residuals. F dopasowuje PCA. Left/Right obraca.",
            ),
            (rect.x + 22, help_y),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=20,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: see what PCA preserves and why fewer dimensions cost information.",
                "Cel: zobacz, co PCA zachowuje i jaki jest koszt mniejszej liczby wymiarów.",
            ),
            (52, 616),
            self._font_body,
            TEXT,
        )

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        draw_panel(surface, rect, PANEL)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self._generate_dataset()
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self._change_noise(-1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self._change_noise(1)
        elif key == pygame.K_LEFT:
            self._rotate_projection(-ANGLE_STEP_DEGREES)
        elif key == pygame.K_RIGHT:
            self._rotate_projection(ANGLE_STEP_DEGREES)
        elif key == pygame.K_f:
            self._toggle_fit_mode()
        elif key == pygame.K_c:
            self.show_residuals = not self.show_residuals
        elif key == pygame.K_r:
            self.projection_mode = PCAProjectionMode.MANUAL
            self.projection_angle_degrees = DEFAULT_PROJECTION_ANGLE_DEGREES
        elif key == pygame.K_n:
            self.sample_seed += 1
            self._generate_dataset()

    def _change_noise(self, delta: int) -> None:
        self.noise_level = max(MIN_NOISE_LEVEL, min(MAX_NOISE_LEVEL, self.noise_level + delta))
        self._generate_dataset()

    def _generate_dataset(self) -> None:
        self._points = self._build_preview_points()

    def _rotate_projection(self, delta_degrees: int) -> None:
        if self.projection_mode == PCAProjectionMode.FIT:
            self.projection_angle_degrees = self._fitted_pca_angle_degrees()
            self.projection_mode = PCAProjectionMode.MANUAL
        self.projection_angle_degrees = (self.projection_angle_degrees + delta_degrees) % 180

    def _toggle_fit_mode(self) -> None:
        if self.projection_mode == PCAProjectionMode.FIT:
            self.projection_mode = PCAProjectionMode.MANUAL
            return

        self.projection_mode = PCAProjectionMode.FIT

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.left + round(step * rect.width / 4)
            y = rect.top + round(step * rect.height / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        draw_text(surface, text, position, font, color)

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
        draw_wrapped_text(
            surface,
            text,
            position,
            width,
            font,
            color,
            line_height=line_height,
        )

    def _to_screen(self, point: tuple[float, float], rect: pygame.Rect) -> tuple[int, int]:
        x, y = point
        return (
            rect.left + round((x + 1.0) * 0.5 * rect.width),
            rect.top + round((1.0 - (y + 1.0) * 0.5) * rect.height),
        )

    def _projection_score(self, point: tuple[float, float]) -> float:
        x, y = point
        direction_x, direction_y = self._projection_direction()
        return direction_x * x + direction_y * y

    def _projection_direction(self) -> tuple[float, float]:
        return self._direction_for_angle(self._active_projection_angle_degrees())

    def _direction_for_angle(self, angle_degrees: int) -> tuple[float, float]:
        angle = math.radians(angle_degrees)
        return (math.cos(angle), math.sin(angle))

    def _active_projection_angle_degrees(self) -> int:
        if self.projection_mode == PCAProjectionMode.FIT:
            return self._fitted_pca_angle_degrees()

        return self.projection_angle_degrees

    def _fitted_pca_angle_degrees(self) -> int:
        centered = self._centered_points()
        variance_x = sum(x * x for x, _y in centered)
        variance_y = sum(y * y for _x, y in centered)
        covariance_xy = sum(x * y for x, y in centered)
        angle_radians = 0.5 * math.atan2(2.0 * covariance_xy, variance_x - variance_y)
        return round(math.degrees(angle_radians)) % 180

    def _centered_points(self) -> tuple[tuple[float, float], ...]:
        mean_x = sum(point[0] for point in self._points) / len(self._points)
        mean_y = sum(point[1] for point in self._points) / len(self._points)
        return tuple((point[0] - mean_x, point[1] - mean_y) for point in self._points)

    def _point_mean(self) -> tuple[float, float]:
        return (
            sum(point[0] for point in self._points) / len(self._points),
            sum(point[1] for point in self._points) / len(self._points),
        )

    def _project_point_to_active_axis(self, point: tuple[float, float]) -> tuple[float, float]:
        mean_x, mean_y = self._point_mean()
        centered_point = (point[0] - mean_x, point[1] - mean_y)
        score = self._projection_score(centered_point)
        direction_x, direction_y = self._projection_direction()
        return (mean_x + score * direction_x, mean_y + score * direction_y)

    def _explained_variance_ratio(self) -> float:
        centered = self._centered_points()
        total_variance = sum(x * x + y * y for x, y in centered)
        if total_variance <= 0.0:
            return 0.0

        projected_variance = sum(self._projection_score(point) ** 2 for point in centered)
        return max(0.0, min(1.0, projected_variance / total_variance))

    def _status_rows(self) -> tuple[tuple[str, str], ...]:
        kept = round(self._explained_variance_ratio() * 100)
        if self.show_residuals:
            residuals_label = self._label("on", "wł.")
        else:
            residuals_label = self._label("off", "wył.")
        return (
            (self._label("mode", "tryb"), self._mode_label()),
            (self._label("data", "dane"), self.preset.for_language(self._language)),
            (self._label("noise", "noise"), str(self.noise_level)),
            (self._label("residuals", "residuals"), residuals_label),
            (self._label("angle", "kąt"), f"{self._active_projection_angle_degrees()}°"),
            (self._label("kept variance", "zachowana wariancja"), f"{kept}%"),
            (self._label("recon error", "błąd rekonstrukcji"), f"{100 - kept}%"),
        )

    def _mode_label(self) -> str:
        if self.projection_mode == PCAProjectionMode.FIT:
            return "fit PCA"

        return "manual"

    def _explained_variance_status_start_y(self, rect: pygame.Rect) -> int:
        bars_bottom = rect.y + 90 + 2 * 52
        return bars_bottom + 18

    def _explained_variance_help_y(self, rect: pygame.Rect) -> int:
        return self._explained_variance_status_start_y(rect) + len(self._status_rows()) * 18 + 12

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl

        return en

    def _build_preview_points(self) -> tuple[tuple[float, float], ...]:
        rng = random.Random(self.sample_seed + self.preset_index * 1009 + self.noise_level * 37)
        noise_scale = self.noise_level * 0.035
        points: list[tuple[float, float]] = []
        for index in range(36):
            t = -0.92 + index * 1.84 / 35
            base_noise = rng.gauss(0.0, noise_scale)
            if self.preset.key == "bands":
                band = -0.25 if index % 2 == 0 else 0.25
                x = t + rng.gauss(0.0, noise_scale * 0.45)
                y = 0.44 * t + band + base_noise
            elif self.preset.key == "noisy":
                x = t + rng.gauss(0.0, noise_scale)
                y = 0.36 * t + 0.18 * math.sin(index * 1.3) + base_noise * 1.5
            else:
                x = t + rng.gauss(0.0, noise_scale * 0.45)
                y = 0.56 * t + 0.12 * math.sin(index * 1.7) + base_noise
            points.append((_clamp(x, -0.95, 0.95), _clamp(y, -0.95, 0.95)))
        return tuple(points)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def create_pca_lab_scene(context: AppContext) -> PCALabScene:
    """Create the unified shell PCA Lab scene."""
    return PCALabScene(context)

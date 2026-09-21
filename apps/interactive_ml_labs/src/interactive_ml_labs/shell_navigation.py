"""Navigation rules for the unified app shell."""

from __future__ import annotations

from enum import StrEnum


class ScreenName(StrEnum):
    """Top-level shell screens."""

    LANGUAGE = "language"
    HOME = "home"
    COURSE_MAP = "course_map"
    PATHS = "paths"
    LESSONS = "lessons"
    LEVELS = "levels"
    DEMOS = "demos"
    INTRO = "intro"
    THEORY = "theory"
    DEMO = "demo"
    LESSON_COMPLETE = "lesson_complete"
    PATH_COMPLETE = "path_complete"
    PROGRESS_REPORT = "progress_report"
    BADGES = "badges"
    SETTINGS = "settings"
    PAUSE = "pause"


DEFAULT_BACK_TARGETS: dict[ScreenName, ScreenName] = {
    ScreenName.HOME: ScreenName.LANGUAGE,
    ScreenName.COURSE_MAP: ScreenName.HOME,
    ScreenName.PATHS: ScreenName.HOME,
    ScreenName.LESSONS: ScreenName.COURSE_MAP,
    ScreenName.LEVELS: ScreenName.HOME,
    ScreenName.DEMOS: ScreenName.LEVELS,
    ScreenName.INTRO: ScreenName.DEMOS,
    ScreenName.LESSON_COMPLETE: ScreenName.LESSONS,
    ScreenName.PATH_COMPLETE: ScreenName.PATHS,
    ScreenName.PROGRESS_REPORT: ScreenName.HOME,
    ScreenName.BADGES: ScreenName.HOME,
}

SETTINGS_BLOCKED_SCREENS: frozenset[ScreenName] = frozenset(
    {
        ScreenName.DEMO,
        ScreenName.SETTINGS,
        ScreenName.THEORY,
    },
)
THEORY_ALLOWED_SCREENS: frozenset[ScreenName] = frozenset(
    {
        ScreenName.INTRO,
        ScreenName.DEMO,
        ScreenName.PAUSE,
    },
)


def default_back_target(screen_name: ScreenName) -> ScreenName | None:
    """Return the default Back/Esc target for a screen, if it has one."""
    return DEFAULT_BACK_TARGETS.get(screen_name)


def can_open_settings(screen_name: ScreenName) -> bool:
    """Return whether the settings screen can be opened from a screen."""
    return screen_name not in SETTINGS_BLOCKED_SCREENS


def can_open_theory(screen_name: ScreenName) -> bool:
    """Return whether the theory screen can be opened from a screen."""
    return screen_name in THEORY_ALLOWED_SCREENS

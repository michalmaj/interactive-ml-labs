"""Tests for unified shell navigation rules."""

from interactive_ml_labs.shell_navigation import (
    ScreenName,
    can_open_settings,
    can_open_theory,
    default_back_target,
)


def test_default_back_target_covers_menu_flow() -> None:
    """Back targets should describe the main menu hierarchy."""
    assert default_back_target(ScreenName.HOME) == ScreenName.LANGUAGE
    assert default_back_target(ScreenName.COURSE_MAP) == ScreenName.HOME
    assert default_back_target(ScreenName.PATHS) == ScreenName.HOME
    assert default_back_target(ScreenName.LESSONS) == ScreenName.COURSE_MAP
    assert default_back_target(ScreenName.LEVELS) == ScreenName.HOME
    assert default_back_target(ScreenName.DEMOS) == ScreenName.LEVELS
    assert default_back_target(ScreenName.INTRO) == ScreenName.DEMOS
    assert default_back_target(ScreenName.BADGES) == ScreenName.HOME


def test_default_back_target_leaves_modal_screens_to_shell_state() -> None:
    """Stateful modal screens should be handled by the app shell itself."""
    assert default_back_target(ScreenName.LANGUAGE) is None
    assert default_back_target(ScreenName.THEORY) is None
    assert default_back_target(ScreenName.DEMO) is None
    assert default_back_target(ScreenName.SETTINGS) is None
    assert default_back_target(ScreenName.PAUSE) is None


def test_settings_opening_is_blocked_for_stateful_screens() -> None:
    """Settings should avoid interrupting active demo/theory/settings screens."""
    assert can_open_settings(ScreenName.HOME) is True
    assert can_open_settings(ScreenName.LEVELS) is True
    assert can_open_settings(ScreenName.DEMO) is False
    assert can_open_settings(ScreenName.SETTINGS) is False
    assert can_open_settings(ScreenName.THEORY) is False


def test_theory_opening_is_limited_to_lesson_contexts() -> None:
    """Theory should only open from screens with a selected lesson context."""
    assert can_open_theory(ScreenName.INTRO) is True
    assert can_open_theory(ScreenName.DEMO) is True
    assert can_open_theory(ScreenName.PAUSE) is True
    assert can_open_theory(ScreenName.HOME) is False
    assert can_open_theory(ScreenName.LEVELS) is False

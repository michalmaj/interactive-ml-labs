"""Smoke tests for the Boosting Mistake Lab Pygame UI modules."""

from boosting_mistake_lab.pygame_app import BoostingPygameApp
from boosting_mistake_lab.renderer import BoostingRenderer, BoostingRenderState


def test_pygame_ui_modules_are_importable() -> None:
    """Pygame UI modules should be importable."""

    assert BoostingPygameApp is not None
    assert BoostingRenderer is not None
    assert BoostingRenderState is not None

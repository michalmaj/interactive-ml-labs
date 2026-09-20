"""Small shared shell types."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(slots=True)
class MenuItem:
    """Clickable/selectable menu item."""

    label: str
    rect: pygame.Rect
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class BadgeItem:
    """One visible badge in the guided learning gallery."""

    label: str
    unlocked: bool

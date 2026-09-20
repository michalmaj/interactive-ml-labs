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

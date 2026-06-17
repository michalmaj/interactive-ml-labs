"""Tests for the native Anomaly Detection Lab scene."""

import pygame
from interactive_ml_labs.anomaly_detection_scene import (
    DEFAULT_THRESHOLD_INDEX,
    PRESETS,
    AnomalyDetectionLabScene,
    create_anomaly_detection_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_anomaly_detection_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Anomaly Detection scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_anomaly_detection_lab_scene(AppContext())

        assert isinstance(scene, AnomalyDetectionLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_anomaly_detection_scene_updates_threshold_and_resets(monkeypatch) -> None:
    """Threshold keys, preset keys, score toggle, and R should update the preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_anomaly_detection_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX
        assert scene._threshold_label() == "60% (2/3)"
        assert scene._precision_label()
        assert scene._recall_label()

        default_counts = scene.confusion_counts()
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

        assert command.kind == SceneCommandKind.NONE
        assert scene._threshold_label() == "45% (1/3)"
        assert scene.confusion_counts().false_positive >= default_counts.false_positive
        assert scene.confusion_counts().false_negative <= default_counts.false_negative

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset is PRESETS[2]
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
        assert scene.show_scores is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX
        assert scene.show_scores is True
    finally:
        pygame.quit()


def test_anomaly_detection_scene_strict_threshold_misses_more_anomalies(monkeypatch) -> None:
    """Raising threshold should reduce alert volume and usually miss more anomalies."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_anomaly_detection_lab_scene(AppContext())
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        loose_counts = scene.confusion_counts()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        strict_counts = scene.confusion_counts()

        loose_alerts = loose_counts.true_positive + loose_counts.false_positive
        strict_alerts = strict_counts.true_positive + strict_counts.false_positive

        assert strict_alerts <= loose_alerts
        assert strict_counts.false_negative >= loose_counts.false_negative
    finally:
        pygame.quit()


def test_anomaly_detection_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize operational labels while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_anomaly_detection_lab_scene(context)

        assert scene._diagnosis_label() in {
            "pominięte anomalie",
            "szum alertów",
            "użyteczny threshold",
        }
        assert "Threshold" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_anomaly_detection_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The anomaly detection preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_anomaly_detection_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()

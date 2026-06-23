# Unified App Shell

This document describes the unified Pygame application shell for Interactive ML Labs.

The project has moved from a collection of separately launched demos toward one guided educational application, while keeping existing demo packages, entry points, tests, and workflows intact.

## Entry Point

The recommended guided entry point is:

```text
interactive-ml-labs
```

From the workspace:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

The guided flow is:

```text
Interactive ML Labs
-> language and settings
-> level selection
-> demo selection
-> demo intro screen
-> theory or experiment
-> pause menu / help / objectives / controls
```

Individual demos still remain runnable on their own for development, testing, teaching, and debugging.

## Architecture

The shell lives as a separate application package:

```text
apps/interactive_ml_labs/
```

It is intentionally not part of `ml_lab_core`. `ml_lab_core` should stay focused on reusable interfaces and small shared utilities. The shell is a concrete application with concrete navigation, settings, manifests, and Pygame behavior.

## Existing Entry Points

Existing demo entry points stay.

Examples:

```text
boosting-mistake-lab-ui
gradient-descent-playground
knn-vote-map
```

The unified shell is the recommended guided experience, but it does not delete or replace standalone demo launchers.

This keeps the project safer to evolve:

- old workflows keep working,
- demos remain easy to debug in isolation,
- tests do not need to be rewritten at once,
- each demo integration can happen in a separate pull request.

## Scene Contract

Shell scenes follow a small lifecycle:

```python
class Scene:
    def handle_event(self, event) -> SceneCommand: ...
    def update(self, dt) -> SceneCommand: ...
    def render(self, surface): ...
```

Current behavior:

- scenes return `SceneCommand`,
- demo scenes are owned by a stack-based `SceneManager`,
- `Esc` and `H` work as global shell shortcuts before demo event handling,
- demo scenes can request pause, restart, back to demo selection, or quit,
- shell overlays use manifest content for objectives, controls, tips, and theory.

## App Context And Settings

The shell uses a small app context for global state.

Important settings:

- language,
- resolution,
- fullscreen,
- adaptive window size,
- fixed-scene scaling,
- sound placeholder,
- current level and selected demo,
- theme.

Settings are mutable through the shell and persisted between launches. Demos may read settings, but global settings should still be changed through explicit shell actions.

## Localization

Localization stays deliberately simple.

Current approach:

- Python dataclasses and dictionaries for app strings,
- global language setting,
- English and Polish UI,
- Polish-capable fonts for diacritics,
- no heavy i18n framework.

Polish UI copy should sound natural rather than literal.

Style rules:

- keep algorithm names in English when they are recognizable names, such as `k-NN`, `Random Forest`, or `Boosting`,
- keep common ML terms in English when Polish versions sound forced, such as `learning rate`, `loss`, `split`, or `decision boundary`,
- translate surrounding explanations into natural Polish,
- prefer short UI phrases over exact sentence-by-sentence translation.

## Demo Manifest

Each demo is described by a manifest.

Core fields:

```python
id
level
title
summary
objectives
controls
difficulty
tags
theory
create_scene
```

The manifest registry drives level and demo selection. The shell should not hardcode level contents in multiple places.

Current state:

- Level 1 has 10 demos,
- Level 2 has 10 demos,
- Level 3 has 7 demos,
- all current demos provide theory content,
- all current demos can be launched through the shell.

## Intro, Theory, Pause, And Help

The shell generates consistent screens and overlays from manifest metadata.

Intro screens show:

- title,
- summary,
- learning objectives,
- controls,
- difficulty,
- tags,
- start and theory actions.

Theory screens give students enough context without forcing them to switch to Markdown files during a live session.

Global behavior:

```text
Esc = pause menu
H = help overlay
T = theory screen from intro/demo flow
L = toggle language
```

Pause and help content is owned by the shell. Demos provide content through their manifests and scenes.

## Resolution, Scaling, And Sound

Resolution is represented in settings.

Current default:

```text
1280x720
```

Supported windowed presets:

```text
1280x720
1320x780
1600x900
1920x1080
```

Fullscreen is optional, not the default.

The settings menu includes:

- adaptive window size,
- fullscreen,
- fixed-scene scaling.

Fixed-size demo scenes are centered or scaled inside the available app window instead of forcing a renderer rewrite.

Sound remains deferred. The settings model may carry a `sound_enabled` placeholder, but there are no audio assets or mixer behavior yet.

## Existing Renderers

Existing demo renderers stay standalone.

They do not need to inherit from a shared renderer base class.

The shell provides a Pygame surface and scene lifecycle. Demo renderers can keep their own drawing logic.

## Testing And Workflow

The migration preserves the current workflow.

Rules:

- old tests stay,
- old entry points stay,
- shell tests stay first-class,
- every demo integration should be a separate pull request,
- no single commit should rewrite all demos,
- algorithmic code should remain testable without opening a Pygame window.

## Current Decisions

The following are accepted as current direction:

- The shell is a separate application package.
- Existing demo entry points remain.
- The shell generates default intro, theory, pause, and help screens from manifests.
- Language is global.
- Localization uses simple Python data structures.
- Settings are persisted.
- Default app resolution is `1280x720`.
- Adaptive sizing, fullscreen, and fixed-scene scaling are opt-in settings.
- Sound is deferred.
- Existing renderers remain standalone.
- Level selection is dynamic from manifests.
- Demo scenes use `SceneCommand` and `SceneManager` for shell navigation.
- Keyboard and mouse support are both part of the shell experience.

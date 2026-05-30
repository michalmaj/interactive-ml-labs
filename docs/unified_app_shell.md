# Unified App Shell

This document describes the planned unified Pygame application shell for Interactive ML Labs.

The goal is to move from a collection of separately launched demos toward one guided educational application, while keeping existing demo packages, entry points, tests, and workflows intact.

## Vision

Interactive ML Labs should eventually have one recommended entry point:

```text
interactive-ml-labs
```

The guided flow should be:

```text
Interactive ML Labs
-> language selection
-> level selection
-> demo selection
-> demo intro screen
-> game / simulation / experiment
-> pause menu / help / objectives / controls
```

Individual demos should still remain runnable on their own for development, testing, teaching, and debugging.

## Architectural Direction

The unified shell should be added as a separate application package, not folded into `ml_lab_core` at the beginning.

Preferred initial location:

```text
apps/interactive_ml_labs/
```

Alternative considered locations:

- `packages/ml_lab_app/`
- `packages/ml_lab_core/src/ml_lab_core/ui/`

The shell should not become part of `ml_lab_core` too early. `ml_lab_core` should stay focused on reusable interfaces and small shared utilities. The shell is a concrete application with concrete navigation, settings, and Pygame behavior.

## Existing Entry Points

Existing demo entry points should stay.

Examples:

```text
boosting-mistake-lab-ui
gradient-descent-playground
knn-vote-map
```

The unified shell should become the recommended guided experience, but it should not delete or replace standalone demo launchers.

This keeps the project safer to evolve:

- old workflows keep working,
- demos remain easy to debug in isolation,
- tests do not need to be rewritten at once,
- each demo integration can happen in a separate pull request.

## Demo Integration Strategy

The first integration should avoid destructive rewrites.

Preferred approach:

1. Add a shell-level demo scene contract.
2. Add a manifest registry.
3. Integrate one demo through an adapter or thin scene wrapper.
4. Keep the existing `pygame_app.py` entry point working.

Boosting Mistake Lab is a good first real integration target because it already has a mature Pygame UI, challenge mode, explanation text, export behavior, and tests.

## Scene Contract

The minimal scene contract should look like this:

```python
class Scene:
    def handle_event(self, event) -> SceneCommand: ...
    def update(self, dt) -> SceneCommand: ...
    def render(self, surface): ...
```

Implemented first slice:

- scenes return `SceneCommand`,
- demo scenes are owned by a small stack-based `SceneManager`,
- `Esc` and `H` stay global shell shortcuts before demo event handling,
- demo scenes can request pause, restart, back to demo selection, or quit.

Still open:

- whether scene-specific state should survive deeper overlay stacks,
- how export commands should be represented,
- how much of the shell menu system should eventually become scene-based too.

## App Context

The initial `AppContext` should stay small and in memory.

Likely fields:

```python
language
resolution
sound_enabled
current_level
selected_demo
theme
```

Initial rules:

- settings are mutable in memory,
- no config file in the first slice,
- demos may read settings,
- demos should not freely mutate global settings unless the shell exposes an explicit action.

Persistent settings can be added later.

## Localization

Localization should start simple.

Preferred initial options:

- dataclasses with `pl` and `en` fields,
- Python dictionaries for shell strings,
- no full i18n framework.

The language should be global, not per-demo.

Open decisions:

- default language: `en` or `pl`,
- whether `L` changes language at runtime,
- whether language selection appears before the main menu,
- which technical terms should stay in English.

Current direction:

- global language setting,
- language chosen in the shell,
- runtime language switching can be added if it stays simple.

## Demo Manifest

Each demo should be described by a manifest.

Minimal fields:

```python
id
level
title
summary
objectives
controls
create_scene
```

Likely near-term extensions:

```python
difficulty
tags
```

The manifest registry should drive level and demo selection. The shell should not hardcode level contents in multiple places.

## Intro Screen

The shell should generate the default demo intro screen from the demo manifest.

This keeps the experience consistent across demos.

The generated intro screen can show:

- title,
- summary,
- learning objectives,
- controls,
- difficulty,
- tags,
- start action.

Later, individual demos may override the default intro if they need a special presentation.

## Pause And Help

Global behavior should be consistent:

```text
Esc = pause menu
H = help overlay
```

The pause menu should eventually contain:

- Resume,
- Help,
- Objectives,
- Controls,
- Restart,
- Back to demos,
- Quit.

The shell should own the overlay behavior. Demos should provide content through their manifest or a small help provider.

## Resolution And Sound

Resolution should be represented in settings from the beginning, but the first implementation can support only one fixed resolution.

Initial direction:

```text
1280x720
```

Fullscreen and multiple resolutions can come later.

Sound should not be implemented in the first slice. A `sound_enabled` placeholder may exist in settings if it helps the architecture, but no audio assets or mixer behavior are required yet.

## Fonts And Polish Text

The shell should support Polish diacritics in UI text.

Current direction:

- keep source files as UTF-8,
- use real Polish text in PL strings,
- create UI fonts through a helper that prefers fonts with broad Unicode coverage,
- add a bundled font later if system font selection becomes inconsistent across platforms.

## Existing Renderers

Existing demo renderers should stay standalone.

They do not need to inherit from a shared renderer base class.

The shell should provide a Pygame surface and scene lifecycle. Demo renderers can keep their own drawing logic.

## Testing And Workflow

The migration must preserve the current workflow.

Rules:

- old tests stay,
- old entry points stay,
- new shell gets its own smoke and unit tests,
- every demo integration should be a separate pull request,
- no single commit should rewrite all demos,
- algorithmic code should remain testable without opening a Pygame window.

## Vertical Slice Plan

Recommended pull request sequence:

1. Add minimal shell skeleton with placeholder screens.
2. Add demo manifest and registry.
3. Integrate Boosting Mistake Lab through an adapter or scene wrapper.
4. Add generated intro screen from manifest.
5. Add pause/help overlay.
6. Add language switching and translated shell strings.
7. Integrate additional demos one by one.

The first implementation request should be intentionally small:

```text
Implement the minimal app shell skeleton only.
```

It should not attempt to implement the entire unified app at once.

## Naming

Preferred entry point:

```text
interactive-ml-labs
```

This is the most natural name for the guided application.

## Current Decisions

The following are accepted as initial direction, unless later implementation work proves otherwise:

- The shell is a separate application package.
- Existing demo entry points remain.
- Boosting Mistake Lab is the first serious integration candidate.
- The shell generates default intro screens from manifests.
- Language is global.
- Localization starts with simple Python data structures.
- Settings are in memory at first.
- One fixed resolution is enough for the first slice.
- Sound is deferred.
- Existing renderers remain standalone.
- Level selection is dynamic from manifests.
- Demo scenes use `SceneCommand` and `SceneManager` for shell navigation.
- Keyboard support comes first; mouse support should be added early.

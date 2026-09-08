# App Shell Decomposition

This document tracks the main post-alpha technical debt in the unified app shell:
`apps/interactive_ml_labs/src/interactive_ml_labs/pygame_app.py` and its
`UnifiedAppShell` class.

The shell works today and is covered by tests, but it has grown into a large
coordination point. That makes it harder to add course features without touching
many unrelated behaviors at once.

## Current Shape

`UnifiedAppShell` currently owns several responsibilities:

- the Pygame event loop,
- screen routing and scene-manager integration,
- shell screen rendering,
- keyboard and mouse input,
- scroll state for multiple panels,
- settings and progress persistence,
- language switching,
- intro, theory, pause, help, badge, concept-check, and course-map flows.

This is acceptable for the `v0.0.9a` alpha because the app is usable, the release
scope is student-facing, and broad refactoring would add risk late in the cycle.
It should still be treated as technical debt number one after the alpha release.

## Goals

- Make app-shell changes smaller and easier to review.
- Keep behavior stable while moving code out of `pygame_app.py`.
- Preserve the current manifest and `Scene` contracts.
- Keep standalone demo entry points working.
- Add focused tests around each extracted helper or module.
- Make future course features easier to add without editing one central class.

## Non-Goals

- Do not rewrite the app shell in one pull request.
- Do not move the shell into `ml_lab_core`.
- Do not force all demo scenes to inherit from one renderer base class.
- Do not remove standalone demo entry points.
- Do not solve conceptual-demo accuracy work here, such as replacing simplified
  t-SNE / UMAP or data-leakage visualizations with full algorithms.
- Do not introduce packaged student distribution as part of this refactor.

## Proposed Boundaries

Possible extraction targets:

- `shell_navigation.py`: screen names, route requests, back-stack behavior, and
  selected course/path/demo state transitions.
- `shell_state.py`: selected indices, scroll offsets, drag state, and other
  mutable UI state that is not part of persisted progress.
- `shell_scrolling.py`: scroll bounds, wheel handling, scrollbar geometry, and
  click/drag calculations.
- `shell_persistence.py`: loading and saving app settings and learning progress.
- `screens/`: screen-specific renderers for home, course map, guided paths,
  level selection, demo selection, intro, theory, settings, badges, pause, and
  concept checks.
- `shell_input.py`: mapping keyboard and mouse events to shell-level commands.
- `shell_text.py`: shell-owned copy/layout helpers, kept separate from demo scene
  UI helpers.

The exact filenames can change during implementation. The important rule is that
each extraction should create a clear boundary and reduce the amount of unrelated
state in `UnifiedAppShell`.

## Incremental Sequence

1. Extract pure helpers first.
   Start with functions that do not need Pygame display state: scrollbar
   calculations, scroll clamping, selected-item visibility, and small text-layout
   helpers.

2. Extract persistence behind a narrow facade.
   Move settings/progress file handling out of `UnifiedAppShell`, while keeping
   the public data model unchanged.

3. Extract one screen renderer at a time.
   Begin with less coupled screens such as badges, concept checks, or settings.
   Each move should keep the same keyboard/mouse behavior.

4. Extract navigation state after renderers are smaller.
   Screen transitions are easier to isolate once fewer render methods depend on
   app-wide mutable fields.

5. Revisit registry dependencies last.
   The app currently imports the global registry directly. Dependency injection
   may help tests later, but it should come after the shell has smaller modules.

## Acceptance Checks For Each Refactor PR

- Existing app behavior remains unchanged.
- Standalone demo entry points remain unchanged.
- `uv run ruff format --check .` passes.
- `uv run ruff check .` passes.
- `uv run --all-packages pytest` passes.
- New extracted pure helpers get focused tests.
- Layout-sensitive shell behavior keeps at least one regression test when moved.
- The PR description names which responsibility moved and what intentionally
  stayed in `UnifiedAppShell`.

## Release Position

This is post-`v0.0.9a` engineering work. The alpha should not be blocked on this
decomposition as long as the current app remains stable and release checks pass.

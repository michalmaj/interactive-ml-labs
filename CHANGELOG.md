# Changelog

All notable changes to this project will be documented in this file.

The project follows a lightweight changelog format inspired by Keep a Changelog.

## Unreleased

### Planned

- Prepare final `v0.0.9a` release assets and GitHub release notes.
- Add screenshots or short GIFs when the first public-facing alpha is published.
- Start the post-alpha student distribution track.
- Decompose the large unified app shell entry point in small post-alpha pull
  requests.

## v0.0.9a - Student-Facing Alpha

### Added

- Unified Pygame app as the recommended guided learning experience.
- Course map above individual guided paths.
- Five guided learning paths with lesson tasks, theory status, progress, badges,
  completion summaries, and concept checks.
- Built-in theory screens, generated intros, pause menus, and help overlays.
- Level 1, Level 2, and Level 3 demos available through the unified app.
- Persistent app settings and learning progress.
- English and Polish shell copy.
- Classroom comfort settings for larger text, high contrast, and a
  colorblind-friendly palette.
- MIT license, GitHub issue forms, and release-readiness docs.
- App shell decomposition plan for the main post-alpha technical debt.

### Changed

- Root README now shows first-run requirements, `uv sync`, and the recommended
  app launch command near the top.
- App documentation now treats the unified app as the main student path while
  keeping standalone demo entry points supported.
- Native app-only scenes now share small UI helpers for repeated panel, text, and
  readout patterns.

### Kept

- Original standalone demo commands remain supported for focused teaching,
  debugging, and development workflows.

### Known Limits

- The alpha is still local-first and launched with `uv run`.
- Packaged Windows, macOS, and Linux builds are planned after this release.
- Some advanced demos are conceptual visualizations and should be documented as
  such when the final release notes are prepared.

### Verification

- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run --all-packages pytest`

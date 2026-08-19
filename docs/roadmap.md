# Roadmap

This roadmap describes the development direction for Interactive ML Labs.

The project is intentionally developed in small pull requests. Each pull request should introduce one logical change and keep existing standalone demo workflows intact.

## Current Status

Interactive ML Labs now has a unified Pygame app as the recommended guided experience:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Current registry coverage:

- Level 1: 10 fundamentals demos,
- Level 2: 10 practical ML demos,
- Level 3: 7 advanced/showcase demos.

Current guided lesson coverage:

- Level 1: 7 guided lessons,
- Level 2: 11 guided lessons,
- Level 3: 4 guided lessons.

The shell currently supports:

- English and Polish UI,
- level and demo selection from manifests,
- scrollable menu lists,
- generated intro screens,
- built-in theory screens,
- pause and help overlays,
- guided learning paths with lesson tasks, prerequisite/next-lesson guidance, completion badges, visible theory status, task checklists in intro/pause flow, and persisted progress,
- fullscreen, adaptive window size, and fixed-scene scaling settings,
- persistent app settings,
- standalone demo entry points for existing demo packages.

CI runs Ruff, the workspace-aware root pytest suite, package-specific app/core
tests, and the standalone demo package tests.
The root pytest suite also checks that local artifacts such as caches, bytecode,
macOS metadata, and accidental shell files are not tracked in git.

## Near-Term Milestones

- [x] Run app tests in CI and root pytest.
- [x] Keep project docs aligned with the current unified app.
- [x] Clean up small repository artifacts and naming drift.
- [x] Extract shared UI helpers for repeated demo-scene panel, text, and wrapping patterns.
- [x] Introduce learning paths, lesson/task manifests, progress, and meaningful completion badges.
- [x] Expand learning paths beyond the first end-to-end path.
- [x] Add checkable task completion to the second learning path.
- [x] Implement the third guided path: From good scores to trustworthy models.
- [x] Implement the fourth guided path: From features to model decisions.
- [x] Plan the first Level 3 guided path: From representation to model behavior.
- [x] Add the first Level 3 lesson hooks for Explained Variance and t-SNE / UMAP.
- [x] Add Time Series Forecasting lesson hooks for the planned Level 3 path.
- [x] Register the Level 3 guided path with shared Calibration and Monitoring lessons.
- [x] Run a Polish copy and shell-details pass for the Level 3 guided path.
- [x] Complete the `v0.0.9a` product prerequisites: course map, concept checks, and comfort settings.
- [x] Prepare `v0.0.9a` release docs and student alpha notes.
- [ ] Prepare the `v0.0.9a` student-facing alpha release.

The current release plan is captured in [release_v0_0_9a.md](release_v0_0_9a.md).
The release runbook is captured in
[release_checklist_v0_0_9a.md](release_checklist_v0_0_9a.md). The next priority
is final release preparation: license, issue templates, repository metadata, and
the GitHub tag/release.

Required product work before `v0.0.9a`:

- [x] Add a course-level map above individual learning paths.
- [x] Improve student feedback with concept checks and stronger summaries.
- [x] Add accessibility and classroom comfort settings.

Release preparation after those product slices:

- prepare `v0.0.9a` release docs and student alpha notes,
- add a license and GitHub issue templates for student feedback,
- document repository metadata and GitHub home-page settings,
- add screenshots or short GIFs for the main README and app docs,
- refresh the README for the release state,
- prepare repository metadata and wiki structure.

## Next Engineering Themes

### Learning Platform Layer

The app now has five guided learning paths with checkable tasks:

- How models learn from error,
- From distance to clusters,
- From good scores to trustworthy models,
- From features to model decisions,
- From representation to model behavior.

All paths connect lesson manifests to real scene interactions and persisted progress while keeping the standalone demo browser intact. The third path, **From good scores to trustworthy models**, connects train/validation/test discipline, leakage, class imbalance, calibration, and production monitoring in one registered guided sequence.

The fourth path, **From features to model decisions**, is now registered with task hooks
for feature scale, feature signal, tree splits, ensemble voting, and model-family
assumptions. A post-registration Polish copy and shell-selection test pass keeps the
new path aligned with the guided course UI.

The fifth path, **From representation to model behavior**, is now registered as a
Level 3 sequence. It reuses the existing Calibration and Monitoring lessons so
confidence and drift progress carry across learning contexts instead of forcing
duplicate work. A post-registration Polish copy and shell-details test pass keeps
the new path readable in the guided course UI.

The direction is captured in [learning_platform.md](learning_platform.md). Future slices should improve course-level guidance, add more useful instructor/student feedback, and only introduce richer gamification where it reinforces actual understanding.

Recent shell work made lesson progress visible throughout the student flow: path details point to the next lesson, lesson lists show compact progress markers, and home/path/lesson/intro/pause screens use consistent progress summaries and bars.

The course map now sits above individual guided paths. It shows the recommended
route, explains why each path follows the previous one, and keeps the full path
browser available for free exploration.

Lesson completion summaries now include concept-level understanding checks. The
checks stay lightweight and supportive: they ask students whether they can
explain the key idea and name the visible signal from the demo, without storing
graded answers.

The path details screen still acts as a compact lesson map, so students can see
lesson status and task progress before entering a specific demo.

### Shared UI Helpers

Native demo scenes now use small shared helpers for repeated panel drawing, text rendering, text wrapping, and readout panels. The helpers were intentionally introduced only after the same patterns appeared across many scenes.

Current boundaries:

- demo scenes share small drawing primitives through `ui_helpers.py` and `readout_panel.py`,
- scene-specific geometry stays near each scene,
- shell screens keep their own rendering path for menus, scrollbars, intro screens, theory viewports, and overlays.

Avoid forcing every scene into one renderer inheritance model. Existing demo renderers can remain standalone.

### Documentation And Teaching Flow

The app should continue reducing context switching for students.

Near-term docs work:

- keep `USAGE.md` and `USAGE.pl.md` aligned with the app,
- keep `docs/levels.md` aligned with the registry,
- add screenshots/GIFs as part of `v0.0.9a` release preparation,
- keep Polish copy natural rather than literal.

The release preparation should also add GitHub issue templates so students can
report broken behavior, confusing lessons, missing topics, and rough classroom
UX without needing to know the codebase.

### Distribution

Student-friendly distribution should follow the `v0.0.9a` learning-platform
polish. The initial direction is to use GitHub Actions to build downloadable
artifacts for Windows, macOS, and Linux and attach them to GitHub releases.

The release should first be worth installing: course guidance, feedback, and
classroom comfort come before packaged binaries.

### Demo Growth And Level Balance

New demos should be added in narrow vertical slices:

1. scene and core interaction,
2. manifest metadata,
3. intro/theory/help content,
4. focused tests,
5. small polish/UX pass after trying it in the shell.

The current demo registry is balanced enough to pause before adding more demos by
default. Guided lesson coverage is now healthier after registering a Level 3 path
for representation, calibration, monitoring, and forecasting. Future demo growth
should be driven by clear teaching gaps rather than by balancing counts alone.

## Completed Historical Phases

The following phases were the original build-out path and are now mostly complete:

- project foundation,
- `ml_lab_core` abstractions,
- Gradient Descent Playground,
- k-NN Vote Map,
- Logistic Regression Boundary Lab,
- Decision Tree Splitter,
- Random Forest Bagging Lab,
- Boosting Mistake Lab,
- unified Pygame app shell,
- manifest-driven level/demo selection,
- generated intro, theory, pause, and help screens,
- Polish UI support,
- display settings and persistent app settings,
- Level 3 transition from placeholder to real demos.

Historical notes remain useful as context, but the current source of truth for app behavior is the registry, tests, and usage docs.

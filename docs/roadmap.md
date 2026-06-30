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

## Near-Term Milestones

- [ ] Run app tests in CI and root pytest.
- [ ] Keep project docs aligned with the current unified app.
- [ ] Clean up small repository artifacts and naming drift.
- [x] Extract shared UI helpers for repeated demo-scene panel, text, and wrapping patterns.
- [x] Introduce learning paths, lesson/task manifests, progress, and meaningful completion badges.
- [x] Expand learning paths beyond the first end-to-end path.
- [x] Add checkable task completion to the second learning path.
- [ ] Implement the third guided path: From good scores to trustworthy models.
- [ ] Continue balancing Level 1, Level 2, and Level 3 with small focused demos.
- [ ] Add screenshots or short GIFs for the main README and app docs.

## Next Engineering Themes

### Learning Platform Layer

The app now has two guided learning paths with checkable tasks:

- How models learn from error,
- From distance to clusters.

Both paths connect lesson manifests to real scene interactions and persisted progress while keeping the standalone demo browser intact.

The planned third path, **From good scores to trustworthy models**, will connect train/validation/test discipline, leakage, class imbalance, calibration, and production monitoring. Its narrative, measurable tasks, badge drafts, and staged rollout are defined in [learning_platform.md](learning_platform.md). Train / Validation / Test Split Lab and Data Leakage Lab now have task hooks, but the path will not be registered until every lesson can be completed end to end.

The direction is captured in [learning_platform.md](learning_platform.md). Future slices should improve progress summaries, add more useful instructor/student feedback, and only introduce richer gamification where it reinforces actual understanding.

Recent shell work made lesson progress visible throughout the student flow: path details point to the next lesson, lesson lists show compact progress markers, and home/path/lesson/intro/pause screens use consistent progress summaries and bars.

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
- add screenshots/GIFs once the UI stabilizes further,
- keep Polish copy natural rather than literal.

### Demo Growth

New demos should be added in narrow vertical slices:

1. scene and core interaction,
2. manifest metadata,
3. intro/theory/help content,
4. focused tests,
5. small polish/UX pass after trying it in the shell.

The current priority is to keep Level 1, Level 2, and Level 3 balanced rather than overbuilding one level. The next guided path should be chosen for narrative value, not just because a demo already exists.

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

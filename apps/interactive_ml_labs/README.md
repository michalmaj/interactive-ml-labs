# Interactive ML Labs App

Unified Pygame app for browsing and running Interactive ML Labs demos.

The app is the recommended guided experience:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Standalone demo entry points remain supported for development, debugging, and focused teaching sessions.

## Current Scope

- global language selection: English and Polish,
- guided learning paths with persisted lesson, task, theory, and badge progress,
- home-screen learning progress with keyboard and mouse continue actions,
- compact progress bars on home, path details, lesson details, intro, and pause screens,
- dynamic level and demo selection from manifests,
- scrollable demo lists with keyboard and mouse navigation,
- generated intro screens with objectives and controls,
- built-in theory screens for every current demo,
- pause and help overlays during active demos,
- display settings for fullscreen, adaptive window size, and fixed-scene scaling,
- persistent app settings,
- all current Level 1, Level 2, and Level 3 demos available through the shell.

## Demo Flow

```text
Interactive ML Labs
-> language and settings
-> home progress and learning mode selection
-> guided path lesson selection or level/demo selection
-> demo intro
-> theory or experiment
-> pause / help / objectives / controls
```

On the home screen, press `C` or click the highlighted progress next-step line to continue the next recommended guided lesson.

## Development Notes

The shell lives in `apps/interactive_ml_labs` so `ml_lab_core` can stay focused on small reusable primitives.

Each demo is registered through a manifest. The manifest provides title, summary, objectives, controls, tags, difficulty, theory content, and a scene factory. The shell uses that metadata to build consistent menu, intro, help, and theory screens.

New demos should be added in small pull requests:

1. Add or adapt the demo scene.
2. Register the manifest.
3. Add tests for manifest metadata, scene behavior, and layout-sensitive UI.
4. Keep standalone entry points working when a demo already has one.

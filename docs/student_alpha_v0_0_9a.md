# v0.0.9a Student Alpha Notes

`v0.0.9a` is the first Interactive ML Labs alpha focused on a guided student
experience.

## What This Version Is

Interactive ML Labs is a local Pygame app for building high-level machine
learning intuition through visual experiments. You do not need to write code in
the lessons. The goal is to see what models do, change important parameters, and
explain the result in plain language.

## Recommended Start

Run the unified app from the repository root:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Then follow this route:

1. Choose a language.
2. Open the course map.
3. Start with the first guided learning path.
4. Read the lesson intro.
5. Open theory when you need context.
6. Complete the in-demo tasks.
7. Use the completion summary and concept checks to verify what you can explain.

Free demo browsing is still available from the level browser.

## What Is Included

- Five guided learning paths.
- Level 1, Level 2, and Level 3 demos in the unified app.
- Lesson intros, theory screens, pause/help overlays, and task progress.
- Completion summaries with lightweight concept checks.
- Badges for completed lessons.
- Persistent progress and settings.
- English and Polish UI.
- Comfort settings: large text, high contrast, and colorblind-friendly palette.

## What To Report

Student feedback is especially useful when it says:

- which lesson was confusing,
- which task did not make sense,
- which text was unclear,
- which screen was hard to read,
- which control did not work,
- which ML topic feels missing.

## Known Limits

- The app is still launched with `uv run`; packaged installers are planned after
  this alpha.
- Concept checks are reflection prompts, not graded quizzes.
- Comfort settings currently focus on shell screens. Individual demo scenes can
  still need follow-up polish.
- Some advanced topics are intentionally simplified so the lesson stays visual
  and teachable.

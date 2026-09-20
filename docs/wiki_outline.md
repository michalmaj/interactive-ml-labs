# Wiki Outline

This outline prepares the first GitHub Wiki pages for the `v0.0.9a` student-facing alpha.

The wiki should stay practical and short. It is meant for students and instructors
who want to run the app, understand the guided paths, and report useful feedback
without reading the whole repository.

## Home

Purpose:

- explain that Interactive ML Labs is a guided local-first ML learning app,
- link to the current release notes,
- link to `USAGE.md` and `USAGE.pl.md`,
- point students toward the unified app entry point.

Suggested content:

```text
Start here if you are using Interactive ML Labs in class.

Recommended launch:
uv run --package interactive-ml-labs-app interactive-ml-labs

Use the course map first, then follow a guided path. Standalone demos are still
available for focused classroom demonstrations.
```

## Getting Started In Class

Purpose:

- help students install dependencies,
- explain the launch command,
- describe what to do during the first 10 minutes.

Suggested sections:

- Requirements: Python 3.12+, uv.
- First launch.
- Choose language.
- Open the course map.
- Complete one short lesson.
- Where progress is stored.

## Guided Learning Paths

Purpose:

- explain how paths, lessons, tasks, theory, and badges fit together,
- give instructors a quick overview of the current path order.

Suggested sections:

- How models learn from error.
- From distance to clusters.
- From good scores to trustworthy models.
- From features to model decisions.
- From representation to model behavior.

## Reporting Confusing Lessons

Purpose:

- make feedback easy for students,
- point them toward the "Confusing lesson" issue template.

Suggested prompts:

- Which lesson was confusing?
- What did you expect to happen?
- What did the app show instead?
- Which explanation or task should be clearer?

## Instructor Notes And Classroom Setup

Purpose:

- help instructors run the app during workshops,
- keep classroom-specific guidance separate from the README.

Suggested sections:

- Recommended display settings.
- Projector/high contrast tips.
- Suggested order for a 45-minute class.
- Suggested order for a 90-minute class.
- When to use standalone demos.

## Feedback Triage

Purpose:

- describe how maintainers should group incoming student feedback.

Suggested labels:

- bug,
- confusing lesson,
- missing topic,
- classroom UX,
- docs,
- release follow-up.

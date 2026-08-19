# v0.0.9a Release Checklist

This checklist turns the `v0.0.9a` plan into a concrete release runbook.

## Release Scope

`v0.0.9a` is a student-facing alpha of the guided learning platform. It should
show that Interactive ML Labs is more than a demo browser:

- the unified app is the recommended entry point,
- students can follow a course map and guided learning paths,
- lessons have tasks, theory, progress, completion summaries, and concept
  checks,
- the shell has classroom comfort settings,
- standalone demo entry points remain supported for teaching and development.

This release is still local-first. It does not need packaged binaries yet.

## Before Tagging

- [ ] Confirm `main` is green in GitHub Actions.
- [ ] Run the local release gate:

```bash
uv run ruff format --check .
uv run ruff check .
uv run --all-packages pytest
```

- [ ] Launch the app locally:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

- [ ] Smoke-test the main student flow:
  - language selection,
  - home progress panel,
  - course map,
  - guided path details,
  - lesson intro,
  - theory screen,
  - demo task progress,
  - help overlay,
  - completion summary and concept checks,
  - settings menu comfort options.

- [ ] Review `USAGE.md` and `USAGE.pl.md`.
- [ ] Review the student release notes:
  - [student_alpha_v0_0_9a.md](student_alpha_v0_0_9a.md),
  - [student_alpha_v0_0_9a.pl.md](student_alpha_v0_0_9a.pl.md).

## GitHub Repository Setup

Before publishing the release, prepare the repository page:

- [x] Add a license file.
- [x] Add GitHub issue templates for:
  - bug reports,
  - confusing lesson feedback,
  - missing topic suggestions,
  - classroom usability feedback.
- [x] Document the recommended repository metadata in
  [github_repository_setup.md](github_repository_setup.md).
- [ ] Set repository description:

```text
Interactive visual machine learning labs with guided lessons, tasks, progress, and bilingual course flow.
```

- [ ] Add repository topics:

```text
machine-learning
education
pygame
interactive-learning
python
teaching
visualization
ml-education
```

- [ ] Enable useful GitHub home-page sections:
  - Releases,
  - Packages later, when distribution artifacts exist.
- [ ] Prepare a small wiki outline for instructor/student notes.

## Tag And Release

Suggested tag:

```bash
git tag v0.0.9a
git push origin v0.0.9a
```

Create a GitHub release from the tag and use the student release notes as the
starting point. Keep the release language honest: this is an alpha for local use,
not a polished installer-based product.

## Not In This Release

These items are intentionally after `v0.0.9a`:

- packaged Windows/macOS/Linux binaries,
- one-click student installation,
- graded quizzes or typed answer collection,
- full accessibility pass inside every native demo scene,
- full LMS-style course management.

## After Release

After `v0.0.9a`, the next track is student distribution:

- GitHub Actions artifact builds,
- release-attached platform packages,
- student-friendly install docs without requiring a development workflow,
- feedback triage based on real classroom use.

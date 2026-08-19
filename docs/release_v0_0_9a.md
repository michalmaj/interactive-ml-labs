# v0.0.9a Release Plan

This document captures the current priority for the next public-facing alpha
release of Interactive ML Labs.

The release goal is to make the project feel like a learning platform, not only
a collection of interactive demos. The app already has demos, guided paths,
tasks, badges, persistence, bilingual copy, and a stable unified shell. The next
release should make the learning route clearer for students and easier to use in
class.

## Release Goal

`v0.0.9a` should be the first release that can be described as:

> a guided, student-facing alpha for building high-level machine learning
> intuition through interactive lessons.

This release does not need to become a full learning management system. It should
stay lightweight, local-first, and demo-driven.

## Required Product Work

### 1. Course Feeling Above Individual Paths

Students should understand where to start, what to do next, and why the next
path follows from the previous one.

Target outcome:

- a course-level map or guide above the existing learning paths,
- a recommended starting point,
- clear "next path" guidance,
- short explanations of how each path builds on earlier intuition,
- copy that makes the experience feel guided without hiding free exploration.

The existing path list remains useful, but it should no longer be the highest
level of structure.

### 2. Better Student Feedback

The app should help students notice whether they understood the lesson, not only
whether they clicked through the task checklist.

Target outcome:

- short concept checks or reflection prompts,
- completion summaries that say what the student should now be able to explain,
- path-level summaries that connect completed lessons into one learning story,
- optional export or report planning if it stays small enough for the release.

This should stay supportive rather than exam-like. The goal is understanding and
confidence, not grading.

Current status: lesson completion summaries now include lightweight concept
checks. They are not graded or persisted; they help students pause and verify
whether they can explain the key idea before continuing.

### 3. Accessibility And Classroom Comfort

The app should be more comfortable on projectors, small laptop screens, and in
mixed classroom conditions.

Target outcome:

- larger readable text mode or presentation-friendly sizing,
- high contrast mode,
- colorblind-friendly palette option,
- a first pass at predictable visual comfort settings.

The implementation can stay incremental. The release should at least establish
the settings and one useful visual path through them.

## Release Preparation

After the required product work is complete:

- create a GitHub tag and release for `v0.0.9a`,
- add a project license before the release,
- refresh the root README for the release state,
- add GitHub issue templates for student feedback,
- prepare repository metadata: description, topics, and homepage settings,
- start or prepare a project wiki for student/instructor-facing notes.

## After v0.0.9a

The next major theme after the alpha release is distribution for students.

Target direction:

- GitHub Actions builds downloadable artifacts for Windows, macOS, and Linux,
- releases attach platform-specific packages,
- usage docs gain a student-friendly path that does not require a development
  environment or `uv run`,
- issue templates collect bug reports, lesson suggestions, missing topics, and
  classroom usability feedback.

Distribution should follow the learning-platform polish, not precede it. The
release should first be worth installing.

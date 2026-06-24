# Learning Platform Direction

Interactive ML Labs has grown from a set of interactive demos into the beginning of a learning platform. The next product direction is to make the app feel less like a demo browser and more like a guided course where students build intuition step by step.

The platform is not meant to teach programming first. Its primary goal is to help students understand machine learning at a high level: what models are trying to optimize, why parameters matter, how evaluation can mislead, and how production behavior can drift.

## Core Model

The app should distinguish three concepts:

- **Demo**: one interactive simulation or experiment.
- **Lesson**: a demo plus theory, goals, tasks, completion rules, and optional rewards.
- **Learning path**: an ordered sequence of lessons where later lessons build on ideas from earlier ones.

The current manifest registry is a good foundation for demos. The first lesson layer now adds lesson and learning-path metadata without breaking existing standalone demo entry points.

## Learning Flow

A student should be able to follow a path like this:

1. Pick a learning path.
2. Read a short lesson intro.
3. Open theory when needed.
4. Complete one or more concrete tasks inside the demo.
5. Get visible progress and a meaningful completion state.
6. Move to the next lesson with a clear reason why it follows.

This keeps the current interactive style, but gives it structure. The important shift is that a lesson should not only say "play with this"; it should ask the student to prove that they noticed the key idea.

## Objectives And Tasks

Objectives describe what the lesson is trying to teach. Tasks describe what the student must do.

Good objective:

- Understand how changing the decision threshold trades precision against recall.

Good task:

- Set the threshold so recall becomes higher than precision, then explain what changed.

Tasks should be checkable from app or scene state whenever possible. Text-only reflection can still exist, but the strongest tasks are based on actions and visible outcomes.

Possible task types:

- `visited_theory`
- `started_demo`
- `changed_parameter`
- `reached_metric_threshold`
- `completed_scene_challenge`

Scenes and adapters can report progress through a small API, for example:

```python
context.progress.record("learning_rate_changed")
context.progress.record_metric("loss", value)
context.progress.complete_task("stable_descent_found")
```

## Gamification

Gamification should support learning, not distract from it.

Useful ideas:

- lesson progress,
- task checklists,
- badges for meaningful concepts,
- "intuition unlocked" moments,
- titles such as Fundamentals Explorer, Practical ML Operator, or Model Debugger,
- session summaries that remind students what they learned.

Things to avoid:

- points for random clicking,
- leaderboards,
- grind,
- badges that do not represent real understanding.

A badge should mean that the student demonstrated a concrete idea or skill.

## Initial Data Model

A first lightweight lesson model could look like this:

```python
@dataclass(frozen=True)
class LessonManifest:
    id: str
    title: LocalizedText
    level: int
    demo_id: str
    prerequisites: tuple[str, ...]
    learning_goal: LocalizedText
    tasks: tuple[LessonTask, ...]
    completion_badge: str | None = None


@dataclass(frozen=True)
class LessonTask:
    id: str
    title: LocalizedText
    instruction: LocalizedText
    success_condition: str
    hint: LocalizedText | None = None
```

The first implementation keeps this declarative and small. More advanced validation and richer task types can come after more paths work end to end.

## First Learning Path

The first path reuses strong existing lessons instead of adding another isolated demo.

**How models learn from error**

1. Linear Regression Line Fit Lab
2. Gradient Descent Playground
3. Logistic Regression Boundary Lab
4. Boosting Mistake Lab

Narrative:

- first, students see error by moving a line manually,
- then they see an algorithm minimize loss,
- then they see decisions and thresholds in classification,
- finally, they see an ensemble focus on previous mistakes.

This path is a good first slice because it connects existing Level 1 and Level 2 material into one learning story. It now has checkable tasks, prerequisite and next-lesson guidance, completion badges, persisted lesson progress, and a visible learning-path progress summary.

## Second Learning Path

The second path starts broadening the platform beyond supervised error minimization.

**From distance to clusters**

1. Distance Metrics Lab
2. k-NN Vote Map
3. K-Means Intro Lab
4. Clustering Lab
5. Gaussian Mixture Intro Lab

Narrative:

- first, students see that "close" depends on the chosen distance metric,
- then they see how k-NN turns distance into neighborhood voting,
- then they see K-Means repeat nearest-centroid assignment,
- then they compare centroid-based and density-based clustering,
- finally, they see soft cluster membership in Gaussian Mixture Models.

This path has lesson and task metadata first. The next slices should connect its tasks to real scene interactions, just as the first path did.

## Suggested PR Sequence

1. Done: add `LessonManifest`, `LessonTask`, and a small learning-path registry.
2. Done: add the first learning path with four lessons and non-invasive metadata.
3. Done: add a lesson/path selection screen in the app shell.
4. Done: persist basic lesson progress.
5. Done: add checkable tasks to the first path lessons.
6. Done: show completion badges and aggregate learning-path progress.
7. Done: show prerequisite and next-lesson guidance in the lesson details panel.

Next slices should focus on adding task completion hooks to the second learning path and making progress summaries more useful for students and instructors.

## Non-Goals For The First Slice

- Do not replace the demo browser.
- Do not remove standalone entry points.
- Do not build a heavy LMS.
- Do not add leaderboards.
- Do not require every existing demo to become a full lesson immediately.

The first slice should prove that lessons, tasks, and progress can exist alongside the current app without destabilizing the demo work.

## Open Questions

- Should learning paths be shown before levels, next to levels, or as a separate top-level menu option?
- Should tasks be language-localized in the same manifest style as current demo copy?
- Should progress be stored next to app settings or in a separate progress file?
- Which completion events should be owned by the shell, and which should be reported by individual scenes?
- How much reflection text should the app ask for, if any, without turning the experience into homework software?

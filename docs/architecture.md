# Architecture

Interactive ML Labs is designed as a collection of visual, interactive machine learning demos.

The repository is organized as a Python workspace. Shared code lives in reusable packages, while every demo is developed as a separate, documented module.

## Main principles

### 1. Keep machine learning logic separate from rendering

Algorithms should not depend on Pygame or any other UI framework.

This means that algorithmic code should be testable with regular unit tests, without opening a graphical window.

Good separation:

```text
algorithm -> snapshot -> renderer
```

Bad separation:

```text
algorithm + pygame drawing + input handling in one file
```

### 2. Prefer step-by-step algorithms

Whenever possible, algorithms should expose one logical step at a time.

Examples:

- one gradient descent update,
- one k-means assignment/update phase,
- one decision tree split,
- one Q-learning transition,
- one active learning query.

This makes the demos easier to visualize and easier to explain.

### 3. Every demo should have the same teaching structure

Each demo should contain:

- a data world or simulation world,
- an algorithm or model,
- a renderer,
- controls,
- live metrics,
- a challenge mode,
- an explanation panel,
- documentation.

### 4. Synthetic data first, real data second

Every demo should work out of the box with synthetic or lightweight data.

Real datasets may be added later, but they should not be required for the first launch.

### 5. The repository should teach software engineering too

This project is not only about machine learning.

It should also show students how real projects are developed:

- small branches,
- pull requests,
- automated checks,
- tests,
- documentation,
- readable commits,
- clear project structure.

## Planned high-level structure

```text
interactive-ml-labs/
├── docs/
├── packages/
│   └── ml_lab_core/
├── demos/
│   ├── level_1_fundamentals/
│   ├── level_2_practical_ml/
│   └── level_3_advanced_showcase/
├── tests/
└── .github/workflows/
```

## Shared package

The ml_lab_core package should contain reusable infrastructure, such as:

- base protocols,
- snapshots,
- scene abstractions,
- reusable metrics,
- UI helpers,
- rendering helpers,
- challenge definitions.

It should not contain demo-specific algorithms.

## Unified application shell

The project is expected to grow from separately launched demos into one guided Pygame application.

The shell should live outside `ml_lab_core` at first, because it is a concrete application rather than a small reusable primitive. Existing demo entry points should remain available while the shell becomes the recommended guided experience.

See [Unified App Shell](unified_app_shell.md) for the current architecture direction.

## Demo packages

Each demo should be self-contained and documented.

A typical demo will eventually contain:

```text
demo_name/
├── README.md
├── THEORY.pl.md
├── CHALLENGES.pl.md
├── src/
├── tests/
└── assets/
```

## Definition of done for a demo

A demo is considered complete when it has:

- working run command,
- step-by-step mode,
- run/pause/reset mode,
- live metrics,
- challenge mode,
- explanation panel,
- tests for algorithmic logic,
- README documentation,
- theory explanation,
- student challenges,
- passing Ruff checks,
- passing tests,
- passing CI.

# Usage

This guide explains how to run Interactive ML Labs locally.

## Requirements

- Python 3.12 or newer
- uv

Install or update dependencies from the repository root:

```bash
uv sync
```

## Recommended Guided App

The unified Pygame app is the recommended entry point for browsing demos:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

The app opens as a `1600x900` window by default. Fullscreen is planned as an opt-in setting, not the default launch mode.

Current app shell flow:

```text
language selection
-> level selection
-> demo selection
-> demo intro screen
-> placeholder demo screen
-> pause/help overlay
```

Real demo integrations are being added gradually. Individual demos can still be run directly.

Common shell controls:

| Key / input | Action |
| ----------- | ------ |
| `Up` / `Down` | Move selection |
| Mouse hover | Move selection |
| Mouse click / `Enter` | Activate selected item |
| `Esc` / `Backspace` | Go back or open pause menu |
| `H` | Toggle help overlay |
| `L` | Toggle language |

## Standalone Demos

Each demo remains runnable on its own.

### Gradient Descent Playground

```bash
uv run --package gradient-descent-playground gradient-descent-playground
uv run --package gradient-descent-playground gradient-descent-playground-ui
```

### k-NN Vote Map

```bash
uv run --package knn-vote-map knn-vote-map
uv run --package knn-vote-map knn-vote-map-ui
```

### Logistic Regression Boundary Lab

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab-ui
```

### Decision Tree Splitter

```bash
uv run --package decision-tree-splitter decision-tree-splitter
uv run --package decision-tree-splitter decision-tree-splitter-ui
```

### Random Forest Bagging Lab

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
uv run --package random-forest-bagging-lab random-forest-bagging-lab-ui
```

### Boosting Mistake Lab

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

## Quality Checks

Run formatting and lint checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run tests for the unified app shell:

```bash
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
```

Run tests for one demo package:

```bash
uv run --package boosting-mistake-lab pytest demos/level_2_practical_ml/02_boosting_mistake_lab/tests
```

## Development Workflow

Use a small branch for each logical change.

Typical flow:

```bash
git switch main
git pull
git switch -c feat/my-change
uv run ruff check .
uv run ruff format --check .
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
git push -u origin feat/my-change
```

Then open a pull request using the repository template.

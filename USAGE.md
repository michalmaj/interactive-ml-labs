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

The app opens as a `1280x720` window by default. Fixed-scene scaling is enabled so larger fixed-canvas demos fit safely, and adaptive window sizing plus fullscreen are available from the in-app settings menu.

The app remembers language, fullscreen, adaptive window sizing, and fixed-scene scaling in a small per-user settings file. Window resolution is not persisted, because adaptive sizing is recalculated for the current display.

Current app shell flow:

```text
language selection
-> level selection
-> demo selection
-> demo intro screen
-> demo screen
-> pause/help overlay
```

Real demo integrations are being added gradually. All current Level 1 and Level 2 demos are available through the unified app, and individual demos can still be run directly.

The unified app also includes app-only Level 3 labs:

- Clustering Lab demonstrates K-Means phases, inertia, draggable points, and a DBSCAN comparison mode.
- PCA Lab demonstrates dataset/noise controls, manual projection rotation, fitted PCA direction, explained variance, reconstruction residuals, and reconstruction error.
- Model Comparison Lab demonstrates Logistic Regression, k-NN, and Decision Tree assumptions on shared datasets, with train/test scores, compact confusion details, and highlighted test errors.
- Calibration Lab demonstrates probability calibration with reliability diagrams, score distributions, a raw-vs-scaled score legend, accuracy@0.5, Brier score, ECE, worst-gap highlighting, calibration gap error bars, and temperature scaling.
- t-SNE / UMAP Exploration Lab demonstrates deterministic toy embeddings, raw-vs-embedding comparison, dataset cues, class labels, seed drift, neighborhood tuning, and local-neighbor links.
- Model Monitoring Drift Lab demonstrates data drift, metric drift, monitoring windows, alert thresholds, lead signals, alert rate, persistence, trend readouts, and investigation acknowledgements.

Common shell controls:

| Key / input | Action |
| ----------- | ------ |
| `Up` / `Down` | Move selection |
| Mouse hover | Move selection |
| Mouse click / `Enter` | Activate selected item |
| `Esc` / `Backspace` | Go back or open pause menu |
| `H` | Toggle the selected demo help overlay |
| `L` | Toggle language |
| `S` | Open settings outside active demos |

## Standalone Demos

Each original Level 1 and Level 2 demo remains runnable on its own. Native app-only labs, such as Clustering Lab, PCA Lab, Model Comparison Lab, Calibration Lab, and t-SNE / UMAP Exploration Lab, are launched from the unified app.

### App-only Level 3 Labs

Clustering Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-4` | Switch dataset preset |
| `-` / `=` | Change `k` in K-Means or `eps` in DBSCAN |
| `Space` | Advance K-Means phase or rerun DBSCAN |
| `M` | Switch K-Means / DBSCAN |
| `C` | Toggle point-to-centroid links |
| Mouse drag | Move a data point |

PCA Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch dataset preset |
| `-` / `=` | Change noise |
| `N` | Generate a new sample |
| `Left` / `Right` | Rotate projection direction |
| `F` | Toggle fitted PCA direction |
| `C` | Toggle reconstruction residual lines |

Model Comparison Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Focus Logistic Regression, k-NN, or Decision Tree |
| `D` | Cycle dataset preset |
| `-` / `=` | Change the active model parameter |
| `A` | Show or hide inactive boundaries |
| `E` | Show or hide misclassified test points |
| `R` | Reset the preview |

Calibration Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch calibration preset |
| `-` / `=` | Change temperature scaling |
| `O` | Show or hide raw pre-temperature scores |
| `E` | Show or hide calibration error bars |
| `R` | Reset the preview |

t-SNE / UMAP Exploration Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch dataset preset |
| `M` | Switch t-SNE / UMAP preview |
| `-` / `=` | Change perplexity / neighbors |
| `S` | Change seed variant and inspect drift |
| `L` | Show or hide local-neighbor links |
| `O` | Show or hide raw high-dimensional layout |
| `R` | Reset the preview |

Model Monitoring Drift Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-4` | Switch monitoring preset |
| `D` / `M` | Select data drift / metric drift signal |
| `-` / `=` / `0` | Change or reset alert threshold |
| `A` | Acknowledge an active alert for investigation |
| `R` | Reset the preview |

What to watch in Model Monitoring Drift Lab:

- The bright line is the active signal; the muted line keeps the other signal visible for comparison.
- `windows` compares the baseline window with the current window and adds a compact trend label.
- `gap`, `threshold`, and `severity` explain whether the current change is below, near, or above the selected alert threshold.
- `first alert`, `alert rate`, `persistence`, and `lead signal` help separate a one-off spike from a repeated production signal.
- `investigation` changes after pressing `A`, so the lab models a small monitoring workflow instead of treating every alert as automatic panic.

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

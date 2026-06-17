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

Real demo integrations are being added gradually. All current Level 1 and Level 2 demos are available through the unified app. Original package demos can still be run directly, while native app-only labs are launched from the unified app.

The unified app includes these app-only Level 1 labs:

- Linear Regression Line Fit Lab demonstrates slope, intercept, residuals, MSE loss, and least-squares fitting with a manually adjustable line.
- K-Means Intro Lab demonstrates assignment/update steps, centroids, k, inertia, and why lower inertia is not the whole clustering story.
- Distance Metrics Lab demonstrates query points, nearest neighbors, Euclidean distance, Manhattan distance, Chebyshev distance, and why k-NN depends on the metric.
- SVM Margin Lab demonstrates decision boundaries, support vectors, margin width, and why the widest correct separator matters.
- Activation Functions Lab demonstrates sigmoid, tanh, ReLU, output ranges, saturation, and local gradient flow.
- Neural Network Playground demonstrates a tiny forward pass through inputs, weights, hidden units, activation, probability, target, and loss.

The unified app includes these app-only Level 2 labs:

- Data Leakage Lab demonstrates suspicious features, prediction-time availability, leaky-vs-clean validation scores, and the habit of distrusting metrics that look too perfect.
- Train / Validation / Test Split Lab demonstrates model selection with validation scores, overfitting from train-validation gaps, and keeping test as the final honest check.
- Feature Scaling Lab demonstrates raw-vs-scaled feature ranges, range ratio, scale-sensitive models, and the effect of scaling on accuracy and iterations.
- Gaussian Mixture Intro Lab demonstrates soft responsibilities, hard assignment, mixture weights, component shapes, and overlapping clusters.
- Hyperparameter Tuning Lab demonstrates validation curves, grid-search intuition, train-validation gaps, and choosing parameters by validation score.
- Class Imbalance Lab demonstrates accuracy traps, precision/recall trade-offs, false negatives, and decision threshold tuning when one class is rare.

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

Each original Level 1 and Level 2 demo remains runnable on its own. Native app-only labs, such as Linear Regression Line Fit Lab, K-Means Intro Lab, Distance Metrics Lab, SVM Margin Lab, Activation Functions Lab, Neural Network Playground, Data Leakage Lab, Train / Validation / Test Split Lab, Feature Scaling Lab, Gaussian Mixture Intro Lab, Hyperparameter Tuning Lab, Class Imbalance Lab, Clustering Lab, PCA Lab, Model Comparison Lab, Calibration Lab, and t-SNE / UMAP Exploration Lab, are launched from the unified app.

### App-only Level 1 Labs

Linear Regression Line Fit Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch dataset |
| `Left` / `Right` | Change slope |
| `Up` / `Down` | Change intercept |
| `F` | Jump to the least-squares fit |
| `R` | Reset the lab |

K-Means Intro Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `Space` | Run one assignment or centroid-update step |
| `A` | Start or pause auto-run |
| `-` / `=` | Change `k` |
| `1-3` | Switch dataset |
| `C` | Toggle point-to-centroid links |
| `N` | Generate a new sample |
| `R` | Reset the lab |

Distance Metrics Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch dataset |
| Arrow keys | Move the query point |
| `M` | Cycle distance metric |
| `R` | Reset the lab |

SVM Margin Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch dataset |
| `Left` / `Right` | Rotate boundary |
| `Up` / `Down` | Shift boundary |
| `F` | Jump to a wide-margin fit |
| `R` | Reset the lab |

Activation Functions Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch activation |
| `Left` / `Right` | Move input x |
| `0` | Reset x to zero |
| `R` | Reset the lab |

Neural Network Playground controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch input example |
| `A` | Cycle activation |
| `-` / `=` | Change weight scale |
| `Up` / `Down` | Change hidden bias |
| `R` | Reset the playground |

### App-only Level 2 Labs

Data Leakage Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch leakage scenario |
| `L` | Toggle the suspicious leakage feature |
| `R` | Reset the preview |

Train / Validation / Test Split Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch split scenario |
| `-` / `=` / `0` | Change or reset model complexity |
| `R` | Reset the preview |

Feature Scaling Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch scaling scenario |
| `S` | Toggle feature scaling |
| `M` | Cycle model |
| `R` | Reset the preview |

Gaussian Mixture Intro Lab controls:

| Key / input | Action |
| ----------- | ------ |
| Arrow keys | Move the query point |
| `-` / `=` | Change component count |
| `H` | Toggle hard assignment |
| `D` | Show or hide density ellipses |
| `1-3` | Switch dataset |
| `R` | Reset the lab |

Hyperparameter Tuning Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch tuning scenario |
| `-` / `=` / `0` | Change or reset parameter value |
| `R` | Reset the preview |

Class Imbalance Lab controls:

| Key / input | Action |
| ----------- | ------ |
| `1-3` | Switch imbalance scenario |
| `-` / `=` / `0` | Change or reset decision threshold |
| `R` | Reset the preview |

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

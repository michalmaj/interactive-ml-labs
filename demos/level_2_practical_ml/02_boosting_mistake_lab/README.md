# Boosting Mistake Lab

Interactive teaching demo for understanding boosting, weak learners, sample weights, staged accuracy, and boosted ensemble decision boundaries.

This demo focuses on intuition.

It shows how boosting repeatedly trains weak learners, increases the importance of difficult samples, and combines weak learners into a stronger ensemble using alpha-weighted voting.

## What this demo teaches

Students can explore:

- weak learners,
- weighted classification error,
- learner weight `alpha`,
- sample weight updates,
- multi-round boosting,
- boosted ensemble prediction,
- staged train/test accuracy,
- generalization gap,
- confidence from ensemble margins,
- challenge-based model tuning,
- decision boundary export.

## Run CLI report

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab
```

The CLI compares:

- first weak learner baseline,
- final boosted ensemble,
- train accuracy,
- test accuracy,
- generalization gap,
- best staged test accuracy,
- winner by test accuracy.

## Run Pygame UI

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

The UI shows two main panels:

- left: weak learner from the selected boosting stage,
- right: boosted ensemble built from rounds `1..selected_stage`.

Training samples are shown as circles.

Circle size represents sample weight.

Test samples are shown as squares.

Misclassified test samples are marked with `X`.

Background color shows decision regions.

## Controls

| Key              | Action                                          |
| ---------------- | ----------------------------------------------- |
| `1`–`4`          | Select preset scenario                          |
| `P`              | Switch to next preset                           |
| `Up`             | Increase selected boosting stage                |
| `Down`           | Decrease selected boosting stage                |
| `]` / `+`        | Increase total boosting round count             |
| `[` / `-`        | Decrease total boosting round count             |
| `D`              | Toggle dataset: `axis_aligned` / `xor`          |
| `W`              | Increase weak learner `min_samples_leaf`        |
| `S`              | Decrease weak learner `min_samples_leaf`        |
| `C`              | Toggle boosted confidence view                  |
| `Left`           | Decrease data noise                             |
| `Right`          | Increase data noise                             |
| `N`              | Generate a new dataset seed                     |
| `E`              | Export selected-stage decision boundary to JSON |
| `R`              | Reset to default preset                         |
| `Esc`            | Quit                                            |


## Preset scenarios

| Key | Preset              | Purpose                                                  |
| --- | ------------------- | -------------------------------------------------------- |
| `1` | Easy axis-aligned   | Simple baseline where one stump already performs well    |
| `2` | Noisy XOR           | Nonlinear case where boosted stumps are more interesting |
| `3` | Overfitting watch   | More noise and more rounds for staged accuracy analysis  |
| `4` | Low-round challenge | Small round budget for discussing trade-offs             |


Presets are designed for classroom use.

They provide quick starting points for discussing different boosting behaviors.

## Challenge mode

Current challenge target:

```text
boosted test accuracy >= 0.85
round_count <= 8
generalization gap <= 0.20
```

The goal is not only to maximize training accuracy.

The goal is to find a model that performs well on test data, does not use too many rounds, and does not overfit too strongly.

Challenge status is shown in the UI side panel.

The explanation panel gives short hints when the challenge fails.

## Explanation panel

The bottom panel gives compact feedback about the current state.

It can explain:

- whether challenge mode passed,
- whether test accuracy is too low,
- whether too many rounds are used,
- whether the generalization gap is too high,
- where the best staged test accuracy happened,
- what confidence view means.

## Decision boundary export

Press:

```text
E
```

The export file is written to:

```text
exports/boosting_mistake_lab_decision_boundary.json
```

The JSON contains:

- dataset metadata,
- selected boosting stage,
- trainer metrics,
- staged accuracy history,
- weak learner split summaries,
- learner alpha values,
- decision boundary grid predictions,
- confidence values,
- raw ensemble scores,
- train/test samples,
- initial and final sample weights.

The `exports/` directory is ignored by Git.

## Main implementation modules

| Module                    | Responsibility                        |
| ------------------------- | ------------------------------------- |
| `dataset.py`              | Synthetic weighted datasets           |
| `weak_learner.py`         | Weighted decision-stump weak learner  |
| `weighted_error.py`       | Weighted error and weighted accuracy  |
| `learner_weight.py`       | Learner alpha computation             |
| `sample_weight_update.py` | AdaBoost-style sample weight update   |
| `boosting_round.py`       | One complete boosting round           |
| `trainer.py`              | Multi-round boosting trainer          |
| `boosted_prediction.py`   | Alpha-weighted ensemble prediction    |
| `staged_history.py`       | Staged boosted train/test metrics     |
| `challenge.py`            | Challenge mode evaluation             |
| `explanation.py`          | Human-readable hints and explanations |
| `presets.py`              | Ready-to-use teaching scenarios       |
| `export.py`               | Decision boundary JSON export         |
| `report.py`               | CLI comparison report                 |
| `scene.py`                | Reusable Pygame demo scene            |
| `renderer.py`             | Pygame rendering                      |
| `pygame_app.py`           | Standalone Pygame application loop    |


## Suggested classroom flow

- Start with preset 1.
- Explain why one stump works well on axis-aligned data.
- Switch to preset 2.
- Show that XOR-like data needs multiple weak learners.
- Move through selected stages with Up / Down.
- Discuss staged train/test accuracy.
- Toggle confidence view with C.
- Use preset 3 to discuss overfitting.
- Use preset 4 as a student challenge.
- Export a result with E.

## Testing

Run package tests:

```bash
uv run --package boosting-mistake-lab pytest demos/level_2_practical_ml/02_boosting_mistake_lab/tests
```

Run the full project checks from the repository root:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run --package ml-lab-core pytest packages/ml_lab_core/tests
uv run --package gradient-descent-playground pytest demos/level_1_fundamentals/01_gradient_descent_playground/tests
uv run --package knn-vote-map pytest demos/level_1_fundamentals/02_knn_vote_map/tests
uv run --package logistic-regression-boundary-lab pytest demos/level_1_fundamentals/03_logistic_regression_boundary_lab/tests
uv run --package decision-tree-splitter pytest demos/level_1_fundamentals/04_decision_tree_splitter/tests
uv run --package random-forest-bagging-lab pytest demos/level_2_practical_ml/01_random_forest_bagging_lab/tests
uv run --package boosting-mistake-lab pytest demos/level_2_practical_ml/02_boosting_mistake_lab/tests
```

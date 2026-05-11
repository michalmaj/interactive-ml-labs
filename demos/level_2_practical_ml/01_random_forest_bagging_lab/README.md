# Random Forest Bagging Lab

Random Forest Bagging Lab is an interactive Level 2 demo from **Interactive ML Labs**.

The demo shows how multiple decision trees can be combined into a more stable ensemble classifier.

Students can compare:

- one decision tree,
- a random forest,
- train accuracy,
- test accuracy,
- generalization gap,
- bootstrap sampling,
- majority voting,
- vote confidence,
- challenge mode.

## What this demo teaches

This demo focuses on practical ensemble learning.

The key idea:

```text
dataset
-> train/test split
-> bootstrap samples
-> many decision trees
-> majority voting
-> final prediction
```

A single decision tree can be easy to interpret, but it can also overfit.

A random forest reduces this risk by training many trees on different bootstrap samples and combining their predictions through voting.

## Why this demo comes after Decision Tree Splitter

Decision Tree Splitter shows how one tree learns recursive axis-aligned splits.

Random Forest Bagging Lab extends that idea:

- one decision tree can overfit,
- many trees can vote together,
- bootstrap sampling creates different training subsets,
- different trees can make different errors,
- voting can stabilize predictions,
- test accuracy is more important than training accuracy.

This makes Random Forest a natural first Level 2 demo.

## Implemented features

Implemented:

- package structure,
- workspace configuration,
- command-line entry point,
- synthetic train/test classification dataset,
- axis-aligned dataset layout,
- XOR dataset layout,
- train/test metadata,
- bootstrap sampling utilities,
- bootstrap index generation,
- out-of-bag index detection,
- bootstrap dataset snapshots,
- majority voting utilities,
- vote count matrix,
- vote confidence computation,
- deterministic tie-breaking for equal votes,
- single-tree baseline,
- train accuracy evaluation,
- test accuracy evaluation,
- baseline prediction support,
- random forest model,
- per-tree bootstrap training,
- ensemble prediction through majority voting,
- forest train/test accuracy,
- forest vote confidence,
- CLI comparison report,
- single-tree vs forest accuracy deltas,
- generalization gap reporting,
- model winner summary,
- basic Pygame visualization,
- side-by-side single-tree and random-forest comparison,
- decision region rendering,
- train/test point rendering,
- misclassified test point markers,
- UI controls for dataset kind, tree count, max depth, noise, and seed,
- bootstrap ratio controls in Pygame,
- confidence view for forest decision regions,
- challenge mode for random forest model selection,
- challenge feedback in Pygame side panel,
- challenge explanation in bottom panel,
- explanation helper for challenge and confidence-view feedback,
- tests,
- CI support.

Not implemented yet:

- feature subsampling per split,
- out-of-bag score aggregation,
- tree diversity visualization,
- per-tree vote inspection,
- train/test toggle in visualization,
- export screenshots/GIFs.

## Dataset variants

The demo includes two synthetic train/test dataset variants.

### `axis_aligned`

Two classes are placed mostly on opposite sides of the `x1` axis.

This variant is intentionally simple and useful for comparing one tree with an ensemble.

### `xor`

Two classes occupy diagonal regions.

This variant is harder for shallow trees and useful for showing why tree-based models can benefit from depth and ensemble voting.

## Train/test split

Unlike most Level 1 demos, this demo uses separate training and test splits.

This is intentional.

Random Forest Bagging Lab focuses on practical model evaluation:

```text
train accuracy -> how well the model fits known data
test accuracy  -> how well the model generalizes
```

High train accuracy alone is not enough.

A model should also perform well on test data.

## Bootstrap sampling

Bootstrap sampling means drawing examples from the training dataset with replacement.

This means:

- one sample may appear multiple times,
- some samples may not appear at all,
- every tree can train on a slightly different dataset.

Samples not selected into a given bootstrap sample are called out-of-bag samples.

These OOB samples are useful for explaining how random forests can estimate generalization.

## Bootstrap ratio control

The UI allows changing the bootstrap sample ratio used by the random forest.

A smaller ratio means that every tree trains on a smaller bootstrap sample.

This can increase tree diversity, but each tree also sees less data.

## Majority voting

Predictions from many trees can be combined into one final prediction:

```text
tree_1 -> class_0
tree_2 -> class_1
tree_3 -> class_1

final prediction -> class_1
```

The implementation also computes vote confidence:

```text
vote confidence = winning votes / number of trees
```

For example, if 7 out of 10 trees vote for `class_1`, the final prediction is `class_1` with confidence `0.70`.

When there is a tie, the current implementation selects the lowest class label. This keeps voting deterministic and easy to test.

## Single-tree baseline

The demo includes a single-tree baseline.

The baseline trains one recursive decision tree on the training split and evaluates it on both:

```text
train split
test split
```

This gives future Random Forest models a meaningful point of comparison.

A single tree can fit simple data very well, but it may become unstable when data is noisy or when the decision boundary is complex.

## Random forest model

The demo includes a small educational random forest model.

The model works as follows:

```text
training split
-> bootstrap sample for tree 1
-> bootstrap sample for tree 2
-> bootstrap sample for tree N
-> train one decision tree per bootstrap sample
-> collect tree predictions
-> majority vote
-> final prediction + vote confidence
```

The forest is evaluated on both train and test splits.

This makes it possible to compare:

```text
single tree vs random forest
train accuracy vs test accuracy
individual prediction vs ensemble vote
```

## CLI comparison report

The command-line version includes a readable comparison report.

For each dataset, the report compares:

- single-tree train accuracy,
- single-tree test accuracy,
- random forest train accuracy,
- random forest test accuracy,
- generalization gap,
- forest vote confidence,
- test accuracy delta,
- winner by test accuracy.

This makes the command-line version useful before the Pygame visualization is introduced.

## How to run the CLI

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
```

## How to run the UI

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab-ui
```

## Controls

| Key | Action |
| --- | ------ |
| `D` | Toggle dataset: `axis_aligned` / `xor` |
| `Up` | Increase forest tree count |
| `Down` | Decrease forest tree count |
| `W` | Increase max tree depth |
| `S` | Decrease max tree depth |
| `B` | Increase bootstrap sample ratio |
| `V` | Decrease bootstrap sample ratio |
| `C` | Toggle forest confidence view |
| `Left` | Decrease data noise |
| `Right` | Increase data noise |
| `N` | Generate a new dataset seed |
| `R` | Reset demo |
| `Esc` | Quit |

## Pygame visualization

The UI compares one decision tree with a random forest.

The left panel shows the single-tree baseline.

The right panel shows the random forest.

Visual encoding:

- circles represent training samples,
- squares represent test samples,
- X marks show misclassified test samples,
- background color shows predicted decision regions.

This makes it possible to compare how one model and an ensemble divide the same feature space.

## Confidence view

Confidence view changes how forest decision regions are drawn.

When confidence view is off, regions show only the final predicted class.

When confidence view is on, pale regions mean weaker agreement between trees, while stronger regions mean higher vote confidence.

This makes ensemble uncertainty easier to see.

## Challenge mode

The UI includes a Random Forest challenge mode.

Current target:

```text
forest test accuracy >= 0.90
tree_count <= 25
generalization gap <= 0.15
```

The goal is not only to get a good result.

The goal is to build a forest that generalizes well without using an unnecessarily large number of trees.

This makes the task closer to practical model selection.

## Explanation helper

The demo includes a small explanation helper for the bottom UI panel.

It builds short student-facing messages for:

- challenge success,
- challenge failure,
- confidence view enabled,
- confidence view disabled,
- compact challenge targets.

This keeps the renderer focused on drawing, while explanation logic remains testable.

## Suggested classroom flow

A recommended classroom flow:

1. Start with the CLI report.
2. Compare single-tree and random-forest metrics.
3. Explain train accuracy and test accuracy.
4. Explain generalization gap.
5. Run the Pygame UI.
6. Start with `axis_aligned`.
7. Switch to `xor`.
8. Compare left and right panels.
9. Increase and decrease tree count.
10. Change max depth.
11. Change bootstrap ratio.
12. Enable confidence view.
13. Increase noise.
14. Try to satisfy challenge mode.

## Related concepts

- ensemble learning,
- bagging,
- bootstrap sampling,
- random forest,
- majority voting,
- vote confidence,
- variance reduction,
- overfitting,
- generalization,
- train/test split,
- model comparison,
- generalization gap,
- model selection.
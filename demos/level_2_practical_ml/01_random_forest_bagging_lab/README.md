# Random Forest Bagging Lab

Random Forest Bagging Lab is an interactive Level 2 demo from **Interactive ML Labs**.

The demo will show how multiple decision trees can be combined into a more stable ensemble classifier.

## What this demo will teach

This demo focuses on ensemble learning.

The key idea:

```text
dataset -> bootstrap samples -> many trees -> voting -> final prediction
```

# Random Forest Bagging Lab

A single decision tree can be easy to interpret, but it can also overfit.

A random forest reduces this risk by training many trees on different bootstrap samples and combining their predictions through voting.

## Why this demo comes after Decision Tree Splitter

Decision Tree Splitter shows how one tree learns recursive splits.

Random Forest Bagging Lab extends that idea:

- one decision tree can overfit,
- many trees can vote together,
- bootstrap sampling creates different training subsets,
- feature randomness creates diverse trees,
- voting stabilizes predictions,
- ensemble accuracy can be better than a single tree.

This makes Random Forest a natural first Level 2 demo.

## Current status

### Implemented

- package structure,
- workspace configuration,
- command-line placeholder,
- smoke test,
- connection to `ml_lab_core`.

### Not implemented yet

- synthetic train/test dataset,
- bootstrap sampling,
- single-tree reuse or adapter,
- forest training,
- majority voting,
- out-of-bag intuition,
- decision boundary visualization,
- tree diversity visualization,
- challenge mode,
- explanation panel.

## How to run

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
```

## Planned interactions

The interactive version will likely support:

- changing number of trees,
- changing max depth,
- changing bootstrap ratio,
- changing feature sampling,
- changing data noise,
- comparing one tree with a forest,
- showing vote confidence,
- showing train/test accuracy,
- visualizing ensemble decision regions.

## Planned learning goals

Students should learn:

- what bagging is,
- what bootstrap sampling is,
- why tree diversity matters,
- how majority voting works,
- why ensembles can improve stability,
- why random forests often generalize better than one deep tree,
- how model complexity and ensemble size affect results.

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
- train/test split.
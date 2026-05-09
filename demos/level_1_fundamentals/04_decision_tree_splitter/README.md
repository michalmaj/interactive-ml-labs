# Decision Tree Splitter

Decision Tree Splitter is an interactive Level 1 demo from **Interactive ML Labs**.

The demo shows how decision trees classify data by recursively splitting the feature space into simpler regions.

Students can observe:

- two-dimensional binary classification data,
- axis-aligned splits,
- Gini impurity,
- entropy,
- information gain,
- manual split evaluation,
- decision stump behavior,
- recursive decision tree behavior,
- decision regions,
- overfitting and underfitting intuition,
- challenge mode,
- explanation panel feedback.

## What this demo teaches

This demo focuses on tree-based classification intuition.

The key idea:

```text
data -> choose split -> split region -> repeat -> leaf prediction
```

A decision tree does not learn one smooth boundary like logistic regression. Instead, it asks a sequence of simple questions.

Example:

```text
if x1 <= 0.25:
    go left
else:
    go right
```

For two-dimensional data, a split is visible as a vertical or horizontal line.

## Why this demo comes after logistic regression

The Logistic Regression Boundary Lab shows a global linear decision boundary.

Decision Tree Splitter shows a different kind of decision surface:

- logistic regression learns one linear boundary,
- decision trees build many axis-aligned splits,
- logistic regression uses probabilities and thresholding,
- decision trees use rules and leaves,
- shallow trees are interpretable,
- deep trees can overfit.

This contrast helps students understand that different models can solve classification tasks using very different assumptions.

## Dataset variants

The demo includes two synthetic dataset variants.

### `axis_aligned`

Two classes are placed mostly on opposite sides of the `x1` axis.

This dataset is intentionally easy for a decision stump. A single vertical split can work well when noise is low.

### `xor`

Two classes occupy diagonal regions.

This dataset is intentionally harder. A single split is not enough, which makes it useful for explaining tree depth and recursive splitting.

## Implemented features

Implemented:

- package structure,
- workspace configuration,
- command-line prototype,
- synthetic binary classification dataset,
- axis-aligned dataset layout,
- XOR dataset layout,
- class count helper,
- class probability helper,
- Gini impurity,
- entropy impurity,
- split candidate representation,
- split evaluation with weighted child impurity,
- information gain,
- best split search,
- manual split prototype,
- manual split snapshots with left/right masks,
- manual split metrics and annotations,
- decision stump model,
- root split fitting,
- leaf majority-class prediction,
- stump predictions,
- stump snapshots with training accuracy,
- recursive decision tree model,
- `max_depth` control,
- `min_samples_split` and `min_samples_leaf` constraints,
- tree node representation,
- tree predictions,
- recursive tree snapshots with node and leaf counts,
- basic Pygame visualization,
- decision region rendering,
- recursive split line rendering,
- keyboard controls for dataset kind, criterion, max depth, noise, and seed,
- manual split mode in Pygame,
- manual split feature switching,
- manual threshold controls,
- manual split gain and impurity panel,
- challenge mode for simple trees,
- dataset-specific challenge depth limits,
- explanation panel helper for automatic tree, manual split, and challenge feedback,
- tests,
- CI support.

Not implemented yet:

- graphical sliders,
- tree diagram view,
- pruning visualization,
- train/test split visualization,
- real dataset example,
- export screenshots/GIFs.

## How to run

Run the command-line version:

```bash
uv run --package decision-tree-splitter decision-tree-splitter
```

Run the Pygame visualization:

```bash
uv run --package decision-tree-splitter decision-tree-splitter-ui
```

## Controls

| Key | Action |
| --- | ------ |
| `M` | Toggle mode: automatic tree / manual split |
| `D` | Toggle dataset kind: `axis_aligned` / `xor` |
| `G` | Toggle impurity criterion: `gini` / `entropy` |
| `F` | Toggle manual split feature: `x1` / `x2` |
| `Q` | Move manual split threshold down |
| `E` | Move manual split threshold up |
| `Up` | Increase max tree depth |
| `Down` | Decrease max tree depth |
| `Left` | Decrease data noise |
| `Right` | Increase data noise |
| `S` | Generate a new dataset seed |
| `R` | Reset the demo |
| `Esc` | Quit |

## Impurity metrics

The demo includes two node impurity metrics:

- Gini impurity,
- entropy.

Both metrics are low when a node is pure and higher when classes are mixed.

A pure node contains samples from one class only.

A mixed node contains samples from multiple classes.

## Split scoring

A split is evaluated by comparing impurity before and after the split:

```text
information_gain = parent_impurity - weighted_child_impurity
```

A good split creates child nodes that are more pure than the parent node.

The current implementation supports:

- Gini-based split scoring,
- entropy-based split scoring,
- automatic best split search.

## Manual split mode

Manual split mode allows students to choose a split manually.

The demo evaluates the selected split using:

- parent impurity,
- left child impurity,
- right child impurity,
- weighted child impurity,
- information gain,
- left and right sample counts.

This makes it possible to compare visual intuition with mathematical split quality.

## Decision stump

A decision stump is a one-level decision tree:

```text
root split
├── left leaf
└── right leaf
```

The stump chooses the best root split using information gain. Then each leaf predicts the majority class of samples assigned to that leaf.

This is the simplest real decision tree model.

It works well for simple axis-aligned data, but it cannot solve XOR-like data because one split is not enough.

## Recursive decision tree

A recursive tree can apply multiple splits:

```text
root split
├── left subtree
└── right subtree
```

This is important for datasets such as `xor`.

A decision stump cannot solve XOR because one split is not enough. A depth-2 tree can split once by `x1` and then split child regions by `x2`, which is enough to solve the low-noise XOR dataset.

## Challenge mode

The demo includes a dataset-specific challenge mode.

Current challenge targets:

```text
axis_aligned: accuracy >= 0.95 with max_depth <= 1
xor:          accuracy >= 0.95 with max_depth <= 2
```

The goal is not only to get high accuracy. The goal is to use a tree that is simple enough for the dataset.

This teaches that model complexity should match problem complexity.

## Suggested classroom flow

A recommended classroom flow:

1. Start with `axis_aligned`.
2. Set `max_depth = 1`.
3. Show that one split can solve the task.
4. Switch to `xor`.
5. Keep `max_depth = 1`.
6. Show that one split is not enough.
7. Increase `max_depth` to 2.
8. Show recursive splitting.
9. Switch to manual split mode.
10. Let students move the split and compare information gain.
11. Increase noise and discuss why the task becomes harder.
12. Discuss why deeper trees may overfit.

## Related concepts

- classification,
- decision trees,
- tree nodes,
- leaves,
- splits,
- Gini impurity,
- entropy,
- information gain,
- decision stump,
- recursive tree,
- max depth,
- overfitting,
- underfitting,
- pruning,
- interpretability,
- model complexity.
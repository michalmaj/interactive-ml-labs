# Decision Tree Splitter

Decision Tree Splitter is an interactive Level 1 demo from **Interactive ML Labs**.

The demo will show how decision trees classify data by recursively splitting the feature space into simpler regions.

## What this demo will teach

This demo focuses on tree-based classification intuition.

Students should understand that a decision tree does not learn a smooth linear boundary like logistic regression. Instead, it asks a sequence of simple questions.

The key idea:

```text
data -> choose split -> split region -> repeat -> leaf prediction
```

For two-dimensional data, a split can be visualized as a vertical or horizontal line.

## Why this demo comes after logistic regression

The Logistic Regression Boundary Lab shows a global linear decision boundary.

Decision trees show a different kind of decision surface:

- Logistic regression learns one linear boundary.
- Decision trees build many axis-aligned splits.
- Logistic regression produces probabilities through sigmoid.
- Decision trees classify by following rules.
- Shallow trees are interpretable.
- Deep trees can overfit.

This makes decision trees a strong next step in Level 1.

## Current status

### Implemented

- Package structure
- Workspace configuration
- Command-line placeholder
- Smoke test
- Connection to `ml_lab_core`
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
- manual split metrics and annotations.

### Not implemented yet

- Pygame visualization
- Keyboard controls
- Challenge mode
- Explanation panel

## How to run

```bash
uv run --package decision-tree-splitter decision-tree-splitter
```

## How to run the UI

```bash
uv run --package decision-tree-splitter decision-tree-splitter-ui
```

| Key     | Action                                        |
| ------- | --------------------------------------------- |
| `D`     | Toggle dataset kind: `axis_aligned` / `xor`   |
| `G`     | Toggle impurity criterion: `gini` / `entropy` |
| `Up`    | Increase max tree depth                       |
| `Down`  | Decrease max tree depth                       |
| `Left`  | Decrease data noise                           |
| `Right` | Increase data noise                           |
| `S`     | Generate a new dataset seed                   |
| `R`     | Reset the demo                                |
| `Esc`   | Quit                                          |


## Planned interactions

The interactive version will likely support:

- Adding a manual vertical or horizontal split
- Stepping through automatic split selection
- Changing max tree depth
- Changing minimum samples per leaf
- Changing data noise
- Changing dataset seed
- Comparing manual splits with algorithm-selected splits
- Resetting the tree

## Planned learning goals

Students should learn:

- What a decision tree is
- What a split is
- Why trees use axis-aligned cuts
- What Gini impurity means
- What entropy means
- What information gain means
- Why shallow trees are interpretable
- Why deep trees can overfit
- How max depth affects generalization

## Related concepts

- Classification
- Decision trees
- Tree nodes
- Leaves
- Splits
- Gini impurity
- Entropy
- Information gain
- Overfitting
- Underfitting
- Pruning
- Interpretability


## Dataset variants

The demo starts with two synthetic dataset variants:

### `axis_aligned`

Two classes are placed mostly on opposite sides of the `x1` axis.

This dataset is intentionally easy for a decision stump. A single vertical split can work well when noise is low.

### `xor`

Two classes occupy diagonal regions.

This dataset is intentionally harder. A single split is not enough, which makes it useful for explaining tree depth and recursive splitting.

## Impurity metrics

The demo now includes two node impurity metrics:

- Gini impurity,
- entropy.

Both metrics are low when a node is pure and higher when classes are mixed.

These functions will be used in the next step to score candidate splits and compute information gain.

## Manual split prototype

The demo now includes a manual split prototype.

A selected split is evaluated with:

- parent impurity,
- left child impurity,
- right child impurity,
- weighted child impurity,
- information gain,
- left and right sample counts.

This prototype will later be connected to the Pygame UI, where students will be able to move a split line and compare their split with the algorithm-selected best split.
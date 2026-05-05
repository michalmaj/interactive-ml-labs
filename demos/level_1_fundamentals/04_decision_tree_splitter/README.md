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

### Not implemented yet

- Synthetic classification dataset
- Gini impurity
- Entropy
- Information gain
- Best split search
- Decision tree model
- Manual split mode
- Pygame visualization
- Keyboard controls
- Challenge mode
- Explanation panel

## How to run

```bash
uv run --package decision-tree-splitter decision-tree-splitter
```

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

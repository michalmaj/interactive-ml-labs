# Logistic Regression Boundary Lab

Logistic Regression Boundary Lab is an interactive Level 1 demo from **Interactive ML Labs**.

The demo will show how logistic regression solves binary classification problems by estimating class probabilities and applying a decision threshold.

## What this demo will teach

This demo focuses on the basic intuition behind probabilistic classification.

Students should understand that logistic regression does not directly output only a class label. It first estimates a probability and then converts that probability into a class using a threshold.

The key idea:

```text
linear score -> sigmoid -> probability -> threshold -> predicted class
```

## Why this demo comes after k-NN

The k-NN demo shows classification through distance and neighbor voting.

This demo will show a different view of classification:

- k-NN uses nearby examples,
- logistic regression learns a decision boundary,
- k-NN voting is local,
- logistic regression produces a global linear boundary,
- logistic regression can expose probabilities and thresholds.

## Current status

### Implemented

- package structure,
- workspace configuration,
- command-line placeholder,
- smoke test,
- connection to `ml_lab_core`.
- synthetic binary classification dataset,
- sigmoid helper,
- binary cross-entropy loss,
- probability thresholding,
- confusion matrix counts,
- accuracy, precision, and recall metrics,
- stepwise logistic regression model,
- gradient descent training for logistic regression,
- probability prediction,
- threshold-based class prediction,
- model snapshots with loss, metrics, weights, and probabilities,
- basic Pygame visualization,
- decision boundary rendering,
- live loss history,
- keyboard controls for learning rate, threshold, noise, and seed.

### Not implemented yet

- decision threshold control,
- confusion matrix metrics,
- Pygame visualization,
- keyboard controls,
- challenge mode,
- explanation panel.

## How to run

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
```

## How to run the UI

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab-ui
```

| Key     | Action                          |
| ------- | ------------------------------- |
| `Space` | Run or pause automatic training |
| `N`     | Perform one training step       |
| `R`     | Reset the demo                  |
| `Up`    | Increase learning rate          |
| `Down`  | Decrease learning rate          |
| `Q`     | Decrease threshold              |
| `E`     | Increase threshold              |
| `Left`  | Decrease data noise             |
| `Right` | Increase data noise             |
| `S`     | Generate a new dataset seed     |
| `Esc`   | Quit                            |


## Planned controls

The interactive version will likely support:

- Increasing or decreasing learning rate
- Increasing or decreasing threshold
- Changing data noise
- Changing dataset seed
- Toggling probability view
- Toggling predicted class view
- Resetting the demo

## Planned learning goals

Students should learn:

- What a binary classifier does
- What a probability estimate means
- How sigmoid transforms a score into a probability
- How a threshold changes predicted classes
- Why accuracy alone can be misleading
- What false positives and false negatives mean
- How a linear decision boundary behaves

## Related concepts

- Binary classification
- Logistic regression
- Sigmoid function
- Decision boundary
- Decision threshold
- Probability
- Binary cross-entropy
- Confusion matrix
- Precision
- Recall
- False positives
- False negatives

## Dataset

The first dataset version is synthetic and contains two 2D Gaussian-like point clouds:

- `class_0`,
- `class_1`.

The dataset is intentionally simple because the first version of the demo should focus on probability, thresholding, and a linear decision boundary.


# Logistic Regression Boundary Lab

Logistic Regression Boundary Lab is an interactive Level 1 demo from **Interactive ML Labs**.

The demo shows how logistic regression solves binary classification problems by learning a linear decision boundary, estimating class probabilities, and applying a decision threshold.

Students can observe:

- two-dimensional binary classification data,
- logistic regression training with gradient descent,
- probability background for `class_1`,
- decision boundary controlled by threshold,
- binary cross-entropy loss,
- accuracy, precision, and recall,
- true positives, true negatives, false positives, and false negatives,
- precision-recall challenge mode,
- explanation panel feedback.

## What this demo teaches

This demo focuses on probabilistic binary classification.

The key idea:

```text
linear score -> sigmoid -> probability -> threshold -> predicted class
```

Logistic regression does not only produce a class. It first estimates the probability of the positive class. The final class depends on the selected decision threshold.

This makes the demo useful for explaining why classification is not only about accuracy, but also about decision costs, false positives, false negatives, precision, and recall.

## Why this demo comes after k-NN

The k-NN Vote Map demo shows classification through distance and neighbor voting.

This demo shows a different classification mechanism:

- k-NN uses nearby examples,
- logistic regression learns model parameters,
- k-NN voting is local,
- logistic regression produces a global linear boundary,
- logistic regression exposes probabilities,
- logistic regression makes threshold tuning visible.

This contrast helps students understand that different classifiers can solve similar tasks using very different ideas.

## Current features

Implemented:

- package structure,
- workspace configuration,
- command-line prototype,
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
- probability background grid for class `1`,
- live loss history,
- confusion matrix panel in the Pygame UI,
- keyboard controls for learning rate, threshold, noise, and seed,
- precision-recall challenge mode using a hidden synthetic test set,
- explanation panel helper for model, threshold, and challenge feedback,
- tests,
- CI support.

Not implemented yet:

- non-linear datasets,
- regularization,
- polynomial feature expansion,
- real dataset example,
- ROC/PR curve visualization,
- graphical sliders.

## How to run

Run the command-line version:

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
```

Run the Pygame visualization:

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab-ui
```

## Controls

| Key | Action |
| --- | ------ |
| `Space` | Run or pause automatic training |
| `N` | Perform one training step |
| `R` | Reset the demo |
| `Up` | Increase learning rate |
| `Down` | Decrease learning rate |
| `Q` | Decrease threshold |
| `E` | Increase threshold |
| `Left` | Decrease data noise |
| `Right` | Increase data noise |
| `S` | Generate a new dataset seed |
| `Esc` | Quit |

Changing learning rate, threshold, noise, or seed resets the model.

## Dataset

The demo uses a synthetic binary classification dataset.

The dataset contains two 2D Gaussian-like point clouds:

- `class_0`,
- `class_1`.

The dataset is intentionally simple because the demo focuses on the following ideas:

- probability estimation,
- thresholding,
- binary classification metrics,
- false positives and false negatives,
- linear decision boundaries.

The noise parameter controls how strongly the two classes overlap.

## Probability background

The colored background shows the predicted probability of `class_1`.

This is not the same thing as the predicted class.

The model first computes:

```text
probability = sigmoid(linear_score)
```

Then the threshold converts probability into class:

```text
if probability >= threshold:
    predicted class = class_1
else:
    predicted class = class_0
```

Changing the learned weights changes the probability background.

Changing the threshold changes the decision boundary, but it does not directly change the learned probabilities.

## Challenge mode

The current challenge is:

```text
Reach recall >= 0.90 while keeping precision >= 0.80.
```

The challenge is evaluated on a hidden synthetic test set.

This is intentionally not a plain accuracy challenge. Logistic regression is a good place to teach that practical classifiers often require balancing different types of errors.

The side panel shows:

- challenge status,
- current precision,
- target precision,
- current recall,
- target recall.

## Suggested classroom use

A recommended classroom flow:

1. Start with the default parameters.
2. Run a few individual training steps with `N`.
3. Observe how the decision boundary changes.
4. Run automatic training with `Space`.
5. Observe how loss decreases.
6. Compare accuracy, precision, and recall.
7. Change threshold using `Q` and `E`.
8. Observe how false positives and false negatives change.
9. Try to complete the precision-recall challenge.
10. Increase noise and discuss why the task becomes harder.

## Related concepts

- binary classification,
- logistic regression,
- sigmoid function,
- probability,
- decision threshold,
- decision boundary,
- binary cross-entropy,
- gradient descent,
- confusion matrix,
- true positives,
- true negatives,
- false positives,
- false negatives,
- accuracy,
- precision,
- recall,
- hidden test set,
- model evaluation.

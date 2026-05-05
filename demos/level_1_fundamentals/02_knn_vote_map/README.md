# k-NN Vote Map

k-NN Vote Map is an interactive Level 1 demo from **Interactive ML Labs**.

The demo shows how the k-nearest neighbors algorithm classifies points based on nearby labeled examples.

Students can observe:

- two-dimensional classification data,
- nearest neighbors of a query point,
- majority voting,
- decision regions,
- the effect of changing `k`,
- the effect of data noise,
- hidden-test accuracy,
- challenge mode feedback.

## What this demo teaches

This demo focuses on the basic intuition behind distance-based classification.

Unlike gradient descent models, k-NN does not learn model parameters such as weights or biases. Instead, it stores training examples and classifies a new point by looking at nearby examples.

The key idea:

```text
new point -> find k nearest neighbors -> vote -> predicted class
```

## Current features

Implemented:

- synthetic binary classification dataset,
- Euclidean distance metric,
- vectorized distance computation from many points to one query point,
- k-nearest neighbors classifier,
- nearest-neighbor voting for one query point,
- command-line prediction prototype,
- basic Pygame visualization,
- random query point classification,
- mouse-based query point selection,
- nearest-neighbor link visualization,
- decision background grid showing predicted class regions,
- keyboard controls for `k`, noise, seed, and reset,
- accuracy challenge mode using a hidden synthetic test set,
- explanation panel for voting and challenge feedback,
- tests,
- CI support.

Not implemented yet:

- alternative distance metrics,
- feature scaling visualization,
- train/test point rendering,
- multi-class datasets,
- real dataset example,
- graphical sliders.

## How to run

Run the command-line version:

```bash
uv run --package knn-vote-map knn-vote-map
```

Run the Pygame visualization:

```bash
uv run --package knn-vote-map knn-vote-map-ui
```

## Controls

| Action       | Description                              |
| ------------ | ---------------------------------------- |
| Click on map | Classify clicked point                   |
| `N`          | Sample and classify a random query point |
| `R`          | Reset the demo                           |
| `Up`         | Increase `k`                             |
| `Down`       | Decrease `k`                             |
| `Left`       | Decrease data noise                      |
| `Right`      | Increase data noise                      |
| `S`          | Generate a new dataset seed              |
| `Esc`        | Quit                                     |


Changing `k`, noise, or seed resets the classifier and recomputes the decision background.

## Challenge mode

The current challenge is:

```text
Reach hidden-test accuracy >= 0.90.
```

The demo evaluates the current k-NN configuration on a separate synthetic test set.

The side panel shows:

- challenge status,
- current accuracy,
- target accuracy,
- number of correct predictions.

This helps students understand that a model should not only look good on visible training points. It should also perform well on unseen data.

## Suggested classroom use

A recommended classroom flow:

- Start with the default parameters.
- Click several points in clearly separated regions.
- Click points close to the decision boundary.
- Change `k` and observe how neighbor voting changes.
- Compare small `k` and large `k`.
- Increase noise and discuss why classification becomes harder.
- Ask students to find settings that keep hidden-test accuracy above the challenge target.
- Discuss why a smoother decision boundary is not always better.

## Related concepts

- classification,
- nearest neighbors,
- distance metrics,
- Euclidean distance,
- majority voting,
- decision boundaries,
- overfitting,
- underfitting,
- hidden test set,
- accuracy,
- feature scaling.

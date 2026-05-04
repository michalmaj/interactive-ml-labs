# k-NN Vote Map

k-NN Vote Map is an interactive Level 1 demo from **Interactive ML Labs**.

The demo will show how the k-nearest neighbors algorithm classifies points based on nearby labeled examples.

## What this demo will teach

Students should understand that k-NN does not learn model parameters in the same way as gradient descent.

Instead, it stores examples and classifies a new point by looking at its nearest neighbors.

The key idea:

```text
new point -> find k nearest neighbors -> vote -> predicted class
```

## Current status

Implemented:

- package structure,
- workspace configuration,
- command-line placeholder,
- smoke test,
- connection to ml_lab_core,
- synthetic binary classification dataset,
- Euclidean distance metric,
- vectorized distance computation from many points to one query point.

Not implemented yet:

- k-nearest neighbors classifier,
- vote visualization,
- Pygame UI,
- keyboard controls,
- challenge mode,
- explanation panel.

## How to run

```bash
uv run --package knn-vote-map knn-vote-map
```

## Planned controls

The interactive version will likely support:

- increasing or decreasing k,
- changing the dataset seed,
- changing noise level,
- adding a test point,
- toggling neighbor visualization,
- resetting the demo.

## Related concepts

- classification,
- nearest neighbors,
- distance metrics,
- majority voting,
- decision boundaries,
- overfitting,
- underfitting,
- feature scaling.


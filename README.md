# Interactive ML Labs

[![CI](https://github.com/michalmaj/interactive-ml-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/michalmaj/interactive-ml-labs/actions/workflows/ci.yml)

Interactive ML Labs is a collection of visual, interactive machine learning demos designed for teaching and experimentation.

The project is built as a professional Python repository with:

- clear project structure,
- automated tests,
- linting and formatting,
- continuous integration,
- documentation for each demo,
- step-by-step educational explanations.

## Project goals

The main goal is to help students understand machine learning algorithms through interactive visualizations, live metrics, and small challenge-based tasks.

Each demo should answer three questions:

1. What problem does the algorithm solve?
2. How does the algorithm work step by step?
3. What happens when important parameters change?

## Planned demo levels

The recommended way to explore the demos is the unified Pygame app:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Individual demo entry points are intentionally still supported. They are useful for focused development, isolated testing, and classroom situations where one demo should be launched directly.

### Level 1 — Fundamentals

Core machine learning intuition:

- gradient descent,
- logistic regression,
- k-nearest neighbors,
- decision trees,
- support vector machines,
- neural networks basics,
- k-means,
- DBSCAN,
- PCA,
- reinforcement learning gridworld.

### Level 2 — Practical ML

Practical model evaluation, robustness, and interpretation:

- random forests,
- boosting,
- Gaussian mixtures,
- hierarchical clustering,
- anomaly detection,
- imbalanced learning,
- calibration,
- feature importance,
- active learning,
- semi-supervised learning.

### Level 3 — Advanced / Showcase

Advanced and visually rich topics:

- Clustering Lab: K-Means phases and DBSCAN comparison,
- Naive Bayes spam detection,
- CNN filter visualization,
- t-SNE / UMAP,
- association rules,
- recommender systems,
- time-series forecasting,
- Hidden Markov Models,
- hyperparameter optimization,
- autoencoders,
- graph machine learning.

## Development approach

The project follows a lightweight GitHub Flow:

1. Create a branch for each logical change.
2. Open a pull request.
3. Describe what changed and how to test it.
4. Run automated checks.
5. Merge only working changes into `main`.

This repository is also intended to demonstrate how real software projects are developed.

## Current status

The unified app is now the recommended guided experience. It includes all implemented Level 1 and Level 2 demos:

- Gradient Descent Playground,
- k-NN Vote Map,
- Logistic Regression Boundary Lab,
- Decision Tree Splitter,
- Random Forest Bagging Lab,
- Boosting Mistake Lab.

It also includes the first real Level 3 lab:

- Clustering Lab, with K-Means assignment/update phases, point-to-centroid links, inertia trend, draggable points, and a DBSCAN comparison mode.

The original Level 1 and Level 2 demos can still be run as standalone commands. The unified app is the recommended path for students, while standalone entry points remain part of the supported development and teaching workflow.

## Next milestones

Near-term work focuses on turning Level 3 into a stronger guided learning track while keeping the shell stable:

- collect review feedback on Clustering Lab after classroom-style use,
- improve the Level 3 learning flow from theory to interaction to mini-challenges,
- decide the next Level 3 demo candidate, likely PCA / dimensionality reduction or model comparison,
- continue polishing Polish copy inside demos without forcing translations of standard ML terms,
- keep app settings, scaling, and help overlays stable as new Level 3 scenes are added.

## Usage

See [USAGE.md](USAGE.md) for local run commands.

Polish version: [USAGE.pl.md](USAGE.pl.md).

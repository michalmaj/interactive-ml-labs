# Roadmap

This roadmap describes the planned development order.

The project is intentionally developed in small pull requests. Each pull request should introduce one logical change.

## Phase 0 — Project foundation

Goal: create a professional project foundation before implementing demos.

Status:

- [x] Initialize repository structure
- [x] Configure uv workspace
- [x] Configure Ruff and pytest
- [x] Add GitHub Actions CI
- [x] Add architecture and roadmap documentation
- [x] Add demo template

## Phase 1 — Core abstractions

Goal: create the minimal shared infrastructure required by demos.

Planned pull requests:

- [x] Add algorithm snapshots
- [x] Add dataset snapshots
- [x] Add stepwise algorithm protocol
- [x] Add base demo scene protocol
- [x] Add basic metrics history utilities

## Phase 2 — First reference demo

Goal: implement the first complete demo as a reference for future demos.

Reference demo:

- Gradient Descent Playground

Planned pull requests:

- [x] Add synthetic regression dataset
- [x] Add mean squared error metric
- [x] Add stepwise gradient descent implementation
- [x] Add algorithm tests
- [x] Add CLI prototype
- [x] Add Pygame renderer
- [x] Add controls
- [ ] Add live metrics panel
- [ ] Add challenge mode
- [ ] Add explanation panel
- [ ] Add README, theory, and challenges documents

## Phase 3 — First three demos

Goal: validate the architecture on three different machine learning problem types.

Planned demos:

- Gradient Descent Playground
- k-NN Vote Map
- k-means Arena

These cover:

- regression and optimization,
- classification,
- clustering.

## Phase 4 — Complete Level 1

Goal: implement all Level 1 fundamental demos.

Planned demos:

- Logistic Regression Boundary Lab
- Decision Tree Splitter
- SVM Kernel Duel
- XOR Neural Playground
- DBSCAN Density Radar
- PCA Projector
- RL Gridworld

## Phase 5 — Level 2

Goal: add practical machine learning demos.

Topics include:

- ensemble learning,
- boosting,
- anomaly detection,
- imbalanced learning,
- calibration,
- interpretability,
- active learning,
- semi-supervised learning.

## Phase 6 — Level 3

Goal: add advanced and showcase demos.

Topics include:

- text classification,
- CNNs,
- embeddings,
- recommender systems,
- time series,
- HMMs,
- hyperparameter optimization,
- autoencoders,
- graph machine learning.
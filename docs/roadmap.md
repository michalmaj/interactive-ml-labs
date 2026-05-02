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
- [ ] Add architecture and roadmap documentation
- [ ] Add demo template

## Phase 1 — Core abstractions

Goal: create the minimal shared infrastructure required by demos.

Planned pull requests:

- [ ] Add algorithm snapshots
- [ ] Add dataset snapshots
- [ ] Add stepwise algorithm protocol
- [ ] Add base demo scene protocol
- [ ] Add basic metrics history utilities

## Phase 2 — First reference demo

Goal: implement the first complete demo as a reference for future demos.

Reference demo:

- Gradient Descent Playground

Planned pull requests:

- [ ] Add synthetic regression dataset
- [ ] Add mean squared error metric
- [ ] Add stepwise gradient descent implementation
- [ ] Add algorithm tests
- [ ] Add CLI prototype
- [ ] Add Pygame renderer
- [ ] Add controls
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
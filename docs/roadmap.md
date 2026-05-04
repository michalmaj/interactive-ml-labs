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

- [x] Add Gradient Descent Playground package skeleton
- [x] Add synthetic regression dataset
- [x] Add mean squared error metric
- [x] Add stepwise gradient descent implementation
- [x] Add algorithm tests
- [x] Add CLI prototype
- [x] Add basic Pygame renderer
- [x] Split Pygame renderer from app loop
- [x] Add keyboard parameter controls
- [x] Add challenge mode
- [x] Polish demo documentation
- [ ] Add screenshots or GIF
- [ ] Add graphical parameter controls
- [x] Add explanation panel

## Phase 3 — Second Level 1 demo: k-NN Vote Map

Goal: implement the second interactive demo focused on classification intuition.

Planned pull requests:

- [x] Add k-NN Vote Map package skeleton
- [x] Add synthetic classification dataset
- [ ] Add distance metrics
- [ ] Add k-nearest neighbors classifier
- [ ] Add vote map CLI prototype
- [ ] Add Pygame visualization
- [ ] Add keyboard controls
- [ ] Add challenge mode
- [ ] Polish demo documentation


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
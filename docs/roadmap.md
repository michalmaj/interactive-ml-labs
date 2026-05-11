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
- [x] Add distance metrics
- [x] Add k-nearest neighbors classifier
- [x] Add vote map CLI prototype
- [x] Add Pygame visualization
- [x] Add keyboard controls
- [x] Add mouse query point selection
- [x] Add decision background grid
- [x] Add challenge mode
- [x] Add explanation panel
- [x] Polish demo documentation


## Phase 4 — Third Level 1 demo: Logistic Regression Boundary Lab

Goal: implement the third interactive demo focused on probabilistic binary classification.

Planned pull requests:

- [x] Add Logistic Regression Boundary Lab package skeleton
- [x] Add synthetic binary classification dataset
- [x] Add sigmoid and classification metrics
- [x] Add logistic regression model
- [x] Add CLI prototype
- [x] Add Pygame visualization
- [x] Add keyboard controls for learning rate and threshold
- [x] Add probability background
- [x] Add confusion matrix metrics
- [x] Add challenge mode
- [x] Add explanation panel
- [x] Polish demo documentation

## Phase 5 — Fourth Level 1 demo: Decision Tree Splitter

Goal: implement the fourth interactive demo focused on interpretable rule-based classification.

Planned pull requests:

- [x] Add Decision Tree Splitter package skeleton
- [x] Add synthetic classification dataset
- [x] Add impurity metrics: Gini and entropy
- [x] Add split scoring with information gain
- [x] Add manual split prototype
- [x] Add decision stump model
- [x] Add recursive decision tree model
- [x] Add CLI prototype
- [x] Add Pygame visualization
- [x] Add manual split mode
- [x] Add tree depth controls
- [x] Add challenge mode
- [x] Add explanation panel
- [x] Polish demo documentation

## Phase 6 — First Level 2 demo: Random Forest Bagging Lab

Goal: implement the first Level 2 demo focused on ensemble learning, bagging, and model stability.

Planned pull requests:

- [x] Add Random Forest Bagging Lab package skeleton
- [x] Add synthetic train/test classification dataset
- [x] Add bootstrap sampling utilities
- [x] Add majority voting and vote confidence
- [x] Add single-tree baseline
- [x] Add random forest model
- [x] Add CLI prototype
- [x] Add Pygame visualization
- [x] Add controls for tree count and max depth
- [x] Add train/test accuracy comparison
- [x] Add challenge mode
- [ ] Add explanation panel
- [ ] Polish demo documentation
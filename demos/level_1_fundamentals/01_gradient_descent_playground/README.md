# Gradient Descent Playground

Gradient Descent Playground is an interactive demo planned for Level 1 of Interactive ML Labs.

The goal of this demo is to help students understand how a model can learn by gradually minimizing a loss function.

## Current status

This package currently contains only the initial project skeleton.

Implemented:

- package structure,
- workspace configuration,
- command-line entry point,
- smoke test,
- placeholder connection to `ml_lab_core`,
- synthetic linear regression dataset.
- mean squared error metric.

Not implemented yet:

- gradient descent algorithm,
- loss history visualization,
- Pygame renderer,
- parameter controls,
- challenge mode,
- explanation panel.

## How to run

```bash
uv run --package gradient-descent-playground gradient-descent-playground
```

## Planned learning goals

Students should learn:

- what a loss function is,
- what gradient descent does,
- how learning rate affects optimization,
- why too small learning rate is slow,
- why too large learning rate may make training unstable,
- how model parameters change during training.


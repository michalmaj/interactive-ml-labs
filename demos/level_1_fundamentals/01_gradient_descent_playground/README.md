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
- stepwise linear regression trained with gradient descent.
- basic Pygame visualization,
- keyboard controls for run, pause, step, reset, and quit.
- separated Pygame application loop and renderer.
- keyboard controls for learning rate, noise level, and dataset seed.

Not implemented yet:

- loss history visualization,
- challenge mode,
- explanation panel.

## How to run

Run the command-line version:

```bash
uv run --package gradient-descent-playground gradient-descent-playground
```

Run the Pygame visualization:

```bash
uv run --package gradient-descent-playground gradient-descent-playground-ui
```

## Controls

- `Space` — run or pause automatic learning,
- `N` — perform one gradient descent step,
- `R` — reset the demo,
- `Up` / `Down` — increase or decrease learning rate,
- `Left` / `Right` — decrease or increase data noise,
- `S` — generate a new dataset seed,
- `Esc` — quit.

## Planned learning goals

Students should learn:

- what a loss function is,
- what gradient descent does,
- how learning rate affects optimization,
- why too small learning rate is slow,
- why too large learning rate may make training unstable,
- how model parameters change during training.


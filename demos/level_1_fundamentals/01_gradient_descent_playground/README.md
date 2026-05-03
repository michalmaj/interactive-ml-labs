# Gradient Descent Playground

Gradient Descent Playground is an interactive Level 1 demo from **Interactive ML Labs**.

The demo shows how a simple linear regression model learns by minimizing mean squared error using gradient descent.

Students can observe:

- synthetic regression data,
- the current regression line,
- loss decreasing over time,
- model parameters changing step by step,
- the effect of learning rate,
- the effect of data noise,
- a simple challenge mode.

## What this demo teaches

This demo focuses on the basic intuition behind optimization in machine learning.

Students should understand that a model does not magically “know” the correct answer. It starts with initial parameters and gradually changes them to reduce prediction error.

The demo is intentionally simple:

```text
y_pred = weight * x + bias
```

Only two parameters are learned:

- `weight`,
- `bias`.

This makes it possible to see clearly what gradient descent does.

## Current features

Implemented:

- synthetic linear regression dataset,
- mean squared error metric,
- stepwise linear regression trained with gradient descent,
- command-line prototype,
- Pygame visualization,
- keyboard controls,
- learning rate adjustment,
- data noise adjustment,
- dataset seed changes,
- live loss history,
- challenge mode,
- tests,
- CI support.

Not implemented yet:

- graphical sliders,
- batch size control,
- polynomial regression,
- train/test split visualization,
- comparison with closed-form linear regression.

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

| Key     | Action                            |
| ------- | --------------------------------- |
| `Space` | Run or pause automatic learning   |
| `N`     | Perform one gradient descent step |
| `R`     | Reset the demo                    |
| `Up`    | Increase learning rate            |
| `Down`  | Decrease learning rate            |
| `Left`  | Decrease data noise               |
| `Right` | Increase data noise               |
| `S`     | Generate a new dataset seed       |
| `Esc`   | Quit                              |

Changing learning rate, noise, or seed resets the demo.

## Challenge mode

The current challenge is:

```text
Reach loss <= 1.0 before 80 gradient descent steps.
```

The challenge status is shown in the side panel.

Possible statuses:

- active — the challenge is still running,
- success — the target loss was reached,
- failed — the step limit was reached before the target loss.

## Suggested classroom use

A recommended classroom flow:

1. Start with the default parameters.
2. Press `N` several times and discuss what changes after each step.
3. Press `Space` and observe automatic learning.
4. Increase learning rate and compare convergence speed.
5. Set learning rate too high and observe instability.
6. Increase data noise and discuss why perfect fitting becomes harder.
7. Ask students to beat the challenge for several seeds.

## Related concepts

- linear regression,
- loss function,
- mean squared error,
- gradient,
- learning rate,
- optimization,
- convergence,
- overfitting intuition,
- noisy data.


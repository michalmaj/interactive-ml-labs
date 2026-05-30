# Uzycie

Ten dokument wyjasnia, jak uruchamiac Interactive ML Labs lokalnie.

## Wymagania

- Python 3.12 albo nowszy
- uv

Zainstaluj albo zaktualizuj zaleznosci z katalogu glownego repozytorium:

```bash
uv sync
```

## Rekomendowana aplikacja

Unified Pygame app jest rekomendowanym wejsciem do przegladania dem:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Obecny przeplyw shell-a:

```text
wybor jezyka
-> wybor poziomu
-> wybor demo
-> ekran startowy demo
-> placeholder ekranu demo
-> pauza / pomoc
```

Prawdziwe integracje dem beda dodawane stopniowo. Kazde demo nadal mozna uruchamiac osobno.

## Osobne dema

Kazde demo nadal dziala jako osobny pakiet.

### Gradient Descent Playground

```bash
uv run --package gradient-descent-playground gradient-descent-playground
uv run --package gradient-descent-playground gradient-descent-playground-ui
```

### k-NN Vote Map

```bash
uv run --package knn-vote-map knn-vote-map
uv run --package knn-vote-map knn-vote-map-ui
```

### Logistic Regression Boundary Lab

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab-ui
```

### Decision Tree Splitter

```bash
uv run --package decision-tree-splitter decision-tree-splitter
uv run --package decision-tree-splitter decision-tree-splitter-ui
```

### Random Forest Bagging Lab

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
uv run --package random-forest-bagging-lab random-forest-bagging-lab-ui
```

### Boosting Mistake Lab

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

## Kontrola jakosci

Uruchom lint i sprawdzenie formatowania:

```bash
uv run ruff check .
uv run ruff format --check .
```

Uruchom testy unified app shell:

```bash
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
```

Uruchom testy jednego demo:

```bash
uv run --package boosting-mistake-lab pytest demos/level_2_practical_ml/02_boosting_mistake_lab/tests
```

## Workflow developerski

Uzywaj malej galezi dla kazdej logicznej zmiany.

Typowy przeplyw:

```bash
git switch main
git pull
git switch -c feat/my-change
uv run ruff check .
uv run ruff format --check .
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
git push -u origin feat/my-change
```

Potem otworz pull request zgodnie z szablonem repozytorium.

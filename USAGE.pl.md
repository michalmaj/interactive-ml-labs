# Użycie

Ten dokument wyjaśnia, jak uruchamiać Interactive ML Labs lokalnie.

## Wymagania

- Python 3.12 albo nowszy
- uv

Zainstaluj albo zaktualizuj zależności z katalogu głównego repozytorium:

```bash
uv sync
```

## Rekomendowana aplikacja

Unified Pygame app to zalecany sposób przechodzenia przez laboratoria:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Domyślnie aplikacja otwiera okno `1280x720`. Skalowanie scen o stałym rozmiarze jest włączone, żeby większe dema mieściły się bezpiecznie w oknie, a adaptacyjny rozmiar okna i fullscreen są dostępne w menu ustawień.

Aplikacja zapamiętuje język, fullscreen, adaptacyjny rozmiar okna i skalowanie scen o stałym rozmiarze w małym pliku ustawień użytkownika. Sama rozdzielczość okna nie jest zapisywana, bo adaptive sizing powinien przeliczać ją pod aktualny ekran.

Obecny przepływ aplikacji:

```text
wybór języka
-> wybór poziomu
-> wybór demo
-> ekran startowy demo
-> ekran demo
-> pauza / pomoc
```

Kolejne dema będą podpinane stopniowo. Wszystkie obecne dema z Level 1 i Level 2 działają już w unified app, a każde z nich nadal można uruchomić osobno.

Unified app zawiera też natywne laboratoria z Level 3:

- Clustering Lab pokazuje fazy K-Means, inertia, przesuwanie punktów oraz tryb porównawczy DBSCAN.
- PCA Lab pokazuje presety danych, noise, ręczne obracanie projekcji, dopasowany kierunek PCA, explained variance, residuals rekonstrukcji i reconstruction error.
- Model Comparison Lab pokazuje założenia Logistic Regression, k-NN i Decision Tree na tych samych datasetach, z train/test score, kompaktowym confusion summary i podświetlaniem błędów testowych.
- Calibration Lab pokazuje calibration prawdopodobieństw przez reliability diagram, rozkład score, legendę raw-vs-scaled score, accuracy@0.5, Brier score, ECE, podświetlenie worst gap, error bars kalibracji i temperature scaling.
- t-SNE / UMAP Exploration Lab pokazuje deterministyczne toy embeddingi, porównanie raw-vs-embedding, wskazówki datasetów, etykiety klas, seed drift, strojenie sąsiedztwa i lokalne połączenia sąsiadów.
- Model Monitoring Drift Lab to natywny prototyp Level 3 dla data drift, metric drift, monitoring windows i alert thresholds.

Podstawowe sterowanie:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `Up` / `Down` | Przesuń zaznaczenie |
| Ruch myszy | Przesuń zaznaczenie na wskazaną pozycję |
| Klik myszy / `Enter` | Aktywuj zaznaczoną pozycję |
| `Esc` / `Backspace` | Wróć albo otwórz pauzę |
| `H` | Pokaż lub ukryj pomoc dla wybranego demo |
| `L` | Zmień język |
| `S` | Otwórz ustawienia poza aktywnym demo |

## Osobne dema

Każde oryginalne demo z Level 1 i Level 2 nadal działa jako osobny pakiet i może być uruchamiane bez unified app. Natywne laby, takie jak Clustering Lab, PCA Lab, Model Comparison Lab, Calibration Lab i t-SNE / UMAP Exploration Lab, uruchamia się z unified app.

### Natywne laby Level 3

Sterowanie w Clustering Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-4` | Zmień preset danych |
| `-` / `=` | Zmień `k` w K-Means albo `eps` w DBSCAN |
| `Space` | Wykonaj fazę K-Means albo ponownie uruchom DBSCAN |
| `M` | Przełącz K-Means / DBSCAN |
| `C` | Pokaż lub ukryj linie punkt-centroid |
| Przeciąganie myszy | Przesuń punkt danych |

Sterowanie w PCA Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień preset danych |
| `-` / `=` | Zmień noise |
| `N` | Wygeneruj nową próbkę |
| `Left` / `Right` | Obróć kierunek projekcji |
| `F` | Przełącz dopasowany kierunek PCA |
| `C` | Pokaż lub ukryj residual lines rekonstrukcji |

Sterowanie w Model Comparison Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Wybierz Logistic Regression, k-NN albo Decision Tree |
| `D` | Zmień preset danych |
| `-` / `=` | Zmień parametr aktywnego modelu |
| `A` | Pokaż albo ukryj nieaktywne granice |
| `E` | Pokaż albo ukryj błędnie sklasyfikowane punkty testowe |
| `R` | Zresetuj podgląd |

Sterowanie w Calibration Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień preset kalibracji |
| `-` / `=` | Zmień temperature scaling |
| `O` | Pokaż albo ukryj raw score sprzed temperature scaling |
| `E` | Pokaż albo ukryj error bars kalibracji |
| `R` | Zresetuj podgląd |

Sterowanie w t-SNE / UMAP Exploration Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień preset danych |
| `M` | Przełącz podgląd t-SNE / UMAP |
| `-` / `=` | Zmień perplexity / neighbors |
| `S` | Zmień wariant seed i sprawdź drift |
| `L` | Pokaż albo ukryj lokalne połączenia sąsiadów |
| `O` | Pokaż albo ukryj raw high-dimensional layout |
| `R` | Zresetuj podgląd |

Sterowanie w Model Monitoring Drift Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-4` | Zmień preset monitoringu |
| `D` / `M` | Wybierz sygnał data drift / metric drift |
| `-` / `=` | Zmień alert threshold |
| `A` | Potwierdź aktywny alert do analizy |
| `R` | Zresetuj podgląd |

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

## Kontrola jakości

Uruchom lint i sprawdzenie formatowania:

```bash
uv run ruff check .
uv run ruff format --check .
```

Uruchom testy unified app:

```bash
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
```

Uruchom testy jednego demo:

```bash
uv run --package boosting-mistake-lab pytest demos/level_2_practical_ml/02_boosting_mistake_lab/tests
```

## Workflow developerski

Używaj małej gałęzi dla każdej logicznej zmiany.

Typowy przepływ:

```bash
git switch main
git pull
git switch -c feat/my-change
uv run ruff check .
uv run ruff format --check .
uv run --package interactive-ml-labs-app pytest apps/interactive_ml_labs/tests
git push -u origin feat/my-change
```

Potem otwórz pull request zgodnie z szablonem repozytorium.

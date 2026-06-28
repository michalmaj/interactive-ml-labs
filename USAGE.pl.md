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
-> ekran główny z postępem nauki
-> ścieżka nauki albo wybór poziomu
-> wybór lekcji albo demo
-> ekran startowy demo
-> ekran demo
-> pauza / pomoc
```

Ekran główny pokazuje zbiorczy postęp w prowadzonych ścieżkach nauki. Naciśnij `C` albo kliknij podświetloną linię z następnym krokiem w panelu postępu, żeby od razu kontynuować sugerowaną lekcję. Możesz też wejść w "Prowadzone ścieżki nauki", żeby ręcznie przejrzeć ścieżki, lekcje, zadania, odznaki i postęp w teorii.

Unified app jest teraz główną ścieżką przechodzenia przez laboratoria. Wszystkie obecne dema z Level 1 i Level 2 działają już w appce, a Level 3 ma rosnący zestaw zaawansowanych labów app-only. Oryginalne pakiety demo nadal można uruchomić osobno, a natywne laby app-only uruchamia się z unified app.

Unified app zawiera te natywne laby z Level 1:

- Linear Regression Line Fit Lab pokazuje slope, intercept, residuals, MSE loss i least-squares fitting przez ręczne ustawianie prostej.
- K-Means Intro Lab pokazuje kroki assignment/update, centroidy, k, inertia i to, czemu niższa inertia nie jest całą historią clusteringu.
- Distance Metrics Lab pokazuje query points, nearest neighbors, Euclidean distance, Manhattan distance, Chebyshev distance i to, czemu k-NN zależy od metryki.
- SVM Margin Lab pokazuje decision boundaries, support vectors, szerokość margin i to, czemu liczy się najszerszy poprawny separator.
- Activation Functions Lab pokazuje sigmoid, tanh, ReLU, zakresy outputu, saturation i przepływ local gradient.
- Neural Network Playground pokazuje mały forward pass przez inputs, weights, hidden units, activation, probability, target i loss.

Unified app zawiera te natywne laby z Level 2:

- Data Leakage Lab pokazuje podejrzane cechy, dostępność w czasie predykcji, leaky-vs-clean validation scores i nawyk nieufania metrykom, które wyglądają zbyt idealnie.
- Train / Validation / Test Split Lab pokazuje model selection przez validation scores, overfitting przez train-validation gaps i zostawianie test jako finalnego uczciwego checku.
- Feature Scaling Lab pokazuje raw-vs-scaled feature ranges, range ratio, modele wrażliwe na skalę oraz wpływ scaling na accuracy i iterations.
- Feature Importance Lab pokazuje permutation/model importance, skorelowane cechy, leakage warnings i stabilność rankingu.
- Gaussian Mixture Intro Lab pokazuje soft responsibilities, hard assignment, mixture weights, kształty komponentów i nakładające się klastry.
- Anomaly Detection Lab pokazuje anomaly scores, thresholds, szum alertów, false positives i pominięte anomalie.
- Hyperparameter Tuning Lab pokazuje validation curves, intuicję grid search, train-validation gaps i wybór parametrów po validation score.
- Class Imbalance Lab pokazuje pułapki accuracy, trade-offs precision/recall, false negatives i strojenie decision threshold, gdy jedna klasa jest rzadka.

Unified app zawiera też natywne laboratoria z Level 3:

- Clustering Lab pokazuje fazy K-Means, inertia, przesuwanie punktów oraz tryb porównawczy DBSCAN.
- PCA Lab pokazuje presety danych, noise, ręczne obracanie projekcji, dopasowany kierunek PCA, explained variance, residuals rekonstrukcji i reconstruction error.
- Model Comparison Lab pokazuje założenia Logistic Regression, k-NN i Decision Tree na tych samych datasetach, z train/test score, kompaktowym confusion summary i podświetlaniem błędów testowych.
- Calibration Lab pokazuje calibration prawdopodobieństw przez reliability diagram, rozkład score, legendę raw-vs-scaled score, accuracy@0.5, Brier score, ECE, podświetlenie worst gap, error bars kalibracji i temperature scaling.
- t-SNE / UMAP Exploration Lab pokazuje deterministyczne toy embeddingi, porównanie raw-vs-embedding, wskazówki datasetów, etykiety klas, seed drift, strojenie sąsiedztwa i lokalne połączenia sąsiadów.
- Model Monitoring Drift Lab pokazuje data drift, metric drift, monitoring windows, alert thresholds, lead signal, alert rate, persistence, trend i potwierdzanie analizy.
- Time Series Forecasting Lab pokazuje holdout forecasts, modele naive/moving-average/trend-seasonal, forecast horizon, residuals, MAE/RMSE, bias i uncertainty bands.

Podstawowe sterowanie:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `Up` / `Down` | Przesuń zaznaczenie |
| Ruch myszy | Przesuń zaznaczenie na wskazaną pozycję |
| Klik myszy / `Enter` | Aktywuj zaznaczoną pozycję |
| `C` na ekranie głównym | Kontynuuj następną lekcję ze ścieżki nauki |
| Klik w następną lekcję w panelu postępu | Kontynuuj następną lekcję ze ścieżki nauki |
| `Esc` / `Backspace` | Wróć albo otwórz pauzę |
| `H` | Pokaż lub ukryj pomoc dla wybranego demo |
| `L` | Zmień język |
| `S` | Otwórz ustawienia poza aktywnym demo |

## Osobne dema

Każde oryginalne demo z Level 1 i Level 2 nadal działa jako osobny pakiet i może być uruchamiane bez unified app. Natywne laby, takie jak Linear Regression Line Fit Lab, K-Means Intro Lab, Distance Metrics Lab, SVM Margin Lab, Activation Functions Lab, Neural Network Playground, Data Leakage Lab, Train / Validation / Test Split Lab, Feature Scaling Lab, Feature Importance Lab, Gaussian Mixture Intro Lab, Anomaly Detection Lab, Hyperparameter Tuning Lab, Class Imbalance Lab, Clustering Lab, PCA Lab, Model Comparison Lab, Calibration Lab, t-SNE / UMAP Exploration Lab i Time Series Forecasting Lab, uruchamia się z unified app.

### Natywne laby Level 1

Sterowanie w Linear Regression Line Fit Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień dataset |
| `Left` / `Right` | Zmień slope |
| `Up` / `Down` | Zmień intercept |
| `F` | Przejdź do least-squares fit |
| `R` | Zresetuj lab |

Sterowanie w K-Means Intro Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `Space` | Wykonaj jeden krok assignment albo update centroidów |
| `A` | Uruchom albo zatrzymaj auto-run |
| `-` / `=` | Zmień `k` |
| `1-3` | Zmień dataset |
| `C` | Pokaż albo ukryj linie punkt-centroid |
| `N` | Wygeneruj nową próbkę |
| `R` | Zresetuj lab |

Sterowanie w Distance Metrics Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień dataset |
| Strzałki | Przesuń query point |
| `M` | Zmień metrykę distance |
| `R` | Zresetuj lab |

Sterowanie w SVM Margin Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień dataset |
| `Left` / `Right` | Obróć boundary |
| `Up` / `Down` | Przesuń boundary |
| `F` | Przejdź do wide-margin fit |
| `R` | Zresetuj lab |

Sterowanie w Activation Functions Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień activation |
| `Left` / `Right` | Przesuń input x |
| `0` | Zresetuj x do zera |
| `R` | Zresetuj lab |

Sterowanie w Neural Network Playground:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień przykład input |
| `A` | Zmień activation |
| `-` / `=` | Zmień weight scale |
| `Up` / `Down` | Zmień hidden bias |
| `R` | Zresetuj playground |

### Natywne laby Level 2

Sterowanie w Data Leakage Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz leakage |
| `L` | Włącz albo usuń podejrzaną cechę leakage |
| `R` | Zresetuj podgląd |

Sterowanie w Train / Validation / Test Split Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz splitu |
| `-` / `=` / `0` | Zmień albo zresetuj complexity modelu |
| `R` | Zresetuj podgląd |

Sterowanie w Feature Scaling Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz scaling |
| `S` | Włącz albo wyłącz feature scaling |
| `M` | Zmień model |
| `R` | Zresetuj podgląd |

Sterowanie w Feature Importance Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz importance |
| `M` | Zmień metodę importance |
| `C` | Pokaż albo ukryj grupy korelacji |
| `L` | Pokaż albo ukryj ostrzeżenie leakage |
| `R` | Zresetuj lab |

Sterowanie w Gaussian Mixture Intro Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| Strzałki | Przesuń query point |
| `-` / `=` | Zmień liczbę komponentów |
| `H` | Przełącz hard assignment |
| `D` | Pokaż albo ukryj density ellipses |
| `1-3` | Zmień dataset |
| `R` | Zresetuj lab |

Sterowanie w Anomaly Detection Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz anomalii |
| `-` / `=` / `0` | Zmień albo zresetuj threshold anomalii |
| `S` | Pokaż albo ukryj pierścienie score |
| `R` | Zresetuj lab |

Sterowanie w Hyperparameter Tuning Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz tuningu |
| `-` / `=` / `0` | Zmień albo zresetuj wartość parametru |
| `R` | Zresetuj podgląd |

Sterowanie w Class Imbalance Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz imbalance |
| `-` / `=` / `0` | Zmień albo zresetuj decision threshold |
| `R` | Zresetuj podgląd |

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
| `-` / `=` / `0` | Zmień albo zresetuj alert threshold |
| `A` | Potwierdź aktywny alert do analizy |
| `R` | Zresetuj podgląd |

Na co patrzeć w Model Monitoring Drift Lab:

- Jasna linia to aktywny sygnał; przygaszona linia pokazuje drugi sygnał do porównania.
- `okna` porównują baseline window z current window i dodają krótką etykietę trendu.
- `luka`, `threshold` i `severity` pokazują, czy obecna zmiana jest poniżej, blisko czy powyżej wybranego alert threshold.
- `pierwszy alert`, `alert rate`, `persistence` i `lead signal` pomagają odróżnić jednorazowy pik od powtarzalnego sygnału produkcyjnego.
- `analiza` zmienia się po naciśnięciu `A`, więc lab pokazuje mały workflow monitoringu zamiast traktować każdy alert jak automatyczną panikę.

Sterowanie w Time Series Forecasting Lab:

| Klawisz / wejście | Akcja |
| ----------------- | ----- |
| `1-3` | Zmień scenariusz time series |
| `M` | Zmień model forecastingu |
| `-` / `=` | Zmień forecast horizon |
| `U` | Pokaż albo ukryj uncertainty band |
| `E` | Pokaż albo ukryj residuals |
| `R` | Zresetuj lab |

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

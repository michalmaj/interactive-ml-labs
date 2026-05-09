# Decision Tree Splitter — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia jest zrozumienie, jak drzewo decyzyjne dzieli przestrzeń danych.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest split,
- czym jest węzeł decyzyjny,
- czym jest liść,
- czym jest Gini impurity,
- czym jest entropy,
- czym jest information gain,
- czym jest decision stump,
- czym jest pełne drzewo rekurencyjne,
- dlaczego głębokie drzewa mogą się przeuczać.

## 2. Uruchomienie

Wersja tekstowa:

```bash
uv run --package decision-tree-splitter decision-tree-splitter
```

Wersja interaktywna:

```bash
uv run --package decision-tree-splitter decision-tree-splitter-ui
```

## 3. Sterowanie

| Klawisz | Działanie |
| ------- | --------- |
| `M` | Przełącz tryb: automatyczne drzewo / manualny split |
| `D` | Przełącz dataset: `axis_aligned` / `xor` |
| `G` | Przełącz kryterium: `gini` / `entropy` |
| `F` | Przełącz cechę manualnego splitu: `x1` / `x2` |
| `Q` | Przesuń manualny threshold w dół |
| `E` | Przesuń manualny threshold w górę |
| `Up` | Zwiększ `max_depth` |
| `Down` | Zmniejsz `max_depth` |
| `Left` | Zmniejsz poziom szumu |
| `Right` | Zwiększ poziom szumu |
| `S` | Wygeneruj nowy dataset przez zmianę seeda |
| `R` | Zresetuj demo |
| `Esc` | Zamknij okno |

## 4. Zadania podstawowe

### Zadanie 1 — uruchomienie demo

1. Uruchom UI.
2. Sprawdź, gdzie znajdują się punkty obu klas.
3. Sprawdź panel boczny.
4. Sprawdź panel dolny z wyjaśnieniem.

Odpowiedz:

- Ile klas widać na planszy?
- Jakie metryki są pokazane w panelu bocznym?
- Co oznaczają kolorowe regiony?
- Co oznaczają linie splitów?

### Zadanie 2 — dataset axis_aligned

1. Ustaw dataset `axis_aligned`.
2. Ustaw `max_depth = 1`.
3. Obserwuj split i regiony decyzyjne.
4. Sprawdź accuracy.

Odpowiedz:

- Czy jeden split wystarcza?
- Która cecha jest najważniejsza?
- Czy drzewo jest proste i interpretowalne?

### Zadanie 3 — dataset XOR

1. Przełącz dataset na `xor`.
2. Ustaw `max_depth = 1`.
3. Sprawdź accuracy.
4. Zwiększ `max_depth` do 2.
5. Porównaj regiony decyzyjne.

Odpowiedz:

- Dlaczego jeden split nie wystarcza?
- Co zmienia się po zwiększeniu głębokości?
- Ile liści pojawia się przy rozwiązaniu XOR?

## 5. Metryki nieczystości

### Zadanie 4 — Gini i entropy

Porównaj trzy węzły:

```text
A: [class_0, class_0, class_0, class_0]
B: [class_0, class_0, class_1, class_1]
C: [class_0, class_0, class_0, class_1]
```

Odpowiedz:

- Który węzeł jest najczystszy?
- Który węzeł jest najbardziej wymieszany?
- Który węzeł powinien mieć największą nieczystość?
- Dlaczego split powinien zmniejszać nieczystość?

## 6. Ocena splitu

### Zadanie 5 — information gain

Dany jest węzeł rodzic:

```text
parent: [class_0, class_0, class_1, class_1]
```

Split A:

```text
left:  [class_0, class_0]
right: [class_1, class_1]
```

Split B:

```text
left:  [class_0, class_1]
right: [class_0, class_1]
```

Odpowiedz:

- Który split ma większy information gain?
- Który split tworzy czystsze dzieci?
- Dlaczego Split A jest lepszy dla drzewa decyzyjnego?

## 7. Manual split mode

### Zadanie 6 — ręczne szukanie splitu

1. Uruchom UI.
2. Naciśnij `M`, aby przejść do manual split mode.
3. Ustaw dataset `axis_aligned`.
4. Przesuwaj threshold klawiszami `Q` i `E`.
5. Przełącz cechę splitu klawiszem `F`.
6. Porównaj gain dla splitu po `x1` i `x2`.

Odpowiedz:

- Który split ma największy gain dla `axis_aligned`?
- Dlaczego split po złej cesze daje słaby gain?
- Czy najlepszy split jest zgodny z intuicją wizualną?

### Zadanie 7 — manual split na XOR

1. Przełącz dataset na `xor`.
2. Pozostań w manual split mode.
3. Spróbuj znaleźć jeden dobry split.
4. Obserwuj information gain.

Odpowiedz:

- Dlaczego jeden manualny split nie rozwiązuje całego XOR?
- Czy split po `x1` albo `x2` wystarcza?
- Dlaczego potrzebne są kolejne poziomy drzewa?

## 8. Decision stump

### Zadanie 8 — decision stump na axis_aligned

1. Ustaw dataset `axis_aligned`.
2. Ustaw `max_depth = 1`.
3. Sprawdź accuracy.
4. Sprawdź liczbę liści.

Odpowiedz:

- Dlaczego decision stump działa dobrze?
- Ile splitów wykonuje stump?
- Ile liści ma stump?

### Zadanie 9 — decision stump na XOR

1. Ustaw dataset `xor`.
2. Ustaw `max_depth = 1`.
3. Sprawdź accuracy.
4. Porównaj z `max_depth = 2`.

Odpowiedz:

- Dlaczego stump nie rozwiązuje XOR?
- Co musi zrobić głębsze drzewo?
- Dlaczego XOR jest dobrym przykładem potrzeby rekurencji?

## 9. Pełne drzewo rekurencyjne

### Zadanie 10 — wpływ max_depth

1. Ustaw dataset `xor`.
2. Sprawdź wyniki dla `max_depth = 1`.
3. Sprawdź wyniki dla `max_depth = 2`.
4. Sprawdź wyniki dla `max_depth = 3`.

Odpowiedz:

- Przy jakiej głębokości model rozwiązuje XOR?
- Czy większa głębokość daje zawsze widoczną poprawę?
- Co dzieje się z liczbą węzłów i liści?

### Zadanie 11 — zbyt głębokie drzewo

1. Zwiększ poziom szumu.
2. Ustaw wysokie `max_depth`.
3. Obserwuj regiony decyzyjne.
4. Porównaj z płytkim drzewem.

Odpowiedz:

- Czy regiony stają się bardziej złożone?
- Czy model może dopasowywać się do szumu?
- Dlaczego głębsze drzewo może mieć overfitting?

## 10. Challenge mode

### Zadanie 12 — challenge dla axis_aligned

1. Ustaw dataset `axis_aligned`.
2. Ustaw `max_depth = 1`.
3. Sprawdź, czy challenge ma status `success`.
4. Zwiększ `max_depth`.
5. Sprawdź, czy challenge nadal jest spełniony.

Odpowiedz:

- Dlaczego `axis_aligned` powinien być rozwiązany prostym drzewem?
- Dlaczego challenge nie premiuje zbyt głębokiego drzewa?
- Czy wysoka accuracy wystarcza?

### Zadanie 13 — challenge dla XOR

1. Ustaw dataset `xor`.
2. Ustaw `max_depth = 1`.
3. Sprawdź accuracy i challenge status.
4. Zwiększ `max_depth` do 2.
5. Sprawdź, czy challenge ma status `success`.

Odpowiedz:

- Dlaczego `xor` wymaga większej głębokości?
- Dlaczego `max_depth = 2` jest wystarczające przy małym szumie?
- Czy `max_depth = 3` jest potrzebne?

## 11. Panel wyjaśnień

### Zadanie 14 — interpretacja panelu

1. Uruchom UI.
2. Przełącz tryb `auto_tree` i `manual_split` klawiszem `M`.
3. Obserwuj tekst w dolnym panelu.
4. Ustaw manualny split tak, aby był poprawny.
5. Przesuń threshold tak daleko, aby split stał się niepoprawny.
6. Porównaj komunikaty.

Odpowiedz:

- Czy panel pomaga zrozumieć aktualny tryb pracy?
- Czy komunikat o niepoprawnym splicie jest zgodny z tym, co widać na planszy?
- Czy challenge status pomaga dobrać odpowiednią głębokość drzewa?

## 12. Pytania kontrolne

1. Czym jest split?
2. Czym jest węzeł?
3. Czym jest liść?
4. Co oznacza Gini impurity?
5. Co oznacza entropy?
6. Co oznacza information gain?
7. Dlaczego split powinien zmniejszać nieczystość?
8. Czym jest decision stump?
9. Czym jest pełne drzewo rekurencyjne?
10. Czym `max_depth = 1` różni się od `max_depth = 2`?
11. Dlaczego XOR wymaga więcej niż jednego splitu?
12. Dlaczego głębokie drzewo może się przeuczać?
13. Dlaczego challenge mode wymaga prostego drzewa?
14. Czym decision tree różni się od logistic regression?
15. Czym decision tree różni się od k-NN?

## 13. Zadanie dodatkowe

Zaproponuj własny challenge mode dla drzewa decyzyjnego.

Przykłady:

- osiągnij accuracy powyżej 0.95 przy maksymalnie 3 liściach,
- rozwiąż XOR przy jak najmniejszej liczbie węzłów,
- znajdź najlepszy manualny split dla `axis_aligned`,
- uzyskaj wysoką accuracy przy dużym poziomie szumu,
- porównaj Gini i entropy dla tego samego datasetu.

Opisz:

- cel challenge,
- ograniczenia,
- sposób punktacji,
- czego student może się z niego nauczyć.
# Decision Tree Splitter — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia będzie zrozumienie, jak drzewo decyzyjne dzieli przestrzeń danych.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest split,
- czym jest węzeł decyzyjny,
- czym jest liść,
- czym jest Gini impurity,
- czym jest entropy,
- czym jest information gain,
- dlaczego głębokie drzewa mogą się przeuczać.

## 2. Uruchomienie

Aktualna wersja zawiera tylko placeholder:

```bash
uv run --package decision-tree-splitter decision-tree-splitter
```

Interaktywna wersja **Pygame** zostanie dodana w kolejnych PR-ach.

### 3. Planowane zadania podstawowe

#### Zadanie 1 — pierwszy split

Wybierz pionowy lub poziomy split.

Podziel dane na dwie części.

Sprawdź, czy po jednej stronie jest więcej punktów jednej klasy.

Odpowiedz:

- Czy split poprawia separację klas?
- Czy jedna strona splitu jest bardziej jednorodna?
- Czy split wydaje się intuicyjny?

#### Zadanie 2 — porównanie splitów

Wybierz kilka różnych splitów.

Porównaj ich **Gini impurity**.

Sprawdź, który split daje największą poprawę.

Odpowiedz:

- Który split jest najlepszy?
- Czy najlepszy split zawsze jest oczywisty wizualnie?
- Dlaczego algorytm potrzebuje miary jakości splitu?

#### Zadanie 3 — głębokość drzewa

Zbuduj płytkie drzewo.

Zbuduj głębsze drzewo.

Porównaj granicę decyzyjną.

Odpowiedz:

- Czy głębsze drzewo lepiej dopasowuje się do danych treningowych?
- Czy granica decyzyjna staje się bardziej złożona?
- Kiedy może pojawić się **overfitting**?

### 4. Planowany challenge mode

Przykładowy challenge:

```text
Osiągnij accuracy >= 0.90 przy max_depth <= 3.
```

Alternatywny challenge:

```text
Znajdź najmniejsze drzewo, które osiąga dobrą klasyfikację.
```

### 5. Pytania kontrolne

1. Czym jest split?
2. Czym jest węzeł?
3. Czym jest liść?
4. Co oznacza **Gini impurity**?
5. Co oznacza **entropy**?
6. Co oznacza **information gain**?
7. Dlaczego drzewo może mieć **overfitting**?
8. Dlaczego ograniczenie `max depth` może pomagać?
9. Czym **decision tree** różni się od **logistic regression**?
10. Czym **decision tree** różni się od **k-NN**?

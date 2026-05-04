# k-NN Vote Map — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia będzie zrozumienie, jak algorytm k-NN podejmuje decyzję klasyfikacyjną.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest klasyfikacja,
- co oznacza parametr `k`,
- jak działa głosowanie sąsiadów,
- dlaczego małe `k` może prowadzić do overfittingu,
- dlaczego duże `k` może prowadzić do underfittingu,
- dlaczego skala cech ma znaczenie.

## 2. Uruchomienie

Aktualna wersja zawiera tylko placeholder:

```bash
uv run --package knn-vote-map knn-vote-map
```

Interaktywna wersja Pygame zostanie dodana w kolejnych PR-ach.
Aktualna wersja generuje już syntetyczny zbiór danych klasyfikacyjnych, ale nie zawiera jeszcze klasyfikatora k-NN ani wizualizacji Pygame.

## 3. Planowane zadania podstawowe

### Zadanie 1 — klasyfikacja jednego punktu

1. Wybierz punkt testowy.
2. Ustaw `k = 1`.
3. Sprawdź najbliższego sąsiada.
4. Odczytaj przewidywaną klasę.

Odpowiedz:

- Który punkt był najbliższy?
- Jaką miał klasę?
- Dlaczego model wybrał taką predykcję?

### Zadanie 2 — głosowanie sąsiadów

1. Ustaw większe `k`, np. `k = 5`.
2. Sprawdź klasy najbliższych sąsiadów.
3. Policz głosy dla każdej klasy.

Odpowiedz:

- Która klasa wygrała głosowanie?
- Czy wynik różni się od `k = 1`?
- Dlaczego?

## 4. Planowane eksperymenty

### Eksperyment 1 — małe k

Sprawdź działanie modelu dla małych wartości `k`.

Odpowiedz:

- Czy granica decyzyjna jest gładka czy poszarpana?
- Czy pojedyncze punkty silnie wpływają na decyzję?
- Czy model wydaje się wrażliwy na szum?

### Eksperyment 2 — duże k

Sprawdź działanie modelu dla dużych wartości `k`.

Odpowiedz:

- Czy granica decyzyjna staje się gładsza?
- Czy model ignoruje lokalne szczegóły?
- Czy zbyt duże k może pogorszyć wynik?

### Eksperyment 3 — szum w danych

Dodaj więcej szumu do danych.

Odpowiedz:

- Czy klasy są trudniejsze do rozdzielenia?
- Czy małe `k` nadal działa dobrze?
- Czy większe `k` pomaga?

## 5. Planowany challenge mode

Przykładowy challenge:

```text
Osiągnij accuracy >= 90% na punktach testowych.
```

Ograniczenia:

- maksymalna liczba zmian parametru `k`,
- losowy seed danych,
- określony poziom szumu.

## 6. Pytania kontrolne

1. Co oznacza k w k-NN?
2. Dlaczego k = 1 może prowadzić do overfittingu?
3. Dlaczego bardzo duże k może prowadzić do underfittingu?
4. Co oznacza głosowanie sąsiadów?
5. Dlaczego odległość jest kluczowa w k-NN?
6. Dlaczego skalowanie cech może zmienić wynik?
7. Czy k-NN ma etap uczenia podobny do regresji liniowej?
8. Kiedy k-NN może być dobrym wyborem?
9. Kiedy k-NN może działać słabo?
10. Jak można ocenić jakość klasyfikatora?

## Klikany punkt testowy

W wersji interaktywnej można kliknąć dowolne miejsce na mapie.

Kliknięty punkt staje się punktem testowym, a algorytm k-NN:

1. liczy odległości do punktów treningowych,
2. wybiera `k` najbliższych sąsiadów,
3. pokazuje linie do sąsiadów,
4. wykonuje głosowanie,
5. wyświetla przewidywaną klasę.

Zadanie:

Klikaj punkty blisko granicy między klasami i obserwuj, jak zmiana `k` wpływa na predykcję.

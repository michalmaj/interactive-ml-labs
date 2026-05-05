# Logistic Regression Boundary Lab — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia będzie zrozumienie, jak regresja logistyczna podejmuje decyzję klasyfikacyjną.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest klasyfikacja binarna,
- czym różni się prawdopodobieństwo od klasy,
- jak działa próg decyzyjny,
- czym jest granica decyzyjna,
- czym są false positives i false negatives,
- dlaczego accuracy nie zawsze wystarcza.

## 2. Uruchomienie

Aktualna wersja zawiera tylko placeholder:

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
```

Interaktywna wersja Pygame zostanie dodana w kolejnych PR-ach.

## 3. Planowane zadania podstawowe

### Zadanie 1 — prawdopodobieństwo a klasa

1. Wybierz punkt testowy.
2. Odczytaj prawdopodobieństwo klasy `class_1`.
3. Zastosuj próg `0.5`.
4. Sprawdź przewidywaną klasę.

**Odpowiedz:**

- Czy prawdopodobieństwo jest tym samym co klasa?
- Co oznacza `probability = 0.8`?
- Co oznacza `probability = 0.4`?

### Zadanie 2 — zmiana progu decyzyjnego

1. Ustaw `threshold = 0.5`.
2. Sprawdź predykcje.
3. Zwiększ `threshold`.
4. Zmniejsz `threshold`.

**Odpowiedz:**

- Czy liczba punktów przewidzianych jako `class_1` się zmienia?
- Co dzieje się przy bardzo niskim progu?
- Co dzieje się przy bardzo wysokim progu?

### Zadanie 3 — granica decyzyjna

1. Obserwuj linię granicy decyzyjnej.
2. Klikaj punkty po obu stronach granicy.
3. Sprawdzaj prawdopodobieństwa.

**Odpowiedz:**

- Po której stronie granicy model przewiduje `class_1`?
- Czy punkty blisko granicy mają `probability` blisko `0.5`?
- Czy punkty daleko od granicy mają bardziej pewne predykcje?

## 4. Planowane eksperymenty

### Eksperyment 1 — learning rate

Sprawdź działanie modelu dla różnych wartości `learning rate`.

**Odpowiedz:**

- Czy model uczy się szybciej przy większym `learning rate`?
- Czy zbyt duży `learning rate` może pogorszyć uczenie?
- Czy `loss` maleje stabilnie?

### Eksperyment 2 — threshold i błędy

Sprawdź, jak `threshold` wpływa na `false positives` i `false negatives`.

**Odpowiedz:**

- Kiedy rośnie liczba `false positives`?
- Kiedy rośnie liczba `false negatives`?
- Czy `threshold = 0.5` zawsze jest najlepszy?

### Eksperyment 3 — szum w danych

Dodaj więcej szumu do danych.

**Odpowiedz:**

- Czy klasy trudniej rozdzielić linią?
- Czy model częściej popełnia błędy?
- Czy zmiana `threshold` pomaga?

## 5. Planowany challenge mode

Przykładowy challenge:

```text
Osiągnij recall >= 0.90 przy precision >= 0.80.
```

Alternatywny challenge:

```text
Znajdź threshold minimalizujący koszt false positives i false negatives.
```

## 6. Pytania kontrolne

1. Do czego służy regresja logistyczna?
2. Dlaczego nazwa „regresja” może być myląca?
3. Co oznacza sigmoid?
4. Co oznacza probability?
5. Jak probability zamienia się na klasę?
6. Co robi threshold?
7. Czym jest false positive?
8. Czym jest false negative?
9. Kiedy warto obniżyć threshold?
10. Kiedy warto podwyższyć threshold?
11. Czy accuracy zawsze wystarcza?
12. Czym regresja logistyczna różni się od k-NN?


W `CHALLENGES.pl.md` dopisz:

```md
## Pierwsza wizualizacja Pygame

Wersja interaktywna pokazuje:

- punkty dwóch klas,
- aktualną granicę decyzyjną,
- punkty błędnie sklasyfikowane oznaczone znakiem `X`,
- loss history,
- accuracy, precision i recall.

Zadanie:

1. Uruchom demo.
2. Wykonuj kroki uczenia klawiszem `N`.
3. Obserwuj, jak granica decyzyjna przesuwa się po kolejnych krokach.
4. Zmień threshold klawiszami `Q` i `E`.
5. Sprawdź, jak threshold wpływa na accuracy, precision i recall.

## Tło prawdopodobieństwa

Tło wykresu pokazuje prawdopodobieństwo klasy `class_1`.

Zadanie:

1. Uruchom demo.
2. Wykonaj kilka kroków uczenia.
3. Obserwuj, jak tło zmienia się wraz z uczeniem modelu.
4. Zmień threshold klawiszami `Q` i `E`.
5. Sprawdź, czy zmienia się tło, czy tylko granica decyzyjna.

Pytanie kontrolne:

> Dlaczego zmiana threshold nie zmienia prawdopodobieństw modelu?

## Confusion matrix

Panel boczny pokazuje wartości `TP`, `TN`, `FP` i `FN`.

Zadanie:

1. Uruchom demo.
2. Wykonaj kilkanaście kroków uczenia.
3. Zmień threshold klawiszami `Q` i `E`.
4. Obserwuj, jak zmieniają się `FP` i `FN`.
5. Porównaj te zmiany z precision i recall.

Pytanie kontrolne:

> Dlaczego obniżenie threshold może zmniejszyć liczbę false negatives, ale zwiększyć liczbę false positives?

## Challenge mode — precision i recall

Demo ocenia model na ukrytym syntetycznym zbiorze testowym.

Aktualny cel:

```text
recall >= 0.90
precision >= 0.80
```

## Zadanie

1. Uruchom demo.
2. Wykonaj uczenie modelu przez kilka lub kilkadziesiąt kroków.
3. Zmieniaj `threshold` klawiszami `Q` i `E`.
4. Obserwuj `precision`, `recall`, `FP` i `FN`.
5. Spróbuj uzyskać status `success`.

## Pytania kontrolne

- Czy łatwiej uzyskać wysoki `recall` czy wysoką `precision`?
- Co dzieje się z `FP` po obniżeniu `threshold`?
- Co dzieje się z `FN` po podwyższeniu `threshold`?
- Dlaczego sam wynik `accuracy` nie mówi wszystkiego?

## Panel wyjaśnień

Na dole ekranu znajduje się panel wyjaśnień.

Zadanie:

1. Uruchom demo.
2. Wykonaj kilka kroków uczenia.
3. Zmieniaj threshold klawiszami `Q` i `E`.
4. Obserwuj komunikat w panelu wyjaśnień.
5. Porównaj komunikat z wartościami FP, FN, precision i recall.

Pytanie kontrolne:

> Czy panel wyjaśnień opisuje to, co faktycznie widać w metrykach?

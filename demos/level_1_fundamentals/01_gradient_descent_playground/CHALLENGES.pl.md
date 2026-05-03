# Gradient Descent Playground — zadania

## Sterowanie

- `Space` — uruchom lub zatrzymaj automatyczne uczenie,
- `N` — wykonaj jeden krok gradient descent,
- `R` — zresetuj demo,
- `Up` / `Down` — zwiększ lub zmniejsz learning rate,
- `Left` / `Right` — zmniejsz lub zwiększ poziom szumu w danych,
- `S` — wygeneruj nowy zbiór danych przez zmianę ziarna losowości,
- `Esc` — zamknij okno.

## Eksperymenty z parametrami

1. Ustaw bardzo mały learning rate i obserwuj, jak wolno maleje loss.
2. Ustaw większy learning rate i sprawdź, czy model uczy się szybciej.
3. Zwiększ szum w danych i sprawdź, czy prosta nadal dobrze dopasowuje się do punktów.
4. Zmień seed kilka razy i sprawdź, czy uczenie wygląda podobnie dla różnych zbiorów danych.

## Pierwsze obserwacje

1. Uruchom demo i wykonuj kroki pojedynczo klawiszem `N`.
2. Obserwuj, jak zmienia się czerwona linia regresji.
3. Obserwuj wartość `loss`.
4. Zatrzymaj demo po kilku krokach i spróbuj wyjaśnić, dlaczego linia przesunęła się właśnie w tę stronę.

## Status

Zadania zostaną rozszerzone po dodaniu właściwego algorytmu i interaktywnej wizualizacji.

## Planowane zadania podstawowe

1. Uruchom demo i zaobserwuj, jak zmienia się wartość loss.
2. Ustaw bardzo mały learning rate. Co dzieje się z szybkością uczenia?
3. Ustaw bardzo duży learning rate. Czy model nadal się uczy stabilnie?
4. Porównaj uczenie dla danych z małym i dużym szumem.

## Planowane zadania średnie

1. Osiągnij wartość loss poniżej ustalonego progu w ograniczonej liczbie kroków.
2. Znajdź learning rate, który uczy szybko, ale nie powoduje niestabilności.
3. Porównaj pełny gradient descent z wariantem mini-batch.

## Planowane zadania zaawansowane

1. Wyjaśnij, dlaczego skalowanie danych może wpływać na działanie gradient descent.
2. Porównaj regresję liniową i wielomianową.
3. Zaproponuj metrykę challenge mode dla tego demo.

## Challenge mode

Aktualny challenge:

- osiągnij `loss <= 1.0`,
- zanim minie `80` kroków gradient descent.

Spróbuj wykonać challenge dla różnych ustawień:

1. domyślny learning rate i domyślny szum,
2. większy learning rate,
3. bardzo mały learning rate,
4. większy poziom szumu,
5. kilka różnych seedów danych.

Pytanie kontrolne:

> Czy większy learning rate zawsze pomaga wygrać challenge?

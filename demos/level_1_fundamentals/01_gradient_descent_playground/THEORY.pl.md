# Gradient Descent — teoria

## Cel algorytmu

Gradient descent jest metodą optymalizacji używaną do minimalizacji funkcji kosztu lub funkcji straty.

W kontekście uczenia maszynowego funkcja straty mierzy, jak bardzo predykcje modelu różnią się od oczekiwanych wyników.

## Intuicja

Algorytm można rozumieć jako schodzenie po powierzchni błędu.

Model zaczyna z pewnymi parametrami początkowymi. Następnie sprawdzane jest, w którą stronę należy te parametry zmienić, aby błąd był mniejszy.

## Najważniejsze pojęcia

- **loss** — wartość błędu modelu,
- **gradient** — kierunek największego wzrostu funkcji,
- **learning rate** — długość kroku wykonywanego podczas aktualizacji parametrów,
- **iteration** — jedna aktualizacja parametrów.

## Co będzie pokazywać demo?

Docelowo demo będzie pokazywać:

- punkty danych,
- aktualną linię regresji,
- zmianę parametrów modelu,
- wykres wartości loss,
- wpływ learning rate na stabilność uczenia,
- tryb krokowy, w którym jedna akcja odpowiada jednej aktualizacji parametrów.

## Typowe błędy interpretacyjne

### Błąd 1: większy learning rate zawsze oznacza szybszą naukę

Zbyt duży learning rate może powodować niestabilność. Model może przeskakiwać minimum funkcji straty albo całkowicie się rozbiegać.

### Błąd 2: malejący loss zawsze oznacza dobry model

Loss może maleć na danych treningowych, ale model nadal może źle generalizować na nowych danych.

### Błąd 3: gradient descent zawsze znajduje najlepsze rozwiązanie

W prostych problemach wypukłych często działa bardzo dobrze. W bardziej złożonych modelach może utknąć w słabszych rozwiązaniach lokalnych albo być wrażliwy na skalowanie danych.
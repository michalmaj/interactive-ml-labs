# k-NN Vote Map — teoria

## 1. Cel demo

Celem demo jest pokazanie intuicji stojącej za algorytmem k-nearest neighbors, czyli k najbliższych sąsiadów.

Algorytm k-NN jest jedną z najprostszych metod klasyfikacji.

Dla nowego punktu algorytm:

1. oblicza odległość do punktów treningowych,
2. wybiera `k` najbliższych sąsiadów,
3. sprawdza ich klasy,
4. przypisuje nowemu punktowi klasę wybraną przez głosowanie.

## 2. Klasyfikacja

Klasyfikacja polega na przypisaniu obiektu do jednej z dostępnych klas.

Przykłady:

- wiadomość: spam albo nie spam,
- pacjent: zdrowy albo chory,
- obraz: kot, pies albo samochód,
- punkt 2D: klasa niebieska albo pomarańczowa.

W tym demo zaczniemy od klasyfikacji punktów 2D.

## 3. Najważniejsza intuicja

k-NN opiera się na prostym założeniu:

> podobne obiekty znajdują się blisko siebie.

Jeżeli nowy punkt znajduje się blisko punktów należących do klasy A, to prawdopodobnie też należy do klasy A.

## 4. Parametr k

Parametr `k` określa, ilu sąsiadów bierze udział w głosowaniu.

Dla `k = 1` algorytm patrzy tylko na najbliższy punkt.

Dla większego `k` algorytm bierze pod uwagę większą grupę sąsiadów.

## 5. Małe k i overfitting

Jeżeli `k` jest bardzo małe, model może być bardzo wrażliwy na pojedyncze punkty i szum.

Może to prowadzić do overfittingu, czyli zbyt mocnego dopasowania do danych treningowych.

## 6. Duże k i underfitting

Jeżeli `k` jest bardzo duże, model może zbyt mocno uśredniać decyzje.

Może to prowadzić do underfittingu, czyli zbyt prostego modelu, który ignoruje lokalną strukturę danych.

## 7. Metryka odległości

k-NN wymaga sposobu mierzenia odległości między punktami.

Najczęściej używana jest odległość euklidesowa.

W przyszłych wersjach demo można porównać różne metryki odległości.

## 8. Co będzie pokazywać demo?

Docelowo demo będzie pokazywać:

- punkty treningowe różnych klas,
- punkt testowy,
- najbliższych sąsiadów,
- głosowanie sąsiadów,
- mapę decyzji,
- wpływ parametru `k`,
- wpływ szumu w danych.

## 9. Typowe błędy interpretacyjne

### Błąd 1: k-NN niczego się nie uczy

To częściowo prawda, ale wymaga doprecyzowania.

k-NN nie uczy parametrów tak jak regresja liniowa albo sieci neuronowe. Jednak wykorzystuje dane treningowe jako pamięć przykładów.

### Błąd 2: k = 1 zawsze daje najlepsze wyniki

Nie zawsze. `k = 1` może działać dobrze na prostych danych, ale jest bardzo wrażliwe na szum.

### Błąd 3: większe k zawsze jest stabilniejsze i lepsze

Większe `k` może zmniejszyć wpływ szumu, ale zbyt duże `k` może rozmyć granicę między klasami.

### Błąd 4: odległość zawsze ma oczywiste znaczenie

W danych wielowymiarowych odległość zależy od skali cech. Jeżeli jedna cecha ma dużo większy zakres wartości niż druga, może dominować wynik.
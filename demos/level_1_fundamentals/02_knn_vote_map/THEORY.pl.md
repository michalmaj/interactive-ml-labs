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

## 7a. Metryka odległości

k-NN wymaga sposobu mierzenia odległości między punktami.

Najczęściej używana jest odległość euklidesowa.

W przyszłych wersjach demo można porównać różne metryki odległości.

## 7b. Odległość euklidesowa

Pierwsza wersja demo wykorzystuje odległość euklidesową.

Dla dwóch punktów 2D:

```text
A = (x1, y1)
B = (x2, y2)
```

odległość euklidesowa oznacza długość prostej linii między tymi punktami.

Intuicyjnie jest to zwykła odległość geometryczna znana z układu współrzędnych.

W k-NN odległość euklidesowa pozwala odpowiedzieć na pytanie:

> Które punkty treningowe znajdują się najbliżej nowego punktu?

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

## Dane syntetyczne w pierwszej wersji demo

Pierwsza wersja demo wykorzystuje prosty syntetyczny zbiór danych 2D.

Dane składają się z dwóch klas:

- `class_0`,
- `class_1`.

Każda klasa jest generowana jako chmura punktów wokół własnego centrum.

Parametry danych:

- `samples_per_class` — liczba punktów w każdej klasie,
- `class_distance` — odległość między centrami klas,
- `noise_std` — poziom szumu, czyli rozrzut punktów wokół centrum,
- `seed` — ziarno losowości umożliwiające odtwarzalność danych.

Dane syntetyczne są użyte celowo, ponieważ pozwalają jasno pokazać wpływ szumu i odległości między klasami na działanie k-NN.
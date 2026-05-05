# k-NN Vote Map — teoria

## 1. Cel demo

Celem demo jest pokazanie intuicji stojącej za algorytmem k-nearest neighbors, czyli k najbliższych sąsiadów.

Algorytm k-NN jest jedną z najprostszych metod klasyfikacji.

Dla nowego punktu algorytm:

1. oblicza odległość do punktów treningowych,
2. wybiera `k` najbliższych sąsiadów,
3. sprawdza ich klasy,
4. przypisuje nowemu punktowi klasę wybraną przez głosowanie.

W demo student może kliknąć dowolne miejsce na mapie i zobaczyć, jak algorytm podejmuje decyzję.

## 2. Klasyfikacja

Klasyfikacja polega na przypisaniu obiektu do jednej z dostępnych klas.

Przykłady:

- wiadomość: spam albo nie spam,
- pacjent: zdrowy albo chory,
- obraz: kot, pies albo samochód,
- punkt 2D: klasa niebieska albo pomarańczowa.

W tym demo klasyfikowane są punkty 2D należące do jednej z dwóch klas:

- `class_0`,
- `class_1`.

## 3. Dane syntetyczne

Demo wykorzystuje syntetyczny zbiór danych 2D.

Dane składają się z dwóch chmur punktów. Każda chmura odpowiada jednej klasie.

Parametry danych:

- `samples_per_class` — liczba punktów w każdej klasie,
- `class_distance` — odległość między centrami klas,
- `noise_std` — poziom szumu, czyli rozrzut punktów wokół centrum,
- `seed` — ziarno losowości umożliwiające odtwarzalność danych.

Dane syntetyczne są użyte celowo, ponieważ pozwalają jasno pokazać wpływ szumu i odległości między klasami na działanie k-NN.

## 4. Najważniejsza intuicja k-NN

k-NN opiera się na prostym założeniu:

> podobne obiekty znajdują się blisko siebie.

Jeżeli nowy punkt znajduje się blisko punktów należących do klasy `class_0`, to prawdopodobnie też należy do `class_0`.

Jeżeli znajduje się blisko punktów klasy `class_1`, to prawdopodobnie należy do `class_1`.

W praktyce jakość tego założenia zależy od danych, sposobu mierzenia odległości oraz skali cech.

## 5. Parametr k

Parametr `k` określa, ilu sąsiadów bierze udział w głosowaniu.

Dla `k = 1` algorytm patrzy tylko na najbliższy punkt.

Dla większego `k` algorytm bierze pod uwagę większą grupę sąsiadów.

W demo parametr `k` można zmieniać klawiszami `Up` i `Down`.

## 6. Małe k i overfitting

Jeżeli `k` jest bardzo małe, model może być bardzo wrażliwy na pojedyncze punkty i szum.

Dla `k = 1` każdy pojedynczy punkt treningowy może silnie wpływać na lokalną decyzję modelu.

Objawy małego `k`:

- granica decyzyjna może być poszarpana,
- pojedyncze odstające punkty mogą zmieniać predykcję,
- model może zbyt mocno dopasować się do danych treningowych.

To jest intuicja overfittingu.

## 7. Duże k i underfitting

Jeżeli `k` jest bardzo duże, model może zbyt mocno uśredniać decyzje.

Objawy dużego `k`:

- granica decyzyjna jest gładsza,
- model jest mniej wrażliwy na pojedyncze punkty,
- lokalna struktura danych może zostać zignorowana.

To jest intuicja underfittingu.

Zbyt duże `k` może spowodować, że model nie zauważa ważnych lokalnych różnic między klasami.

## 8. Odległość euklidesowa

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

## 9. Głosowanie sąsiadów

Po znalezieniu `k` najbliższych sąsiadów algorytm sprawdza ich etykiety klas.

Każdy sąsiad oddaje jeden głos na swoją klasę.

Predykcją zostaje klasa z największą liczbą głosów.

Przykład:

```text
k = 5
sąsiedzi: class_0, class_0, class_1, class_0, class_1

głosy:
class_0 -> 3
class_1 -> 2

predykcja -> class_0
```

W przypadku remisu aktualna implementacja wybiera klasę o mniejszej etykiecie liczbowej. Dzięki temu wynik jest deterministyczny i łatwiejszy do testowania.

## 10. Mapa decyzji

Tło wykresu pokazuje, jaką klasę przewidziałby k-NN w różnych obszarach przestrzeni.

Kolor tła odpowiada przewidywanej klasie.

Dzięki temu można obserwować:

- gdzie przebiega granica decyzyjna,
- jak zmienia się granica po zmianie `k`,
- jak szum wpływa na separację klas,
- kiedy model staje się bardziej lokalny,
- kiedy model staje się zbyt wygładzony.

Mapa decyzji jest szczególnie przydatna do porównania małych i dużych wartości `k`.

## 11. Klikany punkt testowy

W demo można kliknąć dowolne miejsce na mapie.

Kliknięty punkt staje się punktem testowym.

Po kliknięciu demo pokazuje:

- przewidywaną klasę,
- linie do najbliższych sąsiadów,
- głosy sąsiadów,
- krótkie wyjaśnienie decyzji.

To pozwala analizować lokalne zachowanie modelu.

Warto szczególnie klikać punkty:

- blisko środka jednej klasy,
- daleko od obu klas,
- w pobliżu granicy decyzyjnej,
- w miejscach, gdzie tło zmienia kolor.

## 12. Challenge mode

Demo zawiera challenge mode oparty na accuracy na osobnym zbiorze testowym.

Aktualny cel:

```text
osiągnij accuracy >= 0.90
```

Zbiór testowy nie jest tym samym zbiorem, który widać jako punkty treningowe. Jest generowany osobno, z innym seedem.

Dzięki temu student może zobaczyć, że model powinien działać nie tylko na danych treningowych, ale też na nowych danych.

## 13. Accuracy

Accuracy oznacza odsetek poprawnych predykcji.

Przykład:

```text
liczba przykładów testowych: 100
liczba poprawnych predykcji: 93

accuracy = 93 / 100 = 0.93
```

Accuracy jest intuicyjna, ale nie zawsze wystarcza. W problemach niezbalansowanych klasowo może być myląca.

W tym demo klasy są zbalansowane, dlatego accuracy jest dobrym pierwszym wskaźnikiem jakości.

## 14. Szum w danych

Szum oznacza losowe rozrzucenie punktów wokół centrum klasy.

Mały szum:

- klasy są dobrze rozdzielone,
- model łatwiej osiąga wysoką accuracy,
- granica decyzyjna jest zwykle stabilna.

Duży szum:

- klasy zaczynają się nakładać,
- najbliżsi sąsiedzi mogą pochodzić z różnych klas,
- challenge może stać się trudniejszy,
- granica decyzyjna może być mniej oczywista.

W demo szum można zmieniać klawiszami `Left` i `Right`.

## 15. Typowe błędy interpretacyjne

### Błąd 1: k-NN niczego się nie uczy

To zależy od znaczenia słowa „uczy”.

k-NN nie uczy parametrów tak jak regresja liniowa albo sieć neuronowa. Nie wykonuje gradient descent. Jednak przechowuje dane treningowe i używa ich w momencie predykcji.

### Błąd 2: k = 1 zawsze daje najlepsze wyniki

Nie zawsze. `k = 1` może bardzo dobrze dopasować się do danych treningowych, ale jest wrażliwe na szum.

### Błąd 3: większe k zawsze jest lepsze

Nie zawsze. Większe `k` może wygładzić granicę decyzyjną, ale zbyt duże `k` może ignorować lokalne struktury danych.

### Błąd 4: odległość zawsze ma oczywiste znaczenie

Nie zawsze. W danych wielowymiarowych odległość zależy od skali cech.

Jeżeli jedna cecha ma zakres od 0 do 10000, a druga od 0 do 1, pierwsza cecha może zdominować odległość.

### Błąd 5: wysoka accuracy na treningu wystarczy

Nie wystarczy. Model może dobrze działać na danych treningowych, ale gorzej na nowych danych.

Dlatego demo pokazuje challenge oparty o ukryty zbiór testowy.

## 16. Co warto obserwować w demo?

Podczas pracy z demo warto zwrócić uwagę na:

- jak zmienia się granica decyzyjna po zmianie `k`,
- jak zmienia się accuracy challenge,
- czy małe `k` daje bardziej poszarpane tło,
- czy duże `k` wygładza decyzję,
- jak szum wpływa na przewidywania,
- czy kliknięty punkt ma sąsiadów jednej klasy czy obu klas,
- jak głosy sąsiadów prowadzą do predykcji.

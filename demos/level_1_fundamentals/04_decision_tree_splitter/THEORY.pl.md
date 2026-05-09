# Decision Tree Splitter — teoria

## 1. Cel demo

Celem demo jest pokazanie, jak działa drzewo decyzyjne w problemie klasyfikacji.

Drzewo decyzyjne podejmuje decyzję przez sekwencję prostych pytań.

Przykład pytania:

```text
Czy x1 <= 2.5?
```

Jeżeli odpowiedź brzmi tak, przykład przechodzi do lewej gałęzi.

Jeżeli odpowiedź brzmi nie, przykład przechodzi do prawej gałęzi.

## 2. Klasyfikacja za pomocą reguł

Drzewo decyzyjne można interpretować jako zestaw reguł.

Przykład:

```text
jeżeli x1 <= 2.5:
    class_0
w przeciwnym razie:
    class_1
```

Bardziej złożone drzewo zawiera wiele takich pytań.

Każde pytanie dzieli dane na mniejsze podzbiory.

## 3. Split

Split to pojedynczy podział danych.

W danych 2D split może być:

- pionowy — na podstawie cechy `x1`,
- poziomy — na podstawie cechy `x2`.

Przykład splitu pionowego:

```text
x1 <= threshold
```

Przykład splitu poziomego:

```text
x2 <= threshold
```

W wizualizacji split jest widoczny jako linia przecinająca przestrzeń danych.

## 4. Dane syntetyczne w demo

Demo zawiera dwa warianty syntetycznych danych 2D.

### Axis-aligned

Wariant `axis_aligned` zawiera dwie klasy rozmieszczone głównie po lewej i prawej stronie przestrzeni.

Przy małym szumie jeden pionowy split może dobrze rozdzielić klasy.

Ten wariant jest dobry do pokazania pierwszego splitu i działania decision stump.

### XOR

Wariant `xor` zawiera klasy rozmieszczone po przekątnych.

W tym przypadku jeden split zwykle nie wystarcza.

Taki wariant dobrze pokazuje, po co drzewo wykonuje wiele kolejnych podziałów i dlaczego głębokość drzewa ma znaczenie.

## 5. Węzeł i liść

Drzewo decyzyjne składa się z węzłów i liści.

Węzeł decyzyjny zawiera pytanie, np.:

```text
x1 <= 1.7
```

Liść zawiera końcową predykcję klasy.

Przykład:

```text
leaf -> class_0
```

Klasyfikacja nowego punktu polega na przejściu od korzenia drzewa do jednego z liści.

## 6. Nieczystość węzła

Drzewo próbuje tworzyć takie podziały, aby w liściach znajdowały się możliwie jednorodne klasy.

Jeżeli w węźle są prawie same punkty jednej klasy, węzeł jest czysty.

Jeżeli w węźle jest mieszanina wielu klas, węzeł jest nieczysty.

Do mierzenia nieczystości używa się między innymi:

- Gini impurity,
- entropy.

## 7. Gini impurity

Gini impurity mierzy, jak bardzo wymieszane są klasy w danym węźle.

Intuicyjnie:

- Gini bliskie 0 oznacza czysty węzeł,
- większe Gini oznacza większe wymieszanie klas.

Dla klasyfikacji binarnej Gini jest niskie, gdy prawie wszystkie próbki należą do jednej klasy.

Dla węzła zbalansowanego, np.:

```text
class_0: 50%
class_1: 50%
```

Gini wynosi `0.5`.

## 8. Entropy

Entropy jest inną miarą nieczystości.

Podobnie jak Gini:

- niska entropy oznacza bardziej czysty węzeł,
- wysoka entropy oznacza większe wymieszanie klas.

Dla zbalansowanego węzła binarnego entropy wynosi `1.0`.

W praktyce Gini i entropy często prowadzą do podobnych drzew, ale nie zawsze wybierają identyczne splity.

## 9. Metryki nieczystości w implementacji

Aktualna implementacja zawiera:

- `class_counts`,
- `class_probabilities`,
- `gini_impurity`,
- `entropy_impurity`.

Dla czystego węzła, w którym wszystkie próbki należą do jednej klasy, zarówno Gini, jak i entropy wynoszą `0`.

Dla zbalansowanego węzła binarnego:

```text
class_0: 50%
class_1: 50%
```

Gini wynosi `0.5`, a entropy wynosi `1.0`.

Te metryki są używane do oceny, czy dany split poprawia jakość podziału.

## 10. Information gain

Information gain mówi, ile nieczystości udało się zmniejszyć dzięki splitowi.

Dobry split:

- dzieli dane na bardziej jednorodne grupy,
- zmniejsza Gini albo entropy,
- zwiększa information gain.

Wzór intuicyjny:

```text
information_gain = parent_impurity - weighted_child_impurity
```

Jeżeli split poprawia jakość podziału, information gain jest dodatni.

Jeżeli split tworzy dzieci podobnie wymieszane jak rodzic, information gain jest niski.

## 11. Tryb ręcznego splitu

Aktualna implementacja zawiera tryb ręcznego splitu.

Oznacza to, że można wskazać split, np.:

```text
x1 <= 0.0
```

a system obliczy:

- nieczystość rodzica,
- nieczystość lewego dziecka,
- nieczystość prawego dziecka,
- ważoną nieczystość dzieci,
- information gain.

Ten tryb jest ważny dydaktycznie, ponieważ pozwala porównać intuicję studenta z oceną matematyczną splitu.

## 12. Decision stump

Decision stump to drzewo decyzyjne o głębokości 1.

Składa się z:

- jednego splitu w korzeniu,
- lewego liścia,
- prawego liścia.

Schemat:

```text
root split
├── left leaf
└── right leaf
```

Każdy liść przewiduje klasę większościową wśród próbek, które do niego trafiły.

Decision stump dobrze pokazuje pierwszy krok uczenia drzewa, ale ma ograniczenia. Dla danych typu XOR jeden split nie wystarcza, dlatego potrzebne jest pełne drzewo rekurencyjne.

## 13. Pełne drzewo rekurencyjne

Pełne drzewo rekurencyjne może wykonać split w korzeniu, a następnie wykonywać kolejne splity w dzieciach.

Schemat:

```text
root
├── left child
│   ├── left leaf
│   └── right leaf
└── right child
    ├── left leaf
    └── right leaf
```

Najważniejszy parametr to `max_depth`.

Jeżeli `max_depth = 1`, model zachowuje się jak decision stump.

Jeżeli `max_depth = 2`, model może wykonać split w korzeniu i kolejne splity w dzieciach.

To pozwala rozwiązać problem XOR, którego nie da się dobrze rozwiązać jednym splitem.

## 14. Warunki zatrzymania drzewa

Drzewo nie może splitować w nieskończoność.

Aktualna implementacja zatrzymuje splitowanie, gdy:

- osiągnięto `max_depth`,
- węzeł ma za mało próbek,
- węzeł jest czysty,
- nie znaleziono poprawnego splitu,
- split nie spełnia ograniczeń `min_samples_leaf`,
- split ma zbyt mały information gain.

Dzięki temu można kontrolować złożoność modelu.

## 15. Overfitting i underfitting

Płytkie drzewo może mieć underfitting.

Oznacza to, że model jest zbyt prosty, aby uchwycić strukturę danych.

Przykład:

```text
XOR + max_depth = 1
```

Głębokie drzewo może mieć overfitting.

Oznacza to, że model może zbyt dokładnie dopasować się do danych treningowych, w tym do szumu.

Przykład:

```text
duży noise + bardzo wysokie max_depth
```

W praktyce trzeba dobrać głębokość drzewa do złożoności problemu.

## 16. Wizualizacja Pygame

Wizualizacja pokazuje:

- punkty danych,
- regiony decyzyjne liści,
- linie splitów,
- liczbę węzłów,
- liczbę liści,
- głębokość drzewa,
- accuracy na danych treningowych,
- challenge status,
- panel wyjaśnień.

Wariant `axis_aligned` pokazuje, że płytkie drzewo może wystarczyć, gdy dane są dobrze rozdzielone jednym splitem.

Wariant `xor` pokazuje, że pojedynczy split nie wystarcza. Po zwiększeniu `max_depth` do 2 model może wykonać kolejne splity i poprawnie rozdzielić klasy.

## 17. Manual split mode w Pygame

Tryb manual split pozwala samodzielnie ustawić pojedynczy split.

Student może:

- wybrać cechę splitu (`x1` albo `x2`),
- przesuwać threshold,
- obserwować lewy i prawy region,
- sprawdzać Gini albo entropy,
- porównywać intuicyjny split z matematyczną oceną information gain.

Ten tryb jest szczególnie przydatny dydaktycznie, ponieważ pokazuje, że split wizualnie „ładny” nie zawsze musi dawać największy information gain.

## 18. Challenge mode

Challenge mode premiuje nie tylko wysoką accuracy, ale także prostotę drzewa.

Dla datasetu `axis_aligned` celem jest:

```text
accuracy >= 0.95
max_depth <= 1
```

Dla datasetu `xor` celem jest:

```text
accuracy >= 0.95
max_depth <= 2
```

To pokazuje ważną zasadę: model powinien być wystarczająco złożony, aby rozwiązać problem, ale nie bardziej złożony niż trzeba.

Zbyt płytkie drzewo może mieć underfitting.

Zbyt głębokie drzewo może mieć overfitting.

## 19. Panel wyjaśnień

Demo zawiera testowalny helper generujący tekst panelu wyjaśnień.

Panel dolny informuje studenta:

- czy działa tryb automatycznego drzewa,
- czy działa tryb manualnego splitu,
- czy manualny split jest poprawny,
- czy challenge został spełniony,
- jak accuracy, liczba liści i max_depth odnoszą się do aktualnego modelu.

Dzięki temu renderer odpowiada głównie za rysowanie, a logika komunikatów jest osobno testowana.

## 20. Dlaczego decision tree jest ważne?

Drzewa decyzyjne są ważne, ponieważ:

- są intuicyjne,
- są interpretowalne,
- dobrze pokazują pojęcie podziału przestrzeni,
- stanowią bazę dla Random Forest,
- stanowią bazę dla Gradient Boosting,
- dobrze pokazują problem złożoności modelu.

To demo przygotowuje studentów do późniejszych metod ensemble.

## 21. Typowe błędy interpretacyjne

### Błąd 1: głębsze drzewo zawsze jest lepsze

Nie zawsze. Głębsze drzewo może lepiej dopasować się do danych treningowych, ale gorzej generalizować.

### Błąd 2: split zawsze ma oczywisty sens

Nie zawsze. Algorytm wybiera split na podstawie kryterium matematycznego, np. Gini albo entropy.

### Błąd 3: decision tree tworzy dowolne granice

Nie dokładnie. Klasyczne drzewo tworzy podziały osiowe, czyli pionowe lub poziome w przestrzeni 2D.

### Błąd 4: interpretowalność zawsze zostaje zachowana

Nie zawsze. Małe drzewo jest czytelne, ale bardzo głębokie drzewo może być trudne do interpretacji.

### Błąd 5: wysoka accuracy wystarczy

Nie zawsze. Challenge mode pokazuje, że ważna jest także prostota drzewa.

### Błąd 6: jeden split powinien rozwiązać każdy problem

Nie. XOR pokazuje, że czasem potrzebna jest rekurencja i większa głębokość drzewa.
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

### 3. Split

**Split** to pojedynczy podział danych.

W danych 2D split może być:

- **pionowy** — na podstawie cechy `x1`,
- **poziomy** — na podstawie cechy `x2`.

Przykład splitu pionowego:

```text
x1 <= threshold
```

Przykład splitu poziomego:

```text
x2 <= threshold
```

W wizualizacji split będzie widoczny jako linia przecinająca przestrzeń danych.

## 4. Węzeł i liść

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

Klasyfikacja nowego punktu polega na przejściu od **korzenia drzewa** do jednego z **liści**.

### 5. Nieczystość węzła

Drzewo próbuje tworzyć takie podziały, aby w liściach znajdowały się możliwie jednorodne klasy.

Jeżeli w węźle są prawie same punkty jednej klasy, węzeł jest **czysty**.

Jeżeli w węźle jest mieszanina wielu klas, węzeł jest **nieczysty**.

Do mierzenia nieczystości używa się między innymi:

- **Gini impurity**,
- **entropy**.

### 6. Gini impurity

**Gini impurity** mierzy, jak bardzo wymieszane są klasy w danym węźle.

Intuicyjnie:

- Gini bliskie `0` oznacza czysty węzeł,
- większe Gini oznacza większe wymieszanie klas.

Dla klasyfikacji binarnej Gini jest niskie, gdy prawie wszystkie próbki należą do jednej klasy.

### 7. Entropy

**Entropy** jest inną miarą nieczystości.

Podobnie jak Gini:

- niska entropy oznacza bardziej czysty węzeł,
- wysoka entropy oznacza większe wymieszanie klas.

W praktyce Gini i entropy często prowadzą do podobnych drzew, ale nie zawsze wybierają identyczne splity.

### 8. Information gain

**Information gain** mówi, ile nieczystości udało się zmniejszyć dzięki splitowi.

Dobry split:

- dzieli dane na bardziej jednorodne grupy,
- zmniejsza Gini albo entropy,
- zwiększa information gain.

Drzewo wybiera split, który daje największą poprawę jakości podziału.

### 9. Głębokość drzewa

**Głębokość drzewa** określa, ile kolejnych pytań może zadać model.

Płytkie drzewo:

- jest prostsze,
- jest łatwiejsze do interpretacji,
- może mieć **underfitting**.

Głębokie drzewo:

- może dokładnie dopasować się do danych treningowych,
- może tworzyć bardzo szczegółowe reguły,
- może mieć **overfitting**.

### 10. Dlaczego decision tree jest ważne?

Drzewa decyzyjne są ważne, ponieważ:

- są intuicyjne,
- są interpretowalne,
- dobrze pokazują pojęcie podziału przestrzeni,
- stanowią bazę dla **Random Forest**,
- stanowią bazę dla **Gradient Boosting**.

To demo przygotuje studentów do późniejszych metod ensemble.

### 11. Typowe błędy interpretacyjne

#### Błąd 1: głębsze drzewo zawsze jest lepsze

Nie zawsze. Głębsze drzewo może lepiej dopasować się do danych treningowych, ale gorzej generalizować.

#### Błąd 2: split zawsze ma oczywisty sens

Nie zawsze. Algorytm wybiera split na podstawie kryterium matematycznego, np. Gini albo entropy.

#### Błąd 3: decision tree tworzy dowolne granice

Nie dokładnie. Klasyczne drzewo tworzy podziały osiowe, czyli pionowe lub poziome w przestrzeni 2D.

#### Błąd 4: interpretowalność zawsze zostaje zachowana

Nie zawsze. Małe drzewo jest czytelne, ale bardzo głębokie drzewo może być trudne do interpretacji.

## Dane syntetyczne w demo

Demo zawiera dwa warianty syntetycznych danych 2D.

### Axis-aligned

Wariant `axis_aligned` zawiera dwie klasy rozmieszczone głównie po lewej i prawej stronie przestrzeni.

Przy małym szumie jeden pionowy split może dobrze rozdzielić klasy.

Ten wariant jest dobry do pokazania pierwszego splitu i działania decision stump.

### XOR

Wariant `xor` zawiera klasy rozmieszczone po przekątnych.

W tym przypadku jeden split zwykle nie wystarcza.

Taki wariant dobrze pokazuje, po co drzewo wykonuje wiele kolejnych podziałów i dlaczego głębokość drzewa ma znaczenie.
# Random Forest Bagging Lab — teoria

## 1. Cel demo

Celem demo jest pokazanie, jak działa Random Forest, czyli las losowy.

Random Forest składa się z wielu drzew decyzyjnych.

Każde drzewo podejmuje własną decyzję, a końcowa predykcja powstaje przez głosowanie.

## 2. Dlaczego jedno drzewo nie zawsze wystarcza?

Pojedyncze drzewo decyzyjne jest intuicyjne i interpretowalne.

Ma jednak wadę: może łatwo dopasować się zbyt mocno do danych treningowych.

Takie zjawisko nazywa się overfittingiem.

Głębokie drzewo może nauczyć się szczegółowych reguł, które dobrze działają na danych treningowych, ale gorzej na nowych danych.

## 3. Idea ensemble learning

Ensemble learning polega na łączeniu wielu modeli.

Zamiast polegać na jednym modelu, system zbiera predykcje wielu modeli i łączy je w jedną decyzję.

W przypadku klasyfikacji często używa się głosowania.

Przykład:

```text
tree_1 -> class_0
tree_2 -> class_1
tree_3 -> class_1

final prediction -> class_1
```

## 4. Bagging

Bagging oznacza bootstrap aggregating.

To technika, w której wiele modeli trenuje się na różnych losowych próbkach danych.

Każda próbka powstaje przez losowanie z powtórzeniami.

Dzięki temu każde drzewo widzi trochę inny zbiór treningowy.

## 5. Bootstrap sampling

Bootstrap sampling oznacza losowanie próbek z powtórzeniami.

Jeżeli oryginalny dataset ma `N` próbek, bootstrap sample również może mieć `N` próbek, ale niektóre przykłady mogą pojawić się wiele razy, a inne mogą nie pojawić się wcale.

Przykład:

```text
oryginalne indeksy: [0, 1, 2, 3, 4]
bootstrap:          [2, 2, 0, 4, 4]
```

W tym przykładzie próbki `1` i `3` nie zostały wylosowane.

Takie próbki nazywa się out-of-bag, czyli OOB.

## 6. Bootstrap ratio

Parametr `bootstrap_sample_ratio` określa, ile losowań wykonuje się względem liczby próbek treningowych.

Przykład:

```text
bootstrap_sample_ratio = 1.0
```

oznacza, że liczba losowań jest równa liczbie próbek w zbiorze treningowym.

Przykład:

```text
bootstrap_sample_ratio = 0.5
```

oznacza, że każde drzewo otrzymuje mniejszy bootstrap sample.

Mniejszy bootstrap ratio może zwiększyć różnorodność drzew, ale pojedyncze drzewa widzą mniej danych.

## 7. Dane train/test

Demo generuje osobne zbiory:

- train,
- test.

Zbiór train jest używany do uczenia modeli.

Zbiór test jest używany do oceny generalizacji.

W praktycznym uczeniu maszynowym ważne są dwa pytania:

```text
Czy model dobrze dopasował się do danych treningowych?
Czy model działa dobrze na nowych danych?
```

Dlatego w tym demo od początku przygotowano osobne dane treningowe i testowe.

## 8. Train accuracy i test accuracy

Train accuracy mierzy jakość modelu na danych treningowych.

Test accuracy mierzy jakość modelu na danych, których model nie używał do treningu.

Wysoka train accuracy nie wystarcza.

Jeżeli train accuracy jest wysoka, ale test accuracy niska, model prawdopodobnie przeuczył się do danych treningowych.

## 9. Generalization gap

Generalization gap oznacza różnicę:

```text
train accuracy - test accuracy
```

Mały gap oznacza, że model zachowuje się podobnie na danych treningowych i testowych.

Duży gap może sugerować overfitting.

Przykład:

```text
train accuracy = 1.00
test accuracy  = 0.75
gap            = 0.25
```

Taki wynik może oznaczać, że model zapamiętuje dane treningowe, ale gorzej generalizuje.

## 10. Single-tree baseline

Demo zawiera baseline jednego drzewa decyzyjnego.

Baseline oznacza prosty model odniesienia.

W tym demo baseline:

- trenuje jedno drzewo na zbiorze train,
- oblicza train accuracy,
- oblicza test accuracy,
- pozwala później porównać jedno drzewo z Random Forest.

To ważne, ponieważ Random Forest nie powinien być oceniany w próżni.

Trzeba mieć punkt odniesienia:

```text
Czy las działa lepiej niż jedno drzewo?
Czy las jest stabilniejszy niż jedno drzewo?
Czy test accuracy poprawia się względem baseline?
```

## 11. Głosowanie większościowe

W klasyfikacji Random Forest najczęściej stosuje głosowanie większościowe.

Każde drzewo oddaje głos na klasę.

Końcowa klasa to ta, która dostała najwięcej głosów.

Przykład:

```text
tree_1: class_0
tree_2: class_1
tree_3: class_1

final prediction: class_1
```

## 12. Vote confidence

Vote confidence oznacza udział drzew, które głosowały na zwycięską klasę.

Przykład:

```text
10 drzew
7 głosów na class_1
confidence = 7 / 10 = 0.70
```

Confidence nie jest tym samym co accuracy.

Confidence mówi, jak zgodny jest las w danej predykcji.

Accuracy mówi, czy predykcja była poprawna.

## 13. Random Forest w implementacji

Implementacja Random Forest działa w kilku krokach:

1. Tworzony jest bootstrap sample dla każdego drzewa.
2. Każde drzewo trenuje się na swoim bootstrap sample.
3. Każde drzewo przewiduje klasy dla danych train i test.
4. Predykcje drzew są łączone przez majority voting.
5. Obliczana jest train accuracy.
6. Obliczana jest test accuracy.
7. Obliczana jest vote confidence.

Schemat:

```text
bootstrap samples -> decision trees -> tree predictions -> majority vote
```

To jest etap, w którym wszystkie wcześniejsze elementy demo łączą się w pełny model ensemble.

## 14. Dlaczego Random Forest działa?

Random Forest działa dobrze, ponieważ zmniejsza wariancję pojedynczego drzewa.

Pojedyncze głębokie drzewo może być niestabilne.

Mała zmiana danych może prowadzić do innego drzewa.

Las składający się z wielu różnych drzew zwykle jest bardziej stabilny.

Każde drzewo może popełniać inne błędy.

Głosowanie sprawia, że pojedyncze błędy mogą zostać przegłosowane przez większość.

## 15. Raport porównawczy CLI

Implementacja zawiera raport porównujący jedno drzewo z Random Forest.

Raport pokazuje:

- train accuracy pojedynczego drzewa,
- test accuracy pojedynczego drzewa,
- train accuracy lasu,
- test accuracy lasu,
- generalization gap,
- różnicę test accuracy między lasem a baseline,
- model zwycięski według test accuracy.

Winner wybierany jest według test accuracy, ponieważ test accuracy lepiej opisuje generalizację niż train accuracy.

## 16. Wizualizacja Pygame

Demo zawiera wizualizację Pygame.

Wizualizacja pokazuje dwa modele obok siebie:

- single-tree baseline,
- Random Forest.

Lewy panel pokazuje pojedyncze drzewo.

Prawy panel pokazuje las losowy.

Oznaczenia:

- kółka oznaczają próbki treningowe,
- kwadraty oznaczają próbki testowe,
- znak X oznacza błędnie sklasyfikowaną próbkę testową,
- tło pokazuje regiony decyzyjne modelu.

Panel boczny pokazuje:

- dataset,
- poziom szumu,
- seed,
- liczbę drzew,
- max depth,
- bootstrap ratio,
- winner według test accuracy,
- train/test accuracy,
- generalization gap,
- mean vote confidence dla lasu,
- challenge status.

## 17. Confidence view

Confidence view pokazuje siłę głosowania lasu.

Jeżeli wiele drzew głosuje zgodnie, region decyzyjny jest bardziej wyraźny.

Jeżeli drzewa są mniej zgodne, region jest bledszy.

To pozwala odróżnić:

- samą predykcję klasy,
- pewność głosowania ensemble.

Blade regiony nie muszą oznaczać błędnej predykcji.

Oznaczają raczej, że drzewa nie są w pełni zgodne.

## 18. Challenge mode

Demo zawiera challenge mode dla Random Forest.

Challenge sprawdza trzy rzeczy:

```text
forest test accuracy >= 0.90
tree_count <= 25
generalization gap <= 0.15
```

Oznacza to, że model powinien:

- dobrze działać na danych testowych,
- nie używać zbyt wielu drzew,
- nie mieć zbyt dużej różnicy między train accuracy i test accuracy.

To jest ważne, ponieważ w praktyce nie chodzi tylko o maksymalizację accuracy.

Dobry model powinien być skuteczny, stabilny i rozsądnie złożony.

## 19. Panel wyjaśnień

Demo zawiera osobny helper do generowania tekstów wyjaśniających.

Panel dolny informuje studenta:

- czy challenge został spełniony,
- dlaczego challenge nie został spełniony,
- co oznacza confidence view,
- jak interpretować blade regiony w widoku lasu.

Dzięki temu renderer odpowiada głównie za rysowanie, a teksty dydaktyczne są osobno testowane.

## 20. Typowe błędy interpretacyjne

### Błąd 1: więcej drzew zawsze oznacza lepszy model

Nie zawsze. Więcej drzew może stabilizować wynik, ale po pewnym czasie poprawa może być mała.

### Błąd 2: train accuracy wystarczy

Nie. W praktyce ważniejsza jest test accuracy, ponieważ mówi więcej o generalizacji.

### Błąd 3: confidence oznacza poprawność

Nie. Confidence mówi o zgodności drzew, ale nie gwarantuje poprawnej predykcji.

### Błąd 4: Random Forest jest zawsze lepszy od pojedynczego drzewa

Nie zawsze. Na bardzo prostych danych pojedyncze drzewo może wystarczyć.

### Błąd 5: bootstrap sample to zwykły podzbiór

Nie. Bootstrap sampling losuje z powtórzeniami, więc próbki mogą pojawiać się wiele razy.

### Błąd 6: OOB samples to test set

Nie dokładnie. OOB samples są niewylosowane dla konkretnego drzewa, ale nadal pochodzą ze zbioru treningowego.
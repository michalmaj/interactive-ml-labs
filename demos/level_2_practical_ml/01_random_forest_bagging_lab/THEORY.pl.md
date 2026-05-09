# Random Forest Bagging Lab — teoria

## 1. Cel demo

Celem demo jest pokazanie, jak działa Random Forest, czyli las losowy.

Random Forest składa się z wielu drzew decyzyjnych.

Każde drzewo może podejmować własną decyzję, a końcowa predykcja powstaje przez głosowanie.

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

Bagging oznacza *bootstrap aggregating*.

To technika, w której wiele modeli trenuje się na różnych losowych próbkach danych.

Każda próbka powstaje przez losowanie z powtórzeniami.

Dzięki temu każde drzewo widzi trochę inny zbiór treningowy.

## 5. Bootstrap sampling

*Bootstrap sampling* oznacza losowanie próbek z powtórzeniami.

Jeżeli oryginalny dataset ma `N` próbek, bootstrap sample również może mieć `N` próbek, ale niektóre przykłady mogą pojawić się wiele razy, a inne mogą nie pojawić się wcale.

To powoduje, że drzewa w lesie różnią się od siebie.

## 6. Głosowanie

W klasyfikacji Random Forest najczęściej stosuje głosowanie większościowe.

Każde drzewo oddaje głos na klasę.

Końcowa klasa to ta, która dostała najwięcej głosów.

Można też obliczyć *vote confidence*, czyli udział drzew głosujących na zwycięską klasę.

## 7. Dlaczego Random Forest działa?

Random Forest działa dobrze, ponieważ zmniejsza wariancję pojedynczego drzewa.

Pojedyncze głębokie drzewo może być niestabilne.

Mała zmiana danych może prowadzić do innego drzewa.

Las składający się z wielu różnych drzew zwykle jest bardziej stabilny.

## 8. Co będzie pokazane w demo?

Planowane demo pokaże:

- bootstrap sampling,
- wiele drzew uczonych na różnych próbkach,
- głosowanie większościowe,
- vote confidence,
- porównanie jednego drzewa z lasem,
- wpływ liczby drzew,
- wpływ głębokości drzew,
- wpływ szumu w danych,
- różnicę między train accuracy i test accuracy.

## 9. Dlaczego to jest Level 2?

Random Forest jest naturalnym krokiem po drzewach decyzyjnych.

W Level 1 student poznaje pojedyncze modele.

W Level 2 zaczyna poznawać praktyczne techniki poprawiające stabilność, generalizację i odporność modeli.

Random Forest dobrze pokazuje, że czasem lepiej połączyć wiele prostszych modeli niż polegać na jednym bardzo złożonym modelu.
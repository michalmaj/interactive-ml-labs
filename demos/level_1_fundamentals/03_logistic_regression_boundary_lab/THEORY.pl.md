# Logistic Regression Boundary Lab — teoria

## 1. Cel demo

Celem demo jest pokazanie, jak działa regresja logistyczna w problemie klasyfikacji binarnej.

Regresja logistyczna, mimo nazwy, nie jest używana tutaj do regresji, ale do klasyfikacji.

Model ma odpowiadać na pytanie:

```text
Czy punkt należy do klasy 0 czy do klasy 1?
```

## W demo student będzie obserwował:

- punkty dwóch klas,
- granicę decyzyjną,
- prawdopodobieństwa klasy pozytywnej,
- wpływ progu decyzyjnego,
- przypadki false positive i false negative.

## 2. Klasyfikacja binarna

Klasyfikacja binarna oznacza, że model wybiera jedną z dwóch klas.

Przykłady:

- spam albo nie spam,
- pacjent chory albo zdrowy,
- transakcja podejrzana albo normalna,
- punkt należy do klasy 0 albo klasy 1.

W tym demo klasy będą oznaczone jako:

- `class_0`,
- `class_1`.

## 3. Różnica względem k-NN

W k-NN decyzja zależy od najbliższych punktów treningowych.

W regresji logistycznej model uczy parametry granicy decyzyjnej.

Porównanie:

```text
k-NN:
nowy punkt -> najbliżsi sąsiedzi -> głosowanie -> klasa

regresja logistyczna:
nowy punkt -> wynik liniowy -> sigmoid -> prawdopodobieństwo -> próg -> klasa
```

k-NN działa lokalnie, ponieważ decyzja zależy od sąsiedztwa punktu.

Regresja logistyczna uczy globalną granicę decyzyjną.

## 4. Wynik liniowy

Regresja logistyczna zaczyna od obliczenia wyniku liniowego.

Dla danych 2D można zapisać to intuicyjnie jako:

```text
score = w1 * x1 + w2 * x2 + bias
```

gdzie:

- `x1`, `x2` — cechy punktu,
- `w1`, `w2` — wagi modelu,
- `bias` — wyraz wolny,
- `score` — wartość przed przekształceniem przez sigmoid.

Sam `score` nie jest jeszcze prawdopodobieństwem.

## 5. Sigmoid

Funkcja sigmoid przekształca dowolną liczbę rzeczywistą na wartość z zakresu od 0 do 1.

Intuicyjnie:

```text
bardzo ujemny score  -> probability blisko 0
score blisko 0       -> probability blisko 0.5
bardzo dodatni score -> probability blisko 1
```

Dzięki temu wynik modelu można interpretować jako prawdopodobieństwo klasy pozytywnej.

## 6. Prawdopodobieństwo a klasa

Regresja logistyczna najpierw zwraca prawdopodobieństwo.

Przykład:

```text
probability = 0.83
```

To jeszcze nie jest klasa.

Aby otrzymać klasę, trzeba zastosować próg decyzyjny.

Najczęściej używany próg to:

```text
threshold = 0.5
```

Jeżeli:

```text
probability >= threshold
```

model przewiduje `class_1`.

W przeciwnym razie przewiduje `class_0`.

## 7. Próg decyzyjny

Próg decyzyjny decyduje o tym, jak łatwo model przypisuje punkt do klasy pozytywnej.

Niższy próg:

- więcej punktów zostanie uznanych za `class_1`,
- może wzrosnąć recall,
- może wzrosnąć liczba false positives.

Wyższy próg:

- mniej punktów zostanie uznanych za `class_1`,
- może wzrosnąć precision,
- może wzrosnąć liczba false negatives.

To jest bardzo ważne w zastosowaniach praktycznych, np. w medycynie, detekcji spamu albo wykrywaniu fraudów.

## 8. Granica decyzyjna

Granica decyzyjna to miejsce, w którym model jest na granicy decyzji między klasami.

Dla progu 0.5 granica odpowiada zwykle punktom, dla których:

```text
probability = 0.5
```

W przypadku regresji logistycznej z dwiema cechami granica decyzyjna jest linią.

Punkty po jednej stronie linii mają większe prawdopodobieństwo klasy 1.

Punkty po drugiej stronie mają większe prawdopodobieństwo klasy 0.

## 9. Binary cross-entropy

Regresja logistyczna jest zwykle trenowana przez minimalizację funkcji straty nazywanej binary cross-entropy.

Intuicyjnie funkcja ta karze model za przypisywanie niskiego prawdopodobieństwa poprawnej klasie.

Przykład:

- jeżeli prawdziwa klasa to `1`, model powinien dać wysokie prawdopodobieństwo,
- jeżeli prawdziwa klasa to `0`, model powinien dać niskie prawdopodobieństwo.

## Uczenie modelu w aktualnej implementacji

Aktualna implementacja zawiera krokowy model regresji logistycznej.

Model uczy trzy parametry:

- `weight_1`,
- `weight_2`,
- `bias`.

W każdym kroku uczenia:

1. obliczany jest wynik liniowy,
2. sigmoid zamienia wynik na prawdopodobieństwo,
3. binary cross-entropy mierzy błąd,
4. gradient descent aktualizuje wagi i bias,
5. zapisywane są aktualne metryki.

Dzięki temu późniejsza wizualizacja będzie mogła pokazać, jak granica decyzyjna przesuwa się podczas uczenia.

## 10. False positives i false negatives

W klasyfikacji binarnej istnieją różne typy decyzji.

Dla klasy pozytywnej class_1:

- true positive — model poprawnie przewidział `class_1`,
- true negative — model poprawnie przewidział `class_0`,
- false positive — model przewidział `class_1`, ale prawdziwa klasa to `class_0`,
- false negative — model przewidział `class_0`, ale prawdziwa klasa to `class_1`.

Próg decyzyjny wpływa na liczbę false positives i false negatives.

## 11. Dlaczego to demo jest ważne?

Regresja logistyczna jest prostym modelem, ale pokazuje wiele kluczowych pojęć używanych także w bardziej zaawansowanych systemach ML:

- prawdopodobieństwo,
- próg decyzyjny,
- koszt błędów,
- precision,
- recall,
- interpretacja wyniku modelu,
- granica decyzyjna.

Dlatego jest bardzo dobrym krokiem po k-NN.

## 12. Typowe błędy interpretacyjne

### Błąd 1: regresja logistyczna służy do regresji

Nazwa jest myląca. W typowym zastosowaniu regresja logistyczna służy do klasyfikacji.

### Błąd 2: model zwraca od razu klasę

Nie do końca. Model najpierw zwraca prawdopodobieństwo, a dopiero później stosuje się próg decyzyjny.

### Błąd 3: threshold 0.5 zawsze jest najlepszy

Nie zawsze. Optymalny próg zależy od kosztów błędów.

### Błąd 4: accuracy wystarcza do oceny modelu

Nie zawsze. W wielu problemach ważniejsze są precision, recall albo koszt false positives i false negatives.

### Błąd 5: granica zawsze dobrze oddzieli klasy

Nie zawsze. Regresja logistyczna uczy liniową granicę. Jeżeli dane są nieliniowe, model może być zbyt prosty.

## Dane syntetyczne w pierwszej wersji demo

Pierwsza wersja demo wykorzystuje syntetyczny zbiór danych 2D.

Dane składają się z dwóch klas:

- `class_0`,
- `class_1`.

Każda klasa jest generowana jako chmura punktów wokół własnego centrum.

Parametry danych:

- `samples_per_class` — liczba punktów w każdej klasie,
- `class_distance` — odległość między centrami klas,
- `noise_std` — poziom szumu, czyli rozrzut punktów wokół centrum,
- `seed` — ziarno losowości umożliwiające odtwarzalność danych.

Dane syntetyczne są użyte celowo, ponieważ regresja logistyczna w pierwszej wersji demo będzie uczyć liniową granicę decyzyjną. Przy małym szumie klasy są dość łatwe do rozdzielenia linią, a przy większym szumie problem staje się trudniejszy.

## Sigmoid w implementacji demo

Pierwsza wersja implementacji zawiera funkcję `sigmoid`.

Funkcja ta zamienia wynik liniowy modelu na wartość z przedziału od 0 do 1.

Dzięki temu późniejszy model regresji logistycznej będzie mógł zwracać prawdopodobieństwo klasy pozytywnej.

## Metryki klasyfikacji

Demo zawiera już podstawowe metryki klasyfikacji binarnej:

- accuracy,
- precision,
- recall,
- confusion matrix counts.

Accuracy mówi, jaki odsetek wszystkich predykcji jest poprawny.

Precision odpowiada na pytanie:

```text
Spośród punktów przewidzianych jako class_1, ile naprawdę należy do class_1?
```

Recall odpowiada na pytanie:

```text
Spośród wszystkich punktów class_1, ile zostało poprawnie znalezionych?
```

Te metryki będą potrzebne do pokazania, jak threshold wpływa na błędy typu false positive i false negative.

## Tło prawdopodobieństwa

Wizualizacja pokazuje tło prawdopodobieństwa klasy `class_1`.

Jasny kolor po stronie `class_0` oznacza niskie prawdopodobieństwo klasy pozytywnej.

Kolor po stronie `class_1` oznacza wysokie prawdopodobieństwo klasy pozytywnej.

To tło pokazuje wartość przed zastosowaniem progu decyzyjnego.

Dlatego warto rozróżniać dwie rzeczy:

```text
probability -> wartość od 0 do 1
predicted class -> klasa po zastosowaniu threshold
```

Zmiana threshold może przesunąć granicę decyzyjną, ale nie zmienia samych prawdopodobieństw. Uczenie modelu zmienia tło, ponieważ zmieniają się wagi i bias.

## Confusion matrix w wizualizacji

Aktualna wersja demo pokazuje wartości:

- `TP` — true positives,
- `TN` — true negatives,
- `FP` — false positives,
- `FN` — false negatives.

Te wartości pomagają zrozumieć, skąd biorą się precision i recall.

Precision zależy od liczby false positives.

Recall zależy od liczby false negatives.

Zmiana threshold może zmienić liczbę FP i FN, nawet jeśli prawdopodobieństwa modelu pozostają takie same.

## Challenge mode — precision i recall

Aktualna wersja demo zawiera challenge mode oparty na ukrytym zbiorze testowym.

Cel challenge:

```text
recall >= 0.90
precision >= 0.80
```

Ten cel jest bardziej dydaktyczny niż sama accuracy, ponieważ pokazuje kompromis między wykrywaniem klasy pozytywnej a liczbą fałszywych alarmów.

Obniżenie threshold może zwiększyć recall, ale często zwiększa liczbę false positives.

Podwyższenie threshold może zwiększyć precision, ale często zwiększa liczbę false negatives.

Dlatego dobór threshold zależy od kosztów błędów.

## Panel wyjaśnień

Aktualna wersja demo zawiera panel wyjaśnień na dole ekranu.

Panel pomaga interpretować:

- tło prawdopodobieństwa,
- aktualny loss,
- accuracy, precision i recall,
- próg decyzyjny,
- false positives i false negatives,
- status challenge mode.

Dzięki temu student nie tylko widzi zmianę granicy decyzyjnej, ale także otrzymuje krótką interpretację tego, co oznaczają aktualne metryki.
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

## 0. False positives i false negatives

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


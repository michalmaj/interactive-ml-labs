# Gradient Descent — teoria

## 1. Cel demo

Celem demo jest pokazanie, w jaki sposób prosty model regresji liniowej może uczyć się na podstawie danych.

Model nie dostaje bezpośrednio informacji, jakie wartości parametrów są najlepsze. Zamiast tego zaczyna od wartości początkowych i stopniowo je poprawia, aby zmniejszyć błąd predykcji.

Demo pokazuje proces uczenia w sposób krokowy:

```text
jeden krok = jedna aktualizacja parametrów modelu
```

Dzięki temu można obserwować, jak zmieniają się:

- prosta regresji,
- wartość funkcji straty,
- parametr `weight`,
- parametr `bias`.

## 2. Problem regresji liniowej

W regresji liniowej celem jest przewidywanie wartości liczbowej.

W tym demo dane mają jedną cechę wejściową `x` i jedną wartość docelową `y`.

Model ma postać:

```text
y_pred = weight * x + bias
```

gdzie:

- `y_pred` — predykcja modelu,
- `x` — wartość wejściowa,
- `weight` — nachylenie prostej,
- `bias` — wyraz wolny.

Model próbuje dobrać takie wartości `weight` i `bias`, aby jego predykcje były możliwie bliskie wartościom oczekiwanym `y`.

## 3. Dane syntetyczne

Pierwsza wersja demo wykorzystuje dane syntetyczne generowane według równania:

```text
y = true_weight * x + true_bias + noise
```

gdzie:

- `true_weight` — prawdziwe nachylenie prostej użyte do wygenerowania danych,
- `true_bias` — prawdziwy wyraz wolny,
- `noise` — losowy szum dodany do danych.

Dane syntetyczne są użyte celowo, ponieważ pozwalają kontrolować trudność problemu.

Można zmieniać:

- poziom szumu,
- seed generatora losowego,
- liczbę próbek,
- zakres wartości wejściowych.

W aktualnym interfejsie Pygame student może zmieniać poziom szumu i seed.

## 4. Funkcja straty

Aby model mógł się uczyć, potrzebna jest miara błędu.

W tym demo używany jest mean squared error, czyli średni błąd kwadratowy.

Intuicyjnie:

```text
MSE = średnia wartość kwadratów różnic między y i y_pred
```

Jeżeli predykcja jest bardzo blisko wartości oczekiwanej, błąd jest mały.

Jeżeli predykcja jest daleko od wartości oczekiwanej, błąd jest duży. Ponieważ różnica jest podnoszona do kwadratu, większe błędy są karane silniej.

## 5. Gradient descent

Gradient descent jest metodą optymalizacji.

Jego zadaniem jest znalezienie takich parametrów modelu, które zmniejszają funkcję straty.

W tym demo gradient descent zmienia dwa parametry:

- `weight`,
- `bias`.

W każdym kroku algorytm:

1. oblicza predykcje modelu,
2. oblicza błędy predykcji,
3. oblicza gradient względem `weight` i `bias`,
4. aktualizuje parametry w kierunku zmniejszającym loss,
5. zapisuje nową wartość loss.

## 6. Intuicja gradientu

Gradient wskazuje kierunek największego wzrostu funkcji.

Ponieważ celem jest minimalizacja błędu, parametry aktualizowane są w przeciwnym kierunku niż gradient.

W uproszczeniu:

```text
nowy_parametr = stary_parametr - learning_rate * gradient
```

To oznacza, że:

- gradient mówi, w którą stronę zmienić parametr,
- learning rate mówi, jak duży krok wykonać.

## 7. Learning rate

Learning rate jest jednym z najważniejszych parametrów gradient descent.

Jeżeli learning rate jest zbyt mały:

- model uczy się bardzo wolno,
- loss maleje powoli,
- potrzeba wielu kroków, aby osiągnąć dobry wynik.

Jeżeli learning rate jest rozsądny:

- loss maleje stabilnie,
- model szybko dopasowuje prostą do danych.

Jeżeli learning rate jest zbyt duży:

- model może przeskakiwać minimum,
- loss może oscylować,
- uczenie może stać się niestabilne,
- challenge może się nie udać :D

W demo learning rate można zmieniać klawiszami `Up` i `Down`.

## 8. Szum w danych

Szum oznacza losowe odchylenie punktów od idealnej prostej.

Mały szum oznacza, że dane są prawie idealnie liniowe.

Duży szum oznacza, że punkty są bardziej rozrzucone i trudniej dopasować prostą.

Ważna obserwacja:

> Im większy szum, tym trudniej osiągnąć bardzo niski loss.

Nie oznacza to jednak, że model działa źle. W danych zaszumionych pewien poziom błędu jest naturalny.

W demo poziom szumu można zmieniać klawiszami `Left` i `Right`.

## 9. Challenge mode

Challenge mode zamienia obserwację algorytmu w małą grę edukacyjną.

Aktualny cel:

```text
osiągnij loss <= 1.0 przed upływem 80 kroków
```

Student może zmieniać:

- learning rate,
- poziom szumu,
- seed danych.

Dzięki temu można eksperymentować i sprawdzać, kiedy algorytm uczy się szybko, stabilnie i skutecznie.

## 10. Typowe błędy interpretacyjne

### Błąd 1: większy learning rate zawsze jest lepszy

Nie zawsze. Większy learning rate może przyspieszyć uczenie, ale po przekroczeniu pewnego poziomu może spowodować niestabilność.

### Błąd 2: jeżeli loss maleje, model na pewno jest dobry

Loss mierzony na danych treningowych nie mówi wszystkiego o generalizacji. W bardziej zaawansowanych demo zostanie pokazany podział na dane treningowe i testowe.

### Błąd 3: gradient descent zawsze znajduje najlepsze rozwiązanie

W prostej regresji liniowej problem jest łatwy, ale w bardziej złożonych modelach optymalizacja może być trudniejsza.

### Błąd 4: niski loss zawsze jest możliwy

Nie zawsze. Jeżeli dane są bardzo zaszumione, idealne dopasowanie może być nierealistyczne lub oznaczać przeuczenie.

## 11. Co warto obserwować w demo?

Podczas pracy z demo warto zwrócić uwagę na:

- czy loss maleje stabilnie,
- jak szybko czerwona linia zbliża się do punktów,
- jak zmienia się `weight`,
- jak zmienia się `bias`,
- kiedy challenge kończy się sukcesem,
- kiedy challenge kończy się porażką,
- jak szum wpływa na końcowy loss,
- jak learning rate wpływa na stabilność uczenia.


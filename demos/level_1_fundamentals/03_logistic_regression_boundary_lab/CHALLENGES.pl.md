# Logistic Regression Boundary Lab — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia jest zrozumienie, jak regresja logistyczna podejmuje decyzję klasyfikacyjną.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest klasyfikacja binarna,
- czym różni się prawdopodobieństwo od klasy,
- jak działa sigmoid,
- jak działa próg decyzyjny,
- czym jest granica decyzyjna,
- czym jest binary cross-entropy,
- czym są false positives i false negatives,
- jak threshold wpływa na precision i recall,
- dlaczego accuracy nie zawsze wystarcza.

## 2. Uruchomienie

Wersja tekstowa:

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab
```

Wersja interaktywna:

```bash
uv run --package logistic-regression-boundary-lab logistic-regression-boundary-lab-ui
```

## 3. Sterowanie

| Klawisz | Działanie |
| ------- | --------- |
| `Space` | Uruchom albo zatrzymaj automatyczne uczenie |
| `N` | Wykonaj jeden krok uczenia |
| `R` | Zresetuj demo |
| `Up` | Zwiększ learning rate |
| `Down` | Zmniejsz learning rate |
| `Q` | Zmniejsz threshold |
| `E` | Zwiększ threshold |
| `Left` | Zmniejsz poziom szumu |
| `Right` | Zwiększ poziom szumu |
| `S` | Wygeneruj nowy zbiór danych przez zmianę seeda |
| `Esc` | Zamknij okno |

Zmiana learning rate, threshold, poziomu szumu albo seeda resetuje model.

## 4. Zadania podstawowe

### Zadanie 1 — uruchomienie demo

1. Uruchom wersję Pygame.
2. Sprawdź, gdzie znajdują się punkty obu klas.
3. Sprawdź panel boczny z metrykami.
4. Sprawdź panel wyjaśnień na dole ekranu.

Odpowiedz:

- Ile klas widać na ekranie?
- Jakie metryki są pokazywane w panelu bocznym?
- Co oznacza tło wykresu?
- Co oznacza linia granicy decyzyjnej?

### Zadanie 2 — jeden krok uczenia

1. Naciśnij `N`, aby wykonać jeden krok uczenia.
2. Obserwuj zmianę loss.
3. Obserwuj granicę decyzyjną.
4. Obserwuj tło prawdopodobieństwa.

Odpowiedz:

- Czy loss się zmienił?
- Czy granica decyzyjna pojawiła się albo przesunęła?
- Czy tło prawdopodobieństwa zmieniło się po kroku uczenia?
- Dlaczego jeden krok może nie wystarczyć do dobrego dopasowania modelu?

### Zadanie 3 — automatyczne uczenie

1. Naciśnij `Space`, aby uruchomić automatyczne uczenie.
2. Obserwuj loss history.
3. Zatrzymaj uczenie ponownie klawiszem `Space`.

Odpowiedz:

- Czy loss zwykle maleje?
- Czy accuracy rośnie?
- Czy precision i recall zmieniają się tak samo?
- Czy model osiąga stabilny wynik?

## 5. Prawdopodobieństwo a klasa

### Zadanie 4 — tło prawdopodobieństwa

Tło wykresu pokazuje prawdopodobieństwo klasy `class_1`.

1. Uruchom demo.
2. Wykonaj kilka kroków uczenia.
3. Obserwuj, jak tło zmienia się wraz z uczeniem modelu.
4. Zwróć uwagę na obszary niskiego i wysokiego prawdopodobieństwa.

Odpowiedz:

- Który kolor oznacza większe prawdopodobieństwo klasy `class_1`?
- Czy punkty `class_1` znajdują się zwykle w obszarze większego prawdopodobieństwa?
- Czy tło pokazuje klasę czy prawdopodobieństwo?

### Zadanie 5 — probability vs predicted class

1. Wykonaj kilkanaście kroków uczenia.
2. Obserwuj tło prawdopodobieństwa.
3. Obserwuj granicę decyzyjną.
4. Zmień threshold klawiszami `Q` i `E`.

Odpowiedz:

- Czy zmiana threshold zmienia tło prawdopodobieństwa?
- Czy zmiana threshold przesuwa granicę decyzyjną?
- Dlaczego probability i predicted class to nie to samo?

## 6. Próg decyzyjny

### Zadanie 6 — niski threshold

1. Wytrenuj model przez kilkadziesiąt kroków.
2. Zmniejsz threshold klawiszem `Q`.
3. Obserwuj precision, recall, FP i FN.

Odpowiedz:

- Czy więcej punktów jest przewidywanych jako `class_1`?
- Co dzieje się z recall?
- Co dzieje się z false positives?
- Kiedy niski threshold może być użyteczny?

### Zadanie 7 — wysoki threshold

1. Wytrenuj model przez kilkadziesiąt kroków.
2. Zwiększ threshold klawiszem `E`.
3. Obserwuj precision, recall, FP i FN.

Odpowiedz:

- Czy mniej punktów jest przewidywanych jako `class_1`?
- Co dzieje się z precision?
- Co dzieje się z false negatives?
- Kiedy wysoki threshold może być użyteczny?

## 7. Confusion matrix

Panel boczny pokazuje wartości `TP`, `TN`, `FP` i `FN`.

### Zadanie 8 — interpretacja confusion matrix

1. Wytrenuj model.
2. Odczytaj wartości `TP`, `TN`, `FP` i `FN`.
3. Zmień threshold.
4. Odczytaj wartości ponownie.

Odpowiedz:

- Co oznacza `TP`?
- Co oznacza `FP`?
- Co oznacza `FN`?
- Które wartości wpływają na precision?
- Które wartości wpływają na recall?

### Zadanie 9 — FP/FN trade-off

1. Ustaw niski threshold.
2. Zapisz `FP` i `FN`.
3. Ustaw wysoki threshold.
4. Zapisz `FP` i `FN`.

Odpowiedz:

- Czy obniżenie threshold zmniejsza liczbę false negatives?
- Czy obniżenie threshold zwiększa liczbę false positives?
- Czy podwyższenie threshold działa odwrotnie?
- Dlaczego nie zawsze można jednocześnie zmniejszyć FP i FN?

## 8. Learning rate

### Zadanie 10 — mały learning rate

1. Ustaw mały learning rate klawiszem `Down`.
2. Zresetuj demo, jeśli trzeba.
3. Uruchom uczenie.
4. Obserwuj loss history.

Odpowiedz:

- Czy model uczy się wolno?
- Czy loss maleje stabilnie?
- Czy granica decyzyjna przesuwa się powoli?

### Zadanie 11 — większy learning rate

1. Zwiększ learning rate klawiszem `Up`.
2. Uruchom uczenie.
3. Obserwuj loss history i metryki.

Odpowiedz:

- Czy model uczy się szybciej?
- Czy loss maleje szybciej?
- Czy zbyt duży learning rate może pogorszyć stabilność uczenia?

## 9. Szum w danych

### Zadanie 12 — mały szum

1. Zmniejsz noise klawiszem `Left`.
2. Uruchom uczenie.
3. Obserwuj accuracy, precision i recall.

Odpowiedz:

- Czy klasy są łatwiejsze do rozdzielenia?
- Czy model szybciej osiąga dobre wyniki?
- Czy challenge jest łatwiejszy?

### Zadanie 13 — duży szum

1. Zwiększ noise klawiszem `Right`.
2. Uruchom uczenie.
3. Obserwuj błędnie sklasyfikowane punkty oznaczone `X`.

Odpowiedz:

- Czy klasy zaczynają się nakładać?
- Czy liczba błędnych predykcji rośnie?
- Czy sam threshold wystarcza do rozwiązania problemu?
- Czy liniowa granica decyzyjna nadal dobrze pasuje do danych?

## 10. Challenge mode — precision i recall

Demo ocenia model na ukrytym syntetycznym zbiorze testowym.

Aktualny cel:

```text
recall >= 0.90
precision >= 0.80
```

### Zadanie 14 — osiągnij challenge success

1. Uruchom demo.
2. Wytrenuj model.
3. Zmieniaj threshold klawiszami `Q` i `E`.
4. Obserwuj precision, recall, FP i FN.
5. Spróbuj uzyskać status `success`.

Zapisz:

- learning rate,
- threshold,
- noise,
- seed,
- precision,
- recall,
- status challenge.

Odpowiedz:

- Czy łatwiej uzyskać wysoki recall czy wysoką precision?
- Jak threshold wpływa na wynik challenge?
- Czy dobry wynik na treningu zawsze oznacza sukces na ukrytym zbiorze testowym?

### Zadanie 15 — trudniejszy challenge przez szum

1. Zwiększ poziom szumu.
2. Wytrenuj model.
3. Spróbuj osiągnąć challenge success.
4. Porównaj wynik z mniejszym szumem.

Odpowiedz:

- Jaki poziom szumu sprawia, że challenge staje się trudny?
- Czy threshold nadal pomaga?
- Czy model liniowy ma ograniczenia przy dużym szumie?

## 11. Panel wyjaśnień

Na dole ekranu znajduje się panel wyjaśnień.

Panel pokazuje między innymi:

- interpretację tła prawdopodobieństwa,
- aktualne metryki,
- informację o challenge mode,
- wartości FP i FN,
- sugestię strojenia threshold.

### Zadanie 16 — czytanie panelu wyjaśnień

1. Uruchom demo.
2. Wykonaj kilka kroków uczenia.
3. Zmieniaj threshold.
4. Obserwuj komunikat w panelu wyjaśnień.
5. Porównaj komunikat z wartościami FP, FN, precision i recall.

Odpowiedz:

- Czy panel opisuje to, co faktycznie widać w metrykach?
- Czy komunikat pomaga zrozumieć threshold?
- Czy panel wyjaśnia różnicę między probability i predicted class?

## 12. Pytania kontrolne

1. Do czego służy regresja logistyczna?
2. Dlaczego nazwa „regresja” może być myląca?
3. Co oznacza klasyfikacja binarna?
4. Co oznacza wynik liniowy modelu?
5. Co robi sigmoid?
6. Czym różni się probability od predicted class?
7. Co robi threshold?
8. Czym jest granica decyzyjna?
9. Czym jest binary cross-entropy?
10. Co oznacza false positive?
11. Co oznacza false negative?
12. Jak threshold wpływa na FP i FN?
13. Czym różni się precision od recall?
14. Dlaczego accuracy nie zawsze wystarcza?
15. Dlaczego hidden test set jest ważny?
16. Czym regresja logistyczna różni się od k-NN?
17. Jak learning rate wpływa na uczenie?
18. Dlaczego duży szum utrudnia klasyfikację?

## 13. Zadanie dodatkowe

Zaproponuj własny challenge mode dla regresji logistycznej.

Przykłady:

- osiągnij recall powyżej 0.95,
- osiągnij precision powyżej 0.90,
- zminimalizuj liczbę false negatives,
- znajdź threshold minimalizujący koszt błędów,
- osiągnij challenge success przy noise większym niż 2.0.

Opisz:

- cel challenge,
- ograniczenia,
- sposób punktacji,
- czego student może się z niego nauczyć.

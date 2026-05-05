# k-NN Vote Map — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia jest zrozumienie, jak algorytm k-NN podejmuje decyzję klasyfikacyjną.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest klasyfikacja,
- co oznacza parametr `k`,
- jak działa głosowanie sąsiadów,
- dlaczego małe `k` może prowadzić do overfittingu,
- dlaczego duże `k` może prowadzić do underfittingu,
- jak szum wpływa na klasyfikację,
- czym jest granica decyzyjna,
- dlaczego wynik na zbiorze testowym jest ważny.

## 2. Uruchomienie

Wersja tekstowa:

```bash
uv run --package knn-vote-map knn-vote-map
```

Wersja interaktywna:

```bash
uv run --package knn-vote-map knn-vote-map-ui
```

## 3. Sterowanie

| Akcja           | Działanie                                      |
| --------------- | ---------------------------------------------- |
| Kliknięcie mapy | Klasyfikuj kliknięty punkt                     |
| `N`             | Wylosuj i sklasyfikuj punkt testowy            |
| `R`             | Zresetuj demo                                  |
| `Up`            | Zwiększ `k`                                    |
| `Down`          | Zmniejsz `k`                                   |
| `Left`          | Zmniejsz poziom szumu                          |
| `Right`         | Zwiększ poziom szumu                           |
| `S`             | Wygeneruj nowy zbiór danych przez zmianę seeda |
| `Esc`           | Zamknij okno                                   |

Zmiana `k`, poziomu szumu albo seeda resetuje demo i przelicza mapę decyzji.

## 4. Zadania podstawowe

### Zadanie 1 — klasyfikacja jednego punktu

1. Uruchom demo.
2. Kliknij punkt blisko środka niebieskiej klasy.
3. Sprawdź przewidywaną klasę.
4. Sprawdź linie prowadzące do najbliższych sąsiadów.

**Odpowiedz:**

- Jaka klasa została przewidziana?
- Czy najbliżsi sąsiedzi należą głównie do tej samej klasy?
- Czy decyzja wydaje się intuicyjna?

---

### Zadanie 2 — punkt blisko drugiej klasy

1. Kliknij punkt blisko środka pomarańczowej klasy.
2. Sprawdź predykcję.
3. Porównaj wynik z poprzednim zadaniem.

**Odpowiedz:**

- Czy predykcja zmieniła się?
- Czy głosowanie sąsiadów jest jednoznaczne?
- Czy wszystkie najbliższe punkty należą do jednej klasy?

---

### Zadanie 3 — punkt przy granicy decyzyjnej

1. Kliknij kilka punktów blisko granicy między kolorami tła.
2. Obserwuj głosy sąsiadów.
3. Porównaj predykcje po obu stronach granicy.

**Odpowiedz:**

- Czy mała zmiana położenia punktu może zmienić klasę?
- Czy głosowanie jest bardziej wyrównane przy granicy?
- Dlaczego punkty przy granicy są trudniejsze?

---

## 5. Eksperymenty z parametrem `k`

### Zadanie 4 — małe `k`

1. Ustaw małe `k`, np. `k = 1`.
2. Obserwuj tło decyzji.
3. Klikaj punkty blisko granicy.

**Odpowiedz:**

- Czy granica decyzyjna jest gładka czy poszarpana?
- Czy pojedyncze punkty treningowe silnie wpływają na decyzję?
- Czy model wydaje się wrażliwy na szum?

---

### Zadanie 5 — większe `k`

1. Zwiększ `k` kilka razy klawiszem `Up`.
2. Obserwuj zmianę tła decyzji.
3. Klikaj te same okolice co wcześniej.

**Odpowiedz:**

- Czy granica decyzyjna staje się gładsza?
- Czy głosowanie sąsiadów jest bardziej stabilne?
- Czy `accuracy challenge` rośnie czy maleje?

---

### Zadanie 6 — zbyt duże `k`

1. Ustaw bardzo duże `k`.
2. Obserwuj tło decyzji.
3. Klikaj punkty blisko mniejszych lokalnych obszarów.

**Odpowiedz:**

- Czy model ignoruje lokalne szczegóły?
- Czy bardzo duże `k` zawsze pomaga?
- Kiedy duże `k` może prowadzić do underfittingu?

---

## 6. Eksperymenty z szumem

### Zadanie 7 — mały szum

1. Zmniejsz szum klawiszem `Left`.
2. Obserwuj separację klas.
3. Sprawdź `accuracy challenge`.

**Odpowiedz:**

- Czy klasy są łatwiejsze do rozdzielenia?
- Czy challenge jest łatwiejszy?
- Czy granica decyzyjna jest bardziej stabilna?

---

### Zadanie 8 — duży szum

1. Zwiększ szum klawiszem `Right`.
2. Obserwuj, jak klasy zaczynają się mieszać.
3. Sprawdź `accuracy challenge`.

**Odpowiedz:**

- Czy punkty różnych klas zaczynają się nakładać?
- Czy głosowanie sąsiadów częściej jest niejednoznaczne?
- Czy zmiana `k` może poprawić wynik?

---

## 7. Eksperymenty z seedem

### Zadanie 9 — różne zbiory danych

1. Naciśnij `S`, aby zmienić seed.
2. Obserwuj nowy rozkład punktów.
3. Sprawdź `challenge accuracy`.
4. Powtórz kilka razy.

**Odpowiedz:**

- Czy każdy seed daje równie łatwy problem?
- Czy ta sama wartość `k` działa dobrze dla każdego seeda?
- Czy czasami warto zmienić `k` po zmianie danych?

---

## 8. Mapa decyzji

Tło wykresu pokazuje, jaką klasę przewidziałby k-NN w różnych obszarach przestrzeni.

### Zadanie 10 — analiza mapy decyzji

1. Ustaw `k = 1`.
2. Obserwuj kształt granicy decyzyjnej.
3. Zwiększ `k`.
4. Porównaj zmianę tła.

**Odpowiedz:**

- Dlaczego dla małego `k` granica może być bardziej poszarpana?
- Dlaczego większe `k` często wygładza granicę?
- Czy wygładzanie zawsze oznacza lepszy model?

## 9. Challenge mode — accuracy na ukrytym zbiorze testowym

Demo oblicza accuracy na osobnym syntetycznym zbiorze testowym.

Aktualny cel:

```text
osiągnij accuracy >= 0.90
```

### Zadanie 11 — pobij challenge

1. Ustaw domyślny poziom szumu.

2. Zmieniaj `k`.

3. Spróbuj osiągnąć `challenge success`.

**Zapisz:**

- wartość `k`,

- poziom szumu,

- seed,

- accuracy,

- status challenge.

---

### Zadanie 12 — trudniejszy challenge przez szum

1. Zwiększ poziom szumu.

2. Sprawdź, czy challenge nadal jest możliwy.

3. Zmieniaj `k`, aby poprawić wynik.

**Odpowiedz:**

- Jaki poziom szumu sprawia, że challenge staje się trudny?

- Czy małe `k` czy większe `k` działa lepiej?

- Czy przy dużym szumie można zawsze osiągnąć wysoką accuracy?

---

## 10. Panel wyjaśnień

Na dole ekranu znajduje się panel wyjaśnień.

Panel pokazuje między innymi:

- wynik głosowania sąsiadów,

- przewidywaną klasę klikniętego punktu,

- informację o `challenge mode`,

- sugestię zmiany `k`, poziomu szumu albo seeda.

### Zadanie 13 — czytanie decyzji modelu

1. Kliknij punkt blisko granicy.

2. Odczytaj głosy sąsiadów.

3. Zmień `k`.

4. Kliknij podobny punkt jeszcze raz.

**Odpowiedz:**

- Czy głosy sąsiadów zmieniły się po zmianie `k`?

- Czy predykcja się zmieniła?

- Czy panel wyjaśnień pomaga zrozumieć decyzję?

---

## 11. Pytania kontrolne

1. Co oznacza `k` w k-NN?

2. Co oznacza klasyfikacja?

3. Jak działa głosowanie sąsiadów?

4. Dlaczego `k = 1` może prowadzić do overfittingu?

5. Dlaczego bardzo duże `k` może prowadzić do underfittingu?

6. Co oznacza granica decyzyjna?

7. Dlaczego odległość jest kluczowa w k-NN?

8. Dlaczego szum utrudnia klasyfikację?

9. Czym różni się punkt treningowy od punktu testowego?

10. Dlaczego accuracy na ukrytym zbiorze testowym jest ważna?

11. Czy k-NN ma etap uczenia podobny do gradient descent?

12. Kiedy k-NN może działać słabo?

---

## 12. Zadanie dodatkowe

### Zaproponuj własny challenge mode dla k-NN

**Przykłady:**

- osiągnij accuracy powyżej `0.95`,

- osiągnij accuracy powyżej `0.90` przy szumie większym niż `2.0`,

- znajdź `k`, które działa dobrze dla pięciu różnych seedów,

- porównaj `k = 1`, `k = 5`, `k = 15` dla kilku poziomów szumu.

**Opisz:**

- cel challenge,

- ograniczenia,

- sposób punktacji.
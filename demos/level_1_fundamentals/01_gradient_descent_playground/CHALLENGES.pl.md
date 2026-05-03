# Gradient Descent Playground — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia jest zrozumienie, jak gradient descent zmienia parametry prostego modelu regresji liniowej.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest loss,
- co oznacza learning rate,
- co dzieje się w jednym kroku gradient descent,
- jak szum wpływa na jakość dopasowania,
- dlaczego zbyt duży learning rate może być problemem.

## 2. Uruchomienie

Wersja tekstowa:

```bash
uv run --package gradient-descent-playground gradient-descent-playground
```

Wersja interaktywna:

```bash
uv run --package gradient-descent-playground gradient-descent-playground-ui
```

## 3. Sterowanie

| Klawisz | Działanie                                      |
| ------- | ---------------------------------------------- |
| `Space` | Uruchom lub zatrzymaj automatyczne uczenie     |
| `N`     | Wykonaj jeden krok gradient descent            |
| `R`     | Zresetuj demo                                  |
| `Up`    | Zwiększ learning rate                          |
| `Down`  | Zmniejsz learning rate                         |
| `Left`  | Zmniejsz poziom szumu                          |
| `Right` | Zwiększ poziom szumu                           |
| `S`     | Wygeneruj nowy zbiór danych przez zmianę seeda |
| `Esc`   | Zamknij okno                                   |

Zmiana learning rate, poziomu szumu albo seeda resetuje demo.

## 4. Zadania podstawowe

### Zadanie 1 — pierwszy krok uczenia

1. Uruchom demo.
2. Nie naciskaj `Space`.
3. Naciśnij `N` jeden raz.
4. Zaobserwuj, co zmieniło się na ekranie.

Odpowiedz:

- Czy zmieniła się czerwona linia?
- Czy zmienił się loss?
- Czy zmieniły się wartości weight i bias?

### Zadanie 2 — uczenie krok po kroku

1. Naciskaj `N` kilkanaście razy.
2. Obserwuj loss history.
3. Obserwuj czerwoną linię regresji.

Odpowiedz:

- Czy loss maleje w każdym kroku?
- Czy linia zbliża się do punktów?
- Czy zmiany są duże na początku czy pod koniec?

### Zadanie 3 — tryb automatyczny

1. Zresetuj demo klawiszem `R`.
2. Naciśnij `Space`.
3. Obserwuj automatyczne uczenie.

Odpowiedz:

- Ile kroków potrzeba, aby challenge zakończył się sukcesem?
- Czy model zatrzymuje się po osiągnięciu celu?
- Jak zmienia się wykres loss?

## 5. Eksperymenty z learning rate

### Zadanie 4 — mały learning rate

1. Zmniejsz learning rate kilka razy klawiszem `Down`.
2. Uruchom uczenie.
3. bserwuj tempo spadku loss.

Odpowiedz:

- Czy model uczy się szybciej czy wolniej?
- Czy challenge kończy się sukcesem?
- Czy uczenie jest stabilne?

### Zadanie 5 — większy learning rate

1. Zwiększ learning rate kilka razy klawiszem `Up`.
2. Uruchom uczenie.
3. Obserwuj loss i czerwoną linię.

Odpowiedz:

- Czy loss maleje szybciej?
- Czy linia zachowuje się stabilnie?
- Czy challenge jest łatwiejszy?

### Zadanie 6 — zbyt duży learning rate

1. Zwiększ learning rate do wysokiej wartości.
2. Uruchom uczenie.
3. Obserwuj, czy model nadal zachowuje się stabilnie.

Odpowiedz:

- Czy loss nadal maleje?
- Czy czerwona linia zaczyna „skakać”?
- Czy większy learning rate zawsze pomaga

## 6. Eksperymenty z szumem

### Zadanie 7 — mały szum

1. Zmniejsz szum klawiszem `Left`.
2. Uruchom uczenie.
3. Spróbuj wygrać challenge.

Odpowiedz:

- Czy łatwiej osiągnąć niski loss?
- Czy punkty leżą bliżej jednej prostej?
- Czy wynik jest bardziej stabilny?

### Zadanie 8 — duży szum

1. Zwiększ szum klawiszem `Right`.
2. Uruchom uczenie.
3. Spróbuj wygrać challenge.

Odpowiedz:

- Czy trudniej osiągnąć niski loss?
- Czy challenge nadal jest możliwy?
- Czy wysoki loss zawsze oznacza zły model?

## 7. Eksperymenty z seedem

### Zadanie 9 — różne zbiory danych

1. Naciśnij `S`, aby zmienić seed.
2. Uruchom uczenie.
3. Powtórz eksperyment kilka razy.

Odpowiedz:

- Czy każdy zbiór danych jest równie łatwy?
- Czy ten sam learning rate działa dobrze dla każdego seeda?

## 8. Challenge mode

Aktualny challenge:

```text
osiągnij loss <= 1.0 przed upływem 80 kroków
```

### Zadanie 10 — pobij challenge

Znajdź takie ustawienie learning rate, aby wygrać challenge dla domyślnego poziomu szumu.

Zapisz:

- learning rate,
- końcowy loss,
- liczbę kroków,
- status challenge.

### Zadanie 11 — trudniejszy challenge przez szum

1. Zwiększ poziom szumu.
2. Spróbuj wygrać challenge.
3. Zmieniaj learning rate, jeśli to konieczne.

Odpowiedz:

- Jaki poziom szumu sprawia, że challenge staje się trudny?
- Czy da się to naprawić samą zmianą learning rate?
- Co to mówi o jakości danych?

## 9. Pytania kontrolne

1. Co oznacza wartość loss?
2. Dlaczego loss jest wysoki na początku?
3. Co zmienia się w jednym kroku gradient descent?
4. Czym różni się `weight` od `bias`?
5. Co oznacza learning rate?
6. Co się stanie, gdy learning rate jest zbyt mały?
7. Co się stanie, gdy learning rate jest zbyt duży?
8. Dlaczego dane z dużym szumem są trudniejsze?
9. Czy niski loss zawsze jest możliwy?
10. Dlaczego warto testować różne seedy danych?

## 10. Zadanie dodatkowe

Zaproponuj własną wersję challenge mode.

Przykłady:

- osiągnij loss poniżej 0.5,
- osiągnij loss poniżej 1.0 w mniej niż 40 krokach,
- wygraj challenge przy szumie większym niż 2.0,
- znajdź learning rate, który działa dobrze dla pięciu różnych seedów.

Opisz:

- cel challenge,
- ograniczenia,
- sposób punktacji,
- czego student może się z niego nauczyć.


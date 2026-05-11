# Random Forest Bagging Lab — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia jest zrozumienie, jak wiele drzew decyzyjnych może wspólnie tworzyć stabilniejszy klasyfikator.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest Random Forest,
- czym jest bagging,
- czym jest bootstrap sampling,
- jak działa głosowanie większościowe,
- czym jest vote confidence,
- czym jest train/test split,
- czym jest generalization gap,
- dlaczego wiele drzew może być stabilniejsze niż jedno drzewo,
- jak liczba drzew wpływa na wynik,
- jak głębokość drzew wpływa na overfitting.

## 2. Uruchomienie

Wersja CLI:

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
```

Wersja interaktywna:

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab-ui
```

## 3. Sterowanie UI

| Klawisz | Działanie |
| ------- | --------- |
| `D` | Przełącz dataset: `axis_aligned` / `xor` |
| `Up` | Zwiększ liczbę drzew |
| `Down` | Zmniejsz liczbę drzew |
| `W` | Zwiększ `max_depth` |
| `S` | Zmniejsz `max_depth` |
| `B` | Zwiększ bootstrap sample ratio |
| `V` | Zmniejsz bootstrap sample ratio |
| `C` | Włącz/wyłącz confidence view |
| `Left` | Zmniejsz poziom szumu |
| `Right` | Zwiększ poziom szumu |
| `N` | Wygeneruj nowy dataset seed |
| `R` | Zresetuj demo |
| `Esc` | Zamknij okno |

## 4. Zadania podstawowe

### Zadanie 1 — uruchomienie CLI

1. Uruchom wersję CLI.
2. Znajdź wyniki dla `axis_aligned`.
3. Znajdź wyniki dla `xor`.
4. Porównaj single-tree baseline z Random Forest.

Odpowiedz:

- Który model ma większą train accuracy?
- Który model ma większą test accuracy?
- Czy Random Forest zawsze wygrywa?
- Co oznacza winner by test accuracy?

### Zadanie 2 — uruchomienie UI

1. Uruchom UI.
2. Sprawdź lewy panel.
3. Sprawdź prawy panel.
4. Sprawdź panel boczny.
5. Sprawdź panel dolny.

Odpowiedz:

- Co pokazuje lewy panel?
- Co pokazuje prawy panel?
- Co oznaczają kółka?
- Co oznaczają kwadraty?
- Co oznacza znak X?

## 5. Train/test split

### Zadanie 3 — interpretacja train i test

Demo korzysta z osobnych danych treningowych i testowych.

Odpowiedz:

- Do czego służy zbiór treningowy?
- Do czego służy zbiór testowy?
- Dlaczego wysoka train accuracy nie musi oznaczać dobrej generalizacji?
- Dlaczego Random Forest powinien być oceniany także na danych testowych?

## 6. Bootstrap sampling

### Zadanie 4 — indeksy bootstrap

Dany jest zbiór indeksów:

```text
[0, 1, 2, 3, 4, 5]
```

Przykładowy bootstrap sample:

```text
[2, 2, 5, 0, 2, 4]
```

Odpowiedz:

- Które próbki zostały wybrane więcej niż raz?
- Które próbki nie zostały wybrane ani razu?
- Które próbki są out-of-bag?
- Dlaczego różne bootstrap sample mogą prowadzić do różnych drzew?

### Zadanie 5 — bootstrap ratio

1. Uruchom UI.
2. Zmieniaj bootstrap ratio klawiszami `B` i `V`.
3. Obserwuj wyniki w panelu bocznym.
4. Porównaj regiony decyzyjne.

Odpowiedz:

- Co oznacza bootstrap ratio?
- Czy mniejszy bootstrap ratio zwiększa różnorodność drzew?
- Czy większa różnorodność zawsze poprawia test accuracy?
- Co może się stać, jeśli każde drzewo widzi zbyt mało danych?

## 7. Majority voting

### Zadanie 6 — ręczne głosowanie

Dane są predykcje trzech drzew:

```text
tree_1: [0, 1, 1, 0]
tree_2: [0, 1, 0, 0]
tree_3: [1, 1, 1, 0]
```

Odpowiedz:

- Jaka jest finalna predykcja dla każdej próbki?
- Ile głosów dostała klasa `0` dla pierwszej próbki?
- Ile głosów dostała klasa `1` dla trzeciej próbki?
- Dla której próbki confidence wynosi `1.0`?
- Co oznacza confidence równe `2/3`?

## 8. Single tree kontra Random Forest

### Zadanie 7 — porównanie modeli

1. Uruchom UI.
2. Ustaw dataset `axis_aligned`.
3. Porównaj lewy i prawy panel.
4. Przełącz dataset na `xor`.
5. Ponownie porównaj modele.

Odpowiedz:

- Czy single tree i Random Forest tworzą takie same regiony decyzyjne?
- Na którym datasecie różnice są większe?
- Czy pojedyncze drzewo wystarcza dla `axis_aligned`?
- Czy pojedyncze drzewo wystarcza dla `xor`?

## 9. Liczba drzew

### Zadanie 8 — wpływ liczby drzew

1. Ustaw dataset `xor`.
2. Ustaw małą liczbę drzew.
3. Zwiększaj liczbę drzew klawiszem `Up`.
4. Obserwuj test accuracy.
5. Obserwuj confidence.

Odpowiedz:

- Czy więcej drzew zawsze poprawia wynik?
- Czy wynik stabilizuje się po pewnej liczbie drzew?
- Co dzieje się z czasem obliczeń?
- Czy bardzo duża liczba drzew jest zawsze potrzebna?

## 10. Max depth

### Zadanie 9 — wpływ głębokości drzew

1. Ustaw dataset `xor`.
2. Ustaw `max_depth = 1`.
3. Zwiększ `max_depth` do 2.
4. Zwiększ `max_depth` do 3 lub więcej.
5. Obserwuj train/test accuracy.

Odpowiedz:

- Dlaczego `max_depth = 1` może być zbyt małe dla XOR?
- Co poprawia się po zwiększeniu głębokości?
- Czy zbyt duża głębokość może zwiększać overfitting?
- Jak zmienia się generalization gap?

## 11. Noise

### Zadanie 10 — wpływ szumu

1. Ustaw dataset `xor`.
2. Ustaw umiarkowaną liczbę drzew.
3. Zwiększaj noise klawiszem `Right`.
4. Obserwuj błędnie sklasyfikowane test points.
5. Obserwuj test accuracy.

Odpowiedz:

- Czy większy szum pogarsza wynik?
- Czy Random Forest radzi sobie lepiej niż single tree?
- Czy confidence spada w trudniejszych regionach?
- Czy challenge nadal da się spełnić przy dużym szumie?

## 12. Confidence view

### Zadanie 11 — interpretacja confidence

1. Włącz confidence view klawiszem `C`.
2. Obserwuj regiony decyzyjne lasu.
3. Zmieniaj liczbę drzew.
4. Zmieniaj noise.
5. Zmieniaj bootstrap ratio.

Odpowiedz:

- Co oznaczają blade regiony?
- Czy blade regiony zawsze oznaczają błąd?
- Czy confidence jest tym samym co accuracy?
- Dlaczego confidence jest przydatne w modelach ensemble?

## 13. Raport porównawczy CLI

### Zadanie 12 — generalization gap

Uruchom CLI:

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
```

Odpowiedz:

- Który model ma większą train accuracy?
- Który model ma większą test accuracy?
- Co oznacza generalization gap?
- Dlaczego winner powinien być wybierany według test accuracy, a nie train accuracy?
- Czy model z największą train accuracy jest zawsze najlepszy?

## 14. Challenge mode

### Zadanie 13 — spełnienie challenge

Uruchom UI:

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab-ui
```

Warunki challenge:

```text
forest test accuracy >= 0.90
tree_count <= 25
generalization gap <= 0.15
```

Zadanie:

1. Ustaw dataset `xor`.
2. Spróbuj uzyskać challenge status `success`.
3. Zmieniaj liczbę drzew.
4. Zmieniaj `max_depth`.
5. Zmieniaj bootstrap ratio.
6. Obserwuj test accuracy i generalization gap.

Odpowiedz:

- Dlaczego challenge ogranicza liczbę drzew?
- Dlaczego liczy się test accuracy, a nie tylko train accuracy?
- Co oznacza generalization gap?
- Czy większy las zawsze jest lepszy?
- Czy challenge da się spełnić przy dużym szumie?

## 15. Panel wyjaśnień

### Zadanie 14 — komunikaty w UI

1. Włącz i wyłącz confidence view klawiszem `C`.
2. Obserwuj tekst w dolnym panelu.
3. Zmień liczbę drzew tak, aby challenge się nie udał.
4. Zmień parametry tak, aby challenge został spełniony.
5. Porównaj komunikaty.

Odpowiedz:

- Czy panel jasno tłumaczy aktualny stan challenge?
- Czy komunikat o confidence view pomaga interpretować kolory?
- Dlaczego teksty wyjaśnień warto testować automatycznie?

## 16. Pytania kontrolne

1. Czym jest Random Forest?
2. Czym jest bagging?
3. Czym jest bootstrap sampling?
4. Czym są OOB samples?
5. Czym jest majority voting?
6. Czym jest vote confidence?
7. Dlaczego pojedyncze drzewo może mieć overfitting?
8. Dlaczego las może być stabilniejszy?
9. Czym jest train accuracy?
10. Czym jest test accuracy?
11. Czym jest generalization gap?
12. Dlaczego test accuracy jest ważniejsza niż train accuracy?
13. Czy confidence oznacza poprawność predykcji?
14. Dlaczego challenge ogranicza liczbę drzew?
15. Czym Random Forest różni się od pojedynczego decision tree?

## 17. Zadanie dodatkowe

Zaproponuj własny challenge mode dla Random Forest.

Przykłady:

- osiągnij test accuracy powyżej 0.92 przy maksymalnie 15 drzewach,
- uzyskaj jak najmniejszy generalization gap,
- porównaj bootstrap ratio 0.5 i 1.0,
- znajdź minimalną liczbę drzew potrzebną do rozwiązania XOR,
- znajdź konfigurację z najwyższą confidence przy dobrej test accuracy.

Opisz:

- cel challenge,
- ograniczenia,
- sposób punktacji,
- czego student może się z niego nauczyć.
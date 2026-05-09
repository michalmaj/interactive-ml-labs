# Random Forest Bagging Lab — zadania

## 1. Cel ćwiczenia

Celem ćwiczenia będzie zrozumienie, jak wiele drzew decyzyjnych może wspólnie tworzyć stabilniejszy klasyfikator.

Po wykonaniu zadań student powinien umieć odpowiedzieć na pytania:

- czym jest Random Forest,
- czym jest bagging,
- czym jest bootstrap sampling,
- jak działa głosowanie większościowe,
- czym jest vote confidence,
- dlaczego wiele drzew może być stabilniejsze niż jedno drzewo,
- jak liczba drzew wpływa na wynik,
- jak głębokość drzew wpływa na overfitting.

## 2. Uruchomienie

Aktualna wersja zawiera tylko placeholder:

```bash
uv run --package random-forest-bagging-lab random-forest-bagging-lab
```

Interaktywna wersja Pygame zostanie dodana w kolejnych PR-ach.

## 3. Planowane zadania podstawowe

### Zadanie 1 — jedno drzewo kontra las

Wytrenuj jedno drzewo.

Wytrenuj las z wieloma drzewami.

Porównaj granice decyzyjne.

Porównaj accuracy.

Odpowiedz:

- Który model jest stabilniejszy?
- Czy las ma gładszą granicę decyzyjną?
- Czy pojedyncze drzewo łatwiej przeucza się do szumu?

### Zadanie 2 — liczba drzew

Ustaw małą liczbę drzew.

Zwiększ liczbę drzew.

Obserwuj vote confidence.

Obserwuj accuracy.

Odpowiedz:

- Czy więcej drzew zawsze poprawia wynik?
- Czy wynik stabilizuje się po pewnej liczbie drzew?
- Co dzieje się z czasem obliczeń?

### Zadanie 3 — bootstrap sampling

Sprawdź, jak wygląda bootstrap sample.

Zobacz, które próbki pojawiają się wiele razy.

Zobacz, które próbki nie pojawiają się w danym drzewie.

Odpowiedz:

- Dlaczego bootstrap sampling tworzy różne drzewa?
- Dlaczego różnorodność drzew jest ważna?
- Co oznacza losowanie z powtórzeniami?

## 4. Planowany challenge mode

Przykładowy challenge:

```text
Osiągnij test accuracy >= 0.90 przy maksymalnie 25 drzewach.

```

Alternatywny challenge:

```text
Popraw wynik pojedynczego drzewa używając lasu z ograniczoną liczbą drzew.
```

## 5. Pytania kontrolne

- Czym jest Random Forest?
- Czym jest bagging?
- Czym jest bootstrap sampling?
- Dlaczego drzewa w lesie różnią się od siebie?
- Jak działa majority voting?
- Czym jest vote confidence?
- Dlaczego pojedyncze drzewo może mieć overfitting?
- Dlaczego las może być stabilniejszy?
- Czy więcej drzew zawsze oznacza lepszy model?
- Czym Random Forest różni się od pojedynczego decision tree?
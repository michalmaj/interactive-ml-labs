# Boosting Mistake Lab — zadania i pytania kontrolne

Ten plik zawiera pytania oraz zadania do demo **Boosting Mistake Lab**. Celem nie jest wyłącznie uruchomienie gotowego programu, lecz zrozumienie, jak boosting zmienia znaczenie próbek, jak łączy weak learnery oraz jak oceniać model przez pryzmat danych testowych, a nie tylko treningowych.

## 1. Dataset z wagami próbek

Aktualna wersja demo generuje syntetyczne datasety z wagami próbek.

Dostępne są dwa warianty:

```text
axis_aligned
xor
```

Zadanie:

1. Uruchom CLI.
2. Porównaj dataset `axis_aligned` i `xor`.
3. Sprawdź liczbę próbek treningowych i testowych.
4. Sprawdź sumę wag próbek.

Pytania:

- Dlaczego suma wag próbek powinna wynosić `1.0`?
- Dlaczego boosting potrzebuje wag próbek?
- Który dataset jest łatwiejszy dla pojedynczego decision stumpa?
- Dlaczego dataset XOR jest trudniejszy dla prostego splitu osiowego?

## 2. Weak learner baseline

Weak learnerem w tym demo jest prosty decision stump, czyli model z jednym podziałem.

Zadanie:

1. Uruchom weak learner na danych `axis_aligned`.
2. Uruchom weak learner na danych `xor`.
3. Porównaj train accuracy i test accuracy.

Pytania:

- Co oznacza weak learner?
- Dlaczego weak learner może być prosty?
- Dlaczego pojedynczy stump może dobrze działać na `axis_aligned`?
- Dlaczego pojedynczy stump zwykle nie wystarcza dla `xor`?
- Co oznacza `feature_index` i `threshold` w decision stumpie?

## 3. Weighted error

Boosting ocenia weak learnera przy pomocy błędu ważonego.

Dany jest przykład:

```text
sample:      A    B    C    D
is mistake:  no   yes  no   yes
weight:      0.1  0.2  0.3  0.4
```

Zadanie:

- Oblicz weighted error.
- Oblicz weighted accuracy.

Pytania:

- Dlaczego weighted error nie jest tym samym co zwykły error?
- Która błędna próbka ma większy wpływ: `B` czy `D`?
- Dlaczego próbki o większej wadze są ważniejsze dla kolejnych rund boostingu?

## 4. Learner weight alpha

W boostingu każdy weak learner otrzymuje wagę `alpha`.

Dla uproszczonego AdaBoost-like podejścia:

```text
alpha = 0.5 * log((1 - weighted_error) / weighted_error)
```

Zadanie:

Porównaj wartości alpha dla błędów:

```text
weighted_error = 0.10
weighted_error = 0.30
weighted_error = 0.49
```

Pytania:

- Który weak learner otrzyma największe alpha?
- Co dzieje się, gdy weighted error zbliża się do `0.5`?
- Dlaczego model z niskim błędem powinien mieć większy wpływ na ensemble?
- Dlaczego alpha nie powinno być traktowane jako zwykła accuracy?

## 5. Aktualizacja wag próbek

Po każdej rundzie boosting aktualizuje wagi próbek.

Intuicja:

```text
poprawnie sklasyfikowane próbki -> mniejsza waga
błędnie sklasyfikowane próbki   -> większa waga
```

Zadanie:

Dany jest weak learner, który błędnie klasyfikuje dwie próbki. Odpowiedz opisowo:

- Które próbki powinny zwiększyć wagę?
- Które próbki powinny zmniejszyć wagę?
- Dlaczego po aktualizacji trzeba ponownie znormalizować wagi?

Pytania:

- Co oznacza `old_mistake_weight_sum`?
- Co oznacza `updated_mistake_weight_sum`?
- Dlaczego błędne próbki stają się ważniejsze w kolejnej rundzie?
- Czy próbka może mieć dużą wagę mimo że jest tylko jednym punktem w zbiorze?

## 6. Weighted stump split search

Aktualna wersja wybiera split weak learnera na podstawie weighted error.

Zadanie koncepcyjne:

Dany jest liść stumpa z dwiema próbkami:

```text
sample:  A    B
class:   0    1
weight:  0.1  0.9
```

Pytania:

- Jaka klasa powinna zostać wybrana przez zwykłą większość?
- Jaka klasa powinna zostać wybrana przez ważoną większość?
- Dlaczego próbka `B` ma większy wpływ?
- Dlaczego weighted majority jest ważne w boostingu?
- Co może się stać, jeśli stump ignoruje sample weights podczas wyboru splitu?

## 7. Jedna runda boostingu

Aktualna wersja zawiera jedną pełną rundę boostingu.

Zadanie koncepcyjne:

Ułóż w poprawnej kolejności kroki jednej rundy boostingu:

```text
A. aktualizacja wag próbek
B. trening weak learnera
C. obliczenie weighted error
D. utworzenie datasetu dla kolejnej rundy
E. obliczenie alpha
```

Pytania:

- Jaka jest poprawna kolejność?
- Dlaczego weighted error musi być policzony przed alpha?
- Dlaczego alpha jest potrzebne do aktualizacji wag?
- Dlaczego po aktualizacji wag trzeba je znormalizować?
- Co powinno wydarzyć się w kolejnej rundzie?

## 8. Multi-round boosting trainer

Aktualna wersja uruchamia kilka rund boostingu po kolei.

Schemat:

```text
round 1 -> updated weights 1
round 2 -> updated weights 2
round 3 -> updated weights 3
```

Zadanie:

1. Uruchom trainer dla kilku rund.
2. Sprawdź staged learner weights.
3. Sprawdź staged weighted train errors.
4. Sprawdź finalne wagi próbek.

Pytania:

- Dlaczego runda 2 nie powinna zaczynać od wag początkowych?
- Dlaczego runda 2 korzysta z wag po rundzie 1?
- Co oznacza staged weighted train error?
- Co oznacza staged learner weight?
- Dlaczego sama pętla treningowa nie wystarcza jeszcze do finalnej predykcji ensemble?

## 9. Predykcja boosted ensemble

Aktualna wersja oblicza finalną predykcję boosted ensemble.

Dane są trzy weak learnery:

```text
alpha_1 = 0.8, prediction_1 = class 1
alpha_2 = 0.3, prediction_2 = class 0
alpha_3 = 0.4, prediction_3 = class 1
```

Po zamianie klas:

```text
class 0 -> -1
class 1 -> +1
```

Pytania:

- Jaki jest finalny score?
- Jaka jest finalna klasa?
- Który weak learner ma największy wpływ?
- Dlaczego nie jest to zwykłe głosowanie większościowe?
- Co oznacza duży dodatni albo duży ujemny score?
- Co oznacza score bliski zera?

## 10. Staged accuracy history

Aktualna wersja oblicza historię jakości po każdej rundzie boostingu.

Dane są wyniki:

```text
round:          1     2     3     4
train accuracy: 0.70  0.82  0.90  0.96
test accuracy:  0.68  0.80  0.84  0.82
```

Pytania:

- W której rundzie test accuracy jest najlepsze?
- Czy ostatnia runda jest najlepsza?
- Co może oznaczać wzrost train accuracy i spadek test accuracy?
- Dlaczego staged history pomaga wykryć overfitting?
- Dlaczego warto patrzeć na historię, a nie tylko finalny wynik?

## 11. Raport porównawczy CLI

Uruchom CLI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab
```

Zadanie:

1. Porównaj weak learner baseline i boosted ensemble.
2. Sprawdź train accuracy.
3. Sprawdź test accuracy.
4. Sprawdź generalization gap.
5. Sprawdź winner by test accuracy.

Pytania:

- Który model ma większą train accuracy?
- Który model ma większą test accuracy?
- Czy boosted ensemble zawsze wygrywa?
- Co oznacza generalization gap?
- Co oznacza best staged test accuracy?
- Dlaczego winner jest wybierany według test accuracy, a nie train accuracy?

## 12. Wizualizacja Pygame

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Zadanie:

1. Ustaw dataset `xor`.
2. Zmieniaj liczbę rund.
3. Obserwuj prawy panel boosted ensemble.
4. Obserwuj rozmiary punktów treningowych.
5. Włącz i wyłącz confidence view klawiszem `C`.
6. Zwiększ szum klawiszem `Right`.

Pytania:

- Co pokazuje lewy panel?
- Co pokazuje prawy panel?
- Co oznacza rozmiar punktu treningowego?
- Dlaczego niektóre punkty stają się większe?
- Czy więcej rund zawsze poprawia test accuracy?
- Co oznaczają blade regiony w confidence view?

## 13. Selected stage i staged plot

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Zadanie:

1. Ustaw dataset `xor`.
2. Zwiększ całkowitą liczbę rund klawiszem `PageUp`.
3. Zmieniaj selected stage klawiszami `Up` i `Down`.
4. Obserwuj lewy panel.
5. Obserwuj prawy panel.
6. Obserwuj wykres staged accuracy.

Pytania:

- Co zmienia się w lewym panelu po zmianie selected stage?
- Co zmienia się w prawym panelu?
- Dlaczego ensemble po stage 1 może wyglądać podobnie do weak learnera?
- Czy test accuracy rośnie w każdej rundzie?
- W której rundzie test accuracy jest najlepsze?
- Czy finalna runda zawsze jest najlepsza?

## 14. Challenge mode

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Warunki challenge:

```text
boosted test accuracy >= 0.85
round_count <= 8
generalization gap <= 0.20
```

Zadanie:

1. Ustaw dataset `xor`.
2. Dobierz liczbę rund.
3. Zmieniaj `min_samples_leaf`.
4. Zmieniaj noise.
5. Obserwuj staged accuracy plot.
6. Spróbuj uzyskać status `success`.

Pytania:

- Czy zwiększenie liczby rund zawsze pomaga?
- Czy można spełnić challenge przy dużym szumie?
- Dlaczego challenge ogranicza liczbę rund?
- Dlaczego liczy się test accuracy, a nie tylko train accuracy?
- Co oznacza generalization gap?
- W której rundzie uzyskano najlepszą test accuracy?

## 15. Explanation panel

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Zadanie:

1. Ustaw dataset `xor`.
2. Zmieniaj liczbę rund.
3. Zmieniaj selected stage.
4. Zwiększaj i zmniejszaj noise.
5. Obserwuj explanation panel w dolnej części okna.

Pytania:

- Kiedy explanation panel pokazuje sukces?
- Jaką podpowiedź pokazuje, gdy test accuracy jest zbyt niskie?
- Jaką podpowiedź pokazuje, gdy liczba rund jest zbyt duża?
- Co panel mówi o selected stage?
- Jak confidence view wpływa na interpretację regionów decyzyjnych?

## 16. Preset scenarios

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Zadanie:

1. Wciśnij `1` i obejrzyj preset `Easy axis-aligned`.
2. Wciśnij `2` i obejrzyj preset `Noisy XOR`.
3. Wciśnij `3` i obejrzyj preset `Overfitting watch`.
4. Wciśnij `4` i obejrzyj preset `Low-round challenge`.
5. Przełączaj presety klawiszem `P`.

Pytania:

- Który preset jest najłatwiejszy?
- W którym presecie boosting najbardziej pomaga?
- W którym presecie najlepiej widać staged accuracy?
- Kiedy challenge najłatwiej zaliczyć?
- Co zmienia się po ręcznej zmianie noise albo round count?

## 17. Decision boundary export

Uruchom UI:

```bash
uv run --package boosting-mistake-lab boosting-mistake-lab-ui
```

Zadanie:

1. Wybierz preset `Noisy XOR` klawiszem `2`.
2. Zmień selected stage klawiszami `Up` / `Down`.
3. Spróbuj zaliczyć challenge.
4. Naciśnij `E`, aby wyeksportować wynik.
5. Otwórz plik:

```text
exports/boosting_mistake_lab_decision_boundary.json
```

Pytania:

- Jaki `selected_stage` został zapisany?
- Jaka jest `boosted_test_accuracy`?
- Ile weak learnerów zapisano w sekcji `rounds`?
- Jakie wartości alpha mają pierwsze trzy weak learnery?
- Czy confidence jest wysokie w całej siatce?
- Dlaczego eksport może być przydatny w raporcie z ćwiczenia?

## 18. Zadanie końcowe

Celem zadania jest samodzielne dobranie konfiguracji boosted ensemble.

Warunki:

```text
boosted test accuracy >= 0.85
round_count <= 8
generalization gap <= 0.20
```

Instrukcja:

1. Uruchom UI.
2. Wybierz jeden z presetów.
3. Zmieniaj liczbę rund.
4. Zmieniaj selected stage.
5. Zmieniaj noise.
6. Zmieniaj min_samples_leaf.
7. Obserwuj staged accuracy.
8. Spróbuj uzyskać status `success`.
9. Wyeksportuj wynik klawiszem `E`.

Do oddania:

- nazwa presetu początkowego,
- finalna liczba rund,
- selected stage,
- noise,
- min_samples_leaf,
- boosted test accuracy,
- generalization gap,
- best staged round,
- informacja, czy challenge został zaliczony,
- plik `boosting_mistake_lab_decision_boundary.json`,
- krótka interpretacja wyniku.

Pytania końcowe:

- Czy zwiększenie liczby rund zawsze poprawia wynik testowy?
- Czy najlepszy selected stage był ostatnim stage?
- Czy model miał duży generalization gap?
- Które próbki miały największe finalne wagi?
- Czy confidence view pokazywał regiony niepewności?
- Co było trudniejsze: poprawa test accuracy czy ograniczenie overfittingu?

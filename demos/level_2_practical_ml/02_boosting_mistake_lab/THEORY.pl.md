# Boosting Mistake Lab — teoria

Ten dokument opisuje najważniejsze idee stojące za demo **Boosting Mistake Lab**. Materiał jest napisany z perspektywy dydaktycznej: najpierw budowana jest intuicja, a dopiero później pojawiają się szczegóły implementacyjne.

## 1. Cel demo

Boosting Mistake Lab pokazuje, jak z wielu prostych modeli można zbudować silniejszy model zespołowy.

W demo wykorzystywany jest bardzo prosty weak learner: decision stump.

Decision stump to drzewo decyzyjne z jednym podziałem.

Dzięki temu można skupić się na samym mechanizmie boostingu:

```text
weak learner
-> weighted error
-> alpha
-> aktualizacja wag próbek
-> kolejna runda
-> boosted ensemble
```

Główna intuicja:

```text
Boosting zwiększa znaczenie próbek, które wcześniejsze modele klasyfikowały błędnie.
```

## 2. Dataset z wagami próbek

W klasycznym uczeniu nadzorowanym każda próbka często traktowana jest tak samo.

W boostingu każda próbka ma wagę.

Na początku wagi są zwykle jednakowe i znormalizowane:

```text
sum(sample_weights) = 1.0
```

Wagi próbek określają, jak ważna jest dana próbka podczas trenowania kolejnego weak learnera.

Próbka z większą wagą ma większy wpływ na wybór splitu oraz na weighted error.

W demo dostępne są dwa typy datasetów:

```text
axis_aligned
xor
```

Dataset `axis_aligned` jest prostszy dla decision stumpa, ponieważ klasy można często oddzielić jednym podziałem osiowym.

Dataset `xor` jest trudniejszy, ponieważ wymaga złożenia kilku prostych reguł.

## 3. Weak learner

Weak learner to model prosty, który nie musi być bardzo dokładny samodzielnie.

W tym demo weak learnerem jest decision stump.

Decision stump wybiera:

- indeks cechy,
- próg podziału,
- predykcję po lewej stronie progu,
- predykcję po prawej stronie progu.

Przykład:

```text
if x1 <= 0.42:
    predict class 0
else:
    predict class 1
```

Pojedynczy stump ma ograniczoną ekspresyjność.

To ograniczenie jest celowe, ponieważ boosting pokazuje, jak wiele prostych modeli może razem stworzyć silniejsze rozwiązanie.

## 4. Weighted error

Weighted error to błąd klasyfikacji liczony z uwzględnieniem wag próbek.

Zwykły error traktuje każdą próbkę tak samo.

Weighted error sumuje wagi próbek błędnie sklasyfikowanych:

```text
weighted_error = sum(weights_i for misclassified samples)
```

Jeżeli błędnie sklasyfikowana próbka ma dużą wagę, błąd ważony będzie większy.

To jest kluczowe w boostingu, ponieważ kolejne rundy powinny bardziej przejmować się próbkami trudnymi.

Weighted accuracy można zapisać jako:

```text
weighted_accuracy = 1.0 - weighted_error
```

przy założeniu, że wagi próbek sumują się do `1.0`.

## 5. Learner weight alpha

Każdy weak learner otrzymuje własną wagę `alpha`.

Alpha określa, jak silnie dany weak learner wpływa na finalną predykcję ensemble.

Typowa intuicja AdaBoost:

```text
niższy weighted error -> większe alpha
wyższy weighted error -> mniejsze alpha
```

W uproszczonej formie:

```text
alpha = 0.5 * log((1 - weighted_error) / weighted_error)
```

Gdy weighted error jest niski, alpha rośnie.

Gdy weighted error zbliża się do `0.5`, alpha maleje.

Oznacza to, że model niewiele lepszy od losowego zgadywania nie powinien mieć dużego wpływu na ensemble.

## 6. Aktualizacja wag próbek

Po obliczeniu alpha następuje aktualizacja wag próbek.

Intuicja:

```text
próbki poprawnie sklasyfikowane -> mniejsza waga
próbki błędnie sklasyfikowane   -> większa waga
```

Dzięki temu kolejny weak learner bardziej koncentruje się na trudnych próbkach.

Po aktualizacji wagi muszą zostać znormalizowane, aby ich suma ponownie wynosiła `1.0`.

Schemat:

```text
old_weights
-> multiply by factor depending on correctness
-> normalize
-> next_round_weights
```

To właśnie aktualizacja wag sprawia, że boosting jest procesem sekwencyjnym.

Kolejna runda nie jest niezależna od poprzedniej.

## 7. Weighted stump split search

Decision stump w tym demo wybiera split na podstawie weighted training error.

To ważne, ponieważ samo liczenie weighted error po fakcie nie wystarcza.

Weak learner powinien być trenowany tak, aby reagował na aktualne wagi próbek.

Split wybierany jest tak, aby minimalizować:

```text
weighted_error = suma wag błędnie sklasyfikowanych próbek
```

Dodatkowo predykcja w liściach stumpa jest wybierana jako ważona większość klas.

Przykład:

```text
class:   0    1
weight:  0.1  0.9
```

Zwykła większość mogłaby uznać remis.

Ważona większość wybierze klasę `1`, ponieważ próbka tej klasy ma większą wagę.

To jest zgodne z intuicją boostingu: ważniejsze próbki powinny mieć większy wpływ na decyzję modelu.

## 8. Jedna runda boostingu

Jedna pełna runda boostingu wykonuje następujące kroki:

```text
aktualne wagi próbek
-> trening weighted stump
-> obliczenie weighted error
-> obliczenie alpha
-> aktualizacja wag próbek
-> utworzenie datasetu dla kolejnej rundy
```

Pojedyncza runda jeszcze nie tworzy pełnego ensemble.

Jest jednak podstawowym elementem, który później można powtarzać.

Wynikiem jednej rundy są między innymi:

- fitted weak learner,
- weighted train error,
- alpha,
- nowe wagi próbek,
- snapshot z metrykami.

## 9. Multi-round boosting trainer

Trainer powtarza wiele rund boostingu.

Schemat:

```text
round 1 -> nowe wagi
round 2 -> nowe wagi
round 3 -> nowe wagi
...
```

Każda runda korzysta z wag wyprodukowanych przez poprzednią rundę.

To oznacza, że boosting jest procesem sekwencyjnym.

Nie można po prostu trenować wszystkich weak learnerów niezależnie na tych samych wagach początkowych.

Trainer zapisuje między innymi:

- listę weak learnerów,
- alpha dla każdej rundy,
- weighted error dla każdej rundy,
- staged train/test accuracy,
- finalne wagi próbek.

## 10. Predykcja boosted ensemble

Finalna predykcja ensemble powstaje przez ważone głosowanie weak learnerów.

Predykcje klas są zamieniane na wartości znakowane:

```text
class 0 -> -1
class 1 -> +1
```

Następnie liczony jest score:

```text
score = sum(alpha_t * signed_prediction_t)
```

Reguła predykcji:

```text
score > 0  -> class 1
score <= 0 -> class 0
```

Nie jest to zwykłe głosowanie większościowe.

Weak learner z większym alpha ma większy wpływ na wynik końcowy.

Score bliski zera oznacza słabą zgodę ensemble.

Duża wartość bezwzględna score oznacza silniejszą zgodę ensemble.

## 11. Confidence i margin

Confidence w demo jest wyprowadzane ze znormalizowanego marginesu.

Margin opisuje, jak zdecydowane jest głosowanie ensemble.

Intuicja:

```text
mały margin -> model jest mniej pewny
duży margin -> model jest bardziej pewny
```

Confidence nie jest tym samym co accuracy.

Model może być pewny i jednocześnie błędny.

Dlatego confidence należy interpretować jako miarę zgodności ensemble, a nie gwarancję poprawności.

## 12. Staged accuracy history

Staged accuracy pokazuje jakość boosted ensemble po każdej rundzie.

Przykład:

```text
stage 1 -> ensemble z rundy 1
stage 2 -> ensemble z rund 1..2
stage 3 -> ensemble z rund 1..3
```

Staged history pozwala sprawdzić:

- jak zmienia się train accuracy,
- jak zmienia się test accuracy,
- gdzie test accuracy jest najlepsze,
- czy kolejne rundy nadal pomagają,
- czy pojawia się overfitting.

To ważne, ponieważ ostatnia runda nie zawsze musi dawać najlepszy wynik testowy.

## 13. Generalization gap

Generalization gap to różnica między train accuracy i test accuracy:

```text
generalization_gap = train_accuracy - test_accuracy
```

Duży gap może oznaczać overfitting.

W kontekście boostingu warto obserwować, czy kolejne rundy poprawiają test accuracy, czy tylko train accuracy.

Jeżeli train accuracy rośnie, a test accuracy spada, model może coraz bardziej dopasowywać się do zbioru treningowego.

## 14. Raport porównawczy CLI

CLI generuje raport porównujący:

- weak learner baseline,
- boosted ensemble.

Raport pokazuje między innymi:

- train accuracy,
- test accuracy,
- generalization gap,
- weighted train error,
- alpha,
- best staged test accuracy,
- winner by test accuracy.

W praktyce ważniejsze jest porównanie na danych testowych niż na danych treningowych.

Dlatego raport wskazuje zwycięzcę według test accuracy.

## 15. Wizualizacja Pygame

Wizualizacja Pygame pokazuje dwa główne panele.

Lewy panel:

```text
weak learner z wybranego stage
```

Prawy panel:

```text
boosted ensemble zbudowany z rund 1..selected_stage
```

Oznaczenia:

- kółka oznaczają próbki treningowe,
- rozmiar kółka oznacza wagę próbki,
- kwadraty oznaczają próbki testowe,
- znak X oznacza błędną klasyfikację próbki testowej,
- tło oznacza regiony decyzyjne modelu.

Dzięki temu można zobaczyć, jak sample weights i kolejne rundy wpływają na decyzje modelu.

## 16. Selected stage

Selected stage pozwala oglądać boosting krok po kroku.

Przykład:

```text
selected stage = 1 -> tylko pierwszy weak learner
selected stage = 3 -> ensemble z rund 1, 2 i 3
selected stage = 5 -> ensemble z rund 1, 2, 3, 4 i 5
```

Lewy panel pokazuje weak learner z wybranej rundy.

Prawy panel pokazuje ensemble zbudowany z rund od pierwszej do wybranej.

To ułatwia zrozumienie, że boosting nie jest pojedynczym modelem, lecz sekwencją modeli.

## 17. Wykres staged accuracy

Panel boczny zawiera mini-wykres staged accuracy.

Wykres pozwala obserwować:

- train accuracy,
- test accuracy,
- selected stage,
- najlepszą rundę testową.

Wykres jest szczególnie przydatny przy dyskusji o overfittingu.

Jeżeli train accuracy rośnie, ale test accuracy przestaje rosnąć albo spada, dalsze rundy mogą nie poprawiać generalizacji.

## 18. Challenge mode

Challenge mode definiuje konkretne warunki do spełnienia:

```text
boosted test accuracy >= 0.85
round_count <= 8
generalization gap <= 0.20
```

Celem nie jest maksymalizacja train accuracy.

Celem jest znalezienie konfiguracji, która:

- dobrze działa na danych testowych,
- nie używa zbyt wielu rund,
- nie ma zbyt dużego generalization gap.

Challenge mode wymusza myślenie o kompromisie:

```text
skuteczność vs złożoność vs generalizacja
```

## 19. Explanation panel

Explanation panel interpretuje aktualny stan modelu.

Panel może wskazać:

- czy challenge został zaliczony,
- czy test accuracy jest zbyt niskie,
- czy użyto zbyt wielu rund,
- czy generalization gap jest zbyt duży,
- w której rundzie uzyskano najlepszy wynik testowy,
- co oznacza confidence view.

Dzięki temu student nie widzi tylko liczb, lecz otrzymuje krótką interpretację i podpowiedź, co można zmienić.

## 20. Preset scenarios

Presety to gotowe konfiguracje dydaktyczne.

Dostępne scenariusze:

```text
1: Easy axis-aligned
2: Noisy XOR
3: Overfitting watch
4: Low-round challenge
```

Presety pozwalają szybko przełączać się między różnymi sytuacjami:

- prosty przypadek liniowy,
- trudniejszy przypadek XOR,
- przypadek obserwacji overfittingu,
- przypadek ograniczonej liczby rund.

Presety nie zastępują eksperymentowania.

Służą jako punkty startowe do dyskusji.

## 21. Decision boundary export

Demo pozwala wyeksportować aktualny stan modelu do pliku JSON.

Eksport obejmuje:

- konfigurację datasetu,
- wybrany stage boostingu,
- metryki train/test,
- historię staged accuracy,
- parametry weak learnerów,
- wartości alpha,
- siatkę decision boundary,
- confidence,
- raw scores,
- próbki treningowe i testowe,
- wagi próbek.

Eksport jest przydatny w raportach oraz przy porównywaniu wyników studentów.

Pozwala omawiać model poza UI, np. w notebooku albo w sprawozdaniu.

## 22. Podsumowanie całego demo

Boosting Mistake Lab pokazuje pełny przepływ uproszczonego algorytmu boostingowego.

Najważniejsze elementy:

```text
dataset z wagami
-> weighted weak learner
-> weighted error
-> alpha
-> aktualizacja wag próbek
-> wiele rund
-> boosted ensemble
-> staged accuracy
-> challenge mode
-> export wyników
```

Demo jest celowo zbudowane wokół prostego weak learnera.

Dzięki temu można skupić się nie na złożoności modelu bazowego, ale na mechanizmie boostingu.

Najważniejsza intuicja:

```text
Boosting zmienia znaczenie próbek w kolejnych rundach.
```

Próbki błędnie klasyfikowane dostają większą wagę.

Kolejne weak learnery są trenowane z uwzględnieniem tych wag.

Finalny ensemble łączy weak learnery za pomocą wag alpha.

Nie każdy weak learner jest równie ważny.

Weak learner z niższym weighted error otrzymuje większe alpha i ma większy wpływ na finalną predykcję.

Staged accuracy pozwala analizować proces uczenia po każdej rundzie.

Challenge mode wymusza myślenie o kompromisie:

```text
accuracy vs liczba rund vs generalization gap
```

Eksport JSON pozwala zapisać wynik eksperymentu i porównać konfiguracje poza UI.

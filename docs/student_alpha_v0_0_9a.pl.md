# v0.0.9a - notatki dla studentów

`v0.0.9a` to pierwsza alfa Interactive ML Labs skupiona na prowadzonej ścieżce
nauki dla studentów.

## Czym Jest Ta Wersja

Interactive ML Labs to lokalna aplikacja Pygame do budowania intuicji wokół
machine learningu przez wizualne eksperymenty. W lekcjach nie trzeba pisać kodu.
Chodzi o to, żeby zobaczyć, co robi model, zmienić ważne parametry i umieć
wyjaśnić wynik prostym językiem.

## Rekomendowany Start

Uruchom unified app z katalogu głównego repozytorium:

```bash
uv run --package interactive-ml-labs-app interactive-ml-labs
```

Potem przejdź taką trasą:

1. Wybierz język.
2. Otwórz mapę kursu.
3. Zacznij od pierwszej prowadzonej ścieżki nauki.
4. Przeczytaj intro lekcji.
5. Otwórz teorię, kiedy potrzebujesz kontekstu.
6. Wykonaj zadania w demie.
7. Użyj podsumowania i pytań sprawdzających, żeby zobaczyć, co umiesz
   wyjaśnić.

Swobodne przeglądanie dem nadal jest dostępne przez wybór poziomu.

## Co Jest W Środku

- Pięć prowadzonych ścieżek nauki.
- Dema z Level 1, Level 2 i Level 3 w unified app.
- Intro lekcji, ekrany teorii, pauza/pomoc i postęp zadań.
- Podsumowania lekcji z lekkimi pytaniami sprawdzającymi zrozumienie.
- Odznaki za ukończone lekcje.
- Zapamiętywanie postępu i ustawień.
- Interfejs po angielsku i po polsku.
- Opcje komfortu: większy tekst, wysoki kontrast i paleta przyjazna
  daltonizmowi.

## Co Zgłaszać

Najbardziej pomagają zgłoszenia, które mówią:

- która lekcja była niejasna,
- które zadanie nie miało sensu,
- który tekst brzmiał dziwnie albo był niezrozumiały,
- który ekran był trudny do odczytania,
- które sterowanie nie działało,
- którego tematu z machine learningu brakuje.

## Znane Ograniczenia

- Aplikację nadal uruchamia się przez `uv run`; paczki instalacyjne są planowane
  po tej alfie.
- Pytania sprawdzające są promptami do refleksji, a nie ocenianymi quizami.
- Opcje komfortu dotyczą najpierw ekranów shellowych. Pojedyncze dema mogą nadal
  potrzebować osobnego polish passu.
- Część zaawansowanych tematów jest celowo uproszczona, żeby lekcja była
  wizualna i możliwa do omówienia na zajęciach.

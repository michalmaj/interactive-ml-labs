"""Demo manifest registry for the unified shell."""

from __future__ import annotations

from collections.abc import Callable

from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.decision_tree_scene import create_decision_tree_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.knn_scene import create_knn_vote_map_scene
from interactive_ml_labs.logistic_scene import create_logistic_regression_scene
from interactive_ml_labs.manifest import (
    ControlBinding,
    DemoManifest,
    DemoTheory,
    GlossaryTerm,
    LevelManifest,
    LocalizedText,
    TheorySection,
)
from interactive_ml_labs.placeholder_scene import PlaceholderDemoScene
from interactive_ml_labs.random_forest_scene import create_random_forest_scene
from interactive_ml_labs.scene import Scene

LEVEL_MANIFESTS: tuple[LevelManifest, ...] = (
    LevelManifest(
        number=1,
        title=LocalizedText(en="Level 1 - Fundamentals", pl="Poziom 1 - Fundamenty"),
        summary=LocalizedText(
            en="Core machine learning intuition and foundational algorithms.",
            pl="Pierwsze intuicje ML i algorytmy, do których często się wraca.",
        ),
    ),
    LevelManifest(
        number=2,
        title=LocalizedText(en="Level 2 - Practical ML", pl="Poziom 2 - Praktyczne ML"),
        summary=LocalizedText(
            en="Model evaluation, robustness, ensembles, and practical trade-offs.",
            pl="Ewaluacja modeli, odporność, ensemble i kompromisy z praktyki.",
        ),
    ),
    LevelManifest(
        number=3,
        title=LocalizedText(en="Level 3 - Advanced / Showcase", pl="Poziom 3 - Zaawansowane"),
        summary=LocalizedText(
            en="Advanced, specialized, and visually rich machine learning demos.",
            pl="Bardziej zaawansowane tematy i efektowne wizualnie eksperymenty ML.",
        ),
    ),
)

LESSON_CHALLENGES: dict[str, tuple[LocalizedText, ...]] = {
    "gradient_descent_playground": (
        LocalizedText(
            en=(
                "Set learning rate very low. Watch how many steps are needed "
                "before the loss visibly improves."
            ),
            pl=(
                "Ustaw bardzo niski learning rate. Zobacz, ile kroków potrzeba, "
                "zanim loss zacznie wyraźnie spadać."
            ),
        ),
        LocalizedText(
            en=(
                "Set learning rate high enough to overshoot. Notice whether "
                "the path bounces around the minimum."
            ),
            pl=(
                "Ustaw learning rate tak wysoko, żeby kroki przeskakiwały minimum. "
                "Zwróć uwagę, czy ścieżka zaczyna odbijać się na boki."
            ),
        ),
        LocalizedText(
            en=(
                "Change the starting point and compare whether the same "
                "learning rate still feels stable."
            ),
            pl=(
                "Zmień punkt startowy i porównaj, czy ten sam learning rate "
                "nadal zachowuje się stabilnie."
            ),
        ),
    ),
    "knn_vote_map": (
        LocalizedText(
            en=(
                "Try k=1, then a larger odd k. Look for regions where "
                "the vote map becomes smoother but less local."
            ),
            pl=(
                "Porównaj k=1 z większym nieparzystym k. Szukaj miejsc, "
                "gdzie vote map robi się gładsza, ale mniej lokalna."
            ),
        ),
        LocalizedText(
            en=(
                "Move the query point close to a class boundary and watch "
                "which neighbors decide the label."
            ),
            pl=(
                "Przesuń query point blisko granicy klas i zobacz, "
                "którzy sąsiedzi decydują o etykiecie."
            ),
        ),
        LocalizedText(
            en=(
                "Add an outlier near a dense region and check how much it "
                "matters for small and large k."
            ),
            pl=(
                "Dodaj outlier blisko gęstego obszaru i sprawdź, jak mocno "
                "wpływa na wynik przy małym i dużym k."
            ),
        ),
    ),
    "logistic_regression_boundary_lab": (
        LocalizedText(
            en="Move the threshold away from 0.5 and compare precision with recall.",
            pl="Odsuń threshold od 0.5 i porównaj precision z recall.",
        ),
        LocalizedText(
            en=(
                "Add ambiguous points near the boundary and watch how "
                "the probability field changes."
            ),
            pl=(
                "Dodaj niejednoznaczne punkty blisko granicy i obserwuj, "
                "jak zmienia się pole prawdopodobieństwa."
            ),
        ),
        LocalizedText(
            en=(
                "Find a setting where accuracy looks fine, but one class is clearly treated worse."
            ),
            pl=(
                "Znajdź ustawienie, w którym accuracy wygląda dobrze, "
                "ale jedna klasa jest wyraźnie traktowana gorzej."
            ),
        ),
    ),
    "decision_tree_splitter": (
        LocalizedText(
            en=(
                "Make one split that feels visually obvious, then compare "
                "how the impurity score reacts."
            ),
            pl=(
                "Ustaw podział, który wizualnie wydaje się oczywisty, "
                "i sprawdź, jak reaguje impurity score."
            ),
        ),
        LocalizedText(
            en=(
                "Try several manual splits that isolate just a few points. "
                "Notice when the tree starts memorizing noise."
            ),
            pl=(
                "Przetestuj kilka manual splitów izolujących tylko parę punktów. "
                "Zobacz, kiedy drzewo zaczyna zapamiętywać szum."
            ),
        ),
        LocalizedText(
            en=(
                "Switch between views and connect the split line with "
                "the resulting prediction regions."
            ),
            pl=(
                "Przełączaj widoki i połącz linię podziału z regionami "
                "predykcji, które powstają po splicie."
            ),
        ),
    ),
    "random_forest_bagging_lab": (
        LocalizedText(
            en=(
                "Train a few individual trees and compare their boundaries "
                "with the forest boundary."
            ),
            pl=(
                "Wytrenuj kilka pojedynczych drzew i porównaj ich granice "
                "z granicą całego Random Forest."
            ),
        ),
        LocalizedText(
            en="Turn on confidence view and look for places where trees disagree.",
            pl="Włącz confidence view i znajdź obszary, w których drzewa się ze sobą nie zgadzają.",
        ),
        LocalizedText(
            en=(
                "Increase ensemble size and check whether the forest stabilizes "
                "or only becomes more complex."
            ),
            pl=(
                "Zwiększ rozmiar ensemble i sprawdź, czy Random Forest się "
                "stabilizuje, czy tylko robi bardziej złożony."
            ),
        ),
    ),
    "boosting_mistake_lab": (
        LocalizedText(
            en=(
                "Add one weak learner at a time and watch which previously "
                "wrong points get more attention."
            ),
            pl=(
                "Dodawaj weak learner po jednym i obserwuj, które wcześniej "
                "błędne punkty dostają większą uwagę."
            ),
        ),
        LocalizedText(
            en=(
                "Compare train and test accuracy after many rounds. Look for "
                "the moment when the gap becomes suspicious."
            ),
            pl=(
                "Porównuj train i test accuracy po wielu rundach. Szukaj momentu, "
                "w którym różnica zaczyna wyglądać podejrzanie."
            ),
        ),
        LocalizedText(
            en=(
                "Reset and try a simpler sequence. Ask whether a smaller "
                "ensemble explains the pattern well enough."
            ),
            pl=(
                "Zresetuj demo i spróbuj prostszej sekwencji. Sprawdź, "
                "czy mniejszy ensemble wystarczająco dobrze tłumaczy wzorzec."
            ),
        ),
    ),
}

LESSON_GLOSSARY: dict[str, tuple[GlossaryTerm, ...]] = {
    "gradient_descent_playground": (
        GlossaryTerm(
            term="loss",
            definition=LocalizedText(
                en="A number that says how bad the model currently is; lower is better.",
                pl="Liczba mówiąca, jak źle model działa w danej chwili; im niżej, tym lepiej.",
            ),
        ),
        GlossaryTerm(
            term="learning rate",
            definition=LocalizedText(
                en="The step size used when the model updates its parameters.",
                pl="Rozmiar kroku, którym model aktualizuje swoje parametry.",
            ),
        ),
        GlossaryTerm(
            term="gradient",
            definition=LocalizedText(
                en="A direction that points toward the steepest increase of the loss.",
                pl=(
                    "Kierunek najszybszego wzrostu loss; gradient descent idzie w stronę przeciwną."
                ),
            ),
        ),
    ),
    "knn_vote_map": (
        GlossaryTerm(
            term="k",
            definition=LocalizedText(
                en="The number of nearest neighbors allowed to vote.",
                pl="Liczba najbliższych sąsiadów, którzy biorą udział w głosowaniu.",
            ),
        ),
        GlossaryTerm(
            term="query point",
            definition=LocalizedText(
                en="The point whose label the algorithm is trying to predict.",
                pl="Punkt, dla którego algorytm próbuje przewidzieć etykietę.",
            ),
        ),
        GlossaryTerm(
            term="vote map",
            definition=LocalizedText(
                en="A visual map showing which class would win in different parts of the space.",
                pl="Mapa pokazująca, która klasa wygrałaby w różnych miejscach przestrzeni.",
            ),
        ),
    ),
    "logistic_regression_boundary_lab": (
        GlossaryTerm(
            term="threshold",
            definition=LocalizedText(
                en="The probability cutoff used to turn a score into a class label.",
                pl="Próg prawdopodobieństwa, który zamienia wynik modelu na etykietę klasy.",
            ),
        ),
        GlossaryTerm(
            term="decision boundary",
            definition=LocalizedText(
                en="The line or curve where the predicted class changes.",
                pl="Linia albo krzywa, na której zmienia się przewidywana klasa.",
            ),
        ),
        GlossaryTerm(
            term="precision / recall",
            definition=LocalizedText(
                en="Two complementary ways to inspect classification mistakes.",
                pl="Dwa uzupełniające się sposoby patrzenia na błędy klasyfikacji.",
            ),
        ),
    ),
    "decision_tree_splitter": (
        GlossaryTerm(
            term="split",
            definition=LocalizedText(
                en="A rule that sends data points to one side of the tree or the other.",
                pl="Reguła, która wysyła punkty danych na jedną albo drugą stronę drzewa.",
            ),
        ),
        GlossaryTerm(
            term="impurity",
            definition=LocalizedText(
                en="A score describing how mixed the classes are inside a region.",
                pl="Miara pokazująca, jak bardzo klasy są wymieszane w danym regionie.",
            ),
        ),
        GlossaryTerm(
            term="overfitting",
            definition=LocalizedText(
                en=(
                    "When a model learns small accidents in the training data "
                    "instead of the stable pattern."
                ),
                pl=(
                    "Sytuacja, w której model uczy się przypadków z treningu "
                    "zamiast stabilnego wzorca."
                ),
            ),
        ),
    ),
    "random_forest_bagging_lab": (
        GlossaryTerm(
            term="bootstrap",
            definition=LocalizedText(
                en=(
                    "Sampling training examples with replacement so each tree "
                    "sees a slightly different dataset."
                ),
                pl=(
                    "Losowanie przykładów ze zwracaniem, dzięki któremu każde "
                    "drzewo widzi trochę inny zbiór."
                ),
            ),
        ),
        GlossaryTerm(
            term="ensemble",
            definition=LocalizedText(
                en="A group of models whose predictions are combined.",
                pl="Grupa modeli, których predykcje są łączone.",
            ),
        ),
        GlossaryTerm(
            term="confidence view",
            definition=LocalizedText(
                en="A view that highlights where the forest agrees strongly or weakly.",
                pl="Widok pokazujący, gdzie Random Forest zgadza się mocno, a gdzie słabo.",
            ),
        ),
    ),
    "boosting_mistake_lab": (
        GlossaryTerm(
            term="weak learner",
            definition=LocalizedText(
                en="A simple model that is only slightly better than guessing on its own.",
                pl="Prosty model, który sam jest tylko trochę lepszy od zgadywania.",
            ),
        ),
        GlossaryTerm(
            term="boosting",
            definition=LocalizedText(
                en=(
                    "Training weak learners in sequence so later learners focus "
                    "on earlier mistakes."
                ),
                pl=(
                    "Trenowanie weak learnerów po kolei, tak aby kolejne "
                    "skupiały się na wcześniejszych błędach."
                ),
            ),
        ),
        GlossaryTerm(
            term="train/test accuracy",
            definition=LocalizedText(
                en=(
                    "Accuracy measured on data used for training and on held-out "
                    "data used for checking generalization."
                ),
                pl=(
                    "Accuracy mierzona na danych treningowych oraz na osobnych "
                    "danych do sprawdzania generalizacji."
                ),
            ),
        ),
    ),
}

LEVEL_BY_NUMBER: dict[int, LevelManifest] = {
    manifest.number: manifest for manifest in LEVEL_MANIFESTS
}
LEVEL_NAMES: dict[int, LocalizedText] = {
    number: manifest.title for number, manifest in LEVEL_BY_NUMBER.items()
}


def _placeholder_demo(
    *,
    demo_id: str,
    level: int,
    title_en: str,
    title_pl: str,
    summary_en: str,
    summary_pl: str,
    tags: tuple[str, ...],
    objectives: tuple[LocalizedText, ...] | None = None,
    controls: tuple[ControlBinding, ...] | None = None,
    create_scene: Callable[[object], Scene] | None = None,
    difficulty: LocalizedText | None = None,
    theory: DemoTheory | None = None,
) -> DemoManifest:
    """Build a placeholder manifest for a demo."""
    return DemoManifest(
        id=demo_id,
        level=level,
        title=LocalizedText(en=title_en, pl=title_pl),
        summary=LocalizedText(en=summary_en, pl=summary_pl),
        objectives=objectives
        or (
            LocalizedText(
                en="Explore the main intuition behind the algorithm.",
                pl="Zrozum, co jest najważniejsze w działaniu algorytmu.",
            ),
            LocalizedText(
                en="Observe how parameter changes affect model behavior.",
                pl="Sprawdź, jak parametry zmieniają zachowanie modelu.",
            ),
        ),
        controls=controls
        or (
            ControlBinding(
                key="Enter",
                action=LocalizedText(en="start placeholder scene", pl="uruchom ekran demo"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(en="open pause menu or go back", pl="otwórz pauzę albo wróć"),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="toggle help overlay", pl="pokaż lub ukryj pomoc"),
            ),
        ),
        create_scene=create_scene
        or (
            lambda context, demo_id=demo_id: PlaceholderDemoScene(
                context,
                DEMO_BY_ID[demo_id],
            )
        ),
        difficulty=difficulty or LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=tags,
        theory=theory,
    )


DEMO_MANIFESTS: tuple[DemoManifest, ...] = (
    _placeholder_demo(
        demo_id="gradient_descent_playground",
        level=1,
        title_en="Gradient Descent Playground",
        title_pl="Gradient Descent Playground",
        summary_en="Optimization, loss, and learning rate intuition.",
        summary_pl="Optymalizacja, loss i intuicja stojąca za learning rate.",
        objectives=(
            LocalizedText(
                en="Watch gradient descent reduce loss step by step.",
                pl="Zobacz, jak gradient descent krok po kroku zmniejsza loss.",
            ),
            LocalizedText(
                en="Change learning rate and noise to see when training becomes unstable.",
                pl="Zmieniaj learning rate i noise, żeby zobaczyć, kiedy trening traci stabilność.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="start or pause automatic steps", pl="uruchom albo zatrzymaj kroki"
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="perform one gradient descent step", pl="wykonaj jeden krok gradient descent"
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change learning rate", pl="zmień learning rate"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current run", pl="zresetuj aktualny przebieg"),
            ),
        ),
        create_scene=create_gradient_descent_scene,
        tags=("regression", "optimization"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Gradient descent tries to find model parameters that make "
                                "loss smaller."
                            ),
                            pl=(
                                "Gradient descent szuka takich parametrów modelu, "
                                "żeby loss był coraz mniejszy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Here the model is a straight line, so the moving parts are "
                                "weight and bias."
                            ),
                            pl=(
                                "Tutaj model jest prostą, więc najważniejsze parametry "
                                "to weight i bias."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "The gradient points toward the steepest loss increase, "
                                "so training steps in the opposite direction."
                            ),
                            pl=(
                                "Gradient wskazuje kierunek najszybszego wzrostu loss, "
                                "więc trening idzie w stronę przeciwną."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Learning rate decides how large each step is. Too small is slow; "
                                "too large can overshoot."
                            ),
                            pl=(
                                "Learning rate decyduje o wielkości kroku. Za mały działa wolno, "
                                "za duży może przeskakiwać dobre rozwiązanie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Watch the regression line move and compare it with the loss "
                                "history chart."
                            ),
                            pl=(
                                "Patrz, jak przesuwa się linia regresji, i porównuj to "
                                "z wykresem historii loss."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "More noise makes the target harder because the data no longer "
                                "sits close to one clean line."
                            ),
                            pl=(
                                "Większy szum utrudnia zadanie, bo punkty nie leżą już blisko "
                                "jednej czystej prostej."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "A lower loss is not magic; it means predictions are closer "
                                "to the training targets."
                            ),
                            pl=(
                                "Niższy loss to nie magia: oznacza, że predykcje są bliżej "
                                "wartości z danych treningowych."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "If loss jumps around, learning rate is often too aggressive "
                                "for the current data."
                            ),
                            pl=(
                                "Jeśli loss skacze, learning rate często jest zbyt agresywny "
                                "dla aktualnych danych."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["gradient_descent_playground"],
            glossary=LESSON_GLOSSARY["gradient_descent_playground"],
        ),
    ),
    _placeholder_demo(
        demo_id="knn_vote_map",
        level=1,
        title_en="k-NN Vote Map",
        title_pl="k-NN Vote Map",
        summary_en="Distance-based classification and neighborhood voting.",
        summary_pl="Klasyfikacja przez odległość i głosowanie najbliższych sąsiadów.",
        objectives=(
            LocalizedText(
                en="See how k-NN classifies query points by nearby examples.",
                pl="Zobacz, jak k-NN klasyfikuje punkty na podstawie najbliższych przykładów.",
            ),
            LocalizedText(
                en="Change k and noise to compare smoother and more local decision regions.",
                pl="Zmieniaj k i noise, żeby porównać różne decision regions.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Mouse click",
                action=LocalizedText(
                    en="classify a clicked query point", pl="sklasyfikuj kliknięty punkt"
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="classify a random query point", pl="sklasyfikuj losowy punkt"
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change k", pl="zmień k"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current map", pl="zresetuj aktualną mapę"),
            ),
        ),
        create_scene=create_knn_vote_map_scene,
        tags=("classification", "distance"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "k-NN classifies a query point by looking at the labels "
                                "of nearby training examples."
                            ),
                            pl=(
                                "k-NN klasyfikuje punkt query, patrząc na etykiety "
                                "najbliższych przykładów treningowych."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The colored background is a vote map: it shows what class "
                                "the model would choose in each region."
                            ),
                            pl=(
                                "Kolorowe tło to vote map: pokazuje, jaką klasę model "
                                "wybrałby w danym regionie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "The algorithm assumes that nearby points usually belong "
                                "to similar classes."
                            ),
                            pl=(
                                "Algorytm zakłada, że punkty blisko siebie zwykle należą "
                                "do podobnych klas."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The value of k controls how many neighbors get a vote "
                                "before the final prediction is chosen."
                            ),
                            pl=(
                                "Wartość k decyduje, ilu sąsiadów głosuje przed wyborem "
                                "finalnej predykcji."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Small k reacts strongly to local details, so the boundary "
                                "can become jagged."
                            ),
                            pl=(
                                "Małe k mocno reaguje na lokalne szczegóły, więc boundary "
                                "może robić się poszarpane."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Larger k smooths the map, but it can ignore small local "
                                "structures in the data."
                            ),
                            pl=(
                                "Większe k wygładza mapę, ale może ignorować małe lokalne "
                                "struktury w danych."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "k-NN does not train parameters in the usual sense; "
                                "it stores data and uses distance at prediction time."
                            ),
                            pl=(
                                "k-NN nie trenuje parametrów w klasycznym sensie; "
                                "przechowuje dane i używa odległości podczas predykcji."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A high accuracy for one seed does not guarantee stability "
                                "when noise or data layout changes."
                            ),
                            pl=(
                                "Wysoka accuracy dla jednego seedu nie gwarantuje stabilności, "
                                "gdy zmieni się szum albo układ danych."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["knn_vote_map"],
            glossary=LESSON_GLOSSARY["knn_vote_map"],
        ),
    ),
    _placeholder_demo(
        demo_id="logistic_regression_boundary_lab",
        level=1,
        title_en="Logistic Regression Boundary Lab",
        title_pl="Logistic Regression Boundary Lab",
        summary_en="Probabilities, thresholds, and decision boundaries.",
        summary_pl="Prawdopodobieństwa, progi i decision boundary w praktyce.",
        objectives=(
            LocalizedText(
                en="Watch logistic regression turn a linear score into class probabilities.",
                pl=(
                    "Zobacz, jak logistic regression zamienia liniowy score "
                    "na prawdopodobieństwa klas."
                ),
            ),
            LocalizedText(
                en=(
                    "Change learning rate, threshold, and noise to compare "
                    "stable and unstable training."
                ),
                pl=(
                    "Zmieniaj learning rate, threshold i noise, żeby porównać "
                    "stabilny i niestabilny trening."
                ),
            ),
            LocalizedText(
                en="Use precision and recall to reason about threshold trade-offs.",
                pl=(
                    "Używaj precision i recall, żeby rozmawiać o kompromisach "
                    "przy wyborze threshold."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="start or pause automatic training",
                    pl="uruchom albo zatrzymaj trening",
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="perform one gradient descent step",
                    pl="wykonaj jeden krok gradient descent",
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change learning rate", pl="zmień learning rate"),
            ),
            ControlBinding(
                key="Q / E",
                action=LocalizedText(
                    en="change decision threshold",
                    pl="zmień threshold decyzyjny",
                ),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current run", pl="zresetuj aktualny przebieg"),
            ),
        ),
        create_scene=create_logistic_regression_scene,
        tags=("classification", "probability"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Logistic regression turns a linear score into a probability "
                                "for class_1."
                            ),
                            pl=(
                                "Logistic regression zamienia liniowy score na probability "
                                "dla klasy class_1."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The decision boundary appears where probability crosses "
                                "the selected threshold."
                            ),
                            pl=(
                                "Decision boundary pojawia się tam, gdzie probability "
                                "przekracza wybrany threshold."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "The model learns weights and bias, then uses sigmoid "
                                "to squash the score into a 0..1 value."
                            ),
                            pl=(
                                "Model uczy się weights i bias, a potem używa sigmoid, "
                                "żeby zamienić score na wartość 0..1."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Gradient descent changes those parameters to reduce "
                                "classification loss."
                            ),
                            pl=(
                                "Gradient descent zmienia te parametry, żeby zmniejszać "
                                "classification loss."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Changing threshold can improve precision while hurting recall, "
                                "or the other way around."
                            ),
                            pl=(
                                "Zmiana threshold może poprawić precision kosztem recall "
                                "albo odwrotnie."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The probability background changes during training before "
                                "the final class labels are chosen."
                            ),
                            pl=(
                                "Tło probability zmienia się w trakcie treningu, zanim "
                                "model wybierze finalne klasy."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Probability is not the same thing as correctness; "
                                "a confident model can still be wrong."
                            ),
                            pl=(
                                "Probability nie jest tym samym co poprawność; "
                                "pewny model nadal może się mylić."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Accuracy alone can hide FP/FN trade-offs, so precision "
                                "and recall matter."
                            ),
                            pl=(
                                "Sama accuracy może ukrywać kompromis FP/FN, dlatego "
                                "precision i recall są ważne."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["logistic_regression_boundary_lab"],
            glossary=LESSON_GLOSSARY["logistic_regression_boundary_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="decision_tree_splitter",
        level=1,
        title_en="Decision Tree Splitter",
        title_pl="Decision Tree Splitter",
        summary_en="Splits, impurity, and interpretable rules.",
        summary_pl="Splity, impurity i reguły, które da się wyjaśnić człowiekowi.",
        objectives=(
            LocalizedText(
                en="Compare automatic tree splits with a manual split you can move yourself.",
                pl=(
                    "Porównuj automatyczne splity drzewa z manualnym splitem, "
                    "który możesz przesuwać samodzielnie."
                ),
            ),
            LocalizedText(
                en="Change max depth, dataset noise, and split criterion to see how trees behave.",
                pl=(
                    "Zmieniaj max depth, noise w danych i split criterion, "
                    "żeby zobaczyć, jak zachowują się drzewa."
                ),
            ),
            LocalizedText(
                en=(
                    "Switch between axis-aligned and XOR data to discuss "
                    "what tree splits can express."
                ),
                pl=(
                    "Przełączaj dane axis-aligned i XOR, żeby omówić, "
                    "co mogą wyrazić splity w drzewie."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="M",
                action=LocalizedText(
                    en="toggle automatic tree and manual split modes",
                    pl="przełącz automatic tree i manual split",
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change max depth", pl="zmień max depth"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="D",
                action=LocalizedText(en="toggle dataset kind", pl="przełącz rodzaj danych"),
            ),
            ControlBinding(
                key="G",
                action=LocalizedText(en="toggle Gini and entropy", pl="przełącz Gini i entropy"),
            ),
            ControlBinding(
                key="F",
                action=LocalizedText(
                    en="toggle manual split feature",
                    pl="zmień cechę manual split",
                ),
            ),
            ControlBinding(
                key="Q / E",
                action=LocalizedText(
                    en="move manual split threshold",
                    pl="przesuń threshold manual split",
                ),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current setup", pl="zresetuj aktualny układ"),
            ),
        ),
        create_scene=create_decision_tree_scene,
        tags=("classification", "trees"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "A decision tree classifies points by asking a sequence "
                                "of simple feature questions."
                            ),
                            pl=(
                                "Decision tree klasyfikuje punkty, zadając serię prostych "
                                "pytań o cechy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Each split divides the space into smaller regions that "
                                "can receive different predictions."
                            ),
                            pl=(
                                "Każdy split dzieli przestrzeń na mniejsze regiony, "
                                "które mogą dostać różne predykcje."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "The tree searches for splits that make child nodes "
                                "more pure than the parent node."
                            ),
                            pl=(
                                "Drzewo szuka splitów, które sprawiają, że child nodes "
                                "są bardziej jednorodne niż parent node."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Gini and entropy are two ways to score impurity; information gain "
                                "measures the improvement."
                            ),
                            pl=(
                                "Gini i entropy to dwa sposoby mierzenia impurity; "
                                "information gain mierzy poprawę po splicie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Manual split mode lets you compare your intuition with "
                                "the measured gain."
                            ),
                            pl=("Manual split pozwala porównać własną intuicję z policzonym gain."),
                        ),
                        LocalizedText(
                            en=(
                                "XOR data is hard for a shallow axis-aligned tree, "
                                "because one split is not enough."
                            ),
                            pl=(
                                "Dane XOR są trudne dla płytkiego axis-aligned tree, "
                                "bo jeden split nie wystarcza."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "A deeper tree can fit training data very well, but it may "
                                "generalize worse."
                            ),
                            pl=(
                                "Głębsze drzewo może świetnie dopasować train data, "
                                "ale gorzej generalizować."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A split that looks good visually is not always the best one "
                                "by impurity reduction."
                            ),
                            pl=(
                                "Split, który wygląda dobrze wizualnie, nie zawsze jest najlepszy "
                                "według redukcji impurity."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["decision_tree_splitter"],
            glossary=LESSON_GLOSSARY["decision_tree_splitter"],
        ),
    ),
    _placeholder_demo(
        demo_id="random_forest_bagging_lab",
        level=2,
        title_en="Random Forest Bagging Lab",
        title_pl="Random Forest Bagging Lab",
        summary_en="Bootstrap sampling, voting, variance, and stability.",
        summary_pl="Bootstrap sampling, voting, wariancja i stabilność predykcji.",
        objectives=(
            LocalizedText(
                en="Compare a single tree baseline with a random forest on train/test data.",
                pl=("Porównuj single tree baseline z random forest na danych train/test."),
            ),
            LocalizedText(
                en="Change tree count, max depth, noise, and bootstrap ratio to study variance.",
                pl=(
                    "Zmieniaj tree count, max depth, noise i bootstrap ratio, "
                    "żeby zobaczyć wpływ na wariancję."
                ),
            ),
            LocalizedText(
                en="Use confidence view to discuss ensemble voting and uncertainty.",
                pl=("Używaj confidence view do rozmowy o ensemble voting i niepewności predykcji."),
            ),
        ),
        controls=(
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change forest tree count", pl="zmień tree count"),
            ),
            ControlBinding(
                key="W / S",
                action=LocalizedText(en="change max depth", pl="zmień max depth"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="change dataset noise", pl="zmień noise w danych"),
            ),
            ControlBinding(
                key="B / V",
                action=LocalizedText(
                    en="change bootstrap sample ratio",
                    pl="zmień bootstrap sample ratio",
                ),
            ),
            ControlBinding(
                key="D",
                action=LocalizedText(en="toggle dataset kind", pl="przełącz rodzaj danych"),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="toggle confidence view",
                    pl="włącz albo wyłącz confidence view",
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="generate another dataset seed", pl="wygeneruj nowy seed danych"
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the current setup", pl="zresetuj aktualny układ"),
            ),
        ),
        create_scene=create_random_forest_scene,
        tags=("ensemble", "classification"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Random forest combines many decision trees instead of trusting "
                                "one tree."
                            ),
                            pl=(
                                "Random forest łączy wiele decision trees zamiast ufać "
                                "jednemu drzewu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Each tree votes, and the forest prediction is the class "
                                "with the most votes."
                            ),
                            pl=(
                                "Każde drzewo głosuje, a predykcją forest jest klasa "
                                "z największą liczbą głosów."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "Bootstrap sampling gives trees slightly different training sets, "
                                "so their mistakes are less identical."
                            ),
                            pl=(
                                "Bootstrap sampling daje drzewom trochę inne zbiory treningowe, "
                                "więc ich błędy są mniej identyczne."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Voting can reduce variance: one unstable tree matters less "
                                "inside a larger ensemble."
                            ),
                            pl=(
                                "Voting może zmniejszać wariancję: jedno niestabilne drzewo "
                                "ma mniejszy wpływ w większym ensemble."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Compare train/test accuracy and gap for the single tree "
                                "versus the forest."
                            ),
                            pl=("Porównuj train/test accuracy i gap dla single tree oraz forest."),
                        ),
                        LocalizedText(
                            en=(
                                "Confidence view shows where trees agree strongly and where "
                                "the vote is less certain."
                            ),
                            pl=(
                                "Confidence view pokazuje, gdzie drzewa mocno się zgadzają, "
                                "a gdzie głosowanie jest mniej pewne."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "More trees usually improve stability, but they do not fix "
                                "bad data or a poor setup automatically."
                            ),
                            pl=(
                                "Więcej drzew zwykle poprawia stabilność, ale nie naprawia "
                                "automatycznie złych danych ani słabej konfiguracji."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "High confidence means trees agree; it does not guarantee "
                                "that the prediction is correct."
                            ),
                            pl=(
                                "Wysokie confidence oznacza zgodność drzew; nie gwarantuje, "
                                "że predykcja jest poprawna."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["random_forest_bagging_lab"],
            glossary=LESSON_GLOSSARY["random_forest_bagging_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="boosting_mistake_lab",
        level=2,
        title_en="Boosting Mistake Lab",
        title_pl="Boosting Mistake Lab",
        summary_en=(
            "Watch boosting combine weak learners, update sample weights, "
            "and track train/test accuracy across rounds."
        ),
        summary_pl=(
            "Zobacz, jak boosting łączy weak learners, zmienia wagi próbek "
            "i śledzi train/test accuracy po kolejnych rundach."
        ),
        objectives=(
            LocalizedText(
                en="See how each weak learner focuses on mistakes from previous rounds.",
                pl="Zobacz, jak kolejne weak learners skupiają się na wcześniejszych błędach.",
            ),
            LocalizedText(
                en="Compare staged train/test accuracy and the generalization gap.",
                pl="Porównuj staged train/test accuracy i generalization gap.",
            ),
            LocalizedText(
                en="Use confidence view and presets to discuss overfitting and robustness.",
                pl="Używaj confidence view i presetów do rozmowy o overfittingu i odporności.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(
                    en="change selected boosting stage",
                    pl="zmień wybraną rundę boostingu",
                ),
            ),
            ControlBinding(
                key="+ / -",
                action=LocalizedText(
                    en="change total round count",
                    pl="zmień liczbę rund",
                ),
            ),
            ControlBinding(
                key="1-4 / P",
                action=LocalizedText(
                    en="switch preset scenario",
                    pl="przełącz preset",
                ),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="toggle confidence view",
                    pl="włącz albo wyłącz confidence view",
                ),
            ),
            ControlBinding(
                key="E",
                action=LocalizedText(
                    en="export selected-stage decision boundary",
                    pl="wyeksportuj decision boundary dla wybranej rundy",
                ),
            ),
        ),
        create_scene=create_boosting_mistake_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("ensemble", "classification", "boosting"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Boosting builds an ensemble one weak learner at a time, "
                                "with each round reacting to earlier mistakes."
                            ),
                            pl=(
                                "Boosting buduje ensemble po jednym weak learnerze, "
                                "a każda runda reaguje na wcześniejsze błędy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The selected stage lets you inspect how the ensemble changes "
                                "after each boosting round."
                            ),
                            pl=(
                                "Selected stage pozwala zobaczyć, jak ensemble zmienia się "
                                "po każdej rundzie boostingu."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Why it works", pl="Dlaczego to działa"),
                    body=(
                        LocalizedText(
                            en=(
                                "After a weak learner makes mistakes, boosting increases "
                                "the weight of examples that need more attention."
                            ),
                            pl=(
                                "Po błędach weak learnera boosting zwiększa wagi przykładów, "
                                "które wymagają większej uwagi."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Learners with lower weighted error receive stronger influence "
                                "in the final vote."
                            ),
                            pl=(
                                "Learners z niższym weighted error dostają większy wpływ "
                                "na finalne głosowanie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Watch staged train/test accuracy: train can improve while "
                                "test starts to flatten or drop."
                            ),
                            pl=(
                                "Obserwuj staged train/test accuracy: train może rosnąć, "
                                "gdy test zaczyna się wypłaszczać albo spadać."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Confidence view helps identify regions where the ensemble "
                                "has weaker agreement."
                            ),
                            pl=(
                                "Confidence view pomaga znaleźć regiony, w których ensemble "
                                "ma słabszą zgodność."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Adding more rounds is not always better; too many rounds "
                                "can increase overfitting."
                            ),
                            pl=(
                                "Dodawanie rund nie zawsze pomaga; zbyt wiele rund "
                                "może zwiększyć overfitting."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A weak learner is not useless because it is weak; boosting "
                                "uses many small corrections together."
                            ),
                            pl=(
                                "Weak learner nie jest bezużyteczny tylko dlatego, że jest słaby; "
                                "boosting łączy wiele małych korekt."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["boosting_mistake_lab"],
            glossary=LESSON_GLOSSARY["boosting_mistake_lab"],
        ),
    ),
)

DEMO_BY_ID: dict[str, DemoManifest] = {manifest.id: manifest for manifest in DEMO_MANIFESTS}


def levels_from_manifests(manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS) -> tuple[int, ...]:
    """Return sorted level numbers present in the registry."""
    return tuple(sorted({manifest.level for manifest in manifests}))


def demos_for_level(
    level: int,
    manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS,
) -> tuple[DemoManifest, ...]:
    """Return demos belonging to one level."""
    return tuple(manifest for manifest in manifests if manifest.level == level)


def validate_demo_registry(
    *,
    level_manifests: tuple[LevelManifest, ...] = LEVEL_MANIFESTS,
    demo_manifests: tuple[DemoManifest, ...] = DEMO_MANIFESTS,
) -> None:
    """Validate registry consistency.

    Raises:
        ValueError: if the registry contains duplicate or incomplete metadata.
    """
    level_numbers = [level.number for level in level_manifests]
    duplicate_levels = _duplicates(level_numbers)
    if duplicate_levels:
        raise ValueError(f"Duplicate level numbers: {duplicate_levels}")

    known_levels = set(level_numbers)
    demo_ids = [demo.id for demo in demo_manifests]
    duplicate_demo_ids = _duplicates(demo_ids)
    if duplicate_demo_ids:
        raise ValueError(f"Duplicate demo ids: {duplicate_demo_ids}")

    for demo in demo_manifests:
        if demo.level not in known_levels:
            raise ValueError(f"Demo {demo.id!r} references unknown level {demo.level}")
        if not demo.id:
            raise ValueError("Demo id must not be empty")
        if not demo.title.en or not demo.title.pl:
            raise ValueError(f"Demo {demo.id!r} must have localized titles")
        if not demo.summary.en or not demo.summary.pl:
            raise ValueError(f"Demo {demo.id!r} must have localized summaries")
        if not demo.objectives:
            raise ValueError(f"Demo {demo.id!r} must define objectives")
        if not demo.controls:
            raise ValueError(f"Demo {demo.id!r} must define controls")
        if demo.create_scene is None:
            raise ValueError(f"Demo {demo.id!r} must define a scene factory")


def _duplicates(values: list[int] | list[str]) -> tuple[int | str, ...]:
    """Return duplicate values in first-seen order."""
    seen: set[int | str] = set()
    duplicates: list[int | str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return tuple(duplicates)


validate_demo_registry()

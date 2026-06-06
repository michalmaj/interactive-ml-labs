"""Demo manifest registry for the unified shell."""

from __future__ import annotations

from collections.abc import Callable

from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.clustering_scene import create_clustering_lab_scene
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
                "zanim loss zacznie zauważalnie spadać."
            ),
        ),
        LocalizedText(
            en=(
                "Set learning rate high enough to overshoot. Notice whether "
                "the path bounces around the minimum."
            ),
            pl=(
                "Ustaw learning rate tak wysoko, żeby kroki przeskakiwały minimum. "
                "Zobacz, czy ścieżka zaczyna odbijać się na boki."
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
                "którzy sąsiedzi decydują o predykcji."
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
                "ale jedna klasa wypada wyraźnie gorzej."
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
                "i sprawdź, jak zareaguje impurity score."
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
                "Przełączaj widoki i sprawdź, jakie regiony predykcji powstają po wybranym splicie."
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
            pl=("Włącz confidence view i znajdź obszary, w których drzewa nie są zgodne."),
        ),
        LocalizedText(
            en=(
                "Increase ensemble size and check whether the forest stabilizes "
                "or only becomes more complex."
            ),
            pl=(
                "Zwiększ rozmiar ensemble i sprawdź, czy Random Forest się "
                "stabilizuje, czy tylko robi się bardziej złożony."
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
                "źle klasyfikowane punkty dostają większą uwagę."
            ),
        ),
        LocalizedText(
            en=(
                "Compare train and test accuracy after many rounds. Look for "
                "the moment when the gap becomes suspicious."
            ),
            pl=(
                "Porównuj train i test accuracy po wielu rundach. Szukaj momentu, "
                "w którym różnica zaczyna być niepokojąco duża."
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
    "clustering_lab": (
        LocalizedText(
            en=(
                "Start with clean blobs and change k. Notice when K-Means "
                "matches the visible groups and when it starts inventing structure."
            ),
            pl=(
                "Zacznij od clean blobs i zmieniaj k. Zobacz, kiedy K-Means "
                "pasuje do danych, a kiedy zaczyna tworzyć sztuczne podziały."
            ),
        ),
        LocalizedText(
            en=(
                "Switch to moons, compare K-Means with DBSCAN, and find an eps "
                "that follows the curved groups without merging everything."
            ),
            pl=(
                "Przełącz się na moons, porównaj K-Means z DBSCAN i znajdź eps, "
                "który podąża za krzywymi grupami bez łączenia wszystkiego."
            ),
        ),
        LocalizedText(
            en=("Use outliers to compare centroid pull in K-Means with noise detection in DBSCAN."),
            pl=(
                "Użyj outlierów, żeby porównać przesuwanie centroidów w K-Means "
                "z wykrywaniem noise w DBSCAN."
            ),
        ),
    ),
    "level_3_coming_soon": (
        LocalizedText(
            en=(
                "Pick one advanced idea you want to explain visually: dimensionality "
                "reduction, clustering, calibration, or model monitoring."
            ),
            pl=(
                "Wybierz jeden zaawansowany temat do pokazania wizualnie: "
                "dimensionality reduction, clustering, calibration albo model monitoring."
            ),
        ),
        LocalizedText(
            en=(
                "Write down what the student should be able to change, observe, "
                "and explain after five minutes."
            ),
            pl=(
                "Zapisz, co student powinien móc zmienić, zaobserwować "
                "i wyjaśnić po pięciu minutach pracy z demo."
            ),
        ),
        LocalizedText(
            en=(
                "Compare two candidate ideas and choose the one with the clearest "
                "interactive feedback loop."
            ),
            pl=(
                "Porównaj dwa pomysły i wybierz ten, który ma najczytelniejszą "
                "pętlę interakcji: ustawienie, obserwacja, wniosek."
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
                pl="Próg probability, który zamienia wynik modelu na etykietę klasy.",
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
                pl="Dwa uzupełniające się sposoby analizowania błędów klasyfikacji.",
            ),
        ),
    ),
    "decision_tree_splitter": (
        GlossaryTerm(
            term="split",
            definition=LocalizedText(
                en="A rule that sends data points to one side of the tree or the other.",
                pl="Reguła, która kieruje punkty na jedną albo drugą stronę drzewa.",
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
                    "Sytuacja, w której model uczy się przypadkowych szczegółów z treningu "
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
                pl="Widok pokazujący, gdzie drzewa w Random Forest są zgodne, a gdzie nie.",
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
    "clustering_lab": (
        GlossaryTerm(
            term="clustering",
            definition=LocalizedText(
                en=(
                    "Grouping data without class labels, usually by looking for "
                    "points that are close or dense together."
                ),
                pl=(
                    "Grupowanie danych bez etykiet klas, zwykle przez szukanie "
                    "punktów, które są blisko siebie albo tworzą gęste obszary."
                ),
            ),
        ),
        GlossaryTerm(
            term="centroid",
            definition=LocalizedText(
                en="The current center of a K-Means cluster.",
                pl="Aktualny środek klastra w K-Means.",
            ),
        ),
        GlossaryTerm(
            term="inertia",
            definition=LocalizedText(
                en=(
                    "A K-Means score based on distances from points to their "
                    "assigned centroids; lower is usually tighter."
                ),
                pl=(
                    "Miara K-Means oparta na odległościach punktów od przypisanych "
                    "centroidów; niższa zwykle oznacza ciaśniejsze klastry."
                ),
            ),
        ),
        GlossaryTerm(
            term="DBSCAN",
            definition=LocalizedText(
                en=(
                    "A density-based clustering algorithm that groups points in dense "
                    "neighborhoods and can mark sparse points as noise."
                ),
                pl=(
                    "Algorytm clusteringu oparty na gęstości; grupuje punkty w gęstych "
                    "sąsiedztwach i może oznaczać rzadkie punkty jako noise."
                ),
            ),
        ),
        GlossaryTerm(
            term="eps",
            definition=LocalizedText(
                en="The DBSCAN neighborhood radius used to decide which points are close.",
                pl="Promień sąsiedztwa w DBSCAN, który określa, które punkty są blisko.",
            ),
        ),
        GlossaryTerm(
            term="noise",
            definition=LocalizedText(
                en="A point that DBSCAN does not place inside any dense cluster.",
                pl="Punkt, którego DBSCAN nie przypisuje do żadnego gęstego klastra.",
            ),
        ),
    ),
    "level_3_coming_soon": (
        GlossaryTerm(
            term="showcase demo",
            definition=LocalizedText(
                en=(
                    "A visually strong demo that helps students connect an advanced "
                    "ML idea with concrete behavior."
                ),
                pl=(
                    "Efektowne wizualnie demo, które łączy zaawansowaną ideę ML "
                    "z konkretnym zachowaniem modelu."
                ),
            ),
        ),
        GlossaryTerm(
            term="interactive feedback loop",
            definition=LocalizedText(
                en=(
                    "The cycle where a student changes a setting, observes the result, "
                    "and explains why it happened."
                ),
                pl=(
                    "Cykl, w którym student zmienia ustawienie, obserwuje rezultat "
                    "i wyjaśnia, dlaczego tak się stało."
                ),
            ),
        ),
        GlossaryTerm(
            term="advanced experiment",
            definition=LocalizedText(
                en=(
                    "A lab that goes beyond core algorithms and focuses on richer "
                    "model behavior or real-world ML workflow."
                ),
                pl=(
                    "Lab wykraczający poza podstawowe algorytmy, skupiony na bogatszym "
                    "zachowaniu modeli albo praktycznym workflow ML."
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
                pl="Zobacz, jaka intuicja stoi za działaniem algorytmu.",
            ),
            LocalizedText(
                en="Observe how parameter changes affect model behavior.",
                pl="Sprawdź, jak zmiana parametrów wpływa na zachowanie modelu.",
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
        summary_pl="Optymalizacja, loss i praktyczna intuicja stojąca za learning rate.",
        objectives=(
            LocalizedText(
                en="Watch gradient descent reduce loss step by step.",
                pl="Zobacz, jak gradient descent krok po kroku zmniejsza loss.",
            ),
            LocalizedText(
                en="Change learning rate and noise to see when training becomes unstable.",
                pl=(
                    "Zmieniaj learning rate i noise, żeby zobaczyć, "
                    "kiedy trening przestaje być stabilny."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="start or pause automatic steps",
                    pl="uruchom albo zatrzymaj automatyczne kroki",
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
                    en="generate another dataset seed", pl="wylosuj nowy seed danych"
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
                                "Tutaj model jest prostą, więc kluczowe parametry to weight i bias."
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
                                "Obserwuj, jak przesuwa się linia regresji, i porównuj to "
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
                                "Niższy loss to nie magia: oznacza, że predykcje są bliższe "
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
                pl="Zmieniaj k i noise, żeby porównać gładsze i bardziej lokalne decision regions.",
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
                    en="generate another dataset seed", pl="wylosuj nowy seed danych"
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
                                "k-NN klasyfikuje query point, patrząc na etykiety "
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
                                "Wysoka accuracy dla jednego seedu nie gwarantuje, "
                                "że model będzie stabilny, "
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
        summary_pl="Probability, threshold i decision boundary w praktyce.",
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
                    "stabilny trening z niestabilnym."
                ),
            ),
            LocalizedText(
                en="Use precision and recall to reason about threshold trade-offs.",
                pl=(
                    "Używaj precision i recall, żeby analizować kompromisy przy wyborze threshold."
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
                    en="generate another dataset seed", pl="wylosuj nowy seed danych"
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
                                "dla class_1."
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
                                "Tło probability zmienia się w trakcie treningu, jeszcze zanim "
                                "model przypisze finalne klasy."
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
        summary_pl="Splity, impurity i reguły, które da się łatwo wyjaśnić.",
        objectives=(
            LocalizedText(
                en="Compare automatic tree splits with a manual split you can move yourself.",
                pl=(
                    "Porównuj automatyczne splity drzewa z własnym manual split, "
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
                    "Przełączaj dane axis-aligned i XOR, żeby zobaczyć, "
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
                    en="generate another dataset seed", pl="wylosuj nowy seed danych"
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
                                "Każdy split dzieli przestrzeń na mniejsze obszary, "
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
                pl=("Porównuj single tree baseline z Random Forest na danych train/test."),
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
                pl=("Używaj confidence view, żeby omawiać ensemble voting i niepewność predykcji."),
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
                    en="generate another dataset seed", pl="wylosuj nowy seed danych"
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
                                "Random Forest łączy wiele decision trees zamiast ufać "
                                "jednemu drzewu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Each tree votes, and the forest prediction is the class "
                                "with the most votes."
                            ),
                            pl=(
                                "Każde drzewo głosuje, a predykcją Random Forest jest klasa "
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
                            pl=(
                                "Porównuj train/test accuracy i gap dla single tree "
                                "oraz Random Forest."
                            ),
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
                                "Wysokie confidence oznacza zgodność drzew; nie gwarantuje jednak, "
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
            "Zobacz, jak boosting łączy weak learners, zmienia wagi przykładów "
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
                pl="Używaj confidence view i presetów, żeby omawiać overfitting i odporność.",
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
                                "nawet gdy test zaczyna się wypłaszczać albo spadać."
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
    DemoManifest(
        id="clustering_lab",
        level=3,
        title=LocalizedText(en="Clustering Lab", pl="Clustering Lab"),
        summary=LocalizedText(
            en=(
                "Compare K-Means and DBSCAN on unlabeled points to see how "
                "centroids, density, k, and eps shape clustering results."
            ),
            pl=(
                "Porównaj K-Means i DBSCAN na punktach bez etykiet, żeby zobaczyć, "
                "jak centroidy, gęstość, k i eps wpływają na clustering."
            ),
        ),
        objectives=(
            LocalizedText(
                en=(
                    "Connect K-Means assignments with the visible position of "
                    "centroids and data points."
                ),
                pl=("Powiąż przypisania K-Means z położeniem centroidów i punktów na wykresie."),
            ),
            LocalizedText(
                en=(
                    "Use DBSCAN to test whether density-based clustering handles "
                    "moons and outliers differently than K-Means."
                ),
                pl=(
                    "Użyj DBSCAN, żeby sprawdzić, czy clustering oparty na gęstości "
                    "inaczej radzi sobie z moons i outlierami niż K-Means."
                ),
            ),
            LocalizedText(
                en=(
                    "Treat k and eps as modeling decisions that encode different "
                    "assumptions about the data."
                ),
                pl=(
                    "Traktuj k i eps jako decyzje projektowe, które zapisują różne "
                    "założenia o danych."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="1-4",
                action=LocalizedText(
                    en="switch dataset preset",
                    pl="zmień preset danych",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(
                    en="decrease or increase k in K-Means, or eps in DBSCAN",
                    pl="zmniejsz albo zwiększ k w K-Means albo eps w DBSCAN",
                ),
            ),
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="advance one K-Means phase or rerun DBSCAN",
                    pl="wykonaj kolejną fazę K-Means albo ponownie uruchom DBSCAN",
                ),
            ),
            ControlBinding(
                key="A",
                action=LocalizedText(
                    en="toggle auto-run",
                    pl="włącz albo wyłącz auto-run",
                ),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="show or hide point-to-centroid links",
                    pl="pokaż albo ukryj linie punkt-centroid",
                ),
            ),
            ControlBinding(
                key="M",
                action=LocalizedText(
                    en="switch between K-Means and DBSCAN",
                    pl="przełącz między K-Means i DBSCAN",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the active algorithm",
                    pl="zresetuj aktywny algorytm",
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="generate a new dataset sample",
                    pl="wygeneruj nową próbkę danych",
                ),
            ),
            ControlBinding(
                key="Mouse drag",
                action=LocalizedText(
                    en="move a data point",
                    pl="przesuń punkt danych",
                ),
            ),
        ),
        create_scene=create_clustering_lab_scene,
        difficulty=LocalizedText(en="Advanced", pl="Zaawansowane"),
        tags=("clustering", "k-means", "dbscan", "unsupervised", "visualization"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What clustering asks", pl="O co pyta clustering"),
                    body=(
                        LocalizedText(
                            en=(
                                "Clustering looks for structure in data without using class labels."
                            ),
                            pl=(
                                "Clustering szuka struktury w danych bez korzystania "
                                "z etykiet klas."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "That makes the result useful for exploration, but also "
                                "dependent on the assumptions built into the algorithm."
                            ),
                            pl=(
                                "To pomaga w eksploracji danych, ale wynik zależy od "
                                "założeń zapisanych w algorytmie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Lesson path",
                        pl="Ścieżka pracy",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Start in K-Means mode. Step through assignment and "
                                "centroid-update phases so the algorithm feels mechanical, "
                                "not magical."
                            ),
                            pl=(
                                "Zacznij w trybie K-Means. Przejdź przez fazy przypisania "
                                "punktów i przesunięcia centroidów, żeby algorytm był "
                                "mechaniczny, a nie magiczny."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Then switch to DBSCAN and tune eps. Ask whether density "
                                "explains the same data better than nearest-centroid distance."
                            ),
                            pl=(
                                "Potem przełącz się na DBSCAN i dostrój eps. Sprawdź, "
                                "czy gęstość lepiej wyjaśnia te same dane niż odległość "
                                "od najbliższego centroidu."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What K-Means assumes", pl="Co zakłada K-Means"),
                    body=(
                        LocalizedText(
                            en=(
                                "K-Means represents each cluster with a centroid and "
                                "assigns points to the nearest one."
                            ),
                            pl=(
                                "K-Means opisuje każdy klaster przez centroid i przypisuje "
                                "punkty do najbliższego centroidu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "It works best when clusters are compact, roughly round, "
                                "and not wildly different in size."
                            ),
                            pl=(
                                "Najlepiej działa, gdy klastry są zwarte, mniej więcej "
                                "okrągłe i niezbyt różne pod względem rozmiaru."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What DBSCAN changes", pl="Co zmienia DBSCAN"),
                    body=(
                        LocalizedText(
                            en=(
                                "DBSCAN does not ask for k. It grows clusters from dense "
                                "neighborhoods and marks sparse points as noise."
                            ),
                            pl=(
                                "DBSCAN nie pyta o k. Buduje klastry z gęstych sąsiedztw "
                                "i oznacza rzadkie punkty jako noise."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The eps setting controls the neighborhood radius. Too small "
                                "creates too much noise; too large can merge separate groups."
                            ),
                            pl=(
                                "Ustawienie eps kontroluje promień sąsiedztwa. Zbyt małe eps "
                                "tworzy za dużo noise; zbyt duże może połączyć osobne grupy."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Why k and eps are choices",
                        pl="Dlaczego k i eps są wyborami",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "K-Means does not know how many groups are meaningful. "
                                "DBSCAN does not know which neighborhood radius is right."
                            ),
                            pl=(
                                "K-Means nie wie, ile grup ma sens. DBSCAN nie wie, "
                                "jaki promień sąsiedztwa jest właściwy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A lower inertia can look better numerically while still "
                                "splitting a natural group into artificial pieces."
                            ),
                            pl=(
                                "Niższa inertia może wyglądać lepiej liczbowo, ale nadal "
                                "dzielić naturalną grupę na sztuczne kawałki."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="What the interaction shows",
                        pl="Co pokazuje interakcja",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Dragging a point changes the data, not only the display. "
                                "After the move, K-Means recomputes assignments against the "
                                "current centroids."
                            ),
                            pl=(
                                "Przesuwanie punktu zmienia dane, a nie tylko obrazek. "
                                "Po takim ruchu K-Means ponownie liczy przypisania względem "
                                "aktualnych centroidów."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Point-to-centroid links make those assignments explicit: "
                                "each point belongs to the closest centroid."
                            ),
                            pl=(
                                "Linie punkt-centroid pokazują te przypisania wprost: "
                                "każdy punkt należy do najbliższego centroidu."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="How to read inertia",
                        pl="Jak czytać inertia",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "The inertia trend shows whether K-Means is reducing the "
                                "sum of squared point-to-centroid distances."
                            ),
                            pl=(
                                "Trend inertia pokazuje, czy K-Means zmniejsza sumę "
                                "kwadratów odległości punktów od centroidów."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Lower inertia does not always mean a better explanation. "
                                "A larger k can reduce the score while splitting a natural "
                                "group into artificial pieces."
                            ),
                            pl=(
                                "Niższa inertia nie zawsze oznacza lepsze wyjaśnienie danych. "
                                "Większe k może poprawić wynik, a jednocześnie pociąć naturalną "
                                "grupę na sztuczne kawałki."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Where it breaks", pl="Gdzie to się psuje"),
                    body=(
                        LocalizedText(
                            en=(
                                "Moons, uneven densities, and outliers expose the limits "
                                "of centroid-based clustering."
                            ),
                            pl=(
                                "Moons, nierówne gęstości i outliery pokazują ograniczenia "
                                "clusteringu opartego na centroidach."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "That is where density-based methods such as DBSCAN become "
                                "a natural next comparison."
                            ),
                            pl=(
                                "W tym miejscu naturalnym kolejnym porównaniem stają się "
                                "metody oparte na gęstości, takie jak DBSCAN."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="What to compare",
                        pl="Co porównywać",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "On clean blobs, both algorithms can look reasonable. On moons, "
                                "K-Means fights the shape while DBSCAN can follow the curve."
                            ),
                            pl=(
                                "Na clean blobs oba algorytmy mogą wyglądać sensownie. "
                                "Na moons K-Means walczy z kształtem, a DBSCAN może podążyć "
                                "za krzywą."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "On outliers, compare whether the algorithm pulls a centroid "
                                "toward lonely points or leaves them as noise."
                            ),
                            pl=(
                                "Na outlierach sprawdź, czy algorytm przesuwa centroid w stronę "
                                "samotnych punktów, czy zostawia je jako noise."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["clustering_lab"],
            glossary=LESSON_GLOSSARY["clustering_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="level_3_coming_soon",
        level=3,
        title_en="Coming Soon: Advanced Experiments",
        title_pl="Coming soon: zaawansowane eksperymenty",
        summary_en=(
            "A preview slot for richer Level 3 demos: dimensionality reduction, "
            "clustering, calibration, monitoring, and other showcase topics."
        ),
        summary_pl=(
            "Zapowiedź bardziej zaawansowanych dem Level 3: dimensionality reduction, "
            "clustering, calibration, monitoring i innych tematów showcase."
        ),
        objectives=(
            LocalizedText(
                en="Show that Level 3 is reserved for advanced, visually rich ML experiments.",
                pl=(
                    "Pokaż, że Level 3 jest miejscem na zaawansowane, "
                    "wizualnie mocniejsze eksperymenty ML."
                ),
            ),
            LocalizedText(
                en="Keep the guided app shape visible before the first Level 3 demo lands.",
                pl=(
                    "Pokaż docelowy kształt aplikacji, zanim pojawi się "
                    "pierwsze właściwe demo Level 3."
                ),
            ),
            LocalizedText(
                en="Give students a hint about the kinds of workflows coming next.",
                pl="Daj studentom podgląd tematów i workflow, które pojawią się później.",
            ),
        ),
        controls=(
            ControlBinding(
                key="Enter",
                action=LocalizedText(
                    en="open the coming-soon placeholder scene",
                    pl="otwórz placeholder Coming soon",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read Level 3 planning notes",
                    pl="przeczytaj notatki o planie Level 3",
                ),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="go back to the demo list",
                    pl="wróć do listy dem",
                ),
            ),
        ),
        difficulty=LocalizedText(en="Coming soon", pl="W przygotowaniu"),
        tags=("level-3", "showcase", "planning"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this slot shows", pl="Co pokazuje ten ekran"),
                    body=(
                        LocalizedText(
                            en=(
                                "Level 3 is planned as the advanced/showcase layer "
                                "of Interactive ML Labs."
                            ),
                            pl=(
                                "Level 3 jest planowany jako zaawansowana, pokazowa "
                                "warstwa Interactive ML Labs."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The placeholder keeps the learning path visible even before "
                                "the first advanced demo is implemented."
                            ),
                            pl=(
                                "Placeholder pokazuje pełną ścieżkę nauki jeszcze zanim "
                                "powstanie pierwsze zaawansowane demo."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What belongs here", pl="Co tu pasuje"),
                    body=(
                        LocalizedText(
                            en=(
                                "Good Level 3 demos should make a more complex ML idea visible, "
                                "not only configurable."
                            ),
                            pl=(
                                "Dobre demo Level 3 powinno pokazywać bardziej złożoną ideę ML "
                                "w sposób widoczny, a nie tylko konfigurowalny."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Examples: PCA and embeddings, clustering behavior, "
                                "calibration, drift, monitoring, or model comparison workflows."
                            ),
                            pl=(
                                "Przykłady: PCA i embeddings, zachowanie clusteringu, "
                                "calibration, drift, monitoring albo workflow porównywania modeli."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Design principle", pl="Zasada projektowa"),
                    body=(
                        LocalizedText(
                            en=(
                                "A showcase demo should still be teachable: the student changes "
                                "one thing, observes a clear reaction, and can explain it."
                            ),
                            pl=(
                                "Demo showcase nadal musi uczyć: student zmienia jedną rzecz, "
                                "widzi czytelną reakcję i potrafi ją wyjaśnić."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "If the visual result is impressive but hard to reason about, "
                                "it is not ready for the guided app yet."
                            ),
                            pl=(
                                "Jeśli efekt wygląda imponująco, ale trudno go wyjaśnić, "
                                "to demo nie jest jeszcze gotowe do guided app."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["level_3_coming_soon"],
            glossary=LESSON_GLOSSARY["level_3_coming_soon"],
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

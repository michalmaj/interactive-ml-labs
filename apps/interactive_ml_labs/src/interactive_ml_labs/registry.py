"""Demo manifest registry for the unified shell."""

from __future__ import annotations

from collections.abc import Callable

from interactive_ml_labs.activation_scene import create_activation_functions_lab_scene
from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.calibration_scene import create_calibration_lab_scene
from interactive_ml_labs.class_imbalance_scene import create_class_imbalance_lab_scene
from interactive_ml_labs.clustering_scene import create_clustering_lab_scene
from interactive_ml_labs.data_leakage_scene import create_data_leakage_lab_scene
from interactive_ml_labs.decision_tree_scene import create_decision_tree_scene
from interactive_ml_labs.distance_metrics_scene import create_distance_metrics_lab_scene
from interactive_ml_labs.feature_scaling_scene import create_feature_scaling_lab_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.kmeans_intro_scene import create_kmeans_intro_lab_scene
from interactive_ml_labs.knn_scene import create_knn_vote_map_scene
from interactive_ml_labs.linear_regression_scene import create_linear_regression_line_fit_lab_scene
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
from interactive_ml_labs.model_comparison_scene import create_model_comparison_lab_scene
from interactive_ml_labs.monitoring_scene import create_model_monitoring_drift_scene
from interactive_ml_labs.neural_network_scene import create_neural_network_playground_scene
from interactive_ml_labs.pca_scene import create_pca_lab_scene
from interactive_ml_labs.placeholder_scene import PlaceholderDemoScene
from interactive_ml_labs.random_forest_scene import create_random_forest_scene
from interactive_ml_labs.scene import Scene
from interactive_ml_labs.split_lab_scene import create_train_validation_test_lab_scene
from interactive_ml_labs.svm_margin_scene import create_svm_margin_lab_scene
from interactive_ml_labs.tsne_umap_scene import create_tsne_umap_exploration_scene
from interactive_ml_labs.tuning_scene import create_hyperparameter_tuning_lab_scene

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
    "linear_regression_line_fit_lab": (
        LocalizedText(
            en=(
                "Move slope until residuals stop growing from left to right. "
                "Then adjust intercept to balance the errors."
            ),
            pl=(
                "Zmieniaj slope, aż residuals przestaną rosnąć od lewej do prawej. "
                "Potem dopasuj intercept, żeby zbalansować błędy."
            ),
        ),
        LocalizedText(
            en=(
                "Press F to compare your line with the least-squares fit. "
                "Name which parameter was farther away."
            ),
            pl=(
                "Naciśnij F i porównaj swoją prostą z least-squares fit. "
                "Nazwij parametr, który był dalej od optimum."
            ),
        ),
        LocalizedText(
            en=(
                "Switch to the noisy dataset and explain why the best line "
                "still leaves visible residuals."
            ),
            pl=(
                "Przełącz się na zaszumiony dataset i wyjaśnij, czemu najlepsza "
                "prosta nadal zostawia widoczne residuals."
            ),
        ),
    ),
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
    "kmeans_intro_lab": (
        LocalizedText(
            en=(
                "Press Space once and name which centroid owns each visible group. "
                "Then press Space again and describe how centroids move."
            ),
            pl=(
                "Naciśnij Space raz i nazwij, który centroid przejmuje każdą grupę. "
                "Potem naciśnij Space ponownie i opisz, jak przesuwają się centroidy."
            ),
        ),
        LocalizedText(
            en=(
                "Change k on the two-group dataset. Find a setting where K-Means "
                "splits a natural group."
            ),
            pl=(
                "Zmień k na datasecie z dwiema grupami. Znajdź ustawienie, w którym "
                "K-Means dzieli naturalną grupę."
            ),
        ),
        LocalizedText(
            en=(
                "Turn on auto-run and watch inertia. Explain why it usually drops "
                "but does not prove the clusters are meaningful."
            ),
            pl=(
                "Włącz auto-run i obserwuj inertia. Wyjaśnij, czemu zwykle spada, "
                "ale nie dowodzi, że klastry mają sens."
            ),
        ),
    ),
    "distance_metrics_lab": (
        LocalizedText(
            en=(
                "Move the query point and watch which labeled point becomes nearest. "
                "Explain the result before changing the metric."
            ),
            pl=(
                "Przesuwaj query point i obserwuj, który punkt staje się nearest. "
                "Wyjaśnij wynik, zanim zmienisz metrykę."
            ),
        ),
        LocalizedText(
            en=(
                "Switch between Euclidean, Manhattan, and Chebyshev. Find a case "
                "where the nearest point changes."
            ),
            pl=(
                "Przełączaj Euclidean, Manhattan i Chebyshev. Znajdź przypadek, "
                "w którym nearest point się zmienia."
            ),
        ),
        LocalizedText(
            en=(
                "Look at the distance bars and name which axis difference dominates "
                "the chosen metric."
            ),
            pl=(
                "Spójrz na paski distance i nazwij, która różnica na osi dominuje "
                "dla wybranej metryki."
            ),
        ),
    ),
    "svm_margin_lab": (
        LocalizedText(
            en=(
                "Move the boundary until both classes are separated. Then widen the "
                "margin without creating mistakes."
            ),
            pl=(
                "Przesuwaj boundary, aż klasy będą rozdzielone. Potem poszerz margin "
                "bez tworzenia błędów."
            ),
        ),
        LocalizedText(
            en=(
                "Press F and identify which points became support vectors. Explain why "
                "far-away points matter less."
            ),
            pl=(
                "Naciśnij F i wskaż, które punkty stały się support vectors. Wyjaśnij, "
                "czemu dalekie punkty są mniej ważne."
            ),
        ),
        LocalizedText(
            en=(
                "Compare a narrow correct split with a wider correct split. Decide which "
                "one should generalize better."
            ),
            pl=(
                "Porównaj wąski poprawny podział z szerszym poprawnym podziałem. "
                "Zdecyduj, który powinien lepiej generalizować."
            ),
        ),
    ),
    "activation_functions_lab": (
        LocalizedText(
            en=(
                "Move x far left and far right for sigmoid and tanh. Notice where "
                "the local gradient almost disappears."
            ),
            pl=(
                "Przesuń x daleko w lewo i w prawo dla sigmoid oraz tanh. Zobacz, "
                "gdzie local gradient prawie znika."
            ),
        ),
        LocalizedText(
            en=(
                "Switch to ReLU and move x below zero. Explain why the neuron output "
                "and local gradient become zero."
            ),
            pl=(
                "Przełącz na ReLU i ustaw x poniżej zera. Wyjaśnij, czemu output "
                "neuronu i local gradient spadają do zera."
            ),
        ),
        LocalizedText(
            en=("Compare output ranges. Decide which activation is centered around zero."),
            pl=(
                "Porównaj zakresy outputu. Zdecyduj, która activation jest wycentrowana wokół zera."
            ),
        ),
    ),
    "neural_network_playground": (
        LocalizedText(
            en=(
                "Switch activation functions and watch how hidden outputs change "
                "the final probability."
            ),
            pl=(
                "Przełączaj activation functions i obserwuj, jak hidden outputs "
                "zmieniają finalne probability."
            ),
        ),
        LocalizedText(
            en=("Increase weight scale and explain why the same input can become more confident."),
            pl=(
                "Zwiększ weight scale i wyjaśnij, czemu ten sam input może dać "
                "bardziej pewną predykcję."
            ),
        ),
        LocalizedText(
            en=("Move hidden bias up and down. Name which hidden unit changes most."),
            pl=(
                "Przesuwaj hidden bias w górę i w dół. Nazwij hidden unit, "
                "który zmienia się najmocniej."
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
    "data_leakage_lab": (
        LocalizedText(
            en=(
                "Run the leaky version first and explain why a nearly perfect "
                "test score should make you suspicious."
            ),
            pl=(
                "Najpierw uruchom wersję z leakage i wyjaśnij, czemu prawie idealny "
                "test score powinien wzbudzić podejrzenia."
            ),
        ),
        LocalizedText(
            en=(
                "Press L to remove the suspicious feature. Compare the score drop "
                "with the new, more honest validation result."
            ),
            pl=(
                "Naciśnij L, żeby usunąć podejrzaną cechę. Porównaj spadek wyniku "
                "z nowym, uczciwszym wynikiem walidacji."
            ),
        ),
        LocalizedText(
            en=(
                "For each preset, decide whether the suspicious feature would exist "
                "at prediction time."
            ),
            pl=("Dla każdego presetu ustal, czy podejrzana cecha istniałaby w czasie predykcji."),
        ),
    ),
    "class_imbalance_lab": (
        LocalizedText(
            en=(
                "Start with the default threshold and explain why high accuracy "
                "does not guarantee good minority-class performance."
            ),
            pl=(
                "Zacznij od domyślnego threshold i wyjaśnij, czemu wysokie accuracy "
                "nie gwarantuje dobrego wyniku dla klasy mniejszościowej."
            ),
        ),
        LocalizedText(
            en=(
                "Lower the threshold and watch recall improve. Decide whether "
                "the extra false positives are acceptable."
            ),
            pl=(
                "Obniż threshold i obserwuj poprawę recall. Ustal, czy dodatkowe "
                "false positives są akceptowalne."
            ),
        ),
        LocalizedText(
            en=(
                "Raise the threshold and inspect false negatives. Name the business "
                "risk of missing the positive class."
            ),
            pl=(
                "Podnieś threshold i sprawdź false negatives. Nazwij ryzyko biznesowe "
                "przeoczenia klasy pozytywnej."
            ),
        ),
    ),
    "train_validation_test_lab": (
        LocalizedText(
            en=(
                "Compare simple, balanced, and too-flexible models. Pick the model "
                "by validation score before looking at test."
            ),
            pl=(
                "Porównaj model prosty, zbalansowany i zbyt elastyczny. Wybierz model "
                "po validation score, zanim spojrzysz na test."
            ),
        ),
        LocalizedText(
            en=(
                "Find a case where train score improves but validation score drops. "
                "Explain why this is overfitting."
            ),
            pl=(
                "Znajdź przypadek, w którym train score rośnie, ale validation score spada. "
                "Wyjaśnij, czemu to overfitting."
            ),
        ),
        LocalizedText(
            en=(
                "Switch to the small dataset preset and explain why validation becomes "
                "noisier, but is still safer than tuning on test."
            ),
            pl=(
                "Przełącz na mały dataset i wyjaśnij, czemu validation robi się bardziej "
                "szumne, ale nadal jest bezpieczniejsze niż strojenie na test."
            ),
        ),
    ),
    "feature_scaling_lab": (
        LocalizedText(
            en=(
                "Toggle scaling and watch how the range ratio changes. Explain why "
                "raw distance can be misleading."
            ),
            pl=(
                "Przełącz scaling i obserwuj zmianę range ratio. Wyjaśnij, czemu "
                "raw distance może wprowadzać w błąd."
            ),
        ),
        LocalizedText(
            en=(
                "Cycle through models and identify which ones are most sensitive to feature scale."
            ),
            pl=("Przełączaj modele i wskaż, które są najbardziej wrażliwe na skalę cech."),
        ),
        LocalizedText(
            en=(
                "Compare accuracy and iterations before and after scaling. Separate "
                "better geometry from a better model."
            ),
            pl=(
                "Porównaj accuracy i iterations przed oraz po scaling. Oddziel "
                "lepszą geometrię danych od lepszego modelu."
            ),
        ),
    ),
    "hyperparameter_tuning_lab": (
        LocalizedText(
            en=(
                "Move across the validation curve and identify where validation "
                "peaks before checking test."
            ),
            pl=(
                "Przesuwaj się po validation curve i znajdź miejsce, gdzie validation "
                "ma maksimum, zanim sprawdzisz test."
            ),
        ),
        LocalizedText(
            en=(
                "Find a setting with the highest train score. Explain why it is not "
                "the best tuning choice."
            ),
            pl=(
                "Znajdź ustawienie z najwyższym train score. Wyjaśnij, czemu nie jest "
                "najlepszym wyborem tuningu."
            ),
        ),
        LocalizedText(
            en=(
                "Compare underfit, candidate, and overfit regions. Name the signal "
                "that separates them."
            ),
            pl=(
                "Porównaj regiony underfit, candidate i overfit. Nazwij sygnał, który je odróżnia."
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
    "pca_lab": (
        LocalizedText(
            en=(
                "Start with fit PCA, then rotate manually. Watch when the projection "
                "keeps the trend and when residual lines grow."
            ),
            pl=(
                "Zacznij od fit PCA, potem obracaj ręcznie. Obserwuj, kiedy projekcja "
                "zachowuje trend, a kiedy residual lines robią się długie."
            ),
        ),
        LocalizedText(
            en=(
                "Switch between Linear cloud, Noisy cloud, and Two bands. Ask which "
                "patterns PCA can summarize with one line and which it flattens."
            ),
            pl=(
                "Przełączaj Linear cloud, Noisy cloud i Two bands. Sprawdź, które "
                "wzorce PCA umie streścić jedną linią, a które spłaszcza."
            ),
        ),
        LocalizedText(
            en=(
                "Increase noise and use explained variance as a warning label: high "
                "variance kept means less information lost, not zero information lost."
            ),
            pl=(
                "Zwiększ noise i traktuj explained variance jak etykietę ostrzegawczą: "
                "dużo zachowanej wariancji oznacza mniejszą stratę informacji, "
                "nie brak straty."
            ),
        ),
        LocalizedText(
            en=(
                "Toggle residual lines and connect reconstruction error with the "
                "distance from each point to its projection."
            ),
            pl=(
                "Włącz residual lines i połącz błąd rekonstrukcji z odległością "
                "każdego punktu od jego projekcji."
            ),
        ),
    ),
    "model_comparison_lab": (
        LocalizedText(
            en=(
                "Switch between Logistic Regression, k-NN, and Decision Tree. "
                "Describe how each model draws a different decision boundary "
                "for the same points."
            ),
            pl=(
                "Przełączaj Logistic Regression, k-NN i Decision Tree. Opisz, "
                "jak każdy model rysuje inną decision boundary dla tych samych punktów."
            ),
        ),
        LocalizedText(
            en=(
                "Hide the inactive boundaries, inspect one model, then turn all "
                "boundaries back on and compare the shapes again."
            ),
            pl=(
                "Ukryj nieaktywne granice, obejrzyj jeden model, a potem włącz "
                "wszystkie granice i ponownie porównaj ich kształty."
            ),
        ),
        LocalizedText(
            en=(
                "Find the model whose boundary looks simplest. Ask where that "
                "simplicity helps and where it might miss the pattern."
            ),
            pl=(
                "Znajdź model z najprostszą granicą. Zastanów się, gdzie ta prostota "
                "pomaga, a gdzie może przegapić wzorzec."
            ),
        ),
    ),
    "calibration_lab": (
        LocalizedText(
            en=(
                "Compare overconfident, underconfident, and better calibrated score "
                "sets. Ask whether high confidence really means high observed frequency."
            ),
            pl=(
                "Porównaj overconfident, underconfident i lepiej skalibrowane score. "
                "Sprawdź, czy wysokie confidence naprawdę oznacza wysoką częstość trafień."
            ),
        ),
        LocalizedText(
            en=(
                "Use the error bars to find bins where confidence and observed frequency "
                "disagree the most."
            ),
            pl=(
                "Użyj error bars, żeby znaleźć biny, w których confidence i obserwowana "
                "częstość najbardziej się rozmijają."
            ),
        ),
        LocalizedText(
            en=(
                "Compare Brier score with ECE. Notice that both are useful, but they "
                "answer slightly different questions."
            ),
            pl=(
                "Porównaj Brier score z ECE. Zobacz, że oba są przydatne, ale odpowiadają "
                "na trochę inne pytania."
            ),
        ),
    ),
    "tsne_umap_exploration_lab": (
        LocalizedText(
            en=(
                "Sketch which high-dimensional structure should stay visible after "
                "projection: local neighborhoods, clusters, or broad separation."
            ),
            pl=(
                "Naszkicuj, która struktura z wielu wymiarów powinna zostać widoczna "
                "po projekcji: lokalne sąsiedztwa, klastry czy szeroki podział."
            ),
        ),
        LocalizedText(
            en=(
                "Compare what t-SNE and UMAP are usually trusted for, and where "
                "students should avoid over-reading distances."
            ),
            pl=(
                "Porównaj, do czego zwykle ufa się t-SNE i UMAP, a gdzie studenci "
                "nie powinni nadinterpretować odległości."
            ),
        ),
        LocalizedText(
            en=(
                "Plan the first controls: dataset preset, perplexity or neighbors, "
                "random seed, and a way to compare embeddings."
            ),
            pl=(
                "Zaplanuj pierwsze kontrolki: preset danych, perplexity albo neighbors, "
                "random seed i sposób porównywania embeddingów."
            ),
        ),
        LocalizedText(
            en=(
                "Use seed drift and the raw layout to explain when a projection "
                "changed the story and when it only changed the drawing."
            ),
            pl=(
                "Użyj seed drift i raw layout, żeby wyjaśnić, kiedy projekcja "
                "zmieniła opowieść, a kiedy tylko sposób rysowania."
            ),
        ),
    ),
    "model_monitoring_drift_lab": (
        LocalizedText(
            en=(
                "Use preset 2 and decide whether data drift or metric drift becomes "
                "the lead signal. Explain why that matters before checking the metric."
            ),
            pl=(
                "Użyj presetu 2 i ustal, czy lead signal pochodzi z data drift, "
                "czy z metric drift. Wyjaśnij, czemu to ważne przed sprawdzeniem metryki."
            ),
        ),
        LocalizedText(
            en=(
                "Lower and raise the alert threshold. Watch how alert rate, persistence, "
                "severity, and the recommendation change together."
            ),
            pl=(
                "Obniż i podnieś alert threshold. Obserwuj, jak razem zmieniają się "
                "alert rate, persistence, severity i rekomendacja."
            ),
        ),
        LocalizedText(
            en=(
                "Compare the stable baseline with the single-spike preset. Decide when "
                "a trend deserves investigation and when it is only noise."
            ),
            pl=(
                "Porównaj stable baseline z presetem single spike. Ustal, kiedy trend "
                "wymaga analizy, a kiedy jest tylko szumem."
            ),
        ),
        LocalizedText(
            en=(
                "When an alert is active, press A and describe what evidence you would "
                "document before changing the production model."
            ),
            pl=(
                "Gdy alert jest aktywny, naciśnij A i opisz, jakie dowody warto "
                "zapisać przed zmianą modelu na produkcji."
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
    "linear_regression_line_fit_lab": (
        GlossaryTerm(
            term="linear regression",
            definition=LocalizedText(
                en="A regression model that predicts a numeric value with a straight line.",
                pl="Model regresji, który przewiduje wartość liczbową za pomocą prostej.",
            ),
        ),
        GlossaryTerm(
            term="residual",
            definition=LocalizedText(
                en="The vertical error between a point and the model prediction.",
                pl="Pionowy błąd między punktem a predykcją modelu.",
            ),
        ),
        GlossaryTerm(
            term="MSE loss",
            definition=LocalizedText(
                en="Mean squared error: the average squared residual across all points.",
                pl="Mean squared error: średni kwadrat residuals dla wszystkich punktów.",
            ),
        ),
        GlossaryTerm(
            term="least squares",
            definition=LocalizedText(
                en="The line that minimizes the sum of squared residuals.",
                pl="Prosta minimalizująca sumę kwadratów residuals.",
            ),
        ),
    ),
    "kmeans_intro_lab": (
        GlossaryTerm(
            term="centroid",
            definition=LocalizedText(
                en="The current center used to represent one K-Means cluster.",
                pl="Aktualny środek reprezentujący jeden klaster w K-Means.",
            ),
        ),
        GlossaryTerm(
            term="assignment",
            definition=LocalizedText(
                en="The step where each point is attached to the nearest centroid.",
                pl="Krok, w którym każdy punkt trafia do najbliższego centroidu.",
            ),
        ),
        GlossaryTerm(
            term="inertia",
            definition=LocalizedText(
                en=(
                    "The sum of squared distances from points to assigned centroids. "
                    "Lower means tighter clusters, not always better meaning."
                ),
                pl=(
                    "Suma kwadratów odległości punktów od przypisanych centroidów. "
                    "Niższa oznacza ciaśniejsze klastry, ale nie zawsze lepszy sens."
                ),
            ),
        ),
    ),
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
    "distance_metrics_lab": (
        GlossaryTerm(
            term="distance metric",
            definition=LocalizedText(
                en="A rule for turning two points into one distance number.",
                pl="Reguła zamieniająca dwa punkty w jedną liczbę distance.",
            ),
        ),
        GlossaryTerm(
            term="Euclidean distance",
            definition=LocalizedText(
                en="Straight-line distance between two points.",
                pl="Distance po linii prostej między dwoma punktami.",
            ),
        ),
        GlossaryTerm(
            term="Manhattan distance",
            definition=LocalizedText(
                en="Distance counted as horizontal plus vertical movement.",
                pl="Distance liczony jako ruch poziomy plus pionowy.",
            ),
        ),
        GlossaryTerm(
            term="nearest neighbor",
            definition=LocalizedText(
                en="The training point with the smallest distance to the query point.",
                pl="Punkt treningowy z najmniejszym distance do query point.",
            ),
        ),
    ),
    "svm_margin_lab": (
        GlossaryTerm(
            term="margin",
            definition=LocalizedText(
                en="The gap between the decision boundary and the closest training points.",
                pl="Gap między decision boundary a najbliższymi punktami treningowymi.",
            ),
        ),
        GlossaryTerm(
            term="support vector",
            definition=LocalizedText(
                en="A closest point that defines how wide the margin can be.",
                pl="Najbliższy punkt, który definiuje, jak szeroki może być margin.",
            ),
        ),
        GlossaryTerm(
            term="decision boundary",
            definition=LocalizedText(
                en="The line or surface where the model changes its predicted class.",
                pl="Linia albo powierzchnia, przy której model zmienia przewidywaną klasę.",
            ),
        ),
        GlossaryTerm(
            term="maximum margin",
            definition=LocalizedText(
                en="The widest correct separating gap the model can find.",
                pl="Najszerszy poprawny gap rozdzielający klasy.",
            ),
        ),
    ),
    "activation_functions_lab": (
        GlossaryTerm(
            term="activation function",
            definition=LocalizedText(
                en="A nonlinear function that turns a neuron input into its output.",
                pl="Nieliniowa funkcja zamieniająca input neuronu w jego output.",
            ),
        ),
        GlossaryTerm(
            term="local gradient",
            definition=LocalizedText(
                en="How strongly the activation output changes near the current input.",
                pl="To, jak mocno output activation zmienia się przy aktualnym input.",
            ),
        ),
        GlossaryTerm(
            term="saturation",
            definition=LocalizedText(
                en="A flat region where changing input barely changes output.",
                pl="Płaski region, w którym zmiana input prawie nie zmienia outputu.",
            ),
        ),
        GlossaryTerm(
            term="ReLU",
            definition=LocalizedText(
                en=(
                    "An activation that returns zero for negative input "
                    "and input itself above zero."
                ),
                pl="Activation zwracająca zero dla ujemnego input i sam input powyżej zera.",
            ),
        ),
    ),
    "neural_network_playground": (
        GlossaryTerm(
            term="forward pass",
            definition=LocalizedText(
                en="A single computation from inputs through layers to model output.",
                pl="Pojedyncze obliczenie od inputs przez warstwy do outputu modelu.",
            ),
        ),
        GlossaryTerm(
            term="hidden layer",
            definition=LocalizedText(
                en="A layer between inputs and output that builds intermediate features.",
                pl="Warstwa między inputs i output, która buduje pośrednie cechy.",
            ),
        ),
        GlossaryTerm(
            term="weight",
            definition=LocalizedText(
                en="A learned multiplier that controls how strongly one signal is used.",
                pl="Uczony mnożnik kontrolujący, jak mocno używany jest dany sygnał.",
            ),
        ),
        GlossaryTerm(
            term="bias",
            definition=LocalizedText(
                en="A learned offset added before activation.",
                pl="Uczone przesunięcie dodawane przed activation.",
            ),
        ),
        GlossaryTerm(
            term="loss",
            definition=LocalizedText(
                en="A number that says how wrong the prediction is for the current target.",
                pl="Liczba mówiąca, jak bardzo predykcja myli się dla aktualnego targetu.",
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
    "data_leakage_lab": (
        GlossaryTerm(
            term="data leakage",
            definition=LocalizedText(
                en=(
                    "When training or validation data contains information that would "
                    "not be available at prediction time."
                ),
                pl=(
                    "Sytuacja, w której dane treningowe albo walidacyjne zawierają "
                    "informacje niedostępne w czasie predykcji."
                ),
            ),
        ),
        GlossaryTerm(
            term="prediction time",
            definition=LocalizedText(
                en="The moment when the model must make a prediction in the real workflow.",
                pl="Moment, w którym model musi wykonać predykcję w prawdziwym workflow.",
            ),
        ),
        GlossaryTerm(
            term="target proxy",
            definition=LocalizedText(
                en="A feature that almost directly reveals the target label under another name.",
                pl="Cecha, która prawie bezpośrednio ujawnia target pod inną nazwą.",
            ),
        ),
        GlossaryTerm(
            term="honest validation",
            definition=LocalizedText(
                en=(
                    "Validation that uses only information available when "
                    "the model would really run."
                ),
                pl=(
                    "Walidacja używająca tylko informacji dostępnych wtedy, "
                    "gdy model naprawdę będzie działał."
                ),
            ),
        ),
    ),
    "train_validation_test_lab": (
        GlossaryTerm(
            term="train set",
            definition=LocalizedText(
                en="The data used to fit model parameters.",
                pl="Dane używane do dopasowania parametrów modelu.",
            ),
        ),
        GlossaryTerm(
            term="validation set",
            definition=LocalizedText(
                en="The data used to compare model choices during development.",
                pl="Dane używane do porównywania wyborów modelu podczas developmentu.",
            ),
        ),
        GlossaryTerm(
            term="test set",
            definition=LocalizedText(
                en=(
                    "The final held-out data used for an honest estimate after decisions are made."
                ),
                pl="Finalne odłożone dane do uczciwej oceny po podjęciu decyzji.",
            ),
        ),
        GlossaryTerm(
            term="overfitting",
            definition=LocalizedText(
                en="When train score improves by memorizing details that do not generalize.",
                pl="Gdy train score rośnie przez zapamiętywanie szczegółów bez generalizacji.",
            ),
        ),
        GlossaryTerm(
            term="model selection",
            definition=LocalizedText(
                en="Choosing the model setup using validation evidence, not the final test set.",
                pl="Wybór konfiguracji modelu na podstawie validation, a nie finalnego test set.",
            ),
        ),
    ),
    "feature_scaling_lab": (
        GlossaryTerm(
            term="feature scaling",
            definition=LocalizedText(
                en=(
                    "Transforming feature values so numeric ranges are comparable "
                    "before distance, gradient, or coefficient-based learning."
                ),
                pl=(
                    "Przekształcanie wartości cech tak, żeby zakresy liczbowe były "
                    "porównywalne przed użyciem dystansu, gradientu albo coefficients."
                ),
            ),
        ),
        GlossaryTerm(
            term="standardization",
            definition=LocalizedText(
                en="A common scaling method that centers a feature and measures it in std units.",
                pl="Popularna metoda scaling, która centruje cechę i mierzy ją w std units.",
            ),
        ),
        GlossaryTerm(
            term="range ratio",
            definition=LocalizedText(
                en="A quick comparison of the largest feature range to the smallest feature range.",
                pl="Szybkie porównanie największego zakresu cechy z najmniejszym zakresem.",
            ),
        ),
        GlossaryTerm(
            term="feature dominance",
            definition=LocalizedText(
                en="When one raw numeric range is so large that it controls distance or gradients.",
                pl="Sytuacja, gdy jeden surowy zakres liczbowy dominuje dystans albo gradienty.",
            ),
        ),
    ),
    "hyperparameter_tuning_lab": (
        GlossaryTerm(
            term="hyperparameter",
            definition=LocalizedText(
                en="A model setting chosen before or around training, not learned directly.",
                pl=(
                    "Ustawienie modelu wybierane przed treningiem albo wokół niego, "
                    "nie uczone wprost."
                ),
            ),
        ),
        GlossaryTerm(
            term="validation curve",
            definition=LocalizedText(
                en="A plot of train and validation scores across hyperparameter values.",
                pl="Wykres train i validation scores dla kolejnych wartości hyperparameter.",
            ),
        ),
        GlossaryTerm(
            term="grid search",
            definition=LocalizedText(
                en="Trying a planned set of hyperparameter values and comparing validation scores.",
                pl=(
                    "Sprawdzanie zaplanowanych wartości hyperparameter "
                    "i porównywanie validation scores."
                ),
            ),
        ),
        GlossaryTerm(
            term="overfitting",
            definition=LocalizedText(
                en="When train score keeps improving while validation performance gets worse.",
                pl="Gdy train score dalej rośnie, ale validation performance się pogarsza.",
            ),
        ),
    ),
    "class_imbalance_lab": (
        GlossaryTerm(
            term="class imbalance",
            definition=LocalizedText(
                en=(
                    "A dataset where one class is much more common than another, "
                    "so accuracy can hide minority-class mistakes."
                ),
                pl=(
                    "Dataset, w którym jedna klasa jest dużo częstsza od drugiej, "
                    "więc accuracy może ukrywać błędy klasy mniejszościowej."
                ),
            ),
        ),
        GlossaryTerm(
            term="precision",
            definition=LocalizedText(
                en="Among predicted positives, the share that really are positive.",
                pl="Wśród predykcji pozytywnych: udział tych, które naprawdę są pozytywne.",
            ),
        ),
        GlossaryTerm(
            term="recall",
            definition=LocalizedText(
                en="Among real positives, the share that the model successfully found.",
                pl="Wśród prawdziwych pozytywów: udział tych, które model odnalazł.",
            ),
        ),
        GlossaryTerm(
            term="false negative",
            definition=LocalizedText(
                en="A positive case that the model misses.",
                pl="Pozytywny przypadek, którego model nie wykrywa.",
            ),
        ),
        GlossaryTerm(
            term="decision threshold",
            definition=LocalizedText(
                en="The score cutoff used to turn probabilities into predicted labels.",
                pl="Próg score używany do zamiany probability na etykietę predykcji.",
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
    "pca_lab": (
        GlossaryTerm(
            term="PCA",
            definition=LocalizedText(
                en=(
                    "Principal Component Analysis; a linear method that finds directions "
                    "where the data varies the most."
                ),
                pl=(
                    "Principal Component Analysis; liniowa metoda szukająca kierunków, "
                    "w których dane różnią się najbardziej."
                ),
            ),
        ),
        GlossaryTerm(
            term="principal component",
            definition=LocalizedText(
                en="A new axis ordered by how much variance it explains.",
                pl="Nowa oś uporządkowana według tego, ile wariancji wyjaśnia.",
            ),
        ),
        GlossaryTerm(
            term="explained variance",
            definition=LocalizedText(
                en="The share of data spread preserved by selected PCA components.",
                pl="Część rozrzutu danych zachowana przez wybrane komponenty PCA.",
            ),
        ),
        GlossaryTerm(
            term="reconstruction error",
            definition=LocalizedText(
                en=(
                    "The part of a point that is lost when it is projected and rebuilt "
                    "from fewer dimensions."
                ),
                pl=(
                    "Część punktu utracona po projekcji i odtworzeniu go z mniejszej "
                    "liczby wymiarów."
                ),
            ),
        ),
        GlossaryTerm(
            term="residual",
            definition=LocalizedText(
                en="The visual leftover between the original point and its reconstruction.",
                pl="Wizualna reszta między oryginalnym punktem a jego rekonstrukcją.",
            ),
        ),
    ),
    "model_comparison_lab": (
        GlossaryTerm(
            term="model comparison",
            definition=LocalizedText(
                en=(
                    "A workflow for inspecting multiple models on the same data "
                    "before choosing which behavior is useful."
                ),
                pl=(
                    "Workflow porównywania kilku modeli na tych samych danych, "
                    "zanim wybierzemy zachowanie przydatne w zadaniu."
                ),
            ),
        ),
        GlossaryTerm(
            term="decision boundary",
            definition=LocalizedText(
                en="The line, curve, or region edge where a classifier changes its prediction.",
                pl=("Linia, krzywa albo krawędź regionu, w której klasyfikator zmienia predykcję."),
            ),
        ),
        GlossaryTerm(
            term="model assumption",
            definition=LocalizedText(
                en=("A built-in idea about what patterns should be easy for a model to represent."),
                pl=("Wbudowane założenie o tym, jakie wzorce model powinien łatwo reprezentować."),
            ),
        ),
        GlossaryTerm(
            term="bias / variance",
            definition=LocalizedText(
                en=(
                    "A practical trade-off between simple, stable models and flexible "
                    "models that can follow more detail."
                ),
                pl=(
                    "Praktyczny kompromis między prostymi, stabilnymi modelami "
                    "a elastycznymi modelami, które śledzą więcej szczegółów."
                ),
            ),
        ),
    ),
    "calibration_lab": (
        GlossaryTerm(
            term="calibration",
            definition=LocalizedText(
                en=(
                    "How well predicted probabilities match observed frequencies. "
                    "A calibrated 70% score should be right about 70% of the time."
                ),
                pl=(
                    "Dopasowanie predicted probabilities do obserwowanych częstości. "
                    "Skalibrowany score 70% powinien trafiać mniej więcej w 70% przypadków."
                ),
            ),
        ),
        GlossaryTerm(
            term="reliability diagram",
            definition=LocalizedText(
                en=(
                    "A plot comparing average confidence in each bin with the actual "
                    "frequency of positive outcomes."
                ),
                pl=(
                    "Wykres porównujący średnie confidence w binie z rzeczywistą "
                    "częstością pozytywnych wyników."
                ),
            ),
        ),
        GlossaryTerm(
            term="Brier score",
            definition=LocalizedText(
                en="A mean squared error for probabilistic predictions; lower is better.",
                pl="Mean squared error dla predykcji probabilistycznych; niżej znaczy lepiej.",
            ),
        ),
        GlossaryTerm(
            term="ECE",
            definition=LocalizedText(
                en="Expected Calibration Error; an average bin-level calibration gap.",
                pl="Expected Calibration Error; średnia luka kalibracji liczona po binach.",
            ),
        ),
        GlossaryTerm(
            term="worst gap",
            definition=LocalizedText(
                en=(
                    "The bin where confidence and observed frequency differ the most; "
                    "a useful place to inspect first."
                ),
                pl=(
                    "Bin, w którym confidence i obserwowana częstość różnią się najbardziej; "
                    "dobre miejsce do pierwszej inspekcji."
                ),
            ),
        ),
        GlossaryTerm(
            term="temperature scaling",
            definition=LocalizedText(
                en=(
                    "A post-hoc calibration method that softens or sharpens probability "
                    "scores while preserving their ranking."
                ),
                pl=(
                    "Metoda post-hoc calibration, która zmiękcza albo wyostrza "
                    "probability scores, zachowując ich kolejność."
                ),
            ),
        ),
        GlossaryTerm(
            term="accuracy@0.5",
            definition=LocalizedText(
                en=(
                    "Classification accuracy after converting probability scores into labels "
                    "with a 0.5 decision threshold."
                ),
                pl=(
                    "Accuracy klasyfikacji po zamianie probability scores na etykiety "
                    "przez decision threshold 0.5."
                ),
            ),
        ),
    ),
    "tsne_umap_exploration_lab": (
        GlossaryTerm(
            term="embedding",
            definition=LocalizedText(
                en=(
                    "A lower-dimensional representation that tries to preserve useful "
                    "relationships from the original feature space."
                ),
                pl=(
                    "Reprezentacja w mniejszej liczbie wymiarów, która próbuje zachować "
                    "przydatne relacje z oryginalnej przestrzeni cech."
                ),
            ),
        ),
        GlossaryTerm(
            term="t-SNE",
            definition=LocalizedText(
                en=(
                    "A nonlinear dimensionality reduction method often used to inspect "
                    "local neighborhoods in high-dimensional data."
                ),
                pl=(
                    "Nieliniowa metoda dimensionality reduction, często używana do "
                    "oglądania lokalnych sąsiedztw w danych wielowymiarowych."
                ),
            ),
        ),
        GlossaryTerm(
            term="UMAP",
            definition=LocalizedText(
                en=(
                    "A nonlinear embedding method that often preserves more global "
                    "structure than t-SNE while still focusing on neighborhoods."
                ),
                pl=(
                    "Nieliniowa metoda embeddingu, która często lepiej zachowuje "
                    "globalną strukturę niż t-SNE, nadal skupiając się na sąsiedztwach."
                ),
            ),
        ),
        GlossaryTerm(
            term="perplexity",
            definition=LocalizedText(
                en=(
                    "The t-SNE neighborhood scale. Larger values make each point listen "
                    "to more nearby points when shaping the embedding."
                ),
                pl=(
                    "Skala sąsiedztwa w t-SNE. Większe wartości sprawiają, że każdy punkt "
                    "uwzględnia więcej bliskich punktów przy budowaniu embeddingu."
                ),
            ),
        ),
        GlossaryTerm(
            term="neighbors",
            definition=LocalizedText(
                en=(
                    "The UMAP neighborhood scale. It controls how much local detail is "
                    "smoothed into broader structure."
                ),
                pl=(
                    "Skala sąsiedztwa w UMAP. Kontroluje, jak mocno lokalny detal "
                    "jest wygładzany w szerszą strukturę."
                ),
            ),
        ),
        GlossaryTerm(
            term="local trust",
            definition=LocalizedText(
                en=(
                    "A quick cue for whether nearest-neighbor links mostly stay inside "
                    "the same synthetic label."
                ),
                pl=(
                    "Szybka wskazówka, czy połączenia najbliższych sąsiadów najczęściej "
                    "zostają w tej samej syntetycznej etykiecie."
                ),
            ),
        ),
        GlossaryTerm(
            term="global spread",
            definition=LocalizedText(
                en=(
                    "A reading cue for how much of the 2D canvas the embedding uses; "
                    "it is not proof that global distances are faithful."
                ),
                pl=(
                    "Wskazówka, jak mocno embedding wykorzystuje przestrzeń 2D; "
                    "to nie jest dowód, że globalne odległości są wierne."
                ),
            ),
        ),
        GlossaryTerm(
            term="seed drift",
            definition=LocalizedText(
                en=(
                    "How far the current seed variant moved points away from the seed 0 "
                    "baseline for the same settings."
                ),
                pl=(
                    "Miara tego, jak daleko wariant seeda przesunął punkty względem "
                    "baseline seed 0 przy tych samych ustawieniach."
                ),
            ),
        ),
        GlossaryTerm(
            term="raw layout",
            definition=LocalizedText(
                en=(
                    "A simplified reference view of the toy data before comparing the "
                    "t-SNE and UMAP embeddings."
                ),
                pl=(
                    "Uproszczony widok referencyjny toy danych przed porównaniem "
                    "embeddingów t-SNE i UMAP."
                ),
            ),
        ),
    ),
    "model_monitoring_drift_lab": (
        GlossaryTerm(
            term="data drift",
            definition=LocalizedText(
                en=(
                    "A change in the input data distribution after deployment; "
                    "it can make old model assumptions stale."
                ),
                pl=(
                    "Zmiana rozkładu danych wejściowych po wdrożeniu; może sprawić, "
                    "że stare założenia modelu przestaną pasować."
                ),
            ),
        ),
        GlossaryTerm(
            term="metric drift",
            definition=LocalizedText(
                en=(
                    "A change in model quality metrics over time, such as accuracy, "
                    "error rate, calibration, or latency."
                ),
                pl=(
                    "Zmiana metryk jakości modelu w czasie, na przykład accuracy, "
                    "error rate, calibration albo latency."
                ),
            ),
        ),
        GlossaryTerm(
            term="monitoring window",
            definition=LocalizedText(
                en=(
                    "The recent slice of production data used to compare current behavior "
                    "with a baseline period."
                ),
                pl=(
                    "Najnowszy wycinek danych produkcyjnych używany do porównania "
                    "obecnego zachowania z okresem baseline."
                ),
            ),
        ),
        GlossaryTerm(
            term="alert threshold",
            definition=LocalizedText(
                en=(
                    "A boundary where a monitored signal becomes large enough to deserve "
                    "attention or investigation."
                ),
                pl=(
                    "Granica, przy której monitorowany sygnał jest na tyle duży, "
                    "że warto go sprawdzić."
                ),
            ),
        ),
        GlossaryTerm(
            term="first alert",
            definition=LocalizedText(
                en=(
                    "The first time step where a monitored signal crosses the alert "
                    "threshold and deserves a closer look."
                ),
                pl=(
                    "Pierwszy moment, w którym monitorowany sygnał przekracza alert "
                    "threshold i warto mu się przyjrzeć."
                ),
            ),
        ),
        GlossaryTerm(
            term="alert count",
            definition=LocalizedText(
                en=("The number of time-series points that cross the current alert threshold."),
                pl=(
                    "Liczba punktów szeregu czasowego, które przekraczają aktualny alert threshold."
                ),
            ),
        ),
        GlossaryTerm(
            term="alert rate",
            definition=LocalizedText(
                en="The share of time-series points that cross the current alert threshold.",
                pl=(
                    "Udział punktów szeregu czasowego, które przekraczają aktualny alert threshold."
                ),
            ),
        ),
        GlossaryTerm(
            term="persistence",
            definition=LocalizedText(
                en=(
                    "A simple hint that an alert is repeated across enough recent "
                    "points to look more persistent than a one-off spike."
                ),
                pl=(
                    "Prosta wskazówka, że alert powtarza się w tylu ostatnich punktach, "
                    "że wygląda bardziej trwale niż jak pojedynczy pik."
                ),
            ),
        ),
        GlossaryTerm(
            term="spike",
            definition=LocalizedText(
                en=(
                    "A short, sharp threshold crossing that may deserve a look, "
                    "but should not automatically be treated as sustained drift."
                ),
                pl=(
                    "Krótki, ostry skok ponad threshold, któremu warto się przyjrzeć, "
                    "ale nie należy go automatycznie traktować jak trwały drift."
                ),
            ),
        ),
        GlossaryTerm(
            term="lead signal",
            definition=LocalizedText(
                en=(
                    "The monitored signal that crosses the alert threshold first in "
                    "the current scenario."
                ),
                pl=(
                    "Monitorowany sygnał, który w danym scenariuszu jako pierwszy "
                    "przekracza alert threshold."
                ),
            ),
        ),
        GlossaryTerm(
            term="gap",
            definition=LocalizedText(
                en=("The difference between the current window mean and the baseline window mean."),
                pl=("Różnica między średnią current window a średnią baseline window."),
            ),
        ),
        GlossaryTerm(
            term="trend",
            definition=LocalizedText(
                en=(
                    "The direction of change between the baseline window mean "
                    "and the current window mean."
                ),
                pl=("Kierunek zmiany między średnią baseline window a średnią current window."),
            ),
        ),
        GlossaryTerm(
            term="threshold margin",
            definition=LocalizedText(
                en=("How far the current gap sits below or above the selected alert threshold."),
                pl=(
                    "Informacja, jak daleko obecna luka jest poniżej albo powyżej "
                    "wybranego alert threshold."
                ),
            ),
        ),
        GlossaryTerm(
            term="severity",
            definition=LocalizedText(
                en=(
                    "A rough label for how far the current window has moved away from "
                    "baseline after an alert fires."
                ),
                pl=(
                    "Prosta etykieta mówiąca, jak mocno current window odjechało "
                    "od baseline po uruchomieniu alertu."
                ),
            ),
        ),
        GlossaryTerm(
            term="recommendation",
            definition=LocalizedText(
                en=(
                    "A suggested next step that turns a monitoring signal into an "
                    "investigation workflow."
                ),
                pl=(
                    "Sugestia następnego kroku, która zamienia sygnał monitoringu "
                    "w workflow analizy."
                ),
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
        demo_id="linear_regression_line_fit_lab",
        level=1,
        title_en="Linear Regression Line Fit Lab",
        title_pl="Linear Regression Line Fit Lab",
        summary_en="Fit a straight line by adjusting slope and intercept, then compare residuals.",
        summary_pl=("Dopasuj prostą przez zmianę slope i intercept, a potem porównaj residuals."),
        objectives=(
            LocalizedText(
                en="Connect slope and intercept with the visible position of a regression line.",
                pl="Połącz slope i intercept z widocznym położeniem prostej regresji.",
            ),
            LocalizedText(
                en="Use residuals and MSE loss to judge whether a line fits the data.",
                pl="Używaj residuals i MSE loss do oceny, czy prosta pasuje do danych.",
            ),
            LocalizedText(
                en="Compare a manual line with the least-squares fit.",
                pl="Porównaj ręcznie ustawioną prostą z least-squares fit.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch dataset", pl="zmień dataset"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(
                    en="decrease or increase slope", pl="zmniejsz albo zwiększ slope"
                ),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(
                    en="raise or lower intercept",
                    pl="podnieś albo opuść intercept",
                ),
            ),
            ControlBinding(
                key="F",
                action=LocalizedText(
                    en="jump to the least-squares fit",
                    pl="przejdź do least-squares fit",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the lab", pl="zresetuj lab"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read linear regression notes",
                    pl="przeczytaj notatki o linear regression",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_linear_regression_line_fit_lab_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("regression", "loss", "residuals", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Linear regression fits a line that predicts a numeric target "
                                "from one input feature."
                            ),
                            pl=(
                                "Linear regression dopasowuje prostą, która przewiduje liczbowy "
                                "target z jednej cechy wejściowej."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Slope controls the tilt. Intercept controls how high or low "
                                "the line sits."
                            ),
                            pl=(
                                "Slope steruje nachyleniem. Intercept przesuwa prostą "
                                "wyżej albo niżej."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Residuals are the vertical distances from points to the line. "
                                "Large residuals mean large mistakes."
                            ),
                            pl=(
                                "Residuals to pionowe odległości punktów od prostej. "
                                "Duże residuals oznaczają duże błędy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Least squares chooses the line with the smallest squared "
                                "residuals overall."
                            ),
                            pl=(
                                "Least squares wybiera prostą z najmniejszym łącznym "
                                "kwadratem residuals."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not fit one point perfectly if it makes every other "
                                "point worse."
                            ),
                            pl=(
                                "Nie dopasowuj idealnie jednego punktu, jeśli psuje to "
                                "większość pozostałych."
                            ),
                        ),
                        LocalizedText(
                            en=("Noise means even the best simple line can leave visible errors."),
                            pl=(
                                "Noise oznacza, że nawet najlepsza prosta może zostawić "
                                "widoczne błędy."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["linear_regression_line_fit_lab"],
            glossary=LESSON_GLOSSARY["linear_regression_line_fit_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="kmeans_intro_lab",
        level=1,
        title_en="K-Means Intro Lab",
        title_pl="K-Means Intro Lab",
        summary_en="Step through assignments, centroid updates, k, and inertia.",
        summary_pl="Przejdź przez assignments, aktualizacje centroidów, k i inertia.",
        objectives=(
            LocalizedText(
                en="See K-Means as two repeating steps: assign points, then move centroids.",
                pl=(
                    "Zobacz K-Means jako dwa powtarzające się kroki: "
                    "przypisz punkty, potem przesuń centroidy."
                ),
            ),
            LocalizedText(
                en="Change k and notice when compact-looking groups stop matching the data.",
                pl=("Zmieniaj k i obserwuj, kiedy zwarte grupy przestają pasować do danych."),
            ),
            LocalizedText(
                en="Use inertia as a useful signal, not as the only definition of a good cluster.",
                pl=(
                    "Używaj inertia jako przydatnego sygnału, ale nie jako jedynej "
                    "definicji dobrego klastra."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="Space",
                action=LocalizedText(
                    en="run one assignment or centroid-update step",
                    pl="wykonaj jeden krok assignment albo update centroidów",
                ),
            ),
            ControlBinding(
                key="A",
                action=LocalizedText(
                    en="start or pause auto-run",
                    pl="uruchom albo zatrzymaj auto-run",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(en="decrease or increase k", pl="zmniejsz albo zwiększ k"),
            ),
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch dataset", pl="zmień dataset"),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="toggle point-to-centroid links",
                    pl="pokaż albo ukryj linie punkt-centroid",
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(en="generate a new sample", pl="wygeneruj nową próbkę"),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the lab", pl="zresetuj lab"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read K-Means notes",
                    pl="przeczytaj notatki o K-Means",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_kmeans_intro_lab_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("clustering", "k-means", "unsupervised", "inertia", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "K-Means is an unsupervised algorithm: it groups points "
                                "without class labels."
                            ),
                            pl=(
                                "K-Means jest algorytmem unsupervised: grupuje punkty "
                                "bez etykiet klas."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The algorithm alternates between assigning points to the "
                                "nearest centroid and moving each centroid to the mean of "
                                "its assigned points."
                            ),
                            pl=(
                                "Algorytm przełącza się między przypisaniem punktów do "
                                "najbliższego centroidu i przesunięciem każdego centroidu "
                                "do średniej przypisanych punktów."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Changing k changes the question. K-Means will always create "
                                "k clusters, even when that number is not meaningful."
                            ),
                            pl=(
                                "Zmiana k zmienia pytanie. K-Means zawsze utworzy k klastrów, "
                                "nawet kiedy taka liczba nie ma sensu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Inertia often drops as centroids move, but a lower value can "
                                "come from splitting a natural group."
                            ),
                            pl=(
                                "Inertia zwykle spada, gdy centroidy się przesuwają, ale niższa "
                                "wartość może wynikać z podzielenia naturalnej grupy."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not read cluster colors as true labels. They are only "
                                "the current K-Means assignments."
                            ),
                            pl=(
                                "Nie traktuj kolorów klastrów jak prawdziwych etykiet. "
                                "To tylko aktualne assignments z K-Means."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "K-Means likes compact, roughly round groups. Long or curved "
                                "shapes can be misleading."
                            ),
                            pl=(
                                "K-Means lubi zwarte, dość okrągłe grupy. Długie albo "
                                "zakrzywione kształty mogą go mylić."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["kmeans_intro_lab"],
            glossary=LESSON_GLOSSARY["kmeans_intro_lab"],
        ),
    ),
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
        demo_id="distance_metrics_lab",
        level=1,
        title_en="Distance Metrics Lab",
        title_pl="Distance Metrics Lab",
        summary_en=(
            "Move a query point and compare Euclidean, Manhattan, and Chebyshev nearest neighbors."
        ),
        summary_pl=(
            "Przesuwaj query point i porównuj nearest neighbors dla Euclidean, "
            "Manhattan i Chebyshev."
        ),
        objectives=(
            LocalizedText(
                en="See how a distance metric turns feature differences into one number.",
                pl="Zobacz, jak metryka distance zamienia różnice cech w jedną liczbę.",
            ),
            LocalizedText(
                en="Compare nearest neighbors under Euclidean, Manhattan, and Chebyshev distance.",
                pl="Porównaj nearest neighbors dla Euclidean, Manhattan i Chebyshev distance.",
            ),
            LocalizedText(
                en="Connect distance metrics with k-NN classification intuition.",
                pl="Połącz metryki distance z intuicją stojącą za k-NN classification.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch dataset", pl="zmień dataset"),
            ),
            ControlBinding(
                key="Arrow keys",
                action=LocalizedText(en="move the query point", pl="przesuń query point"),
            ),
            ControlBinding(
                key="M",
                action=LocalizedText(en="cycle distance metric", pl="zmień metrykę distance"),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the lab", pl="zresetuj lab"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read distance metric notes",
                    pl="przeczytaj notatki o metrykach distance",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_distance_metrics_lab_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("distance", "knn", "classification", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Many ML methods need a way to decide whether two points are close."
                            ),
                            pl=(
                                "Wiele metod ML potrzebuje sposobu na ocenę, czy dwa punkty "
                                "są blisko."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A distance metric is that rule. Different rules can disagree "
                                "near ambiguous points."
                            ),
                            pl=(
                                "Metryka distance jest taką regułą. Różne reguły mogą się "
                                "nie zgadzać przy niejednoznacznych punktach."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Euclidean measures straight-line closeness. Manhattan counts "
                                "axis-aligned movement."
                            ),
                            pl=(
                                "Euclidean mierzy bliskość po prostej. Manhattan liczy ruch "
                                "wzdłuż osi."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The nearest neighbor is not a universal fact; it depends on "
                                "the chosen metric."
                            ),
                            pl=(
                                "Nearest neighbor nie jest faktem absolutnym; zależy od "
                                "wybranej metryki."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not assume distance is neutral. The metric encodes what "
                                "kind of difference matters."
                            ),
                            pl=(
                                "Nie zakładaj, że distance jest neutralny. Metryka koduje, "
                                "jaki rodzaj różnicy ma znaczenie."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Before using k-NN, ask whether the feature geometry matches "
                                "the problem."
                            ),
                            pl=(
                                "Przed użyciem k-NN zapytaj, czy geometria cech pasuje do problemu."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["distance_metrics_lab"],
            glossary=LESSON_GLOSSARY["distance_metrics_lab"],
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
        demo_id="svm_margin_lab",
        level=1,
        title_en="SVM Margin Lab",
        title_pl="SVM Margin Lab",
        summary_en=(
            "Rotate a decision boundary and see how support vectors define the widest margin."
        ),
        summary_pl=(
            "Obracaj decision boundary i zobacz, jak support vectors definiują najszerszy margin."
        ),
        objectives=(
            LocalizedText(
                en="Separate two classes with a linear decision boundary.",
                pl="Rozdziel dwie klasy liniową decision boundary.",
            ),
            LocalizedText(
                en="Watch how margin changes as the boundary rotates or shifts.",
                pl="Obserwuj, jak margin zmienia się przy obrocie albo przesunięciu boundary.",
            ),
            LocalizedText(
                en="Identify support vectors as the points closest to the boundary.",
                pl="Rozpoznaj support vectors jako punkty najbliższe boundary.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch dataset", pl="zmień dataset"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="rotate boundary", pl="obróć boundary"),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="shift boundary", pl="przesuń boundary"),
            ),
            ControlBinding(
                key="F",
                action=LocalizedText(
                    en="jump to a wide-margin fit",
                    pl="przejdź do wide-margin fit",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the lab", pl="zresetuj lab"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read SVM margin notes",
                    pl="przeczytaj notatki o SVM margin",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_svm_margin_lab_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("classification", "svm", "margin", "boundary", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "An SVM tries to separate classes with a boundary that leaves "
                                "a wide safe gap."
                            ),
                            pl=(
                                "SVM próbuje rozdzielić klasy boundary, która zostawia "
                                "szeroki bezpieczny gap."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Only the closest points define that gap. "
                                "These are support vectors."
                            ),
                            pl=("Ten gap definiują najbliższe punkty. To właśnie support vectors."),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "A boundary can classify all points correctly and still have "
                                "a narrow margin."
                            ),
                            pl=(
                                "Boundary może klasyfikować wszystkie punkty poprawnie, "
                                "a mimo to mieć wąski margin."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Moving far-away points does not matter as much as moving "
                                "support vectors."
                            ),
                            pl=(
                                "Przesunięcie dalekich punktów ma mniejsze znaczenie niż "
                                "przesunięcie support vectors."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not stop at any correct split. SVM prefers the correct split "
                                "with the widest margin."
                            ),
                            pl=(
                                "Nie zatrzymuj się na dowolnym poprawnym podziale. SVM woli "
                                "poprawny podział z najszerszym margin."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "If the classes are not separable, hard-margin intuition is only "
                                "the first step."
                            ),
                            pl=(
                                "Jeśli klas nie da się rozdzielić idealnie, intuicja hard-margin "
                                "jest tylko pierwszym krokiem."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["svm_margin_lab"],
            glossary=LESSON_GLOSSARY["svm_margin_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="activation_functions_lab",
        level=1,
        title_en="Activation Functions Lab",
        title_pl="Activation Functions Lab",
        summary_en=(
            "Compare sigmoid, tanh, and ReLU curves to see output ranges and local gradients."
        ),
        summary_pl=(
            "Porównuj krzywe sigmoid, tanh i ReLU, żeby zobaczyć zakresy outputu "
            "oraz local gradients."
        ),
        objectives=(
            LocalizedText(
                en="See how an activation function transforms one neuron input.",
                pl="Zobacz, jak activation function przekształca pojedynczy input neuronu.",
            ),
            LocalizedText(
                en="Compare output ranges for sigmoid, tanh, and ReLU.",
                pl="Porównaj zakresy outputu dla sigmoid, tanh i ReLU.",
            ),
            LocalizedText(
                en="Notice saturation and local gradient flow.",
                pl="Zauważ saturation i przepływ local gradient.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch activation", pl="zmień activation"),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(en="move input x", pl="przesuń input x"),
            ),
            ControlBinding(
                key="0",
                action=LocalizedText(en="reset x to zero", pl="zresetuj x do zera"),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the lab", pl="zresetuj lab"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read activation function notes",
                    pl="przeczytaj notatki o activation functions",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_activation_functions_lab_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("neural-networks", "activation", "gradient", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "A neural network layer usually applies an activation after "
                                "a weighted sum."
                            ),
                            pl=(
                                "Warstwa neural network zwykle stosuje activation po "
                                "ważonej sumie wejść."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The activation changes both the output signal and the gradient "
                                "used for learning."
                            ),
                            pl=(
                                "Activation zmienia zarówno output signal, jak i gradient "
                                "używany do uczenia."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Sigmoid and tanh flatten at the edges, so their local gradient "
                                "gets tiny."
                            ),
                            pl=(
                                "Sigmoid i tanh wypłaszczają się na krańcach, więc ich local "
                                "gradient robi się bardzo mały."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "ReLU is simple and keeps gradient on the positive side, but "
                                "negative input gives zero output."
                            ),
                            pl=(
                                "ReLU jest proste i utrzymuje gradient po dodatniej stronie, "
                                "ale ujemny input daje output zero."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "An activation is not just decoration; it shapes what the neuron "
                                "can express."
                            ),
                            pl=(
                                "Activation nie jest ozdobą; wpływa na to, co neuron może wyrazić."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Flat regions can slow learning because gradients carry "
                                "less signal."
                            ),
                            pl=(
                                "Płaskie regiony mogą spowalniać uczenie, bo gradients niosą "
                                "mniej sygnału."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["activation_functions_lab"],
            glossary=LESSON_GLOSSARY["activation_functions_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="neural_network_playground",
        level=1,
        title_en="Neural Network Playground",
        title_pl="Neural Network Playground",
        summary_en=("Follow inputs through a tiny hidden layer to probability, target, and loss."),
        summary_pl=("Prześledź inputs przez małą hidden layer aż do probability, targetu i loss."),
        objectives=(
            LocalizedText(
                en="Trace one forward pass through inputs, hidden units, activation, and output.",
                pl="Prześledź forward pass przez inputs, hidden units, activation i output.",
            ),
            LocalizedText(
                en="See how weights and hidden bias change probability and loss.",
                pl="Zobacz, jak weights i hidden bias zmieniają probability oraz loss.",
            ),
            LocalizedText(
                en="Connect activation choice with hidden-layer behavior.",
                pl="Połącz wybór activation z zachowaniem hidden layer.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch input example", pl="zmień przykład input"),
            ),
            ControlBinding(
                key="A",
                action=LocalizedText(en="cycle activation", pl="zmień activation"),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(en="change weight scale", pl="zmień weight scale"),
            ),
            ControlBinding(
                key="Up / Down",
                action=LocalizedText(en="change hidden bias", pl="zmień hidden bias"),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the playground", pl="zresetuj playground"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read neural network notes",
                    pl="przeczytaj notatki o neural networks",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_neural_network_playground_scene,
        difficulty=LocalizedText(en="Introductory", pl="Wprowadzający"),
        tags=("neural-networks", "forward-pass", "classification", "loss", "level-1"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "A neural network is a chain of simple computations: inputs, "
                                "weights, bias, activation, and output."
                            ),
                            pl=(
                                "Neural network to łańcuch prostych obliczeń: inputs, "
                                "weights, bias, activation i output."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "This playground follows one forward pass through a tiny "
                                "hidden layer."
                            ),
                            pl=(
                                "Ten playground śledzi jeden forward pass przez małą hidden layer."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Weights decide which input signals matter. Bias shifts "
                                "hidden units before activation."
                            ),
                            pl=(
                                "Weights decydują, które sygnały wejściowe są ważne. "
                                "Bias przesuwa hidden units przed activation."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The output probability is useful only when you compare it "
                                "with the target and loss."
                            ),
                            pl=(
                                "Output probability ma sens dopiero w porównaniu z targetem i loss."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not treat the network as magic. Every arrow and node is "
                                "part of a calculation."
                            ),
                            pl=(
                                "Nie traktuj sieci jak magii. Każda strzałka i każdy neuron "
                                "są częścią obliczenia."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "High probability is not automatically good; it is good only "
                                "when it points toward the target."
                            ),
                            pl=(
                                "Wysokie probability nie jest automatycznie dobre; jest dobre "
                                "tylko wtedy, gdy wskazuje target."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["neural_network_playground"],
            glossary=LESSON_GLOSSARY["neural_network_playground"],
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
        demo_id="data_leakage_lab",
        level=2,
        title_en="Data Leakage Lab",
        title_pl="Data Leakage Lab",
        summary_en=(
            "Toggle suspicious features to see why perfect validation scores "
            "can be a data problem, not a great model."
        ),
        summary_pl=(
            "Przełączaj podejrzane cechy i zobacz, czemu idealne validation scores "
            "mogą oznaczać problem z danymi, a nie świetny model."
        ),
        objectives=(
            LocalizedText(
                en="Compare leaky and cleaned validation results on the same scenario.",
                pl="Porównaj leaky i cleaned validation results w tym samym scenariuszu.",
            ),
            LocalizedText(
                en="Identify whether a feature would exist at prediction time.",
                pl="Ustal, czy cecha istniałaby w czasie predykcji.",
            ),
            LocalizedText(
                en="Treat suspiciously high scores as a prompt to inspect the dataset.",
                pl="Traktuj podejrzanie wysokie wyniki jako sygnał do sprawdzenia datasetu.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch leakage scenario", pl="zmien scenariusz leakage"),
            ),
            ControlBinding(
                key="L",
                action=LocalizedText(
                    en="toggle suspicious leakage feature",
                    pl="włącz albo usuń podejrzaną cechę leakage",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the preview", pl="zresetuj podgląd"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read data leakage notes", pl="przeczytaj notatki o leakage"
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_data_leakage_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("data-quality", "validation", "leakage", "level-2"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Data leakage happens when the model sees information "
                                "that would not exist at prediction time."
                            ),
                            pl=(
                                "Data leakage pojawia sie wtedy, gdy model widzi informacje, "
                                "których nie będzie w czasie predykcji."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The leaky score can look excellent because the suspicious "
                                "feature already contains part of the answer."
                            ),
                            pl=(
                                "Leaky score może wyglądać świetnie, bo podejrzana cecha "
                                "zawiera już część odpowiedzi."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwowac"),
                    body=(
                        LocalizedText(
                            en=(
                                "When leakage is enabled, train and test accuracy are both "
                                "nearly perfect, so the small gap is not reassuring."
                            ),
                            pl=(
                                "Gdy leakage jest włączone, train i test accuracy są prawie "
                                "idealne, więc mały gap nie powinien uspokajać."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "After removing the leakage feature, the score drops, "
                                "but the validation becomes more honest."
                            ),
                            pl=(
                                "Po usunięciu cechy leakage wynik spada, ale walidacja "
                                "staje się uczciwsza."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not trust a perfect metric until you check when each "
                                "feature is created in the real workflow."
                            ),
                            pl=(
                                "Nie ufaj idealnej metryce, dopóki nie sprawdzisz, kiedy "
                                "każda cecha powstaje w prawdziwym workflow."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A feature can leak even if it has a harmless technical name, "
                                "such as a timestamp or a future aggregate."
                            ),
                            pl=(
                                "Cecha może leakować nawet wtedy, gdy ma niewinną nazwę "
                                "techniczną, jak timestamp albo future aggregate."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["data_leakage_lab"],
            glossary=LESSON_GLOSSARY["data_leakage_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="train_validation_test_lab",
        level=2,
        title_en="Train / Validation / Test Split Lab",
        title_pl="Train / Validation / Test Split Lab",
        summary_en=(
            "Choose model complexity with validation scores, then treat the test "
            "score as the final honest estimate."
        ),
        summary_pl=(
            "Wybieraj complexity modelu na podstawie validation score, a test score "
            "traktuj jako finalną uczciwą ocenę."
        ),
        objectives=(
            LocalizedText(
                en="Compare train, validation, and test scores across model complexity.",
                pl="Porównuj train, validation i test scores przy różnych complexity.",
            ),
            LocalizedText(
                en="Recognize overfitting from a large train-validation gap.",
                pl="Rozpoznawaj overfitting po dużym train-validation gap.",
            ),
            LocalizedText(
                en="Use validation for model selection and keep test for the final check.",
                pl="Używaj validation do wyboru modelu, a test zostaw na finalny check.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch split scenario", pl="zmień scenariusz splitu"),
            ),
            ControlBinding(
                key="- / = / 0",
                action=LocalizedText(
                    en="decrease, increase, or reset model complexity",
                    pl="zmniejsz, zwiększ albo zresetuj complexity modelu",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the preview", pl="zresetuj podgląd"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read train/validation/test notes",
                    pl="przeczytaj notatki o train/validation/test",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_train_validation_test_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("validation", "model-selection", "overfitting", "level-2"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Train data fits the model. Validation data helps choose "
                                "between model setups. Test data waits until the end."
                            ),
                            pl=(
                                "Train data dopasowuje model. Validation data pomaga wybrać "
                                "konfigurację. Test data czeka do końca."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "If you tune repeatedly on test, the test set stops being "
                                "an honest estimate of future performance."
                            ),
                            pl=(
                                "Jeśli stroisz model na test, test set przestaje być "
                                "uczciwą oceną przyszłej jakości."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "A too-simple model has weak train and validation scores, "
                                "so it is underfitting."
                            ),
                            pl=(
                                "Zbyt prosty model ma słaby train i validation score, "
                                "więc jest underfitting."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A too-flexible model can reach a high train score while "
                                "validation drops."
                            ),
                            pl=(
                                "Zbyt elastyczny model może mieć wysoki train score, "
                                "gdy validation spada."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not pick the model with the best test score after trying "
                                "many options; that leaks test feedback into model selection."
                            ),
                            pl=(
                                "Nie wybieraj modelu po najlepszym test score po wielu próbach; "
                                "to wpuszcza feedback z testu do model selection."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A small dataset makes validation noisier, but it is still "
                                "safer than tuning on the final test set."
                            ),
                            pl=(
                                "Mały dataset sprawia, że validation jest bardziej szumne, "
                                "ale nadal jest bezpieczniejsze niż strojenie na test set."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["train_validation_test_lab"],
            glossary=LESSON_GLOSSARY["train_validation_test_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="feature_scaling_lab",
        level=2,
        title_en="Feature Scaling Lab",
        title_pl="Feature Scaling Lab",
        summary_en=(
            "Toggle scaling to see how raw feature ranges distort distance, gradients, "
            "and practical model comparison."
        ),
        summary_pl=(
            "Przełączaj scaling i zobacz, jak surowe zakresy cech zniekształcają "
            "dystans, gradienty i praktyczne porównanie modeli."
        ),
        objectives=(
            LocalizedText(
                en="Compare raw and scaled feature ranges on the same dataset.",
                pl="Porównuj raw i scaled feature ranges na tym samym datasecie.",
            ),
            LocalizedText(
                en="See which models are sensitive to numeric scale.",
                pl="Zobacz, które modele są wrażliwe na skalę liczbową.",
            ),
            LocalizedText(
                en="Use range ratio, accuracy, and iteration count to reason about scaling.",
                pl="Używaj range ratio, accuracy i iterations do analizy scaling.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch scaling scenario", pl="zmień scenariusz scaling"),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(en="toggle feature scaling", pl="włącz albo wyłącz scaling"),
            ),
            ControlBinding(
                key="M",
                action=LocalizedText(en="cycle model", pl="zmień model"),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the preview", pl="zresetuj podgląd"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read feature scaling notes",
                    pl="przeczytaj notatki o feature scaling",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_feature_scaling_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("preprocessing", "scaling", "distance", "optimization", "level-2"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Many ML algorithms read numeric size literally. A feature "
                                "with a huge range can dominate distance or gradient steps."
                            ),
                            pl=(
                                "Wiele algorytmów ML czyta wielkość liczbową dosłownie. "
                                "Cecha o ogromnym zakresie może zdominować dystans albo gradient."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Scaling does not add information; it changes the geometry "
                                "so models can compare features more fairly."
                            ),
                            pl=(
                                "Scaling nie dodaje informacji; zmienia geometrię danych, "
                                "żeby modele uczciwiej porównywały cechy."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "k-NN changes because distance changes. Gradient Descent "
                                "often needs fewer iterations after scaling."
                            ),
                            pl=(
                                "k-NN zmienia się, bo zmienia się dystans. Gradient Descent "
                                "często potrzebuje mniej iterations po scaling."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Logistic Regression coefficients are easier to compare "
                                "when features share a similar scale."
                            ),
                            pl=(
                                "Logistic Regression coefficients łatwiej porównywać, "
                                "gdy cechy mają podobną skalę."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not compare raw coefficients across features measured "
                                "in different units."
                            ),
                            pl=(
                                "Nie porównuj raw coefficients między cechami mierzonymi "
                                "w różnych jednostkach."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Scaling should be fitted on train data and reused on validation, "
                                "test, and production data."
                            ),
                            pl=(
                                "Scaling dopasuj na train data i użyj tej samej transformacji "
                                "na validation, test oraz production data."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["feature_scaling_lab"],
            glossary=LESSON_GLOSSARY["feature_scaling_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="hyperparameter_tuning_lab",
        level=2,
        title_en="Hyperparameter Tuning Lab",
        title_pl="Hyperparameter Tuning Lab",
        summary_en=(
            "Use validation curves to choose k, max depth, or regularization "
            "without chasing train score."
        ),
        summary_pl=(
            "Używaj validation curves do wyboru k, max depth albo regularization "
            "bez gonienia za train score."
        ),
        objectives=(
            LocalizedText(
                en="Compare train, validation, and test scores across parameter values.",
                pl="Porównuj train, validation i test scores dla kolejnych wartości parametru.",
            ),
            LocalizedText(
                en="Pick the setting with the best validation score, not best train score.",
                pl="Wybieraj ustawienie z najlepszym validation score, nie train score.",
            ),
            LocalizedText(
                en="Recognize underfit and overfit regions on a validation curve.",
                pl="Rozpoznawaj regiony underfit i overfit na validation curve.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(en="switch tuning scenario", pl="zmień scenariusz tuningu"),
            ),
            ControlBinding(
                key="- / = / 0",
                action=LocalizedText(
                    en="decrease, increase, or reset parameter value",
                    pl="zmniejsz, zwiększ albo zresetuj wartość parametru",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the preview", pl="zresetuj podgląd"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read hyperparameter tuning notes",
                    pl="przeczytaj notatki o hyperparameter tuning",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_hyperparameter_tuning_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("validation", "tuning", "overfitting", "model-selection", "level-2"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Hyperparameters shape the model before it learns. "
                                "You choose them by comparing validation evidence."
                            ),
                            pl=(
                                "Hyperparameters kształtują model przed uczeniem. "
                                "Wybiera się je przez porównanie validation evidence."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Train score often rewards complexity, so validation is "
                                "the safer signal for model selection."
                            ),
                            pl=(
                                "Train score często nagradza complexity, więc validation "
                                "jest bezpieczniejszym sygnałem do model selection."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "The best validation point is usually between underfit "
                                "and overfit regions."
                            ),
                            pl=(
                                "Najlepszy punkt validation zwykle leży między regionami "
                                "underfit i overfit."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Use test only after choosing the hyperparameter, otherwise "
                                "test feedback leaks into tuning."
                            ),
                            pl=(
                                "Używaj test dopiero po wyborze hyperparameter, inaczej "
                                "feedback z testu przecieka do tuningu."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not pick the point with the highest train score; that "
                                "often chooses the most overfit setup."
                            ),
                            pl=(
                                "Nie wybieraj punktu z najwyższym train score; to często "
                                "wybiera najbardziej overfit setup."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A grid search is only as honest as the validation split "
                                "and the final untouched test set."
                            ),
                            pl=(
                                "Grid search jest tak uczciwy, jak validation split "
                                "i finalny nietknięty test set."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["hyperparameter_tuning_lab"],
            glossary=LESSON_GLOSSARY["hyperparameter_tuning_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="class_imbalance_lab",
        level=2,
        title_en="Class Imbalance Lab",
        title_pl="Class Imbalance Lab",
        summary_en=(
            "Compare accuracy, precision, recall, false negatives, and threshold "
            "trade-offs when one class is rare."
        ),
        summary_pl=(
            "Porównuj accuracy, precision, recall, false negatives i threshold "
            "trade-offs, gdy jedna klasa jest rzadka."
        ),
        objectives=(
            LocalizedText(
                en="See how high accuracy can hide weak recall on the minority class.",
                pl="Zobacz, jak wysokie accuracy może ukrywać słaby recall klasy mniejszościowej.",
            ),
            LocalizedText(
                en="Move the decision threshold and compare false positives with false negatives.",
                pl="Przesuwaj decision threshold i porównuj false positives z false negatives.",
            ),
            LocalizedText(
                en="Choose a threshold based on the cost of mistakes, not accuracy alone.",
                pl="Wybieraj threshold według kosztu błędów, nie tylko według accuracy.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(
                    en="switch imbalance scenario", pl="zmień scenariusz imbalance"
                ),
            ),
            ControlBinding(
                key="- / = / 0",
                action=LocalizedText(
                    en="decrease, increase, or reset decision threshold",
                    pl="zmniejsz, zwiększ albo zresetuj decision threshold",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(en="reset the preview", pl="zresetuj podgląd"),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read class imbalance notes",
                    pl="przeczytaj notatki o class imbalance",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_class_imbalance_lab_scene,
        difficulty=LocalizedText(en="Practical", pl="Praktyczny"),
        tags=("classification", "metrics", "imbalance", "threshold", "level-2"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="What this demo shows", pl="Co pokazuje to demo"),
                    body=(
                        LocalizedText(
                            en=(
                                "Class imbalance means one label is much more common "
                                "than the label you often care about most."
                            ),
                            pl=(
                                "Class imbalance oznacza, że jedna etykieta jest dużo częstsza "
                                "niż klasa, na której zwykle najbardziej nam zależy."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A model can get many easy negatives right and still miss "
                                "too many rare positives."
                            ),
                            pl=(
                                "Model może poprawnie klasyfikować wiele łatwych negatywów, "
                                "a nadal przeoczyć zbyt wiele rzadkich pozytywów."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to notice", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Accuracy answers a broad question, but recall asks whether "
                                "the model found the positive class."
                            ),
                            pl=(
                                "Accuracy odpowiada na szerokie pytanie, ale recall pyta, "
                                "czy model znalazł klasę pozytywną."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Lower thresholds usually improve recall and create more "
                                "false positives; higher thresholds do the opposite."
                            ),
                            pl=(
                                "Niższe thresholds zwykle poprawiają recall i tworzą więcej "
                                "false positives; wyższe robią odwrotnie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Common mistakes", pl="Typowe pułapki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Do not optimize accuracy alone when the minority class is "
                                "the expensive or risky case."
                            ),
                            pl=(
                                "Nie optymalizuj samego accuracy, gdy klasa mniejszościowa "
                                "jest przypadkiem kosztownym albo ryzykownym."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A high recall threshold choice can still be bad if the review "
                                "team cannot handle the false positives."
                            ),
                            pl=(
                                "Threshold z wysokim recall nadal może być zły, jeśli zespół "
                                "nie udźwignie liczby false positives."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["class_imbalance_lab"],
            glossary=LESSON_GLOSSARY["class_imbalance_lab"],
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
    DemoManifest(
        id="pca_lab",
        level=3,
        title=LocalizedText(en="PCA Lab", pl="PCA Lab"),
        summary=LocalizedText(
            en=(
                "Explore PCA by changing datasets and noise, rotating a manual "
                "projection, fitting PCA, and inspecting reconstruction residuals."
            ),
            pl=(
                "Eksperymentuj z PCA: zmieniaj dane i noise, obracaj ręczną projekcję, "
                "dopasuj PCA i analizuj residuals rekonstrukcji."
            ),
        ),
        objectives=(
            LocalizedText(
                en=(
                    "Use manual projection to see why PCA chooses the direction with "
                    "the largest spread."
                ),
                pl=(
                    "Użyj ręcznej projekcji, żeby zobaczyć, dlaczego PCA wybiera "
                    "kierunek o największym rozrzucie."
                ),
            ),
            LocalizedText(
                en=(
                    "Compare dataset presets and noise levels to see when one component "
                    "is a good summary."
                ),
                pl=(
                    "Porównaj presety danych i poziomy noise, żeby zobaczyć, kiedy "
                    "jedna komponenta jest dobrym streszczeniem."
                ),
            ),
            LocalizedText(
                en=(
                    "Read residual lines and reconstruction error as the visible cost "
                    "of compression."
                ),
                pl=("Czytaj residual lines i reconstruction error jako widoczny koszt kompresji."),
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(
                    en="switch dataset preset",
                    pl="zmień preset danych",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(
                    en="decrease or increase noise",
                    pl="zmniejsz albo zwiększ noise",
                ),
            ),
            ControlBinding(
                key="N",
                action=LocalizedText(
                    en="generate a new sample",
                    pl="wygeneruj nową próbkę",
                ),
            ),
            ControlBinding(
                key="Left / Right",
                action=LocalizedText(
                    en="rotate the projection direction",
                    pl="obróć kierunek projekcji",
                ),
            ),
            ControlBinding(
                key="F",
                action=LocalizedText(
                    en="toggle manual direction and fitted PCA direction",
                    pl="przełącz ręczny kierunek i dopasowany kierunek PCA",
                ),
            ),
            ControlBinding(
                key="C",
                action=LocalizedText(
                    en="show or hide reconstruction residual lines",
                    pl="pokaż albo ukryj residual lines rekonstrukcji",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the projection direction",
                    pl="zresetuj kierunek projekcji",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read the PCA lesson notes",
                    pl="przeczytaj notatki lekcyjne o PCA",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(
                    en="open the help overlay",
                    pl="otwórz pomoc",
                ),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_pca_lab_scene,
        difficulty=LocalizedText(en="Advanced preview", pl="Zaawansowany podgląd"),
        tags=("pca", "dimensionality-reduction", "projection", "visualization"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="Why reduce dimensions", pl="Po co redukować wymiary"),
                    body=(
                        LocalizedText(
                            en=(
                                "Many datasets have more features than people can inspect. "
                                "Dimensionality reduction builds a smaller view that keeps "
                                "important structure."
                            ),
                            pl=(
                                "Wiele zbiorów ma więcej cech, niż da się wygodnie oglądać. "
                                "Dimensionality reduction tworzy mniejszy widok, który nadal "
                                "zachowuje ważną strukturę."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The useful question is not whether compression loses information, "
                                "but which information it keeps and which it drops."
                            ),
                            pl=(
                                "Najważniejsze pytanie nie brzmi, czy kompresja traci informację, "
                                "tylko którą informację zachowuje, a którą odrzuca."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Lesson path", pl="Ścieżka pracy"),
                    body=(
                        LocalizedText(
                            en=(
                                "Start by pressing F. The fitted PCA direction is the line that "
                                "keeps the most variance for the current dataset."
                            ),
                            pl=(
                                "Zacznij od F. Dopasowany kierunek PCA to linia, która zachowuje "
                                "najwięcej wariancji dla aktualnego zbioru."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Then rotate manually with Left/Right and watch kept variance, "
                                "reconstruction error, and residual lines change together."
                            ),
                            pl=(
                                "Potem obracaj ręcznie przez Left/Right i obserwuj, jak razem "
                                "zmieniają się wariancja, błąd rekonstrukcji i residual lines."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What PCA keeps", pl="Co zachowuje PCA"),
                    body=(
                        LocalizedText(
                            en=(
                                "PCA finds linear directions ordered by variance. The first "
                                "principal component is where the data spreads out most."
                            ),
                            pl=(
                                "PCA znajduje liniowe kierunki uporządkowane według wariancji. "
                                "Pierwsza principal component to kierunek, w którym dane mają "
                                "największy rozrzut."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Projecting onto one component turns a cloud into a line. "
                                "That can reveal the main trend, but it hides variation sideways."
                            ),
                            pl=(
                                "Projekcja na jedną komponentę zamienia chmurę w linię. "
                                "Może pokazać główny trend, ale ukrywa rozrzut w bok."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Residual lines show the missing sideways part: each line connects "
                                "a point to its reconstruction on the projection axis."
                            ),
                            pl=(
                                "Residual lines pokazują brakującą część w bok: każda linia łączy "
                                "punkt z jego rekonstrukcją na osi projekcji."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Data and noise",
                        pl="Dane i noise",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Linear cloud is the friendly case: one direction explains most "
                                "of the structure. Noisy cloud makes the trade-off less clean."
                            ),
                            pl=(
                                "Linear cloud to przyjazny przypadek: jeden kierunek wyjaśnia "
                                "większość struktury. Noisy cloud zaciemnia kompromis."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Two bands is useful because PCA can keep spread while flattening "
                                "a pattern that still matters visually."
                            ),
                            pl=(
                                "Two bands jest przydatne, bo PCA może zachować rozrzut, ale "
                                "spłaszczyć wzorzec, który wizualnie nadal ma znaczenie."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="How to read explained variance",
                        pl="Jak czytać explained variance",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Explained variance estimates how much of the original spread "
                                "is preserved by selected components."
                            ),
                            pl=(
                                "Explained variance szacuje, jaka część pierwotnego rozrzutu "
                                "zostaje zachowana przez wybrane komponenty."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A high number is useful, but it is not a guarantee that every "
                                "task-relevant pattern survived the projection."
                            ),
                            pl=(
                                "Wysoka wartość jest pomocna, ale nie gwarantuje, że każdy "
                                "ważny dla zadania wzorzec przetrwał projekcję."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Residuals and reconstruction",
                        pl="Residuals i rekonstrukcja",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "A residual line connects an original point with the point rebuilt "
                                "from the 1D projection. Long lines show what the projection lost."
                            ),
                            pl=(
                                "Residual line łączy oryginalny punkt z punktem odtworzonym "
                                "z projekcji 1D. Długie linie pokazują, co projekcja zgubiła."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Reconstruction error is the same idea as a summary number: "
                                "how much variation is left outside the chosen direction."
                            ),
                            pl=(
                                "Reconstruction error to ta sama intuicja zapisana liczbą: "
                                "ile zmienności zostaje poza wybranym kierunkiem."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to compare", pl="Co porównywać"),
                    body=(
                        LocalizedText(
                            en=(
                                "Compare manual and fitted directions on every preset. A good "
                                "explanation should match the plot, not only a high percentage."
                            ),
                            pl=(
                                "Porównuj ręczne i dopasowane kierunki na każdym presecie. Dobre "
                                "wyjaśnienie ma pasować do wykresu, nie tylko do procentu."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["pca_lab"],
            glossary=LESSON_GLOSSARY["pca_lab"],
        ),
    ),
    DemoManifest(
        id="model_comparison_lab",
        level=3,
        title=LocalizedText(en="Model Comparison Lab", pl="Model Comparison Lab"),
        summary=LocalizedText(
            en=(
                "Compare Logistic Regression, k-NN, and Decision Tree on one "
                "classification dataset to see how assumptions shape decision boundaries."
            ),
            pl=(
                "Porównaj Logistic Regression, k-NN i Decision Tree na jednym "
                "zbiorze klasyfikacyjnym, żeby zobaczyć, jak założenia modeli "
                "kształtują decision boundary."
            ),
        ),
        objectives=(
            LocalizedText(
                en=("Compare three classifiers on the same points before judging their metrics."),
                pl=(
                    "Porównaj trzy klasyfikatory na tych samych punktach, zanim "
                    "zaczniesz oceniać ich metryki."
                ),
            ),
            LocalizedText(
                en=(
                    "Connect each decision boundary shape with a model assumption: "
                    "linear, local, or rule-based."
                ),
                pl=(
                    "Połącz kształt każdej decision boundary z założeniem modelu: "
                    "liniowym, lokalnym albo regułowym."
                ),
            ),
            LocalizedText(
                en=(
                    "Treat visual comparison as the first step before richer evaluation workflows."
                ),
                pl=(
                    "Potraktuj porównanie wizualne jako pierwszy krok przed "
                    "pełniejszą ewaluacją modeli."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="1",
                action=LocalizedText(
                    en="focus Logistic Regression",
                    pl="wybierz Logistic Regression",
                ),
            ),
            ControlBinding(
                key="2",
                action=LocalizedText(
                    en="focus k-NN",
                    pl="wybierz k-NN",
                ),
            ),
            ControlBinding(
                key="3",
                action=LocalizedText(
                    en="focus Decision Tree",
                    pl="wybierz Decision Tree",
                ),
            ),
            ControlBinding(
                key="A",
                action=LocalizedText(
                    en="show or hide inactive boundaries",
                    pl="pokaż albo ukryj nieaktywne granice",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(
                    en="decrease or increase the active model parameter",
                    pl="zmniejsz albo zwiększ parametr aktywnego modelu",
                ),
            ),
            ControlBinding(
                key="D",
                action=LocalizedText(
                    en="cycle dataset preset",
                    pl="zmień preset danych",
                ),
            ),
            ControlBinding(
                key="E",
                action=LocalizedText(
                    en="show or hide misclassified test points",
                    pl="pokaż albo ukryj błędnie sklasyfikowane punkty testowe",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the preview",
                    pl="zresetuj podgląd",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read the model comparison lesson notes",
                    pl="przeczytaj notatki o porównywaniu modeli",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(
                    en="open the help overlay",
                    pl="otwórz pomoc",
                ),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_model_comparison_lab_scene,
        difficulty=LocalizedText(en="Advanced preview", pl="Zaawansowany podgląd"),
        tags=(
            "model-comparison",
            "classification",
            "decision-boundary",
            "visualization",
        ),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(
                        en="Why compare models",
                        pl="Po co porównywać modele",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Different models can see the same data through very different "
                                "assumptions. A metric alone often hides that difference."
                            ),
                            pl=(
                                "Różne modele mogą patrzeć na te same dane przez bardzo różne "
                                "założenia. Sama metryka często ukrywa tę różnicę."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "This lab starts with the visible part: how each model divides "
                                "the feature space."
                            ),
                            pl=(
                                "Ten lab zaczyna od rzeczy widocznej: jak każdy model dzieli "
                                "przestrzeń cech."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Lesson path", pl="Ścieżka pracy"),
                    body=(
                        LocalizedText(
                            en=(
                                "Start with Logistic Regression and read the straight boundary. "
                                "Ask which points it explains well and which it cuts too simply."
                            ),
                            pl=(
                                "Zacznij od Logistic Regression i obejrzyj prostą granicę. "
                                "Sprawdź, które punkty wyjaśnia dobrze, a które tnie zbyt prosto."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Then switch to k-NN and Decision Tree. Compare local flexibility "
                                "with axis-aligned rules."
                            ),
                            pl=(
                                "Potem przełącz się na k-NN i Decision Tree. Porównaj lokalną "
                                "elastyczność z regułami ustawionymi wzdłuż osi."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Same data, different assumptions",
                        pl="Te same dane, różne założenia",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Logistic Regression prefers one smooth global split. That makes "
                                "it stable and readable, but limited when the real pattern bends."
                            ),
                            pl=(
                                "Logistic Regression preferuje jeden gładki, globalny podział. "
                                "Dzięki temu jest stabilna i czytelna, ale ograniczona, gdy "
                                "prawdziwy wzorzec się wygina."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "k-NN lets nearby examples vote. Its boundary can bend around "
                                "local detail, but noise can also influence the vote."
                            ),
                            pl=(
                                "k-NN pozwala głosować najbliższym przykładom. Jego granica "
                                "może wyginać się wokół lokalnych szczegółów, ale noise też "
                                "może wpływać na głosowanie."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Decision Tree builds rules that cut the space into rectangles. "
                                "That is interpretable, but the boundary can become blocky."
                            ),
                            pl=(
                                "Decision Tree buduje reguły, które tną przestrzeń na prostokąty. "
                                "To jest interpretowalne, ale granica może robić się blokowa."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="What the first slice shows",
                        pl="Co pokazuje pierwszy wycinek",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "This version stays intentionally small: it focuses on boundary "
                                "shape, dataset presets, train/test accuracy, and compact "
                                "confusion details with precision/recall "
                                "before adding full training controls."
                            ),
                            pl=(
                                "Ta wersja jest celowo mała: skupia się na kształcie granicy, "
                                "presetach danych, train/test accuracy i kompaktowych "
                                "confusion details z precision/recall, zanim dodamy pełne "
                                "kontrolki treningu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Use - / = to change the active model parameter. Watch whether "
                                "more flexibility improves the test score or only bends the "
                                "boundary harder. Use E to connect the metric with the actual "
                                "misclassified test points."
                            ),
                            pl=(
                                "Użyj - / =, żeby zmienić parametr aktywnego modelu. Sprawdź, "
                                "czy większa elastyczność poprawia test score, czy tylko "
                                "mocniej wygina granicę. Użyj E, żeby połączyć metrykę "
                                "z konkretnymi błędnie sklasyfikowanymi punktami testowymi."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="What to add next",
                        pl="Co dodać później",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "A fuller lab can add richer confusion matrix views, class "
                                "threshold controls, and real training controls."
                            ),
                            pl=(
                                "Pełniejszy lab może dodać bogatszy widok confusion matrix, "
                                "kontrolki threshold dla klas i prawdziwe kontrolki treningu."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The comparison should still stay visual: metrics are easier "
                                "to trust when students can see what changed."
                            ),
                            pl=(
                                "Porównanie nadal powinno zostać wizualne: metrykom łatwiej "
                                "zaufać, gdy student widzi, co się zmieniło."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["model_comparison_lab"],
            glossary=LESSON_GLOSSARY["model_comparison_lab"],
        ),
    ),
    DemoManifest(
        id="calibration_lab",
        level=3,
        title=LocalizedText(en="Calibration Lab", pl="Calibration Lab"),
        summary=LocalizedText(
            en=(
                "Compare confidence scores with observed frequencies using a reliability "
                "diagram, Brier score, and Expected Calibration Error."
            ),
            pl=(
                "Porównaj confidence scores z obserwowanymi częstościami przez reliability "
                "diagram, Brier score i Expected Calibration Error."
            ),
        ),
        objectives=(
            LocalizedText(
                en="See why a model can be accurate but still poorly calibrated.",
                pl="Zobacz, dlaczego model może mieć dobrą accuracy, ale słabą kalibrację.",
            ),
            LocalizedText(
                en=(
                    "Compare overconfident, underconfident, and better calibrated "
                    "probability scores."
                ),
                pl=(
                    "Porównaj overconfident, underconfident i lepiej skalibrowane "
                    "probability scores."
                ),
            ),
            LocalizedText(
                en="Use Brier score and ECE as complementary calibration signals.",
                pl="Użyj Brier score i ECE jako uzupełniających sygnałów kalibracji.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(
                    en="switch calibration preset",
                    pl="zmień preset kalibracji",
                ),
            ),
            ControlBinding(
                key="E",
                action=LocalizedText(
                    en="show or hide calibration error bars",
                    pl="pokaż albo ukryj error bars kalibracji",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(
                    en="decrease or increase temperature scaling",
                    pl="zmniejsz albo zwiększ temperature scaling",
                ),
            ),
            ControlBinding(
                key="O",
                action=LocalizedText(
                    en="show or hide raw pre-temperature scores",
                    pl="pokaż albo ukryj raw score sprzed temperature scaling",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the preview",
                    pl="zresetuj podgląd",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read the calibration lesson notes",
                    pl="przeczytaj notatki o kalibracji",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(
                    en="open the help overlay",
                    pl="otwórz pomoc",
                ),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_calibration_lab_scene,
        difficulty=LocalizedText(en="Advanced preview", pl="Zaawansowany podgląd"),
        tags=("calibration", "probability", "evaluation", "visualization"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(en="Why calibration matters", pl="Po co kalibracja"),
                    body=(
                        LocalizedText(
                            en=(
                                "A classifier can rank examples well and still give probability "
                                "scores that are too bold or too cautious."
                            ),
                            pl=(
                                "Klasyfikator może dobrze porządkować przypadki, a mimo to "
                                "dawać probability scores zbyt odważne albo zbyt ostrożne."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Calibration asks whether a 70% prediction is correct about "
                                "70% of the time."
                            ),
                            pl=(
                                "Kalibracja pyta, czy predykcja 70% trafia mniej więcej "
                                "w 70% przypadków."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The side panel also shows accuracy@0.5: the classifier decision "
                                "after turning probabilities into labels with a threshold."
                            ),
                            pl=(
                                "Panel boczny pokazuje też accuracy@0.5: decyzję klasyfikatora "
                                "po zamianie prawdopodobieństw na etykiety przez threshold."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Lesson path", pl="Ścieżka pracy"),
                    body=(
                        LocalizedText(
                            en=(
                                "Start with Overconfident. Look for bins where confidence is "
                                "higher than observed frequency."
                            ),
                            pl=(
                                "Zacznij od Overconfident. Szukaj binów, w których confidence "
                                "jest wyższe niż obserwowana częstość."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Then compare Underconfident and Better calibrated. Ask which "
                                "preset makes the bars stay closer to the diagonal."
                            ),
                            pl=(
                                "Potem porównaj Underconfident i Lepiej skalibrowane. Sprawdź, "
                                "w którym presecie słupki są bliżej przekątnej."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="How to read the diagram", pl="Jak czytać diagram"),
                    body=(
                        LocalizedText(
                            en=(
                                "The diagonal is perfect calibration. A bar above or below it "
                                "means observed outcomes disagree with predicted confidence."
                            ),
                            pl=(
                                "Przekątna oznacza idealną kalibrację. Słupek nad albo pod nią "
                                "oznacza, że obserwowane wyniki rozmijają się z confidence."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Error bars make the gap visible. Larger gaps mean the score "
                                "needs more caution before it is used as a probability."
                            ),
                            pl=(
                                "Error bars pokazują tę lukę. Większa luka oznacza, że score "
                                "wymaga ostrożności, zanim potraktujemy go jak probability."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Brier score and ECE", pl="Brier score i ECE"),
                    body=(
                        LocalizedText(
                            en=(
                                "Brier score punishes probability errors sample by sample. "
                                "ECE summarizes bin-level calibration gaps."
                            ),
                            pl=(
                                "Brier score karze błędy probabilistyczne próbka po próbce. "
                                "ECE streszcza luki kalibracji na poziomie binów."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "They should be read together: one number rarely explains the "
                                "whole behavior of probability scores."
                            ),
                            pl=(
                                "Warto czytać je razem: jedna liczba rzadko wyjaśnia całe "
                                "zachowanie probability scores."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Worst gap points to the bin with the largest "
                                "confidence-vs-outcome difference, so it is a good "
                                "first diagnostic target."
                            ),
                            pl=(
                                "Worst gap wskazuje bin z największą różnicą "
                                "confidence-vs-outcome, więc jest dobrym pierwszym "
                                "celem diagnostycznym."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="Temperature scaling",
                        pl="Temperature scaling",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Temperature scaling is a post-hoc calibration trick: it changes "
                                "how sharp probability scores are without changing their order."
                            ),
                            pl=(
                                "Temperature scaling to prosta metoda post-hoc calibration: "
                                "zmienia ostrość probability scores bez zmiany ich kolejności."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Higher temperature softens scores toward 0.5. Lower temperature "
                                "pushes them closer to 0 or 1."
                            ),
                            pl=(
                                "Wyższa temperatura zmiękcza score w stronę 0.5. Niższa "
                                "wypycha je bliżej 0 albo 1."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Use O to compare raw pre-temperature scores with the active "
                                "scaled scores. The score legend shows which marker is which."
                            ),
                            pl=(
                                "Użyj O, żeby porównać raw score sprzed temperature scaling "
                                "z aktywnymi, przeskalowanymi score. Legenda pokazuje, "
                                "który marker jest który."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["calibration_lab"],
            glossary=LESSON_GLOSSARY["calibration_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="tsne_umap_exploration_lab",
        level=3,
        title_en="t-SNE / UMAP Exploration Lab",
        title_pl="t-SNE / UMAP Exploration Lab",
        summary_en=(
            "Explore an interactive embedding prototype for comparing local neighborhoods, "
            "global structure, and projection stability."
        ),
        summary_pl=(
            "Eksploruj interaktywny prototyp embeddingów do porównywania lokalnych "
            "sąsiedztw, globalnej struktury i stabilności projekcji."
        ),
        objectives=(
            LocalizedText(
                en=(
                    "Compare how nonlinear embeddings can reveal neighborhoods in "
                    "high-dimensional data."
                ),
                pl=(
                    "Porównaj, jak nieliniowe embeddingi mogą pokazywać sąsiedztwa "
                    "w danych wielowymiarowych."
                ),
            ),
            LocalizedText(
                en=("Tune the first controls: perplexity, neighbors, seed, and dataset preset."),
                pl=("Dostrajaj pierwsze kontrolki: perplexity, neighbors, seed i preset danych."),
            ),
            LocalizedText(
                en=(
                    "Keep the lesson honest about what distances and clusters in a "
                    "2D embedding can and cannot prove."
                ),
                pl=(
                    "Pilnuj, żeby lekcja uczciwie mówiła, co odległości i klastry "
                    "w embeddingu 2D mogą, a czego nie mogą dowodzić."
                ),
            ),
        ),
        controls=(
            ControlBinding(
                key="1-3",
                action=LocalizedText(
                    en="switch dataset preset",
                    pl="zmień preset danych",
                ),
            ),
            ControlBinding(
                key="M",
                action=LocalizedText(
                    en="switch t-SNE / UMAP preview",
                    pl="przełącz podgląd t-SNE / UMAP",
                ),
            ),
            ControlBinding(
                key="- / =",
                action=LocalizedText(
                    en="decrease or increase perplexity / neighbors",
                    pl="zmniejsz albo zwiększ perplexity / neighbors",
                ),
            ),
            ControlBinding(
                key="S",
                action=LocalizedText(
                    en="change seed variant and inspect drift",
                    pl="zmień wariant seed i sprawdź drift",
                ),
            ),
            ControlBinding(
                key="L",
                action=LocalizedText(
                    en="show or hide local neighbor links",
                    pl="pokaż albo ukryj lokalne połączenia sąsiadów",
                ),
            ),
            ControlBinding(
                key="O",
                action=LocalizedText(
                    en="show or hide raw high-dimensional layout",
                    pl="pokaż albo ukryj raw high-dimensional layout",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the preview",
                    pl="zresetuj podgląd",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read the t-SNE / UMAP exploration lesson",
                    pl="przeczytaj lekcję eksploracji t-SNE / UMAP",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_tsne_umap_exploration_scene,
        difficulty=LocalizedText(en="Advanced prototype", pl="Prototyp zaawansowany"),
        tags=("tsne", "umap", "embedding", "dimensionality-reduction", "visualization"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(
                        en="Why this lab belongs in Level 3", pl="Dlaczego Level 3"
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "t-SNE and UMAP are useful precisely because they are not "
                                "simple linear projections. That power also makes them easy "
                                "to misread."
                            ),
                            pl=(
                                "t-SNE i UMAP są użyteczne właśnie dlatego, że nie są prostą "
                                "projekcją liniową. Ta siła sprawia też, że łatwo je źle odczytać."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The first real slice should make the trade-off visible: "
                                "local neighborhoods can look clear even when global distances "
                                "are not directly comparable."
                            ),
                            pl=(
                                "Pierwszy prawdziwy wycinek powinien pokazać kompromis: "
                                "lokalne sąsiedztwa mogą być czytelne, nawet gdy globalnych "
                                "odległości nie da się porównywać wprost."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Current controls", pl="Obecne kontrolki"),
                    body=(
                        LocalizedText(
                            en=(
                                "This slice uses fixed synthetic datasets, a seed control, "
                                "and one main neighborhood parameter."
                            ),
                            pl=(
                                "Ten wycinek używa stałych syntetycznych datasetów, kontroli "
                                "seed i jednego głównego parametru sąsiedztwa."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "For t-SNE that parameter can be perplexity; for UMAP it can "
                                "be number of neighbors."
                            ),
                            pl=(
                                "Dla t-SNE tym parametrem może być perplexity; dla UMAP liczba "
                                "neighbors."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The scene now labels seed 0 as the baseline and reports seed "
                                "drift for variant layouts."
                            ),
                            pl=(
                                "Scena opisuje seed 0 jako baseline i pokazuje seed drift "
                                "dla wariantów układu."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="Reading the metrics", pl="Jak czytać metryki"),
                    body=(
                        LocalizedText(
                            en=(
                                "Local trust is a quick cue for whether nearby points mostly "
                                "keep neighbors with the same label."
                            ),
                            pl=(
                                "Local trust szybko pokazuje, czy bliskie punkty najczęściej "
                                "trzymają sąsiadów z tą samą etykietą."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Global spread describes how much of the 2D canvas the embedding "
                                "uses. It is useful as a reading cue, not as a proof that global "
                                "distances are faithful."
                            ),
                            pl=(
                                "Global spread mówi, jak mocno embedding wykorzystuje przestrzeń "
                                "2D. To dobra wskazówka do czytania wykresu, ale nie dowód, "
                                "że globalne odległości są wierne."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The color legend names the synthetic labels. It helps students "
                                "separate label information from the geometry created by the "
                                "embedding."
                            ),
                            pl=(
                                "Legenda kolorów nazywa syntetyczne etykiety. Pomaga oddzielić "
                                "informację o labelach od geometrii tworzonej przez embedding."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What students should compare", pl="Co porównywać"),
                    body=(
                        LocalizedText(
                            en=(
                                "Students should compare whether nearby points stay nearby, "
                                "whether clusters split or merge, and whether a different seed "
                                "changes the story."
                            ),
                            pl=(
                                "Studenci powinni porównywać, czy bliskie punkty pozostają "
                                "blisko, czy klastry się dzielą lub łączą oraz czy inny seed "
                                "zmienia opowieść."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The lab should avoid claiming that every 2D distance reflects "
                                "a true high-dimensional distance."
                            ),
                            pl=(
                                "Lab nie powinien sugerować, że każda odległość 2D odpowiada "
                                "prawdziwej odległości w wielu wymiarach."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Use the raw layout only as a reference: it is a simplified "
                                "projection of the toy data, not ground truth."
                            ),
                            pl=(
                                "Traktuj raw layout tylko jako punkt odniesienia: to uproszczona "
                                "projekcja toy danych, a nie ground truth."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Dataset cues point to the intended question for each preset: "
                                "clean separation, bridge points, or nested neighborhoods."
                            ),
                            pl=(
                                "Wskazówki datasetów pokazują pytanie dla każdego presetu: "
                                "czysty podział, punkty mostu albo zagnieżdżone sąsiedztwa."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to add next", pl="Co dodać później"),
                    body=(
                        LocalizedText(
                            en=(
                                "A later PR can replace the deterministic toy embeddings with "
                                "real algorithm output once the teaching interface is stable."
                            ),
                            pl=(
                                "Późniejszy PR może zastąpić deterministyczne toy embeddingi "
                                "prawdziwym wynikiem algorytmów, gdy interfejs dydaktyczny "
                                "będzie już stabilny."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["tsne_umap_exploration_lab"],
            glossary=LESSON_GLOSSARY["tsne_umap_exploration_lab"],
        ),
    ),
    _placeholder_demo(
        demo_id="model_monitoring_drift_lab",
        level=3,
        title_en="Model Monitoring Drift Lab",
        title_pl="Model Monitoring Drift Lab",
        summary_en=("A Level 3 prototype for watching model behavior change after deployment."),
        summary_pl=(
            "Prototyp Level 3 o obserwowaniu, jak zachowanie modelu zmienia się po wdrożeniu."
        ),
        objectives=(
            LocalizedText(
                en="Compare data drift with metric drift in a production-style workflow.",
                pl="Porównaj data drift z metric drift w workflow podobnym do produkcji.",
            ),
            LocalizedText(
                en=(
                    "Decide which signals should become alerts and which should remain "
                    "context for investigation."
                ),
                pl=(
                    "Ustal, które sygnały powinny stać się alertami, a które tylko "
                    "kontekstem do analizy."
                ),
            ),
            LocalizedText(
                en="Plan a visual dashboard that explains change over time without panic.",
                pl="Zaplanuj dashboard, który pokazuje zmianę w czasie bez siania paniki.",
            ),
        ),
        controls=(
            ControlBinding(
                key="1-4",
                action=LocalizedText(
                    en="switch monitoring preset",
                    pl="zmień preset monitoringu",
                ),
            ),
            ControlBinding(
                key="D / M",
                action=LocalizedText(
                    en="select data drift / metric drift signal",
                    pl="wybierz sygnał data drift / metric drift",
                ),
            ),
            ControlBinding(
                key="- / = / 0",
                action=LocalizedText(
                    en="decrease, increase, or reset alert threshold",
                    pl="zmniejsz, zwiększ albo zresetuj alert threshold",
                ),
            ),
            ControlBinding(
                key="A",
                action=LocalizedText(
                    en="acknowledge an active alert for investigation",
                    pl="potwierdź aktywny alert do analizy",
                ),
            ),
            ControlBinding(
                key="R",
                action=LocalizedText(
                    en="reset the preview",
                    pl="zresetuj podgląd",
                ),
            ),
            ControlBinding(
                key="T",
                action=LocalizedText(
                    en="read model monitoring planning notes",
                    pl="przeczytaj notatki o planie monitoringu modeli",
                ),
            ),
            ControlBinding(
                key="H",
                action=LocalizedText(en="open the help overlay", pl="otwórz pomoc"),
            ),
            ControlBinding(
                key="Esc",
                action=LocalizedText(
                    en="open pause or return to the demo list",
                    pl="otwórz pauzę albo wróć do listy dem",
                ),
            ),
        ),
        create_scene=create_model_monitoring_drift_scene,
        difficulty=LocalizedText(en="Advanced prototype", pl="Prototyp zaawansowany"),
        tags=("monitoring", "drift", "production-ml", "evaluation", "level-3"),
        theory=DemoTheory(
            sections=(
                TheorySection(
                    title=LocalizedText(
                        en="Why monitoring belongs in Level 3",
                        pl="Dlaczego monitoring pasuje do Level 3",
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "Training metrics tell only the beginning of the story. "
                                "After deployment, data and behavior can change."
                            ),
                            pl=(
                                "Metryki treningowe opowiadają tylko początek historii. "
                                "Po wdrożeniu dane i zachowanie modelu mogą się zmieniać."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "A good monitoring lab should make time, baselines, and "
                                "alerts visible without pretending every wiggle is a crisis."
                            ),
                            pl=(
                                "Dobry lab monitoringu powinien pokazywać czas, baseline "
                                "i alerty bez udawania, że każde drgnięcie jest kryzysem."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(en="What to watch", pl="Co obserwować"),
                    body=(
                        LocalizedText(
                            en=(
                                "Data drift asks whether production inputs still resemble "
                                "the baseline period."
                            ),
                            pl=(
                                "Data drift pyta, czy wejścia produkcyjne nadal przypominają "
                                "okres baseline."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Metric drift asks whether model behavior is changing: "
                                "quality, calibration, latency, or error mix."
                            ),
                            pl=(
                                "Metric drift pyta, czy zmienia się zachowanie modelu: "
                                "jakość, calibration, latency albo struktura błędów."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "Compare both signals before reacting: a single spike is "
                                "not the same thing as persistent drift, and a threshold "
                                "is a prompt for inspection rather than proof of failure."
                            ),
                            pl=(
                                "Porównaj oba sygnały przed reakcją: pojedynczy spike "
                                "to nie to samo co trwały drift, a threshold jest "
                                "zaproszeniem do sprawdzenia, nie dowodem awarii."
                            ),
                        ),
                    ),
                ),
                TheorySection(
                    title=LocalizedText(
                        en="First prototype shape", pl="Kształt pierwszego prototypu"
                    ),
                    body=(
                        LocalizedText(
                            en=(
                                "A first slice can show a time series, a baseline window, "
                                "a current window, the gap between them, one alert threshold, "
                                "and a simple lead signal and severity label."
                            ),
                            pl=(
                                "Pierwszy wycinek może pokazać szereg czasowy, baseline window, "
                                "current window, lukę między nimi, jeden alert threshold, "
                                "lead signal i prostą etykietę severity."
                            ),
                        ),
                        LocalizedText(
                            en=(
                                "The goal is to teach inspection before automation: "
                                "students should explain why an alert fired and where "
                                "the first alert appeared before acknowledging it or "
                                "following a recommendation."
                            ),
                            pl=(
                                "Celem jest uczenie inspekcji przed automatyzacją: "
                                "student powinien umieć wyjaśnić, dlaczego alert zadziałał "
                                "i gdzie pojawił się pierwszy alert, zanim go potwierdzi "
                                "albo pójdzie za rekomendacją."
                            ),
                        ),
                    ),
                ),
            ),
            mini_challenges=LESSON_CHALLENGES["model_monitoring_drift_lab"],
            glossary=LESSON_GLOSSARY["model_monitoring_drift_lab"],
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
                en="Keep the guided app shape visible as more focused Level 3 demos land.",
                pl=(
                    "Pokaż docelowy kształt aplikacji, gdy dochodzą kolejne konkretne dema Level 3."
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

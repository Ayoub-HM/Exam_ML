from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Chemins des fichiers imposés par l'énoncé et dossier des résultats.
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_DIR / "botnet_train.csv.gz"
TEST_FILE = PROJECT_DIR / "botnet_test_features.csv.gz"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_STATE = 42
MAX_ALERTS = 360

# Variables utilisées par le modèle, séparées selon leur type.
NUMERIC_FEATURES = [
    "src_port",
    "dst_port",
    "duration",
    "tot_pkts",
    "tot_bytes",
    "src_bytes",
    "src_tos",
    "dst_tos",
    "src_is_internal",
    "dst_is_internal",
]
CATEGORICAL_FEATURES = ["proto", "direction", "state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge les données et conserve les identifiants pour les prédictions."""
    # Lecture des fichiers compressés avec les noms demandés par le sujet.
    train = pd.read_csv(TRAIN_FILE, compression="gzip")
    test = pd.read_csv(TEST_FILE, compression="gzip")
    test_ids = test.get("capture_id")

    # Vérification des colonnes nécessaires avant de lancer le modèle.
    missing_train = set(FEATURES + [TARGET]) - set(train.columns)
    missing_test = set(FEATURES) - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Colonnes manquantes : train={sorted(missing_train)}, "
            f"test={sorted(missing_test)}"
        )

    return train[FEATURES], train[TARGET], test[FEATURES], test_ids


def build_model(
    model_name: str = "logistic_regression",
    class_weight: str | dict[int, float] | None = "balanced",
    regularization: float = 1.0,
) -> Pipeline:
    """Construit le prétraitement et l'un des quatre modèles imposés."""
    # Les numériques sont imputées par médiane et standardisées pour le modèle linéaire.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    if model_name == "logistic_regression":
        numeric_pipeline.steps.append(("scaler", StandardScaler()))
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    elif model_name in {"decision_tree", "random_forest"}:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    elif model_name == "hist_gradient_boosting":
        # L'encodage ordinal conserve une matrice dense compacte pour HGB.
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # Les catégories manquantes sont imputées puis encodées.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )
    # Le préprocesseur applique chaque traitement au bon groupe de colonnes.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    # Sélection du classificateur demandé par l'énoncé.
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=regularization,
            class_weight=class_weight,
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            class_weight=class_weight,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight=class_weight,
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        classifier = HistGradientBoostingClassifier(
            class_weight=class_weight,
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_exploration(features: pd.DataFrame, labels: pd.Series) -> None:
    """Génère quelques graphiques utiles pour l'analyse du rapport."""
    # Création du dossier puis application d'un style homogène aux graphiques.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Visualisation du déséquilibre de la cible.
    plt.figure(figsize=(7, 5))
    sns.countplot(x=labels)
    plt.title("Répartition de la variable cible")
    plt.xlabel("Label")
    plt.ylabel("Nombre de flux")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Visualisation des pourcentages de valeurs manquantes.
    missing = features.isna().mean().sort_values(ascending=False)
    missing = (missing[missing > 0] * 100).sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing.values, y=missing.index, color="#2878b5")
    plt.title("Valeurs manquantes")
    plt.xlabel("Pourcentage")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
    plt.close()


def select_alert_threshold(probabilities: pd.Series) -> float:
    """Retourne le seuil des MAX_ALERTS probabilités les plus élevées."""
    # Le SOC traite un nombre fixe d'alertes : on sélectionne donc les meilleurs scores.
    sorted_scores = probabilities.sort_values(ascending=False)
    if len(sorted_scores) <= MAX_ALERTS:
        return float(sorted_scores.min())
    return float(sorted_scores.iloc[MAX_ALERTS - 1])


def select_top_alerts(probabilities: pd.Series) -> pd.Series:
    """Sélectionne exactement MAX_ALERTS alertes, même en cas d'égalité."""
    # nlargest applique un départage stable grâce à l'ordre original des lignes.
    selected_indexes = probabilities.nlargest(MAX_ALERTS, keep="first").index
    predictions = pd.Series(0, index=probabilities.index, dtype=int)
    predictions.loc[selected_indexes] = 1
    return predictions


def tune_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[Pipeline, dict[str, object], pd.Series]:
    """Compare quelques réglages et privilégie les détections dans les 360 alertes."""
    # C contrôle la régularisation ; les pondérations traitent le déséquilibre des classes.
    candidates = [
        {"regularization": 0.1, "class_weight": "balanced"},
        {"regularization": 1.0, "class_weight": "balanced"},
        {"regularization": 10.0, "class_weight": "balanced"},
        {"regularization": 1.0, "class_weight": None},
    ]
    best_model = None
    best_config = None
    best_probabilities = None
    best_key = (-1, -1.0)

    for config in candidates:
        candidate = build_model(**config)
        candidate.fit(x_train, y_train)
        candidate_probabilities = pd.Series(
            candidate.predict_proba(x_valid)[:, 1], index=y_valid.index
        )
        threshold = select_alert_threshold(candidate_probabilities)
        candidate_predictions = select_top_alerts(candidate_probabilities)
        true_positives = int(((candidate_predictions == 1) & (y_valid == 1)).sum())
        average_precision = average_precision_score(y_valid, candidate_probabilities)
        key = (true_positives, average_precision)
        print(
            f"Réglage C={config['regularization']}, "
            f"class_weight={config['class_weight']} : "
            f"{true_positives}/{MAX_ALERTS} attaques dans les alertes, "
            f"AP={average_precision:.4f}"
        )
        if key > best_key:
            best_key = key
            best_model = candidate
            best_config = config
            best_probabilities = candidate_probabilities

    if best_model is None or best_config is None or best_probabilities is None:
        raise RuntimeError("Aucun réglage de modèle n'a pu être entraîné.")
    return best_model, best_config, best_probabilities


def save_threshold_plot(labels: pd.Series, probabilities: pd.Series) -> None:
    """Enregistre la courbe précision-rappel pour le rapport."""
    # Cette courbe est plus informative que l'accuracy avec une classe rare.
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe précision-rappel - régression logistique")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close()


def main() -> None:
    # Chargement et première exploration des flux réseau.
    features, labels, test_features, test_ids = load_data()
    save_exploration(features, labels)
    print(f"Données d'entraînement : {features.shape[0]} lignes")
    print(f"Répartition des labels : {labels.value_counts().sort_index().to_dict()}")

    # Séparation stratifiée pour conserver la proportion de botnets en validation.
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # Comparaison des réglages de la régression logistique.
    logistic_model, selected_config, probabilities = tune_model(
        x_train, y_train, x_valid, y_valid
    )

    # Comparaison des trois autres modèles imposés sur la même validation.
    model_results = [("logistic_regression", logistic_model, selected_config, probabilities)]
    for model_name in ["decision_tree", "random_forest", "hist_gradient_boosting"]:
        candidate = build_model(model_name=model_name, class_weight="balanced")
        candidate.fit(x_train, y_train)
        candidate_probabilities = pd.Series(
            candidate.predict_proba(x_valid)[:, 1], index=y_valid.index
        )
        candidate_threshold = select_alert_threshold(candidate_probabilities)
        candidate_predictions = select_top_alerts(candidate_probabilities)
        true_positives = int(((candidate_predictions == 1) & (y_valid == 1)).sum())
        candidate_average_precision = average_precision_score(
            y_valid, candidate_probabilities
        )
        print(
            f"Modèle {model_name} : {true_positives}/{MAX_ALERTS} attaques dans les alertes, "
            f"AP={candidate_average_precision:.4f}"
        )
        model_results.append((model_name, candidate, {"class_weight": "balanced"}, candidate_probabilities))

    # Priorité au nombre d'attaques détectées, puis à l'Average Precision.
    best_name, model, selected_config, probabilities = max(
        model_results,
        key=lambda result: (
            int(((select_top_alerts(result[3]) & (y_valid == 1)).sum())),
            average_precision_score(y_valid, result[3]),
        ),
    )
    print(f"\nMeilleur modèle opérationnel : {best_name}")

    # Évaluation indépendante du seuil par défaut, puis seuil opérationnel du SOC.
    default_predictions = (probabilities >= 0.5).astype(int)
    threshold = select_alert_threshold(probabilities)
    soc_predictions = select_top_alerts(probabilities)
    print(f"\nROC-AUC : {roc_auc_score(y_valid, probabilities):.4f}")
    print(f"Average Precision : {average_precision_score(y_valid, probabilities):.4f}")
    print("\nRapport avec le seuil 0.5 :")
    print(classification_report(y_valid, default_predictions, zero_division=0))
    print(f"Seuil SOC retenu : {threshold:.6f}")
    print(f"Alertes validation au seuil SOC : {soc_predictions.sum()}")
    print(f"Réglage retenu : {selected_config}")
    print("Matrice de confusion au seuil SOC :")
    print(confusion_matrix(y_valid, soc_predictions))
    save_threshold_plot(y_valid, probabilities)

    # Réentraînement sur toutes les données avant la prédiction finale.
    model = build_model(model_name=best_name, **selected_config)
    model.fit(features, labels)
    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_predictions = select_top_alerts(pd.Series(test_probabilities))
    output = pd.DataFrame({"prediction": test_predictions, "probability": test_probabilities})
    if test_ids is not None:
        output.insert(0, "capture_id", test_ids.to_numpy())
    output.to_csv(RESULTS_DIR / "logistic_regression_predictions.csv", index=False)
    print(f"\nRésultats enregistrés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_DIR / "botnet_train.csv.gz"
MAX_ALERTS = 360
RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "src_port", "dst_port", "duration", "tot_pkts", "tot_bytes",
    "src_bytes", "src_tos", "dst_tos", "src_is_internal", "dst_is_internal",
]
CATEGORICAL_FEATURES = ["proto", "direction", "state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Charge les variables explicatives et la cible du fichier d'entraînement."""
    train = pd.read_csv(TRAIN_FILE, compression="gzip")
    return train[FEATURES], train["label"]


def build_model(model_name: str) -> Pipeline:
    """Construit le preprocessing et le modèle demandé."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    elif model_name in {"decision_tree", "random_forest"}:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    elif model_name == "hist_gradient_boosting":
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", categorical_encoder),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear",
            random_state=RANDOM_STATE,
        )
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            class_weight="balanced", max_depth=20, min_samples_leaf=2,
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced", n_estimators=100, max_depth=20,
            min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
        )
    else:
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.08, max_iter=150,
            max_leaf_nodes=31, l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def top_alert_predictions(probabilities: pd.Series) -> pd.Series:
    """Marque exactement les 360 flux les plus suspects comme alertes."""
    predictions = pd.Series(0, index=probabilities.index, dtype=int)
    predictions.loc[probabilities.nlargest(MAX_ALERTS, keep="first").index] = 1
    return predictions


def print_metrics(title: str, labels: pd.Series, predictions: pd.Series, probabilities: pd.Series) -> None:
    """Affiche la matrice et les métriques principales."""
    print(f"\n--- {title} ---")
    print("Matrice de confusion [ [TN, FP], [FN, TP] ] :")
    print(confusion_matrix(labels, predictions))
    print(f"Accuracy  : {accuracy_score(labels, predictions):.4f}")
    print(f"Precision : {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(labels, predictions, zero_division=0):.4f}")
    print(f"F1-score  : {f1_score(labels, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(labels, probabilities):.4f}")
    print(f"Average Precision : {average_precision_score(labels, probabilities):.4f}")
    print("\nRapport détaillé :")
    print(classification_report(labels, predictions, zero_division=0))


def run(model_name: str, display_name: str) -> None:
    """Entraîne un seul modèle et affiche ses deux évaluations."""
    features, labels = load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    model = build_model(model_name)
    print(f"Modèle testé : {display_name}")
    print("Entraînement sur 80 % et validation sur 20 % du train.")
    model.fit(x_train, y_train)
    probabilities = pd.Series(model.predict_proba(x_valid)[:, 1], index=y_valid.index)

    default_predictions = (probabilities >= 0.5).astype(int)
    print_metrics("Seuil 0.5", y_valid, default_predictions, probabilities)

    alert_predictions = top_alert_predictions(probabilities)
    print(f"Alertes sélectionnées : {alert_predictions.sum()} / {MAX_ALERTS}")
    print_metrics("Top 360 alertes SOC", y_valid, alert_predictions, probabilities)


if __name__ == "__main__":
    raise SystemExit("Lancez main_1.py, main_2.py, main_3.py ou main_4.py.")
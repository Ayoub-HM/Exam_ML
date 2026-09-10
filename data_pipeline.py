# Outils pour construire les chemins sans dépendre du dossier courant.
from pathlib import Path

# Manipulation et lecture des données tabulaires.
import pandas as pd
# Composants scikit-learn utilisés pour créer le prétraitement reproductible.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Dossier du projet et fichiers imposés par le sujet.
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_DIR / "botnet_train.csv.gz"
TEST_FILE = PROJECT_DIR / "botnet_test_features.csv.gz"
TARGET = "label"

# Variables numériques qui seront imputées puis standardisées.
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
# Variables texte qui seront imputées puis encodées en One-Hot.
CATEGORICAL_FEATURES = ["proto", "direction", "state"]
# Liste complète des variables utilisées par le modèle.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the required files and separate features from the training label."""
    # Lecture des deux jeux compressés avec les noms imposés par le sujet.
    train = pd.read_csv(TRAIN_FILE, compression="gzip")
    test = pd.read_csv(TEST_FILE, compression="gzip")

    # Vérification précoce pour éviter une erreur difficile à comprendre plus tard.
    missing_train = set(FEATURES + [TARGET]) - set(train.columns)
    missing_test = set(FEATURES) - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns: train={sorted(missing_train)}, "
            f"test={sorted(missing_test)}"
        )

    # On retire la cible des variables d'entrée et l'identifiant n'est pas utilisé.
    return train[FEATURES], train[TARGET], test[FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing fitted only on the training data."""
    # Pipeline numérique : médiane pour les absences, puis mise à l'échelle.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Pipeline catégoriel : modalité fréquente, puis colonnes binaires.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Application des bons traitements aux bons groupes de colonnes.
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    # Contrôle rapide : l'entraînement et le test doivent avoir la même dimension.
    train_features, labels, test_features = load_data()
    preprocessor = build_preprocessor()
    train_matrix = preprocessor.fit_transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    print(f"Training rows: {len(labels)}")
    print(f"Transformed train shape: {train_matrix.shape}")
    print(f"Transformed test shape: {test_matrix.shape}")
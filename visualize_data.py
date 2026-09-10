from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Localisation des données et des images générées.
PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_DIR / "botnet_train.csv.gz"
FIGURES_DIR = PROJECT_DIR / "figures"


def main() -> None:
    # Lecture du jeu d'entraînement pour explorer les données et la cible.
    train = pd.read_csv(TRAIN_FILE, compression="gzip")
    FIGURES_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Visualisation de la répartition des classes légitime et botnet.
    plt.figure(figsize=(7, 5))
    sns.countplot(data=train, x="label")
    plt.title("Répartition de la variable cible")
    plt.xlabel("Label")
    plt.ylabel("Nombre de flux")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Visualisation de la proportion de valeurs manquantes par variable.
    missing = train.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0] * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing.values, y=missing.index, color="#2878b5")
    plt.title("Valeurs manquantes dans le jeu d'entraînement")
    plt.xlabel("Pourcentage de valeurs manquantes")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
    plt.close()

    # Comparaison des distributions numériques selon le label.
    numeric_features = [
        "duration",
        "tot_pkts",
        "tot_bytes",
        "src_bytes",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.ravel(), numeric_features):
        sns.histplot(
            data=train,
            x=feature,
            hue="label",
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            ax=axis,
        )
        axis.set_title(f"Distribution de {feature}")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=150)
    plt.close(figure)

    # Comparaison des catégories principales du protocole et de la direction.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=train, x="proto", hue="label", ax=axes[0])
    axes[0].set_title("Protocoles selon le label")
    axes[0].tick_params(axis="x", rotation=30)
    sns.countplot(data=train, x="direction", hue="label", ax=axes[1])
    axes[1].set_title("Directions selon le label")
    axes[1].tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "categorical_distributions.png", dpi=150)
    plt.close(figure)

    print(f"Figures generated in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
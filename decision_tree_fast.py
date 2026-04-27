import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"
FEATURES = ["minimum_orbit_intersection", "absolute_magnitude"]
TARGET = "hazardous"


def coerce_bool_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "normalized.csv"))
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = coerce_bool_labels(df[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simple model with reasonable depth
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nModel depth: {clf.get_depth()}, leaves: {clf.tree_.n_leaves}")
    print(f"Test F1 score (80/20 split): {test_f1:.4f}")

    fig, ax = plt.subplots(figsize=(12, 18))
    plot_tree(
        clf,
        feature_names=FEATURES,
        class_names=["Not Hazardous", "Hazardous"],
        filled=True,
        rounded=True,
        fontsize=24,
        ax=ax,
    )
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(EXPLORATION_DIR, "decision_tree.png"), dpi=150)
    plt.close()

    print("\nDecision tree plot saved to Exploration/decision_tree.png")


if __name__ == "__main__":
    main()

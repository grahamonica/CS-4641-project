import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"
FEATURES = ["minimum_orbit_intersection", "absolute_magnitude"]
TARGET = "hazardous"


def coerce_bool_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def select_depth(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_depth: int = 1,
    max_depth: int = 8,
    random_state: int = 42,
    tolerance: float = 1e-4,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = []
    for depth in range(min_depth, max_depth + 1):
        model = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        scores.append((depth, f1_scores.mean(), f1_scores.std()))

    best_mean_f1 = max(item[1] for item in scores)
    selected_depth = min(
        depth for depth, mean_f1, _ in scores if mean_f1 >= (best_mean_f1 - tolerance)
    )
    return selected_depth, best_mean_f1, scores


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "normalized.csv"))
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = coerce_bool_labels(df[TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    selected_depth, best_cv_f1, depth_scores = select_depth(X_train, y_train)
    clf = DecisionTreeClassifier(max_depth=selected_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Selected max_depth (from train CV): {selected_depth}")
    print(f"Selected model depth: {clf.get_depth()}, leaves: {clf.tree_.n_leaves}")
    print(f"Best 5-fold train CV F1: {best_cv_f1:.4f}")
    print(f"Test F1 score (80/20 split): {test_f1:.4f}")
    print("Depth search (depth, mean_f1, std_f1):")
    for depth, mean_f1, std_f1 in depth_scores:
        print(f"  ({depth}, {mean_f1:.4f}, {std_f1:.4f})")

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=FEATURES,
        class_names=["Not Hazardous", "Hazardous"],
        filled=True,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, "decision_tree.png"), dpi=150)
    plt.close()

    print("Decision tree plot saved to Exploration/decision_tree.png")


if __name__ == "__main__":
    main()

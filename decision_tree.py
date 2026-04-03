import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"
FEATURES = ["estimated_diameter", "relative_velocity", "absolute_magnitude"]


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "normalized.csv"))

    X = df[FEATURES]
    y = df["hazardous"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=FEATURES,
        class_names=["Not Hazardous", "Hazardous"],
        filled=True,
        max_depth=4,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, "decision_tree.png"), dpi=150)
    plt.close()

    print("Decision tree plot saved to Exploration/decision_tree.png")


if __name__ == "__main__":
    main()

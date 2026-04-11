import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"

# All features from post-exploratory data (excluding hazardous)
ALL_FEATURES = [
    "data_arc_in_days",
    "observations_used",
    "minimum_orbit_intersection",
    "eccentricity",
    "perihelion_distance",
    "perihelion_time",
    "absolute_magnitude",
    "relative_velocity",
    "estimated_diameter",
]


def select_significant_features(df: pd.DataFrame, target: str, alpha: float = 0.05):
    """
    Select features with statistically significant correlation to the target.
    Uses Spearman correlation for robustness to non-linear relationships.
    """
    significant_features = []
    
    print(f"Feature Significance Analysis (alpha={alpha}):")
    print("-" * 60)
    
    for feature in ALL_FEATURES:
        corr, p_value = spearmanr(df[feature], df[target])
        is_significant = p_value < alpha
        significance_marker = "***" if is_significant else ""
        print(f"{feature:30s} | r={corr:7.4f} | p-value={p_value:.4e} {significance_marker}")
        
        if is_significant:
            significant_features.append(feature)
    
    print("-" * 60)
    print(f"Selected {len(significant_features)} significant features out of {len(ALL_FEATURES)}")
    print(f"Features: {significant_features}\n")
    
    return significant_features


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "normalized.csv"))

    # Ensure orbit_uncertainty is integer and in 0-9
    y = df["orbit_uncertainty"].astype(int)
    valid_mask = y.between(0, 9)
    df = df[valid_mask].copy()
    y = y[valid_mask]
    print(f"Orbit Uncertainty Class Distribution:")
    print(y.value_counts().sort_index())
    print()

    # Select statistically significant features
    features = select_significant_features(df, "orbit_uncertainty")
    
    if not features:
        print("Error: No statistically significant features found!")
        return
    
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Plot ROC curves for multiclass (One-vs-Rest)
    classes = np.unique(y)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if n_classes == 2:
        # Binary classification ROC
        fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"Class {classes[1]} (AUC = {roc_auc:.2f})")
    else:
        # Multiclass ROC (One-vs-Rest)
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Orbit Uncertainty Prediction (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, "logistic_regression_roc.png"), dpi=150)
    plt.close()

    print("Logistic regression ROC curve saved to Exploration/logistic_regression_roc.png")


if __name__ == "__main__":
    main()


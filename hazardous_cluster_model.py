import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

MOID_COLUMN = "minimum_orbit_intersection"
MAGNITUDE_COLUMN = "absolute_magnitude"
TARGET_COLUMN = "hazardous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-selecting cluster model for hazardous-asteroid prediction using "
            "minimum_orbit_intersection and absolute_magnitude."
        )
    )
    parser.add_argument(
        "--data",
        default=os.path.join("Data", "normalized.csv"),
        help="Path to CSV with minimum_orbit_intersection, absolute_magnitude, and hazardous.",
    )
    parser.add_argument(
        "--plot",
        default=os.path.join("Exploration", "moid_absolute_magnitude_cluster_prediction.png"),
        help="Path for output plot.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test size.")
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum cluster count (tree leaves) to evaluate.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum cluster count (tree leaves) to evaluate.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for choosing the fewest clusters near the best CV F1.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def coerce_bool_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    invalid = ~lowered.isin(truthy | falsy)

    if invalid.any():
        examples = series[invalid].dropna().head(5).tolist()
        raise ValueError(f"Could not parse hazardous labels as boolean. Bad values: {examples}")

    return lowered.isin(truthy)


def select_leaf_count(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_clusters: int,
    max_clusters: int,
    random_state: int,
    tolerance: float,
) -> Tuple[int, float, List[Tuple[int, float, float]]]:
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be greater than or equal to min_clusters.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores: List[Tuple[int, float, float]] = []
    for leaves in range(min_clusters, max_clusters + 1):
        model = DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        scores.append((leaves, cv_scores.mean(), cv_scores.std()))

    best_cv_f1 = max(mean_f1 for _, mean_f1, _ in scores)
    selected = min(
        leaves for leaves, mean_f1, _ in scores if mean_f1 >= (best_cv_f1 - tolerance)
    )
    return selected, best_cv_f1, scores


def save_plot(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    model: DecisionTreeClassifier,
    scores: List[Tuple[int, float, float]],
    selected_clusters: int,
    plot_path: str,
) -> None:
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    x_min = X[MOID_COLUMN].min() - 0.1
    x_max = X[MOID_COLUMN].max() + 0.1
    y_min = X[MAGNITUDE_COLUMN].min() - 0.1
    y_max = X[MAGNITUDE_COLUMN].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 450),
        np.linspace(y_min, y_max, 450),
    )
    grid = pd.DataFrame(
        {
            MOID_COLUMN: xx.ravel(),
            MAGNITUDE_COLUMN: yy.ravel(),
        }
    )

    grid_leaf_ids = model.apply(grid)
    all_leaf_ids = model.apply(X)
    unique_leaf_ids = sorted(np.unique(all_leaf_ids))
    leaf_to_idx = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}
    grid_cluster_idx = np.array([leaf_to_idx[leaf] for leaf in grid_leaf_ids]).reshape(xx.shape)
    data_cluster_idx = np.array([leaf_to_idx[leaf] for leaf in all_leaf_ids])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_clusters = axes[0]
    base_cluster_colors = [
        "#ffd1dc",  # light pink
        "#cfe8ff",  # light blue
        "#d6f5d6",  # light green
        "#fff3bf",  # light yellow
        "#e4d7ff",  # light violet
        "#ffdcb8",  # light orange
        "#d7f5f2",  # light teal
        "#f7d7e8",  # rose
    ]
    cluster_colors = [
        base_cluster_colors[i % len(base_cluster_colors)] for i in range(len(unique_leaf_ids))
    ]
    cluster_cmap = ListedColormap(cluster_colors)
    ax_clusters.contourf(
        xx,
        yy,
        grid_cluster_idx,
        levels=np.arange(-0.5, len(unique_leaf_ids) + 0.5, 1.0),
        cmap=cluster_cmap,
        alpha=0.6,
    )
    # Draw explicit lines at cluster borders.
    if len(unique_leaf_ids) > 1:
        ax_clusters.contour(
            xx,
            yy,
            grid_cluster_idx,
            levels=np.arange(0.5, len(unique_leaf_ids) - 0.5 + 1e-9, 1.0),
            colors="black",
            linewidths=1.2,
            alpha=0.9,
        )
    # Label each cluster directly on the plot.
    for cluster_idx in range(len(unique_leaf_ids)):
        mask = data_cluster_idx == cluster_idx
        if not np.any(mask):
            continue
        center_x = float(X.loc[mask, MOID_COLUMN].median())
        center_y = float(X.loc[mask, MAGNITUDE_COLUMN].median())
        ax_clusters.text(
            center_x,
            center_y,
            f"Cluster {cluster_idx + 1}",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
        )
    hazard_mask = y_true.to_numpy().astype(bool)
    not_hazard_mask = ~hazard_mask
    ax_clusters.scatter(
        X.loc[not_hazard_mask, MOID_COLUMN],
        X.loc[not_hazard_mask, MAGNITUDE_COLUMN],
        c="blue",
        alpha=0.3,
        s=13,
        label="Not Hazardous",
    )
    ax_clusters.scatter(
        X.loc[hazard_mask, MOID_COLUMN],
        X.loc[hazard_mask, MAGNITUDE_COLUMN],
        c="red",
        alpha=0.45,
        s=14,
        label="Hazardous",
    )
    mistakes = y_true.to_numpy() != y_pred.to_numpy()
    if mistakes.any():
        ax_clusters.scatter(
            X.loc[mistakes, MOID_COLUMN],
            X.loc[mistakes, MAGNITUDE_COLUMN],
            facecolors="none",
            edgecolors="black",
            s=30,
            linewidths=0.8,
            label="Misclassified",
        )
    ax_clusters.set_xlabel(MOID_COLUMN)
    ax_clusters.set_ylabel(MAGNITUDE_COLUMN)
    ax_clusters.set_title(f"Self-Selected Clusters (leaves={selected_clusters})")
    cluster_handles = [
        Patch(facecolor=cluster_colors[i], edgecolor="black", label=f"Cluster {i + 1}")
        for i in range(len(unique_leaf_ids))
    ]
    class_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="blue", markersize=7, label="Not Hazardous"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="red", markersize=7, label="Hazardous"),
    ]
    if mistakes.any():
        class_handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color="black",
                markerfacecolor="none",
                markersize=7,
                linewidth=0,
                label="Misclassified",
            )
        )
    ax_clusters.legend(
        handles=cluster_handles + class_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
    )

    ax_scores = axes[1]
    cluster_counts = [item[0] for item in scores]
    f1_scores = [item[1] for item in scores]
    ax_scores.plot(cluster_counts, f1_scores, marker="o", color="black")
    ax_scores.axvline(selected_clusters, color="red", linestyle="--", label="Selected cluster count")
    ax_scores.set_xlabel("Cluster count (leaf nodes)")
    ax_scores.set_ylabel("5-fold CV F1 (train only)")
    ax_scores.set_title("Auto-Selection Curve")
    ax_scores.legend(loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    required_cols = [MOID_COLUMN, MAGNITUDE_COLUMN, TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {args.data}: {missing_cols}. "
            f"Required: {required_cols}"
        )

    df = df.dropna(subset=required_cols).copy()
    df[TARGET_COLUMN] = coerce_bool_labels(df[TARGET_COLUMN])
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Hazard label has only one class after filtering.")

    X = df[[MOID_COLUMN, MAGNITUDE_COLUMN]]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    selected_clusters, best_cv_f1, scores = select_leaf_count(
        X_train,
        y_train,
        args.min_clusters,
        args.max_clusters,
        args.random_state,
        args.selection_tolerance,
    )

    final_model = DecisionTreeClassifier(
        max_leaf_nodes=selected_clusters, random_state=args.random_state
    )
    final_model.fit(X_train, y_train)

    y_test_pred = pd.Series(final_model.predict(X_test), index=y_test.index, dtype=bool)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[False, True]).ravel()

    y_all_pred = pd.Series(final_model.predict(X), index=X.index, dtype=bool)
    actual_clusters = final_model.tree_.n_leaves
    save_plot(
        X,
        y,
        y_all_pred,
        final_model,
        scores,
        actual_clusters,
        args.plot,
    )

    print(f"Rows used: {len(df)}")
    print(f"Selected clusters (leaves): {actual_clusters}")
    print(f"Best 5-fold train CV F1: {best_cv_f1:.4f}")
    print(f"F1 score (hazardous=True) on 20% test: {test_f1:.4f}")
    print("Cluster search (leaves, mean_cv_f1, std_cv_f1):")
    for leaves, mean_f1, std_f1 in scores:
        print(f"  ({leaves}, {mean_f1:.4f}, {std_f1:.4f})")
    print(f"Confusion matrix [TN, FP, FN, TP]: [{tn}, {fp}, {fn}, {tp}]")
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()

import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"

POST_RENAME_MAP = {
    "absolute_magnitude_h": "absolute_magnitude",
    "relative_velocity0": "relative_velocity",
    "is_potentially_hazardous_asteroid": "hazardous",
}

SCATTER_COLUMNS = [
    "data_arc_in_days",
    "observations_used",
    "orbit_uncertainty",
    "minimum_orbit_intersection",
    "eccentricity",
    "perihelion_distance",
    "perihelion_time",
    "absolute_magnitude",
    "relative_velocity",
    "estimated_diameter",
]


def ensure_output_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXPLORATION_DIR, exist_ok=True)


def save_correlation_matrix(df: pd.DataFrame, filename: str, title: str):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, filename), dpi=150)
    plt.close()


def build_post_exploratory_data(processed: pd.DataFrame) -> pd.DataFrame:
    post = processed.copy()
    post["estimated_diameter"] = (
        post["estimated_diameter_kilometers_min"] + post["estimated_diameter_kilometers_max"]
    ) / 2
    post = post.drop(
        columns=[
            "estimated_diameter_kilometers_min",
            "estimated_diameter_kilometers_max",
            "miss_distance_kilometers0",
            "epoch_osculation",
        ]
    )
    post = post.rename(columns=POST_RENAME_MAP)
    return post


def normalize_post_data(post: pd.DataFrame):
    cols_to_normalize = [col for col in post.columns if col != "hazardous"]
    means = post[cols_to_normalize].mean()
    stds = post[cols_to_normalize].std()
    safe_stds = stds.replace(0, 1)

    normalized = post.copy()
    normalized[cols_to_normalize] = (post[cols_to_normalize] - means) / safe_stds

    std_means = pd.DataFrame({"mean": means, "std": stds})
    return normalized, std_means


def save_scatter_plots(normalized: pd.DataFrame):
    colors = normalized["hazardous"].map({True: "red", False: "blue"})
    legend_handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="red", label="Hazardous"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="blue", label="Not Hazardous"),
    ]

    for x_col, y_col in itertools.combinations(SCATTER_COLUMNS, 2):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(normalized[x_col], normalized[y_col], c=colors, alpha=0.3, s=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.legend(handles=legend_handles)
        plt.tight_layout()
        plt.savefig(os.path.join(EXPLORATION_DIR, f"{x_col}_vs_{y_col}.png"), dpi=150)
        plt.close()
        print(f"Saved {x_col}_vs_{y_col}.png")


def main():
    ensure_output_dirs()

    processed = pd.read_csv(os.path.join(DATA_DIR, "processed.csv"))
    save_correlation_matrix(processed, "correlation_matrix.png", "Correlation Matrix (processed.csv)")
    print("Correlation matrix saved to Exploration/correlation_matrix.png")

    post = build_post_exploratory_data(processed)
    post.to_csv(os.path.join(DATA_DIR, "post_exploratory_data.csv"), index=False)
    print("post_exploratory_data.csv saved to Data/")

    save_correlation_matrix(
        post,
        "post_exploratory_correlation_matrix.png",
        "Correlation Matrix (post_exploratory_data.csv)",
    )
    print("Post-exploratory correlation matrix saved to Exploration/post_exploratory_correlation_matrix.png")

    normalized, std_means = normalize_post_data(post)
    normalized.to_csv(os.path.join(DATA_DIR, "normalized.csv"), index=False)
    std_means.to_csv(os.path.join(DATA_DIR, "std_means.csv"))
    print("normalized.csv and std_means.csv saved to Data/")

    save_scatter_plots(normalized)


if __name__ == "__main__":
    main()

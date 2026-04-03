import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "Data"
PLOTS_DIR = "Plots"

OUTLIER_COLUMNS = [
    "estimated_diameter_kilometers_min",
    "estimated_diameter_kilometers_max",
    "relative_velocity0",
    "miss_distance_kilometers0",
    "absolute_magnitude_h",
]

PROCESSED_COLUMNS = [
    "data_arc_in_days",
    "observations_used",
    "orbit_uncertainty",
    "minimum_orbit_intersection",
    "epoch_osculation",
    "eccentricity",
    "perihelion_distance",
    "perihelion_time",
    "absolute_magnitude_h",
    "estimated_diameter_kilometers_min",
    "estimated_diameter_kilometers_max",
    "relative_velocity0",
    "miss_distance_kilometers0",
    "is_potentially_hazardous_asteroid",
]


def ensure_output_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_distribution_plot(values: pd.Series, col: str, lower: float, upper: float, mean: float, median: float):
    bin_width = (values.max() - values.min()) / 500
    if bin_width == 0:
        bin_width = 1
    bins = np.arange(values.min(), values.max() + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(12, 5))
    _, bin_edges, patches = ax.hist(values, bins=bins, color="steelblue", edgecolor="none")

    for patch, left in zip(patches, bin_edges[:-1]):
        if left < lower or left >= upper:
            patch.set_facecolor("salmon")

    ax.axvline(median, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median:.4f}")
    ax.axvline(mean, color="orange", linestyle="-", linewidth=1.5, label=f"Mean: {mean:.4f}")
    ax.axvline(lower, color="red", linestyle=":", linewidth=1.5, label=f"Mean ± 2σ: [{lower:.4f}, {upper:.4f}]")
    ax.axvline(upper, color="red", linestyle=":", linewidth=1.5)

    ax.set_xlim(values.min(), values.max())
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {col} with outlier regions highlighted")

    in_patch = mpatches.Patch(color="steelblue", label="Within 2σ of mean")
    out_patch = mpatches.Patch(color="salmon", label="Outlier (≥2σ from mean)")
    ax.legend(
        handles=[
            in_patch,
            out_patch,
            plt.Line2D([], [], color="green", linestyle="--", label=f"Median: {median:.4f}"),
            plt.Line2D([], [], color="orange", linestyle="-", label=f"Mean: {mean:.4f}"),
            plt.Line2D([], [], color="red", linestyle=":", label="Mean ± 2σ"),
        ],
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{col}_distribution.png"), dpi=150)
    plt.close()


def main():
    ensure_output_dirs()

    raw = pd.read_csv(os.path.join(DATA_DIR, "raw.csv"))
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    raw["IN"] = True

    # Rows missing required inputs cannot flow into the downstream analysis.
    missing_required = raw[PROCESSED_COLUMNS].isna().any(axis=1)
    raw.loc[missing_required, "IN"] = False
    processing = raw.copy()

    for col in OUTLIER_COLUMNS:
        values = processing[col].dropna()
        mean = values.mean()
        std = values.std()
        median = values.median()

        lower = mean - 2 * std
        upper = mean + 2 * std

        save_distribution_plot(values, col, lower, upper, mean, median)

        outlier_mask = processing[col].notna() & ((processing[col] < lower) | (processing[col] > upper))
        processing.loc[outlier_mask, "IN"] = False

        print(
            f"{col}: mean={mean:.4f}, std={std:.4f}, median={median:.4f}, "
            f"outliers marked={outlier_mask.sum()}, total OUT so far={(~processing['IN']).sum()}"
        )

    processing.to_csv(os.path.join(DATA_DIR, "processing.csv"), index=False)

    processed = processing.loc[processing["IN"], PROCESSED_COLUMNS].copy()
    processed.to_csv(os.path.join(DATA_DIR, "processed.csv"), index=False)

    print(f"\nMissing required rows excluded: {missing_required.sum()}")
    print(f"Done. processing.csv rows: {len(processing)}, processed.csv rows: {len(processed)}")


if __name__ == "__main__":
    main()

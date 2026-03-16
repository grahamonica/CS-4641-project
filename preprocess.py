import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_DIR = "Data"
PLOTS_DIR = "Plots"

# Step 1: Load raw.csv and create processing.csv with IN column
raw = pd.read_csv(os.path.join(DATA_DIR, "raw.csv"))
raw["IN"] = True
processing = raw.copy()
processing.to_csv(os.path.join(DATA_DIR, "processing.csv"), index=False)

# Columns to process for outlier removal (using all data each time per step 3)
outlier_cols = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]

# Steps 2 & 3: For each column, plot histogram and mark outliers
for col in outlier_cols:
    values = processing[col]  # Use all rows each time

    mean = values.mean()
    std = values.std()
    median = values.median()

    lower = mean - 2 * std
    upper = mean + 2 * std

    bin_width = (values.max() - values.min()) / 500
    bins = np.arange(values.min(), values.max() + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(12, 5))

    counts, bin_edges, patches = ax.hist(values, bins=bins, color="steelblue", edgecolor="none")

    # Color outlier bars red
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
    ax.legend(handles=[in_patch, out_patch,
                        plt.Line2D([], [], color="green", linestyle="--", label=f"Median: {median:.4f}"),
                        plt.Line2D([], [], color="orange", linestyle="-", label=f"Mean: {mean:.4f}"),
                        plt.Line2D([], [], color="red", linestyle=":", label=f"Mean ± 2σ")],
              loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{col}_distribution.png"), dpi=150)
    plt.close()

    # Mark outlier rows as OUT in processing
    outlier_mask = (processing[col] < lower) | (processing[col] > upper)
    processing.loc[outlier_mask, "IN"] = False

    print(f"{col}: mean={mean:.4f}, std={std:.4f}, median={median:.4f}, "
          f"outliers marked={outlier_mask.sum()}, total OUT so far={(~processing['IN']).sum()}")

# Save updated processing.csv with IN column reflecting all outlier passes
processing.to_csv(os.path.join(DATA_DIR, "processing.csv"), index=False)

# Step 4: Build processed.csv — drop cols 1,2,7,8 (id, name, orbiting_body, sentry_object)
# Keep only IN rows, then drop the IN column
cols_to_drop = ["id", "name", "orbiting_body", "sentry_object", "IN"]
processed = processing[processing["IN"]].drop(columns=cols_to_drop)
processed.to_csv(os.path.join(DATA_DIR, "processed.csv"), index=False)

print(f"\nDone. processing.csv rows: {len(processing)}, processed.csv rows: {len(processed)}")

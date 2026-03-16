import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"

# Step 1: Correlation matrix from processed.csv
processed = pd.read_csv(os.path.join(DATA_DIR, "processed.csv"))

corr = processed.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

ax.set_title("Correlation Matrix (processed.csv)")
plt.tight_layout()
plt.savefig(os.path.join(EXPLORATION_DIR, "correlation_matrix.png"), dpi=150)
plt.close()

print("Correlation matrix saved to Exploration/correlation_matrix.png")

# Step 2: Create post_exploratory_data.csv
# Replace est_diameter_min and est_diameter_max with their average (estimated_diameter)
# Remove miss_distance (0.03 correlation with hazardous — insufficient predictive value)
post = processed.copy()
post["estimated_diameter"] = (post["est_diameter_min"] + post["est_diameter_max"]) / 2
post = post.drop(columns=["est_diameter_min", "est_diameter_max", "miss_distance"])
post.to_csv(os.path.join(DATA_DIR, "post_exploratory_data.csv"), index=False)

print("post_exploratory_data.csv saved to Data/")

# Step 3: Normalize post_exploratory_data.csv (mean=0, std=1) for all columns except hazardous
cols_to_normalize = [c for c in post.columns if c != "hazardous"]

means = post[cols_to_normalize].mean()
stds = post[cols_to_normalize].std()

normalized = post.copy()
normalized[cols_to_normalize] = (post[cols_to_normalize] - means) / stds
normalized.to_csv(os.path.join(DATA_DIR, "normalized.csv"), index=False)

std_means = pd.DataFrame({"mean": means, "std": stds})
std_means.to_csv(os.path.join(DATA_DIR, "std_means.csv"))

print("normalized.csv and std_means.csv saved to Data/")

# Step 4: Scatter plots of each predictor pair, colored by hazardous
predictors = ["estimated_diameter", "relative_velocity", "absolute_magnitude"]
pairs = [(predictors[i], predictors[j]) for i in range(len(predictors)) for j in range(i + 1, len(predictors))]

colors = normalized["hazardous"].map({True: "red", False: "blue"})

for x_col, y_col in pairs:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normalized[x_col], normalized[y_col], c=colors, alpha=0.3, s=5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    red_patch = plt.Line2D([], [], marker="o", color="w", markerfacecolor="red", label="Hazardous")
    blue_patch = plt.Line2D([], [], marker="o", color="w", markerfacecolor="blue", label="Not Hazardous")
    ax.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, f"{x_col}_vs_{y_col}.png"), dpi=150)
    plt.close()
    print(f"Saved {x_col}_vs_{y_col}.png")

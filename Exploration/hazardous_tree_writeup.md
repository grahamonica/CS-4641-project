# Hazardous Asteroid Classification Write-Up

## Goal
Predict `hazardous` from:
- `minimum_orbit_intersection`
- `absolute_magnitude`

Both scripts now use a strict **80/20 split**:
- 80% train
- 20% test

Model complexity is selected automatically from the training set only using 5-fold cross-validation and F1 scoring.

## 1) Decision Tree (`decision_tree.py`)
- Candidate depths searched: 1 to 8.
- Selection rule: choose the **smallest depth** within tolerance of the best mean CV F1.
- Selected depth: **2**.
- Test F1 (hazardous class): **0.9898**.
- Test confusion matrix: `[[4062, 4], [4, 390]]`.

Interpretation:
- The boundary between hazardous and non-hazardous points is effectively captured with a shallow tree.
- This matches the expected behavior that near-perfect separation appears in about two layers.

## 2) Cluster-Labeled Tree Regions (`hazardous_cluster_model.py`)
- Uses tree leaves as data-driven clusters.
- Candidate leaf counts searched: 2 to 8.
- Selection rule: choose the **smallest leaf count** within tolerance of the best mean CV F1.
- Selected clusters (leaves): **3**.
- Test F1 (hazardous class): **0.9898**.
- Test confusion matrix: `[[4062, 4], [4, 390]]`.

Interpretation:
- The model finds a small number of regions that align with the hazard geometry in the MOID vs magnitude plot.
- Cluster labeling and boundaries are generated from learned splits, not hard-coded thresholds.

## Conclusion
Using only `minimum_orbit_intersection` and `absolute_magnitude`, both pipelines achieve near-perfect classification on the 20% held-out test split while selecting low model complexity automatically.

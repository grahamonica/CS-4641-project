import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_DIR = "Data"
EXPLORATION_DIR = "Exploration"

def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv(os.path.join(DATA_DIR, "normalized.csv"))

    X = df.drop("eccentricity", axis=1)
    y = df["eccentricity"]

    # -----------------------------
    # 2. Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. Feature engineering (log transforms)
    # -----------------------------
    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    for col in X_train.columns:
        # log only works for strictly positive values
        if (X_train[col] > 0).all():
            X_train_fe[f"log_{col}"] = np.log(X_train[col])
            X_test_fe[f"log_{col}"] = np.log(X_test[col])

    for col in X_train.columns:
        # sqrt only works for strictly positive values
        if (X_train[col] > 0).all():
            X_train_fe[f"sqrt_{col}"] = np.sqrt(X_train[col])
            X_test_fe[f"sqrt_{col}"] = np.sqrt(X_test[col])

    # -----------------------------
    # 4. Polynomial + interaction features
    # -----------------------------
    poly = PolynomialFeatures(degree=3, include_bias=False)

    X_train_poly = poly.fit_transform(X_train_fe)
    X_test_poly = poly.transform(X_test_fe)
    
    # Get feature names after polynomial transformation
    poly_feature_names = poly.get_feature_names_out(X_train_fe.columns)
    print("Total polynomial features created:", len(poly_feature_names))

    # -----------------------------
    # 5. Scale features (important for Lasso)
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # -----------------------------
    # 6. Lasso feature selection
    # -----------------------------
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    selected_mask = lasso.coef_ != 0
    
    # Track which features were selected
    selected_features = poly_feature_names[selected_mask]
    print(f"\nFeatures selected by Lasso ({len(selected_features)}/{len(poly_feature_names)}):")
    for i, feature in enumerate(selected_features):
        coef_index = np.where(selected_mask)[0][i]
        print(f"  {feature}: coef={lasso.coef_[coef_index]:.6f}")

    X_train_selected = X_train_scaled[:, selected_mask]
    X_test_selected = X_test_scaled[:, selected_mask]

    # -----------------------------
    # 7. Final linear regression model
    # -----------------------------
    final_model = LinearRegression()
    final_model.fit(X_train_selected, y_train)

    preds = final_model.predict(X_test_selected)
    
    # Print final model coefficients
    print(f"\nFinal Linear Regression Model Coefficients:")
    for feature, coef in zip(selected_features, final_model.coef_):
        print(f"  {feature}: {coef:.6f}")
    print(f"  Intercept: {final_model.intercept_:.6f}")

    # -----------------------------
    # 8. Evaluation
    # -----------------------------
    print("Mean Squared Error:")
    print(mean_squared_error(y_test, preds))

    print("Mean Absolute Error:")
    print(mean_absolute_error(y_test, preds))

    print("R^2 score:", r2_score(y_test, preds))

    # Compared to baseline
    y_pred_baseline = np.full_like(y_test, y_train.mean())

    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    print("Baseline MSE:", baseline_mse)

    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    print("Baseline MAE:", baseline_mae)

    baseline_r2 = r2_score(y_test, y_pred_baseline)
    print("Baseline R^2:", baseline_r2)

    # -----------------------------
    # 9. Graph
    # -----------------------------

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6)

    # perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())

    plt.plot([min_val, max_val], [min_val, max_val], color='red')

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Linear Regression)")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLORATION_DIR, "linear_regression.png"), dpi=150)
    plt.close()

    print("Linear regression model saved to Exploration/linear_regression.png")


if __name__ == "__main__":
    main()
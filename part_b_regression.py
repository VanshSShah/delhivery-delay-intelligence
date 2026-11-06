"""Part B: Delivery-Hours Regression pipeline, tree depth diagnostic, models evaluation & CV comparison."""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


def run_part_b():
    os.makedirs("outputs", exist_ok=True)

    # Read modelling table generated in Part A
    df = pd.read_csv(os.path.join("data", "modelling_table.csv"))

    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Train / Validation / Test split (840 / 180 / 180)
    # First split: 70% train (840), 30% temp (360)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # Second split: split temp 50/50 into validation (180) and test (180)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_rows = len(X_train)
    val_rows = len(X_val)
    test_rows = len(X_test)

    # 2. Scaling without leakage
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on training rows ONLY

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = round(float(np.abs(X_train_scaled.mean(axis=0)).max()), 4)
    test_scaled_max_abs_mean = round(float(np.abs(X_test_scaled.mean(axis=0)).max()), 4)

    # Write outputs/part_b_checks.csv
    checks_df = pd.DataFrame([
        ("train_rows", train_rows),
        ("val_rows", val_rows),
        ("test_rows", test_rows),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
    ], columns=["metric", "value"])
    checks_df.to_csv(os.path.join("outputs", "part_b_checks.csv"), index=False)

    # 3. Baseline Model (Predict mean of y_train)
    y_train_mean = y_train.mean()
    y_val_baseline_pred = np.full_like(y_val, fill_value=y_train_mean)

    base_mae = round(mean_absolute_error(y_val, y_val_baseline_pred), 4)
    base_mse = round(mean_squared_error(y_val, y_val_baseline_pred), 4)
    base_rmse = round(np.sqrt(base_mse), 4)
    base_r2 = round(r2_score(y_val, y_val_baseline_pred), 4)

    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_rows = []
    best_depth = None
    lowest_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)

        tr_pred = dt.predict(X_train)
        va_pred = dt.predict(X_val)

        tr_rmse = round(np.sqrt(mean_squared_error(y_train, tr_pred)), 4)
        va_rmse = round(np.sqrt(mean_squared_error(y_val, va_pred)), 4)

        depth_str = "unlimited" if d is None else str(d)
        tree_rows.append({
            "max_depth": depth_str,
            "train_rmse": tr_rmse,
            "val_rmse": va_rmse,
        })

        if va_rmse < lowest_val_rmse:
            lowest_val_rmse = va_rmse
            best_depth = d

    tree_depth_df = pd.DataFrame(tree_rows)
    tree_depth_df.to_csv(os.path.join("outputs", "part_b_tree_depth.csv"), index=False)

    # 5. Four Candidate Models evaluation on validation set
    models_dict = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": base_mae,
            "MSE": base_mse,
            "RMSE": base_rmse,
            "R2": base_r2,
        }
    ]

    fitted_models = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        val_preds = model.predict(X_val)
        mae = round(mean_absolute_error(y_val, val_preds), 4)
        mse = round(mean_squared_error(y_val, val_preds), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_val, val_preds), 4)

        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

    # 6. Learning-rate experiment for Gradient Boosting
    learning_rates = [0.01, 0.1, 1.0]
    lr_rows = []

    for lr in learning_rates:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gbr.fit(X_train, y_train)
        val_preds = gbr.predict(X_val)

        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = round(np.sqrt(val_mse), 4)
        val_r2 = round(r2_score(y_val, val_preds), 4)

        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        })

    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv(os.path.join("outputs", "part_b_gb_learning_rate.csv"), index=False)

    # 7. Cross-validated comparison and final model evaluation
    # Combine train + val sets (1,020 rows) for 5-fold CV
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # Identify top 2 candidates by validation RMSE
    val_metrics_df = pd.DataFrame(metrics_rows)
    val_candidates = val_metrics_df[val_metrics_df["model"] != "mean_baseline"].copy()
    val_candidates = val_candidates.sort_values(by="RMSE", ascending=True)
    top2_names = val_candidates.iloc[:2]["model"].tolist()

    cv_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for name in top2_names:
        model_obj = models_dict[name]
        scores = cross_val_score(model_obj, X_train_val, y_train_val, cv=cv_kf, scoring="r2")
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join("outputs", "part_b_cv.csv"), index=False)

    # Winning candidate on validation RMSE score
    winning_model_name = val_candidates.iloc[0]["model"]
    winning_model = fitted_models[winning_model_name]

    # Evaluate winning model ONCE on test rows
    test_preds = winning_model.predict(X_test)
    test_mae = round(mean_absolute_error(y_test, test_preds), 4)
    test_mse = round(mean_squared_error(y_test, test_preds), 4)
    test_rmse = round(np.sqrt(test_mse), 4)
    test_r2 = round(r2_score(y_test, test_preds), 4)

    metrics_rows.append({
        "model": winning_model_name,
        "split": "test",
        "MAE": test_mae,
        "MSE": test_mse,
        "RMSE": test_rmse,
        "R2": test_r2,
    })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join("outputs", "part_b_metrics.csv"), index=False)

    print(f"Part B execution completed successfully! Winning model: {winning_model_name}")


if __name__ == "__main__":
    run_part_b()
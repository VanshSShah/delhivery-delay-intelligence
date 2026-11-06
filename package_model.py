"""Part D (Task 5): Package the Final Delay Model Artifacts and Perform Reload Verification."""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import clean_shipments, FEATURES


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read raw shipments to obtain median impute values
    raw_df = pd.read_csv(os.path.join("data", "raw_shipments.csv"))
    _, medians = clean_shipments(raw_df)

    # Save models/impute_values.csv
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": medians["distance_km"]},
        {"column": "parcel_weight_kg", "median": medians["parcel_weight_kg"]},
        {"column": "declared_value_inr", "median": medians["declared_value_inr"]},
    ])
    impute_df.to_csv(os.path.join("models", "impute_values.csv"), index=False)

    # 2. Save models/feature_order.csv
    feature_order_df = pd.DataFrame({"feature": FEATURES})
    feature_order_df.to_csv(os.path.join("models", "feature_order.csv"), index=False)

    # 3. Read Part C config to retrieve hyperparameters and threshold
    config_df = pd.read_csv(os.path.join("outputs", "part_c_final_config.csv"))
    config = config_df.iloc[0]

    selector_k = int(config["selector_k"])
    threshold = float(config["threshold"])
    n_estimators = int(config["n_estimators"])
    min_samples_leaf = int(config["min_samples_leaf"])
    max_depth_val = config["max_depth"]

    if str(max_depth_val).lower() == "unlimited" or pd.isna(max_depth_val):
        max_depth = None
    else:
        max_depth = int(max_depth_val)

    # Save models/delay_threshold.csv
    pd.DataFrame([{"threshold": threshold}]).to_csv(
        os.path.join("models", "delay_threshold.csv"), index=False
    )

    # 4. Rebuild Part C Stratified Split & Fit Pipeline
    modelling_table = pd.read_csv(os.path.join("data", "modelling_table.csv"))
    X = modelling_table[FEATURES]
    y = modelling_table["is_delayed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", rf_model),
    ])
    pipeline.fit(X_train, y_train)

    # Save models/delay_model.pkl
    model_path = os.path.join("models", "delay_model.pkl")
    joblib.dump(pipeline, model_path)

    # 5. Reload Model & Verify Identical Output on Test Set
    reloaded_pipeline = joblib.load(model_path)

    in_memory_probs = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]

    prob_diff = float(np.abs(in_memory_probs - reloaded_probs).max())
    reload_max_abs_diff = round(prob_diff, 4)

    reload_check_df = pd.DataFrame([
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", reload_max_abs_diff),
        ("threshold", threshold),
    ], columns=["metric", "value"])

    reload_check_df.to_csv(os.path.join("outputs", "part_d_reload_check.csv"), index=False)

    print("Model packaging and reload check completed successfully!")


if __name__ == "__main__":
    run_package_model()
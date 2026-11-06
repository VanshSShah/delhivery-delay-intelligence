"""Part D (Task 6): End-to-End Prediction Pipeline for New Unlabelled Shipments."""

import os
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import clean_shipments, build_features, FEATURES


def run_predict_delay():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load packaged model artifacts
    model_path = os.path.join("models", "delay_model.pkl")
    impute_path = os.path.join("models", "impute_values.csv")
    thresh_path = os.path.join("models", "delay_threshold.csv")

    pipeline = joblib.load(model_path)

    impute_df = pd.read_csv(impute_path)
    medians_dict = dict(zip(impute_df["column"], impute_df["median"]))

    thresh_df = pd.read_csv(thresh_path)
    threshold = float(thresh_df.iloc[0]["threshold"])

    # 2. Read new unlabelled shipments and hub metadata
    new_shipments_path = os.path.join("data", "new_shipments.csv")
    hubs_path = os.path.join("data", "hubs.csv")

    new_raw = pd.read_csv(new_shipments_path)
    hubs = pd.read_csv(hubs_path)

    # 3. Clean and engineer features using saved medians
    clean_df, _ = clean_shipments(new_raw, medians=medians_dict)
    feature_df = build_features(clean_df, hubs)

    # Extract exact features in pinned order
    X_new = feature_df[FEATURES]

    # 4. Predict delay probabilities and binary alert flags
    delay_probs = pipeline.predict_proba(X_new)[:, 1]
    predicted_alerts = (delay_probs >= threshold).astype(int)

    # 5. Build final output predictions table
    predictions_df = pd.DataFrame({
        "shipment_id": feature_df["shipment_id"],
        "hub_id": feature_df["hub_id"],
        "delay_probability": np.round(delay_probs, 4),
        "predicted_delay_alert": predicted_alerts,
    })

    # Sort by shipment_id to ensure clean order
    predictions_df = predictions_df.sort_values("shipment_id").reset_index(drop=True)

    # Write outputs/predictions.csv
    predictions_df.to_csv(os.path.join("outputs", "predictions.csv"), index=False)

    total_scored = len(predictions_df)
    alert_count = int(predictions_df["predicted_delay_alert"].sum())
    alert_rate = round(alert_count / total_scored, 4)

    print(
        f"Prediction complete! Scored {total_scored} shipments. "
        f"Alerts triggered: {alert_count} ({alert_rate * 100:.2f}%)."
    )


if __name__ == "__main__":
    run_predict_delay()
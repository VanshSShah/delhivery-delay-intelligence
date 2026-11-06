"""Part A: Data Cleaning, Preprocessing, Target Creation, Checks & EDA outputs."""

import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES


def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Read raw datasets
    raw_path = os.path.join("data", "raw_shipments.csv")
    hubs_path = os.path.join("data", "hubs.csv")
    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    raw_rows = len(raw)

    # Step 1: Count metrics prior to cleaning (after dedup)
    dedup_raw = raw.drop_duplicates().reset_index(drop=True)
    rows_after_dedup = len(dedup_raw)
    duplicate_rows_removed = raw_rows - rows_after_dedup

    missing_distance_km_before = dedup_raw["distance_km"].isna().sum()
    missing_parcel_weight_before = (
        dedup_raw["parcel_weight_kg"].isna() | (dedup_raw["parcel_weight_kg"] > 100)
    ).sum() - (dedup_raw["parcel_weight_kg"] > 100).sum()
    
    # Missing weight before count (excluding sentinel values)
    # dedup_raw has NaN and sentinel values
    weight_nans = dedup_raw["parcel_weight_kg"].isna().sum()
    sentinel_weight_rows = (dedup_raw["parcel_weight_kg"] > 100).sum()
    missing_parcel_weight_before = weight_nans

    # Declared value missing before
    missing_declared_value_before = dedup_raw["declared_value"].isna().sum()
    missing_payment_mode_before = dedup_raw["payment_mode"].isna().sum()

    service_type_spellings_before = dedup_raw["service_type"].nunique()

    # Step 2: Apply clean_shipments and build_features
    clean_df, medians = clean_shipments(raw)
    service_type_spellings_after = clean_df["service_type"].nunique()

    modelling_table = build_features(clean_df, hubs)
    rows_after_hub_merge = len(modelling_table)

    # Add target: is_delayed (1 when delivery_hours > promised_hours, else 0)
    modelling_table["is_delayed"] = (
        modelling_table["delivery_hours"] > modelling_table["promised_hours"]
    ).astype(int)

    # Save data/modelling_table.csv
    modelling_table.to_csv(os.path.join("data", "modelling_table.csv"), index=False)

    # -------------------------------------------------------------
    # Outputs 1: outputs/part_a_checks.csv
    # -------------------------------------------------------------
    # Compute nulls over 17 FEATURES + delivery_hours + is_delayed
    cols_to_check = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = modelling_table[cols_to_check].isna().sum().sum()

    delayed_count = modelling_table["is_delayed"].sum()
    delay_rate = round(modelling_table["is_delayed"].mean(), 4)

    hours_q1 = unrounded_q1 = modelling_table["delivery_hours"].quantile(0.25)
    hours_q3 = unrounded_q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = unrounded_q3 - unrounded_q1
    hours_lower_bound = unrounded_q1 - 1.5 * iqr
    hours_upper_bound = unrounded_q3 + 1.5 * iqr

    hours_outlier_count = (
        (modelling_table["delivery_hours"] < hours_lower_bound)
        | (modelling_table["delivery_hours"] > hours_upper_bound)
    ).sum()

    declared_value_mean_inr = round(modelling_table["declared_value_inr"].mean(), 4)
    declared_value_median_inr = round(modelling_table["declared_value_inr"].median(), 4)
    log_declared_value_mean = round(modelling_table["log_declared_value"].mean(), 4)
    log_declared_value_median = round(modelling_table["log_declared_value"].median(), 4)

    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round(hours_q1, 4)),
        ("hours_q3", round(hours_q3, 4)),
        ("hours_lower_bound", round(hours_lower_bound, 4)),
        ("hours_upper_bound", round(hours_upper_bound, 4)),
        ("hours_outlier_count", hours_outlier_count),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("declared_value_median_inr", declared_value_median_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("log_declared_value_median", log_declared_value_median),
    ]

    checks_df = pd.DataFrame(checks_data, columns=["metric", "value"])
    checks_df.to_csv(os.path.join("outputs", "part_a_checks.csv"), index=False)

    # -------------------------------------------------------------
    # Outputs 2: outputs/part_a_delay_by_group.csv
    # -------------------------------------------------------------
    # Bins: 0-200, 201-400, 401-600, 601+
    # Edges: 0, 200, 400, 600, 2000 (right edge inclusive)
    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    # 1. service_type
    for st in ["Economy", "Express", "Standard"]:
        sub = modelling_table[modelling_table["service_type"] == st]
        group_rows.append({
            "dimension": "service_type",
            "group": st,
            "shipments": len(sub),
            "delayed": sub["is_delayed"].sum(),
            "delay_rate": round(sub["is_delayed"].mean(), 4),
            "mean_hours": round(sub["delivery_hours"].mean(), 4),
        })

    # 2. hub_tier
    for ht in [1, 2, 3]:
        sub = modelling_table[modelling_table["hub_tier"] == ht]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(ht),
            "shipments": len(sub),
            "delayed": sub["is_delayed"].sum(),
            "delay_rate": round(sub["is_delayed"].mean(), 4),
            "mean_hours": round(sub["delivery_hours"].mean(), 4),
        })

    # 3. distance_band
    for db in ["0-200", "201-400", "401-600", "601+"]:
        sub = modelling_table[modelling_table["distance_band"] == db]
        group_rows.append({
            "dimension": "distance_band",
            "group": db,
            "shipments": len(sub),
            "delayed": sub["is_delayed"].sum(),
            "delay_rate": round(sub["is_delayed"].mean(), 4),
            "mean_hours": round(sub["delivery_hours"].mean(), 4),
        })

    delay_by_group_df = pd.DataFrame(group_rows)
    delay_by_group_df.to_csv(os.path.join("outputs", "part_a_delay_by_group.csv"), index=False)

    # -------------------------------------------------------------
    # Outputs 3: outputs/part_a_correlations.csv
    # -------------------------------------------------------------
    numeric_df = modelling_table.select_dtypes(include=[np.number])
    corr_df = numeric_df.corr().round(4)
    corr_df.to_csv(os.path.join("outputs", "part_a_correlations.csv"))

    print("Part A execution completed successfully!")


if __name__ == "__main__":
    run_part_a()
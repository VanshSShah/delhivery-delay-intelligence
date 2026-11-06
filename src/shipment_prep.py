"""Shared data preprocessing and feature engineering pipeline for Delhivery Last-Mile Delay Intelligence."""

import numpy as np
import pandas as pd

# Standard promised hours per service type
PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

# Pinned feature list in exact order
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

# Non-informative synthetic features
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Clean raw shipment DataFrame according to Part A specification."""
    df = raw.copy()

    # Step 1: Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Step 2: Clean service_type
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # Step 3: Payment mode missing values -> 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4: Parse declared_value to float (declared_value_inr)
    if "declared_value" in df.columns:
        cleaned_val = (
            df["declared_value"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["declared_value_inr"] = pd.to_numeric(cleaned_val, errors="coerce")
        df = df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch_time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step 6: Treat parcel_weight_kg > 100 as NaN sentinel
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step 7: Calculate or use provided medians
    if medians is None:
        computed_medians = {
            "distance_km": round(float(df["distance_km"].median()), 4),
            "parcel_weight_kg": round(float(df["parcel_weight_kg"].median()), 4),
            "declared_value_inr": round(float(df["declared_value_inr"].median()), 4),
        }
        medians_dict = computed_medians
    else:
        medians_dict = {
            "distance_km": round(float(medians["distance_km"]), 4),
            "parcel_weight_kg": round(float(medians["parcel_weight_kg"]), 4),
            "declared_value_inr": round(float(medians["declared_value_inr"]), 4),
        }

    # Fill missing values using the medians
    df["distance_km"] = df["distance_km"].fillna(medians_dict["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians_dict["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians_dict["declared_value_inr"])

    return df, medians_dict


def build_features(clean, hubs):
    """Attach hub information and build standard modeling features."""
    df = clean.copy()

    # Left merge hub details
    df = df.merge(hubs, on="hub_id", how="left")

    # promised_hours from service_type lookup
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # is_cod: 1 if COD, else 0 (Unknown gives 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # is_weekend: 1 if Saturday (5) or Sunday (6)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # dispatch_hour: integer hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # log_declared_value: natural log of declared_value_inr
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
"""Part D (Task 1-4): Hub Profiling, K-Means Segmentation, Separation Evidence & Mismatch Detection."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_part_d_segments():
    os.makedirs("outputs", exist_ok=True)

    # 1. Hub Profile Calculation
    modelling_table = pd.read_csv(os.path.join("data", "modelling_table.csv"))

    hub_profile = (
        modelling_table.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
    )

    # Sort strictly by ascending hub_id (HUB01 to HUB24)
    hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

    # Round profile metrics to 4 decimal places
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        hub_profile[col] = hub_profile[col].round(4)

    hub_profile.to_csv(os.path.join("outputs", "hub_profile.csv"), index=False)

    # 2 & 3. K-Means Segmentation & Separation Evidence
    profile_features = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    X_hub = hub_profile[profile_features].values

    scaler = StandardScaler()
    X_hub_scaled = scaler.fit_transform(X_hub)

    # Compute across-to-within distance ratio for K in [2, 3, 4, 5]
    n_hubs = len(X_hub_scaled)
    sep_rows = []

    for k_val in [2, 3, 4, 5]:
        kmeans_k = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels_k = kmeans_k.fit_predict(X_hub_scaled)

        across_dists = []
        within_dists = []

        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                dist = float(np.linalg.norm(X_hub_scaled[i] - X_hub_scaled[j]))
                if labels_k[i] == labels_k[j]:
                    within_dists.append(dist)
                else:
                    across_dists.append(dist)

        across_mean = np.mean(across_dists)
        within_mean = np.mean(within_dists)
        ratio = round(float(across_mean / within_mean), 4)

        # Count cluster sizes and join with hyphens descending
        counts = pd.Series(labels_k).value_counts().sort_values(ascending=False).tolist()
        sizes_str = "-".join(map(str, counts))

        sep_rows.append({
            "K": k_val,
            "across_over_within": ratio,
            "cluster_sizes": sizes_str
        })

    pd.DataFrame(sep_rows).to_csv(os.path.join("outputs", "part_d_separation.csv"), index=False)

    # 4. Final K=3 Cluster Assignment, Naming & Mismatch Analysis
    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_3 = kmeans_3.fit_predict(X_hub_scaled)

    hub_profile["cluster_raw"] = labels_3

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_means = (
        hub_profile.groupby("cluster_raw")["delay_rate"]
        .mean()
        .reset_index()
        .sort_values("delay_rate")
        .reset_index(drop=True)
    )

    segment_mapping = {
        cluster_means.loc[0, "cluster_raw"]: "Standard",
        cluster_means.loc[1, "cluster_raw"]: "Watch",
        cluster_means.loc[2, "cluster_raw"]: "Escalate",
    }

    hub_profile["segment"] = hub_profile["cluster_raw"].map(segment_mapping)

    # Load hubs info metadata
    hubs_meta = pd.read_csv(os.path.join("data", "hubs.csv"))
    hub_segments = hub_profile[["hub_id", "segment"]].merge(
        hubs_meta[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
    )
    hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments.to_csv(os.path.join("outputs", "hub_segments.csv"), index=False)

    # Write segment profile table
    segment_prof_rows = []
    for seg_name in ["Standard", "Watch", "Escalate"]:
        sub = hub_profile[hub_profile["segment"] == seg_name]
        segment_prof_rows.append({
            "segment": seg_name,
            "hubs": len(sub),
            "mean_delivery_hours": round(float(sub["mean_delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["delay_rate"].mean()), 4),
            "mean_distance_km": round(float(sub["mean_distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["mean_n_stops"].mean()), 4),
        })

    pd.DataFrame(segment_prof_rows).to_csv(os.path.join("outputs", "segment_profile.csv"), index=False)

    # Detect Tier vs Segment Mismatch
    # Rule: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments["expected_segment"] = hub_segments["hub_tier"].map(tier_implied)

    mismatches = hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]].copy()
    mismatches = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]]
    mismatches.to_csv(os.path.join("outputs", "tier_vs_segment_mismatch.csv"), index=False)

    print(f"Part D Segments execution completed! Detected {len(mismatches)} tier mismatches.")


if __name__ == "__main__":
    run_part_d_segments()
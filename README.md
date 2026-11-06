# Delhivery Last-Mile Delay Intelligence Pipeline

A comprehensive machine learning intelligence system designed for **Delhivery** to predict last-mile delivery delays, profile hub performance, detect operational mismatches, and automate proactive dispatch alerts.

---

## Executive Summary & Key Results

- **Data Integrity & Preprocessing**: Cleaned 1,224 raw shipment records down to 1,200 non-duplicate rows. Successfully handled non-standard text encodings (`₹`), standardized case variations in service types, and replaced scanner fault sentinel weight values (`> 100 kg`) with strict, leakage-free median imputations.
- **Regression Analysis (Part B)**: Built and evaluated multiple regression algorithms to model continuous delivery hours. **Linear Regression** and **Gradient Boosting** achieved top validation performance with $R^2 \ge 0.93$. On the holdout test set, the final regression model achieved an $R^2$ of **0.9493** and RMSE of **4.52 hours**.
- **Classification & Alert System (Part C)**: Feature signal ranking identified distance, hub tier, daily capacity, number of stops, and service promised hours as the primary drivers of delivery delays. Using top-ranked features, a Random Forest classifier was optimized via grid and randomized searches. Evaluating against an alert threshold of **0.50** produced a final test set Precision of **0.6889** and Recall of **0.7561** ($F_1 = 0.7209$).
- **Hub Profiling & Mismatch Detection (Part D)**: K-Means clustering ($K=3$) grouped 24 nationwide hubs into three operational performance tiers: **Standard**, **Watch**, and **Escalate**. The model detected **4 major hub mismatches** where designated hub tier labels failed to represent real operating conditions (`HUB07 Kolkata`, `HUB13 Nagpur`, `HUB15 Kochi`, and `HUB22 Dehradun`).
- **Production Scoring**: Packaged model artifacts allow seamless batch scoring of new unlabelled shipments via `predict_delay.py`, identifying **8 high-risk delay alerts (16.67%)** out of 48 new incoming dispatches.

---

## Project Structure

```text
delhivery-delay-intelligence/
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   ├── __init__.py
│   └── shipment_prep.py
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
├── outputs/
│   ├── part_a_checks.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_a_correlations.csv
│   ├── part_b_checks.csv
│   ├── part_b_tree_depth.csv
│   ├── part_b_metrics.csv
│   ├── part_b_gb_learning_rate.csv
│   ├── part_b_cv.csv
│   ├── part_c_models.csv
│   ├── part_c_feature_scores.csv
│   ├── part_c_selectk.csv
│   ├── part_c_tuning.csv
│   ├── part_c_thresholds.csv
│   ├── part_c_final_config.csv
│   ├── part_c_final_test.csv
│   ├── hub_profile.csv
│   ├── part_d_separation.csv
│   ├── hub_segments.csv
│   ├── segment_profile.csv
│   ├── tier_vs_segment_mismatch.csv
│   ├── part_d_reload_check.csv
│   └── predictions.csv
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md

Setup & Environment Setup
1. Requirements
Python: 3.11.x

Dependencies (pinned in requirements.txt):

pandas==2.2.3

numpy==1.26.4

scikit-learn==1.5.2

joblib==1.4.2

2. Virtual Environment Setup
PowerShell
# Create virtual environment
py -3.11 -m venv .venv

# Activate virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

Pipeline Execution Guide
Run all pipeline modules in sequence from the project root directory:

PowerShell
# 1. Generate synthetic dataset
python generate_data.py

# 2. Run Part A: Data cleaning, preprocessing & EDA
python part_a_clean_eda.py

# 3. Run Part B: Regression modelling for delivery hours
python part_b_regression.py

# 4. Run Part C: Classification, feature selection & threshold tuning
python part_c_classification.py

# 5. Run Part D (Tasks 1-4): Hub segmentation & mismatch detection
python part_d_segments.py

# 6. Run Part D (Task 5): Package trained model & test reload integrity
python package_model.py

# 7. Run Part D (Task 6): Score new unlabelled shipments
python predict_delay.py

Summary of Modeling & Segmentation Results
Part B: Delivery-Hours Regression Performance
Model,Split,MAE (Hours),MSE,RMSE (Hours),R2
Mean Baseline,Validation,15.7952,355.5204,18.8552,-0.0212
Linear Regression,Validation,3.9222,23.9235,4.8912,0.9313
Decision Tree (Depth 6),Validation,5.0971,39.4338,6.2796,0.8867
Random Forest,Validation,4.2882,30.3550,5.5095,0.9128
Gradient Boosting,Validation,3.9166,24.1722,4.9165,0.9306
Linear Regression (Final),Test,3.5671,20.4624,4.5235,0.9493
Part C: Delay Classification Performance
Model / Evaluation,Threshold,Precision,Recall,F1​ Score,Accuracy
Blind Baseline (All On-Time),0.50,0.0000,0.0000,0.0000,0.7667
Logistic Regression,0.50,0.7778,0.8333,0.8046,0.9056
Decision Tree (Depth 6),0.50,0.7561,0.7381,0.7470,0.8833
Default Random Forest,0.50,0.8378,0.7381,0.7848,0.9056
Final Random Forest (Top 7 Features),0.50,0.6889,0.7561,0.7209,0.8667
Part D: Hub Segmentation & Mismatches
Using K-Means clustering (
K=3), hubs are segmented by operational characteristics:

Standard Segment (7 hubs): Low delivery delay rate (2.57%), short route distance (~188.9 km), avg stops ~55.

Watch Segment (8 hubs): Moderate delay rate (15.50%), medium distance (~314.2 km), avg stops ~71.

Escalate Segment (9 hubs): High delay rate (46.00%), long distance (~470.5 km), heavy route density (~84 stops).

Detected Tier Mismatches
HUB07 (Kolkata): Designated as Tier 1, but profiles as Watch (operational lag due to route complexity).

HUB13 (Nagpur): Designated as Tier 2, but profiles as Escalate (excessive load relative to capacity).

HUB15 (Kochi): Designated as Tier 2, but profiles as Escalate (terrain/monsoon weather delays).

HUB22 (Dehradun): Designated as Tier 3, but profiles as Watch (outperforming expectation due to lower volume).

Operations Playbook
Automated Early Dispatch Alerts:
Integrate predict_delay.py into daily warehouse management workflows. Any shipment assigned a delay probability 
≥0.50 should immediately trigger priority handling at origin hubs and express linehaul routing.

Targeted Hub Interventions:
Escalate Hubs (HUB13, HUB15): Immediately reallocate temporary sorting capacity, restrict maximum rider stops per route to 
<70, and adjust promised SLAs during adverse rain conditions.
Watch Hubs (HUB07): Conduct linehaul timing audits to eliminate bottlenecks between arrival and final-mile dispatch.

Tier Label Realignment:
Formally update internal hub SLA targets to reflect actual operational tiers rather than administrative city tiers, preventing systemic SLA breach penalties.
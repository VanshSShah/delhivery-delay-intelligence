# Delhivery Last-Mile Delay Intelligence Pipeline

A comprehensive machine learning intelligence system designed for **Delhivery** to predict last-mile delivery delays, profile hub performance, detect operational mismatches, and automate proactive dispatch alerts.

---

## How to run

### 1. Requirements & Setup
- **Python Version**: `3.11.x`
- **Dependencies**: Pinned in `requirements.txt` (`pandas==2.2.3`, `numpy==1.26.4`, `scikit-learn==1.5.2`, `joblib==1.4.2`).

```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

2. Full Pipeline Execution Sequence
Execute all pipeline modules in sequence from the project root directory:
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

Data findings
Data Cleaning & Integrity: Standardized 1,224 raw shipment records down to 1,200 non-duplicate rows. Corrected text encoding (₹), standardized service level case variants, and replaced scanner fault sentinel weight values (> 100 kg) with strict, leakage-free median imputations (distance_km: 314.95, parcel_weight_kg: 1.83, declared_value_inr: 1121.50).
Hub Profiling & Mismatch Detection: K-Means clustering ($K=3$) grouped 24 nationwide hubs into Standard, Watch, and Escalate performance tiers based on operational characteristics.
Detected Hub Mismatches:
HUB07 (Kolkata): Designated Tier 1, but operating as Watch due to route bottlenecks.
HUB13 (Nagpur): Designated Tier 2, but operating as Escalate due to high volume load relative to capacity.
HUB15 (Kochi): Designated Tier 2, but operating as Escalate due to terrain/weather delays.
HUB22 (Dehradun): Designated Tier 3, but operating as Watch outperforming expectations.
Model results (regression and classification)
1. Delivery-Hours Regression Performance (Part B)
Model,Split,MAE (Hours),MSE,RMSE (Hours),R2
Mean Baseline,Validation,15.7952,355.5204,18.8552,-0.0212
Linear Regression,Validation,3.9222,23.9235,4.8912,0.9313
Decision Tree (Depth 6),Validation,5.0971,39.4338,6.2796,0.8867
Random Forest,Validation,4.2882,30.3550,5.5095,0.9128
Gradient Boosting,Validation,3.9166,24.1722,4.9165,0.9306
Linear Regression (Final),Test,3.5671,20.4624,4.5235,0.9493
2. Delay Classification Performance (Part C)
Model / Evaluation,Threshold,Precision,Recall,F1​ Score,Accuracy
Blind Baseline (All On-Time),0.50,0.0000,0.0000,0.0000,0.7667
Logistic Regression,0.50,0.7778,0.8333,0.8046,0.9056
Decision Tree (Depth 6),0.50,0.7561,0.7381,0.7470,0.8833
Default Random Forest,0.50,0.8378,0.7381,0.7848,0.9056
Final Random Forest (Top 7 Features),0.50,0.6889,0.7561,0.7209,0.8667

Operations recommendation
Automated Priority Dispatching: Integrate delay scoring (predict_delay.py) into daily warehouse dispatches. Any shipment triggering an alert (probability $\ge 0.50$) should be flagged for expedited linehaul routing and priority rider assignment.
Resource Reallocation for Escalate Hubs: Reallocate temporary sorting capacity to HUB13 (Nagpur) and HUB15 (Kochi), and cap maximum rider stops per route to $< 70$ during peak volumes.
SLA & Tier Realignment: Adjust administrative tier labels to match actual operating realities, preventing systemic SLA breach penalties at misclassified hubs like HUB07 and HUB13.

Production plan
Model Artifact Packaging: Models are stored in models/delay_model.pkl alongside feature column orders (feature_order.csv), decision thresholds (delay_threshold.csv), and training medians (impute_values.csv).
Automated Batch Scoring: Unlabelled daily shipments are ingested via predict_delay.py, producing formatted risk probabilities and binary alert triggers in outputs/predictions.csv. Tested reload probability drift is verified at 0.0.
Pipeline Ingestion: Embed predict_delay.py as a daily scheduled cron job or Airflow task operating on new shipment dispatches.

Limitations
Synthetic Data Distributions: The pipeline was trained and evaluated on simulated logistics data; real-world edge cases (e.g., severe weather anomalies, festive surge spikes) may require retraining.
Feature Coverage: Additional operational features such as real-time GPS traffic feeds, individual courier performance metrics, and detailed weather status could further improve alert precision.

"""Part C: Delay Classification, Feature Signal Selection, Hyperparameter Tuning & Alert Threshold Optimization."""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS


def run_part_c():
    os.makedirs("outputs", exist_ok=True)

    # Read modelling table
    df = pd.read_csv(os.path.join("data", "modelling_table.csv"))

    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified Train / Validation / Test split (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 2. Baseline & Initial Classifiers (Decision Rule threshold = 0.5)
    # Blind Baseline (all 0)
    y_val_blind = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_blind).ravel()
    acc = round(accuracy_score(y_val, y_val_blind), 4)
    prec = 0.0  # Specified in task instructions
    rec = round(recall_score(y_val, y_val_blind, zero_division=0), 4)
    f1 = round(f1_score(y_val, y_val_blind, zero_division=0), 4)

    models_rows = [{
        "model": "blind_all_on_time",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }]

    # Logistic Regression
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    models_rows.append({
        "model": "logistic_regression",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall": round(recall_score(y_val, y_pred), 4),
        "f1": round(f1_score(y_val, y_pred), 4)
    })

    # Decision Tree depth tuning
    tree_depths = [2, 3, 4, 5, 6, 8, 10]
    best_tree_depth = None
    best_tree_f1 = -1.0
    best_tree_metrics = None

    for d in tree_depths:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_val)
        current_f1 = f1_score(y_val, pred)
        if current_f1 > best_tree_f1:
            best_tree_f1 = current_f1
            best_tree_depth = d
            tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_val, pred).ravel()
            best_tree_metrics = {
                "model": f"decision_tree_depth_{d}",
                "TN": int(tn_d), "FP": int(fp_d), "FN": int(fn_d), "TP": int(tp_d),
                "accuracy": round(accuracy_score(y_val, pred), 4),
                "precision": round(precision_score(y_val, pred), 4),
                "recall": round(recall_score(y_val, pred), 4),
                "f1": round(current_f1, 4)
            }

    models_rows.append(best_tree_metrics)

    # Default Random Forest (17 features)
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    y_pred_rf = rf_def.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_rf).ravel()
    models_rows.append({
        "model": "random_forest_default",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, y_pred_rf), 4),
        "precision": round(precision_score(y_val, y_pred_rf), 4),
        "recall": round(recall_score(y_val, y_pred_rf), 4),
        "f1": round(f1_score(y_val, y_pred_rf), 4)
    })

    pd.DataFrame(models_rows).to_csv(os.path.join("outputs", "part_c_models.csv"), index=False)

    # 3. Feature Signal Analysis & SelectKBest
    # Fit Default RF on 17 features for importances
    rf_def_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def_all.fit(X_train, y_train)
    rf_importances = rf_def_all.feature_importances_

    # Fit SelectKBest (f_classif, k='all') on training rows only
    selector_all = SelectKBest(score_func=f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": np.round(f_scores, 4),
        "rf_importance": np.round(rf_importances, 4),
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores_df = feature_scores_df.sort_values(by="f_rank").reset_index(drop=True)
    feature_scores_df.to_csv(os.path.join("outputs", "part_c_feature_scores.csv"), index=False)

    # SelectKBest k experiment (k in 5, 7, 10, 12, 17)
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)
        selectk_rows.append({
            "k": k,
            "val_f1": round(f1_score(y_val, pred_val), 4)
        })
    pd.DataFrame(selectk_rows).to_csv(os.path.join("outputs", "part_c_selectk.csv"), index=False)

    # 4. Hyperparameter Tuning on Top 7 Features
    # Transform train and val with top k=7 selector fitted on train
    sel7 = SelectKBest(f_classif, k=7)
    X_train_k7 = sel7.fit_transform(X_train, y_train)
    X_val_k7 = sel7.transform(X_val)

    # Search space definition
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    # Grid Search (cv=5)
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5)
    gs.fit(X_train_k7, y_train)
    gs_pred_val = gs.predict(X_val_k7)
    gs_val_f1 = round(f1_score(y_val, gs_pred_val), 4)

    # Randomized Search (cv=5, n_iter=9, random_state=42)
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
    rs.fit(X_train_k7, y_train)
    rs_pred_val = rs.predict(X_val_k7)
    rs_val_f1 = round(f1_score(y_val, rs_pred_val), 4)

    # Default Forest on same 7 features
    rf_def_k7 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def_k7.fit(X_train_k7, y_train)
    def_pred_val = rf_def_k7.predict(X_val_k7)
    def_val_f1 = round(f1_score(y_val, def_pred_val), 4)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs.best_params_),
            "best_cv_accuracy": round(float(gs.best_score_), 4),
            "val_f1": gs_val_f1
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs.best_params_),
            "best_cv_accuracy": round(float(rs.best_score_), 4),
            "val_f1": rs_val_f1
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "{'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1}",
            "best_cv_accuracy": "",
            "val_f1": def_val_f1
        }
    ]
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(os.path.join("outputs", "part_c_tuning.csv"), index=False)

    # 5. Final Configuration Selection
    # Highest val_f1 (tie-breaker: default_forest_same_7_features -> grid_search -> randomized_search)
    sorted_tuning = tuning_df.sort_values(by="val_f1", ascending=False)
    winning_search_name = sorted_tuning.iloc[0]["search"]

    if winning_search_name == "default_forest_same_7_features":
        final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        n_est, max_d, min_leaf = 100, "unlimited", 1
    elif winning_search_name == "grid_search":
        final_rf = RandomForestClassifier(**gs.best_params_, random_state=42)
        n_est = gs.best_params_["n_estimators"]
        max_d = gs.best_params_["max_depth"]
        min_leaf = gs.best_params_["min_samples_leaf"]
    else:
        final_rf = RandomForestClassifier(**rs.best_params_, random_state=42)
        n_est = rs.best_params_["n_estimators"]
        max_d = rs.best_params_["max_depth"]
        min_leaf = rs.best_params_["min_samples_leaf"]

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", final_rf)
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert Threshold Optimization
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_rows = []
    chosen_threshold = 0.1

    for t in threshold_list:
        preds = (val_probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_val, preds).ravel()
        rec_t = round(recall_score(y_val, preds, zero_division=0), 4)
        prec_t = round(precision_score(y_val, preds, zero_division=0), 4)
        acc_t = round(accuracy_score(y_val, preds), 4)
        f1_t = round(f1_score(y_val, preds, zero_division=0), 4)

        thresh_rows.append({
            "threshold": t,
            "TN": int(tn_t), "FP": int(fp_t), "FN": int(fn_t), "TP": int(tp_t),
            "accuracy": acc_t, "precision": prec_t, "recall": rec_t, "f1": f1_t
        })

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(os.path.join("outputs", "part_c_thresholds.csv"), index=False)

    # Select highest threshold with validation recall >= 0.80 (fallback to 0.1)
    eligible_thresh = thresh_df[thresh_df["recall"] >= 0.80]
    if not eligible_thresh.empty:
        chosen_threshold = float(eligible_thresh["threshold"].max())

    # Write outputs/part_c_final_config.csv
    config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": n_est,
        "max_depth": max_d,
        "min_samples_leaf": min_leaf,
        "threshold": chosen_threshold,
        "chosen_from": winning_search_name
    }])
    config_df.to_csv(os.path.join("outputs", "part_c_final_config.csv"), index=False)

    # 7. One Final Test Evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds_thresh = (test_probs >= chosen_threshold).astype(int)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, test_preds_thresh).ravel()

    final_test_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn_test), "FP": int(fp_test), "FN": int(fn_test), "TP": int(tp_test),
        "accuracy": round(accuracy_score(y_test, test_preds_thresh), 4),
        "precision": round(precision_score(y_test, test_preds_thresh, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_preds_thresh, zero_division=0), 4),
        "f1": round(f1_score(y_test, test_preds_thresh, zero_division=0), 4)
    }])
    final_test_df.to_csv(os.path.join("outputs", "part_c_final_test.csv"), index=False)

    print(f"Part C execution completed successfully! Chosen threshold: {chosen_threshold}")


if __name__ == "__main__":
    run_part_c()
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter

print("Step 1 passed")

from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import NeighbourhoodCleaningRule
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.meta import LearnNSE
from skmultiflow.meta import LeverageBagging
from skmultiflow.trees import HoeffdingTree

print("Step 2 passed")

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             classification_report, matthews_corrcoef)

print("All critical packages are compatible.")

import sys
sys.path.append('/Users/ariasarch/JAWZ_Big_Data/src')

import cryptoaml.datareader as cdr

print('cryptoaml.datareader succesfully loaded')

from cryptoaml.metrics import elliptic_time_indexed_results

print('cryptoaml.metrics succesfully loaded')

from cryptoaml.models import AdaptiveXGBoostClassifier

print('cryptoaml.models AdaptiveXGBoostClassifier succesfully loaded')

from cryptoaml.models import AdaptiveStackedBoostClassifier

print('cryptoaml.models AdaptiveStackedBoostClassifier succesfully loaded')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print('All packages succesfully loaded')

def compute_metrics(y_true, y_pred, y_probs=None):

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute Labels
    labels=np.unique(y_pred)

    # Basic metrics
    metrics_list = [
        ("Accuracy", accuracy_score(y_true, y_pred)),
        ("Precision", precision_score(y_true, y_pred, average='binary', labels=labels)),
        ("Recall", recall_score(y_true, y_pred, average='binary', labels=labels)),
        ("F1 Score", f1_score(y_true, y_pred, average='binary', labels=labels)),
        ("Specificity", tn / (tn + fp) if (tn + fp) != 0 else 0),
        ("NPV", tn / (tn + fn) if (tn + fn) != 0 else 0),
        ("FPR", fp / (fp + tn) if (fp + tn) != 0 else 0),
        ("FDR", fp / (fp + tp) if (fp + tp) != 0 else 0),
        ("MCC", matthews_corrcoef(y_true, y_pred)),
        ("Confusion Matrix", (tn, fp, fn, tp))
    ]

    # Additional metrics that require probability scores
    if y_probs is not None:
        metrics_list.append(("ROC AUC", roc_auc_score(y_true, y_probs)))
        metrics_list.append(("AUCPR", average_precision_score(y_true, y_probs)))
    
    return metrics_list

def evaluate_batch_incremental(model, data, t_eval=0):
    
    results = {}
    results_time = []
    true_test = []
    predictions_test = []
    probabilities_test = []  # List to store probabilities for AUCPR calculation
    
    min_ts = data["ts"].min()
    max_ts = data["ts"].max()
    total_steps = max_ts - min_ts  # Calculate the total number of steps

    # Loop through each timestep, except the last one since there's no following month to predict
    for step, ts in enumerate(np.arange(min_ts, max_ts)):

        print(f"Evaluating timestep {step + 1} of {total_steps}")  # Progress update

        # get training data for the current timestep 
        train_set = data[data["ts"] == ts]
        train_set_X = train_set.iloc[:,:-1]
        train_set_y = train_set["class"]      

        # partially fit model 
        model.partial_fit(train_set_X.values, train_set_y.values)    

        # get test data for the current timestep + 1 
        test_set = data[data["ts"] == ts + 1]
        x_test = test_set.iloc[:,:-1].values
        y_test = test_set["class"].values
    
        # predict test data for the current timestep + 1
        y_pred = model.predict(x_test)

        # Check if the model supports probability prediction and handle accordingly
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
        else:
            y_probs = None  # For models that do not support predict_prob

        # Computing metrics
        metrics = compute_metrics(y_test, y_pred, y_probs)
        for metric_name, metric_value in metrics:
            print(f"{metric_name}: {metric_value}")
        
        true_test.append(y_test)
        predictions_test.append(y_pred)

        # Collect label counts if necessary; consider a default if the class label does not exist
        label_counts = test_set["class"].value_counts().to_dict()
        pos_label_count = label_counts.get(1, 0)  # Default to 0 if the positive class label '1' does not exist

        results_time.append({
            "timestep": ts + 1,
            "metrics": metrics,
            "total_pos_label": pos_label_count
        })
     
    # Aggregate results
    y_true_aggregated = np.concatenate(true_test, axis=0)
    y_pred_aggregated = np.concatenate(predictions_test, axis=0)
    y_probs_aggregated = np.concatenate(probabilities_test, axis=0) if probabilities_test else None

    # Calculate aggregate metrics
    # Use the compute_metrics function for aggregated metrics
    aggregated_metrics = compute_metrics(y_true_aggregated, y_pred_aggregated, y_probs_aggregated)

    results["test_results"] = aggregated_metrics  # Store the computed metrics

    # Store the metrics to a pandas df
    results["time_metrics"] = pd.DataFrame(results_time)
    print(results)
    return results

def evaluate(feature_set, n_eval):
    elliptic = cdr.get_data("elliptic")
    data_eval = elliptic.train_test_split(train_size=0.7, 
                                     feat_set=feature_set, 
                                     inc_meta=False,
                                     inc_unknown=False)

    train_data = data_eval.train_X
    train_data["class"] = data_eval.train_y
    test_data = data_eval.test_X
    test_data["class"] = data_eval.test_y 
    data_eval = train_data.append(test_data, ignore_index=True)
    
    f_set = "elliptic"+"_"+feature_set
    experiment_3_results = {}
    experiment_3_results["ARF"] = {}
    experiment_3_results["AXGBr"] = {}
    experiment_3_results["AXGBp"] = {}
    experiment_3_results["ASXGB"] = {}
    experiment_3_results["ARF"][f_set] = {}
    experiment_3_results["AXGBr"][f_set] = {}
    experiment_3_results["AXGBp"][f_set] = {}
    experiment_3_results["ASXGB"][f_set] = {}

    # 2. Adapative Random Forest
    print("Evaluating ARF")
    arf = AdaptiveRandomForest(performance_metric="kappa")
    experiment_3_results["ARF"][f_set] = evaluate_batch_incremental(arf, data_eval, n_eval)
    
    # 2. Adapative Extreme Gradient Boosting with Replacement
    # 3. Adapative Extreme Gradient Boosting with Push
    # Adaptive XGBoost classifier parameters
    n_estimators = 30       # Number of members in the ensemble
    learning_rate = 0.3     # Learning rate or eta
    max_depth = 6           # Max depth for each tree in the ensemble
    max_window_size = 1000  # Max window size
    min_window_size = 1     # set to activate the dynamic window strategy
    detect_drift = False    # Enable/disable drift detection

    print("Evaluating AXGBr")
    AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      max_window_size=max_window_size,
                                      min_window_size=min_window_size,
                                      detect_drift=detect_drift)
    experiment_3_results["AXGBr"][f_set] = evaluate_batch_incremental(AXGBr, data_eval, n_eval)

    # print("Evaluating AXGBp")
    # AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
    #                                   n_estimators=n_estimators,
    #                                   learning_rate=learning_rate,
    #                                   max_depth=max_depth,
    #                                   max_window_size=max_window_size,
    #                                   min_window_size=min_window_size,
    #                                   detect_drift=detect_drift)
    # experiment_3_results["AXGBp"][f_set] = evaluate_batch_incremental(AXGBp, data_eval, n_eval)

    # 4. Proposed Method
    print("Evaluating ASXGB")
    ASXGB = AdaptiveStackedBoostClassifier()
    experiment_3_results["ASXGB"][f_set] = evaluate_batch_incremental(ASXGB, data_eval, n_eval)
    
    # elliptic_time_indexed_results(experiment_3_results)
    # print(experiment_3_results)

    # After all evaluations are done, save each model's results:
    for model_key, model_results in experiment_3_results.items():
        for feature_key, results in model_results.items():
            if "time_metrics" in results:
                results_filename = f"{model_key}_{feature_key}_results.csv"
                results["time_metrics"].to_csv(results_filename, index=False)
                print(f"Results saved to {results_filename}")


evaluate("AF", 35)

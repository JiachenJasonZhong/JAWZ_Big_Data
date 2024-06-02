import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
import itertools
import time
import matplotlib.pyplot as plt

sys.path.append('/Users/ariasarch/JAWZ_Big_Data/src')

print("Step 1 passed")

from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.neighbors import NearestNeighbors
from skmultiflow.meta import AdaptiveRandomForest
from skmultiflow.meta import LearnNSE
from skmultiflow.meta import LeverageBagging
from skmultiflow.trees import HoeffdingTree

print("Step 2 passed")

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             roc_auc_score, matthews_corrcoef, precision_recall_curve, auc)

print("All critical packages are compatible.")

import cryptoaml.datareader as cdr
from cryptoaml.metrics import elliptic_time_indexed_results
from cryptoaml.models import (AdaptiveXGBoostClassifier, AdaptiveStackedBoostClassifier, 
                              LSTM_AdaptiveStackedBoostClassifier, LSTM_Base)

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

print('Step 3 passed - All packages succesfully loaded')    

def compute_metrics(y_true, y_pred, y_probs):

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute Labels
    labels=np.unique(y_pred)

    # Basic metrics
    metrics_list = [
        ("True Negtaives", tn),
        ("False Positives", fp),
        ("False Negtaives", fn),
        ("True Positives", tp),
        ("Accuracy", accuracy_score(y_true, y_pred)),
        ("Precision", precision_score(y_true, y_pred, average='binary', labels=labels)),
        ("Recall", recall_score(y_true, y_pred, average='binary', labels=labels)),
        ("F1 Score", f1_score(y_true, y_pred, average='binary', labels=labels)),
        ("Specificity", tn / (tn + fp) if (tn + fp) != 0 else 0),
        ("NPV", tn / (tn + fn) if (tn + fn) != 0 else 0),
        ("FPR", fp / (fp + tn) if (fp + tn) != 0 else 0),
        ("FDR", fp / (fp + tp) if (fp + tp) != 0 else 0),
        ("MCC", matthews_corrcoef(y_true, y_pred)),
        ("Confusion Matrix", (tn, fp, fn, tp)),
        ("ROC_AUC", roc_auc_score(y_true, y_probs)),
        ("AUCPR", average_precision_score(y_true, y_probs))
    ]
    
    return metrics_list
   
def evaluate_batch_incremental(model, data):
    
    results = {}
    f1_score = 0
    results_time = []
    true_test = []
    predictions_test = []
    probabilities_test = []  # List to store probabilities for AUC calculation
    timestep_totals = []
    
    min_ts = data["ts"].min()
    max_ts = data["ts"].max()
    total_steps = max_ts - min_ts  # Calculate the total number of steps

    # Loop through each timestep, except the last one since there's no following month to predict
    for step, ts in enumerate(np.arange(min_ts, max_ts)):
        if step % 20 == 0:
            print(f"Evaluating timestep {step + 1} of {total_steps}")  # Progress update

        # get training data for the current timestep 
        train_set = data[data["ts"] == ts]
        train_set_X = train_set.iloc[:,:-1]
        train_set_y = train_set["class"]      

        # partially fit model 
        if isinstance(model, LSTM_Base):
            model.train_lstm(train_set_X.values, train_set_y.values)
        else:
            model.partial_fit(train_set_X.values, train_set_y.values)   

        # get test data for the current timestep + 1 
        test_set = data[data["ts"] == ts + 1]
        x_test = test_set.iloc[:,:-1].values
        y_test = test_set["class"].values

        # calculate the total number of 1s in y_test
        total_ones = np.sum(y_test == 1)
        timestep_totals.append(total_ones) 
    
        # predict test data for the current timestep + 1
        y_pred = model.predict(x_test)

        # Check if the model has the 'predict_proba' method
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
            # Assuming y_probs is the array containing the model's probability outputs
            valid_probs = y_probs[np.isfinite(y_probs)]  # Filter out NaN and infinite values

            if len(valid_probs) > 0:
                median_prob = np.median(valid_probs)  # Compute the median of valid probabilities
            else:
                median_prob = 0.5  # Default if no valid probabilities exist

            # Replace NaN and infinite values with the median probability
            y_probs = np.nan_to_num(y_probs, nan=median_prob, posinf=median_prob, neginf=median_prob)

            # Check for invalid y_probs
            if len(valid_probs) != len(y_probs):
                print(f'Total number of valid_probs: {len(valid_probs)}')
                print(f'Total number of y_probs: {len(y_probs)}')
            if np.any(np.isnan(y_probs)):
                print("NaN values found")
            if np.any(np.isinf(y_probs)):
                print("Inf values found")
        else:
            y_probs = model.eval_proba(x_test)  # Probabilities for the positive class

            # Check the shape of y_probs
            if y_probs.ndim == 2:
                # If y_probs is a 2-dimensional array, assume it contains probabilities for both classes
                y_probs = y_probs[:, 1]  # Select the probabilities for the positive class
            else:
                # If y_probs is a 1-dimensional array, assume it contains probabilities for the positive class
                pass  # No need to modify y_probs

            # Assuming y_probs is the array containing the model's probability outputs
            valid_probs = y_probs[np.isfinite(y_probs)]  # Filter out NaN and infinite values

            if len(valid_probs) > 0:
                median_prob = np.median(valid_probs)  # Compute the median of valid probabilities
            else:
                median_prob = 0.5  # Default if no valid probabilities exist

            # Replace NaN and infinite values with the median probability
            y_probs = np.nan_to_num(y_probs, nan=median_prob, posinf=median_prob, neginf=median_prob)

            # Check for invalid y_probs
            if len(valid_probs) != len(y_probs):
                print(f'Total number of valid_probs: {len(valid_probs)}')
                print(f'Total number of y_probs: {len(y_probs)}')
            if np.any(np.isnan(y_probs)):
                print("NaN values found")
            if np.any(np.isinf(y_probs)):
                print("Inf values found")

        # Computing metrics
        metrics = compute_metrics(y_test, y_pred, y_probs)

        # # Double check metric evalutaion 
        # if step == 0:
        #     for metric_name, metric_value in metrics:
        #         print(f"{metric_name}: {metric_value}")
        
        true_test.append(y_test)
        predictions_test.append(y_pred)
        probabilities_test.append(y_probs)

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
    y_probs_aggregated = np.concatenate(probabilities_test, axis=0)

    # Calculate aggregate metrics
    # Use the compute_metrics function for aggregated metrics
    aggregated_metrics = compute_metrics(y_true_aggregated, y_pred_aggregated, y_probs_aggregated)

    # Extract the F1 score from the aggregated metrics
    results["test_results"] = aggregated_metrics  # Store the computed metrics
    
    # Extract the F1 score from the aggregated metrics
    # f1_score = next((metric_value for metric_name, metric_value in aggregated_metrics if metric_name == "F1 Score"), None)
    # # print(f"Average F1 Score: {f1_score:.4f}")

    # Store the metrics to a pandas df
    results["time_metrics"] = pd.DataFrame(results_time)
    # print(results)

    # print(timestep_totals)

    return results

def evaluate(feature_set):
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
    experiment_results = {}
    experiment_results["ARF"] = {}
    experiment_results["AXGBr"] = {}
    experiment_results["AXGBp"] = {}
    experiment_results["ASXGB"] = {}
    experiment_results["LSTM_Base"] = {}
    experiment_results["LSTM_ASXGB"] = {}
    experiment_results["ARF"][f_set] = {}
    experiment_results["AXGBr"][f_set] = {}
    experiment_results["AXGBp"][f_set] = {}
    experiment_results["ASXGB"][f_set] = {}
    experiment_results["LSTM_Base"][f_set] = {}
    experiment_results["LSTM_ASXGB"][f_set] = {}

    # # 2. Adapative Random Forest
    # print("Evaluating ARF")
    # arf = AdaptiveRandomForest(performance_metric="kappa")
    # experiment_results["ARF"][f_set] = evaluate_batch_incremental(arf, data_eval)
    
    # # 2. Adapative Extreme Gradient Boosting with Replacement
    # # 3. Adapative Extreme Gradient Boosting with Push
    # # Adaptive XGBoost classifier parameters
    # n_estimators = 30       # Number of members in the ensemble
    # learning_rate = 0.3     # Learning rate or eta
    # max_depth = 6           # Max depth for each tree in the ensemble
    # max_window_size = 1000  # Max window size
    # min_window_size = 1     # set to activate the dynamic window strategy
    # detect_drift = False    # Enable/disable drift detection

    # print("Evaluating AXGBr")
    # AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
    #                                   n_estimators=n_estimators,
    #                                   learning_rate=learning_rate,
    #                                   max_depth=max_depth,
    #                                   max_window_size=max_window_size,
    #                                   min_window_size=min_window_size,
    #                                   detect_drift=detect_drift)
    # experiment_results["AXGBr"][f_set] = evaluate_batch_incremental(AXGBr, data_eval)

    # print("Evaluating AXGBp")
    # AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
    #                                   n_estimators=n_estimators,
    #                                   learning_rate=learning_rate,
    #                                   max_depth=max_depth,
    #                                   max_window_size=max_window_size,
    #                                   min_window_size=min_window_size,
    #                                   detect_drift=detect_drift)
    # experiment_results["AXGBp"][f_set] = evaluate_batch_incremental(AXGBp, data_eval)

    # # 4. Proposed Method by the original authors
    # print("Evaluating ASXGB")
    # ASXGB = AdaptiveStackedBoostClassifier()
    # experiment_results["ASXGB"][f_set] = evaluate_batch_incremental(ASXGB, data_eval)

    # # 6. LSTM Method with Xgboost
    # print("Evaluating LSTM + ASXGB") 
    # LSTM_ASXGB = LSTM_AdaptiveStackedBoostClassifier(lstm_units=32,
    #                                                  lstm_epochs=20,
    #                                                  lstm_dropout=0.1,
    #                                                  lstm_grad_clip_threshold=1,
    #                                                  learning_rate=0.000001,
    #                                                  weight_decay=0.1,
    #                                                  target_value=0.69,
    #                                                  scale_factor=0.9,
    #                                                  threshold=0.5)
    # experiment_results["LSTM_ASXGB"][f_set] = evaluate_batch_incremental(LSTM_ASXGB, data_eval)

    # # 6. LSTM Method WITHOUT Xgboost
    print("Evaluating LSTM with no xgboost")

    # Hyperparameter values to search
    hyperparameters = {
        'lstm_units': [32], # 32, 64, 128
        'lstm_epochs': [20], # 5, 10, 20, 40
        'lstm_dropout': [0.2], # 0.2, 0.3, 0.4, 0.5
        'lstm_grad_clip_threshold': [1.0],
        'learning_rate': [0.000001], # 0.0001, 0.00001, 0.000001
        'weight_decay': [0.1], # 0.1, 0.01, 0.001
        'target_value': [0.69],
        'scale_factor': [0.9], # 0.9, 1, 1.1
        'threshold': [0.5]
    }

    # Generate all possible combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*hyperparameters.values())) 
    total_combinations = len(hyperparameter_combinations)   

    # Initialize a dictionary to store the results for each hyperparameter combination
    hyper_parameter_results = {}
    f1_scores = []

    # Iterate over each hyperparameter combination
    for i, combination in enumerate(hyperparameter_combinations, start=1):
        print(f'Performing iteration {i} of {total_combinations} now')

        # Time Estimation
        start_time_forward = time.time()

        # Unpack the hyperparameter values
        lstm_units, lstm_epochs, lstm_dropout, lstm_grad_clip_threshold, learning_rate, weight_decay, target_value, scale_factor, threshold = combination
        print(f"Evaluating LSTM with hyperparameters: {combination}")
        
        # Run the model with the current hyperparameter combination
        LSTM_base = LSTM_Base(
            lstm_units=lstm_units,
            lstm_epochs=lstm_epochs,
            lstm_dropout=lstm_dropout,
            lstm_grad_clip_threshold=lstm_grad_clip_threshold,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            target_value=target_value,
            scale_factor=scale_factor,
            threshold=threshold
        )
        
        # Evaluate the model and store the results
        experiment_results["LSTM_Base"][f_set] = evaluate_batch_incremental(LSTM_base, data_eval)
        
        # Store the results for the current hyperparameter combination
        hyper_parameter_results[str(combination)] = experiment_results["LSTM_Base"][f_set]

        # Extract the F1 score from the experiment results
        f1_score = next(metric_value for metric_name, metric_value in experiment_results["LSTM_Base"][f_set]["test_results"] if metric_name == "F1 Score")
        print(f"F1 Score for iteration {i}: {f1_score:.4f}")

        # Time Estimation
        end_time_forward = time.time()
        tuning_time = end_time_forward - start_time_forward
        remaining_combinations = total_combinations - i
        total_estimated_time = tuning_time * (remaining_combinations)

        # Append the F1 score to the list
        f1_scores.append(f1_score)

        # Create a plot of the F1 scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o')
        plt.xlabel('Hyperparameter Iteration: (Units)')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Hyperparameter Iteration')
        plt.show()

        # Print the total estimated time 
        print("\n" + "*" * 50)
        print(f"*{' ' * 48}*")
        print(f"*{' ' * 15}Total Estimated Time{' ' * 15}*")
        print(f"*{' ' * 48}*")
        print(f"*{' ' * 8}{total_estimated_time:.2f} seconds remaining{' ' * 8}*")
        print(f"*{' ' * 8}{remaining_combinations} combinations remaining{' ' * 8}*")
        print(f"*{' ' * 48}*")
        print("*" * 50 + "\n")


    # After all evaluations are done, save each model's results
    for model_key, model_results in experiment_results.items():
        for feature_key, results in model_results.items():
            if "time_metrics" in results:
                results_filename = f"{model_key}_{feature_key}_results.csv"
                results["time_metrics"].to_csv(results_filename, index=False)
                print(f"Results saved to {results_filename}")

    # Convert the hyperparameter results to a DataFrame and save it
    hyper_parameter_df = pd.DataFrame(hyper_parameter_results).transpose()
    hyper_parameter_df.to_csv("hyperparameter_tuning_results.csv", index=True)
    print("Hyperparameter tuning results saved to hyperparameter_tuning_results.csv")

    print(f1_scores)

def evaluate_batch_incremental_norolling(model, train_data, test_data):
    results = {}
    results_time = []
    true_test = []
    predictions_test = []
    probabilities_test = []  # List to store probabilities for AUC calculation

    train_set = train_data
    train_set_X = train_set.iloc[:,:-1]
    train_set_y = train_set["class"]      

    # partially fit model 
    if isinstance(model, LSTM_Base):
        model.train_lstm(train_set_X.values, train_set_y.values)
    else:
        model.partial_fit(train_set_X.values, train_set_y.values) 

    print('Model successfully trained') 

    # get test data for the current timestep + 1 
    test_set = test_data
    x_test = test_set.iloc[:,:-1].values
    y_test = test_set["class"].values

    # predict test data for the current timestep + 1
    print('Predicting on Test dataset now')
    y_pred = model.predict(x_test)

    # Check if the model has the 'predict_proba' method
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
        # Assuming y_probs is the array containing the model's probability outputs
        valid_probs = y_probs[np.isfinite(y_probs)]  # Filter out NaN and infinite values

        if len(valid_probs) > 0:
            median_prob = np.median(valid_probs)  # Compute the median of valid probabilities
        else:
            median_prob = 0.5  # Default if no valid probabilities exist

        # Replace NaN and infinite values with the median probability
        y_probs = np.nan_to_num(y_probs, nan=median_prob, posinf=median_prob, neginf=median_prob)

        # Check for invalid y_probs
        if len(valid_probs) != len(y_probs):
            print(f'Total number of valid_probs: {len(valid_probs)}')
            print(f'Total number of y_probs: {len(y_probs)}')
        if np.any(np.isnan(y_probs)):
            print("NaN values found")
        if np.any(np.isinf(y_probs)):
            print("Inf values found")
    else:
        y_probs = model.eval_proba(x_test)  # Probabilities for the positive class

        # Check the shape of y_probs
        if y_probs.ndim == 2:
            # If y_probs is a 2-dimensional array, assume it contains probabilities for both classes
            y_probs = y_probs[:, 1]  # Select the probabilities for the positive class
        else:
            # If y_probs is a 1-dimensional array, assume it contains probabilities for the positive class
            pass  # No need to modify y_probs

        # Assuming y_probs is the array containing the model's probability outputs
        valid_probs = y_probs[np.isfinite(y_probs)]  # Filter out NaN and infinite values

        if len(valid_probs) > 0:
            median_prob = np.median(valid_probs)  # Compute the median of valid probabilities
        else:
            median_prob = 0.5  # Default if no valid probabilities exist

        # Replace NaN and infinite values with the median probability
        y_probs = np.nan_to_num(y_probs, nan=median_prob, posinf=median_prob, neginf=median_prob)

        # Check for invalid y_probs
        if len(valid_probs) != len(y_probs):
            print(f'Total number of valid_probs: {len(valid_probs)}')
            print(f'Total number of y_probs: {len(y_probs)}')
        if np.any(np.isnan(y_probs)):
            print("NaN values found")
        if np.any(np.isinf(y_probs)):
            print("Inf values found")

    # Computing metrics
    metrics = compute_metrics(y_test, y_pred, y_probs)

    # Double check metric evalutaion 
    # if step == 0:
    #     for metric_name, metric_value in metrics:
    #         print(f"{metric_name}: {metric_value}")
    
    true_test.append(y_test)
    predictions_test.append(y_pred)
    probabilities_test.append(y_probs)

    # Collect label counts if necessary; consider a default if the class label does not exist
    label_counts = test_set["class"].value_counts().to_dict()
    pos_label_count = label_counts.get(1, 0)  # Default to 0 if the positive class label '1' does not exist

    results_time.append({
        # "timestep": ts + 1,
        "metrics": metrics,
        "total_pos_label": pos_label_count
    })
     
    # Aggregate results
    y_true_aggregated = np.concatenate(true_test, axis=0)
    y_pred_aggregated = np.concatenate(predictions_test, axis=0)
    y_probs_aggregated = np.concatenate(probabilities_test, axis=0)

    # Calculate aggregate metrics
    # Use the compute_metrics function for aggregated metrics
    aggregated_metrics = compute_metrics(y_true_aggregated, y_pred_aggregated, y_probs_aggregated)

    results["test_results"] = aggregated_metrics  # Store the computed metrics

    # Store the metrics to a pandas df
    results["time_metrics"] = pd.DataFrame(results_time)
    print(results)
    return results

def evaluate_norolling(feature_set):
    elliptic = cdr.get_data("elliptic")
    print('Step 4 passed - Data has been retrieved')

    data_eval = elliptic.train_test_split(train_size=0.7, 
                                          feat_set=feature_set, 
                                          inc_meta=False,
                                          inc_unknown=False)

    train_data = data_eval.train_X
    train_data["class"] = data_eval.train_y
    test_data = data_eval.test_X
    test_data["class"] = data_eval.test_y 

    # Count the number of 1s and 0s in the testing dataset
    test_num_ones = np.sum(test_data["class"] == 1)
    test_num_zeros = np.sum(test_data["class"] == 0)

    print("\nTesting Dataset:")
    print(f"Total number of 1s: {test_num_ones}")
    print(f"Total number of 0s: {test_num_zeros}")

    print('Step 5 passed - Data has been split')
    
    f_set = "elliptic"+"_"+feature_set
    experiment_results = {}
    experiment_results["ARF"] = {}
    experiment_results["AXGBr"] = {}
    experiment_results["AXGBp"] = {}
    experiment_results["ASXGB"] = {}
    experiment_results["LSTM_ASXGB"] = {}
    experiment_results["LSTM_Base"] = {}
    experiment_results["ARF"][f_set] = {}
    experiment_results["AXGBr"][f_set] = {}
    experiment_results["AXGBp"][f_set] = {}
    experiment_results["ASXGB"][f_set] = {}
    experiment_results["LSTM_ASXGB"][f_set] = {}
    experiment_results["LSTM_Base"][f_set] = {}

    # # 2. Adapative Random Forest
    # print("Evaluating ARF")
    # arf = AdaptiveRandomForest(performance_metric="kappa")
    # experiment_results["ARF"][f_set] = evaluate_batch_incremental_norolling(arf, train_data, test_data)
    
    # # 2. Adapative Extreme Gradient Boosting with Replacement
    # # 3. Adapative Extreme Gradient Boosting with Push
    # # Adaptive XGBoost classifier parameters
    # n_estimators = 30       # Number of members in the ensemble
    # learning_rate = 0.3     # Learning rate or eta
    # max_depth = 6           # Max depth for each tree in the ensemble
    # max_window_size = 1000  # Max window size
    # min_window_size = 1     # set to activate the dynamic window strategy
    # detect_drift = False    # Enable/disable drift detection

    # print("Evaluating AXGBr")
    # AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
    #                                   n_estimators=n_estimators,
    #                                   learning_rate=learning_rate,
    #                                   max_depth=max_depth,
    #                                   max_window_size=max_window_size,
    #                                   min_window_size=min_window_size,
    #                                   detect_drift=detect_drift)
    # experiment_results["AXGBr"][f_set] = evaluate_batch_incremental_norolling(AXGBr, train_data, test_data)

    # print("Evaluating AXGBp")
    # AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
    #                                   n_estimators=n_estimators,
    #                                   learning_rate=learning_rate,
    #                                   max_depth=max_depth,
    #                                   max_window_size=max_window_size,
    #                                   min_window_size=min_window_size,
    #                                   detect_drift=detect_drift)
    # experiment_results["AXGBp"][f_set] = evaluate_batch_incremental_norolling(AXGBp, train_data, test_data)

    # # 4. Proposed Method by the original authors
    # print("Evaluating ASXGB")
    # ASXGB = AdaptiveStackedBoostClassifier()
    # experiment_results["ASXGB"][f_set] = evaluate_batch_incremental_norolling(ASXGB, train_data, test_data)
    
    # 5. LSTM Method with Xgboost
    print("Evaluating LSTM + ASXGB")
    LSTM_ASXGB = LSTM_AdaptiveStackedBoostClassifier(lstm_units=32,
                                                     lstm_epochs=20,
                                                     lstm_dropout=0.2,
                                                     lstm_grad_clip_threshold=1.0,
                                                     learning_rate=0.000001,
                                                     weight_decay=0.1,
                                                     target_value=0.69,
                                                     scale_factor=0.9,
                                                     threshold=0.5)
    experiment_results["LSTM_ASXGB"][f_set] = evaluate_batch_incremental_norolling(LSTM_ASXGB, train_data, test_data)

    # 6. LSTM Method WITHOUT Xgboost
    print("Evaluating LSTM with no xgboost")

    # Run the model with the current hyperparameter combination
    LSTM_base = LSTM_Base(
        lstm_units=32,
        lstm_epochs=20,
        lstm_dropout=0.2,
        lstm_grad_clip_threshold=1.0,
        learning_rate=0.000001,
        weight_decay=0.1,
        target_value=0.69,
        scale_factor=0.9,
        threshold=0.5
    )
    
    # Evaluate the model and store the results
    experiment_results["LSTM_Base"][f_set] = evaluate_batch_incremental_norolling(LSTM_base, train_data, test_data)

    # After all evaluations are done, save each model's results
    for model_key, model_results in experiment_results.items():
        for feature_key, results in model_results.items():
            if "time_metrics" in results:
                results_filename = f"{model_key}_{feature_key}_results_norolling.csv"
                results["time_metrics"].to_csv(results_filename, index=False)
                print(f"Results saved to {results_filename}")

# Original Rolling Windows by the Authors 
evaluate("AF")
# evaluate("LF")

# Our Split
# evaluate_norolling("AF")
# evaluate_norolling("LF")

print('Experiments complete :)')

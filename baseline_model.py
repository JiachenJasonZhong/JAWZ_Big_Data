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
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

print("All critical packages are compatible.")

import sys
sys.path.append('/Users/ariasarch/aml-crypto-graph/src/')

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

print('packages succesfully loaded')

def evaluate_batch_incremental(model, data, t_eval=35):
    
    results = {}
    results_time = []
    true_test = []
    predictions_test = []
    
    for ts in np.arange(data["ts"].min(), data["ts"].max()):
        
        # get training data for the current timestep 
        train_set = data[data["ts"] == ts]
        train_set_X = train_set.iloc[:,:-1]
        train_set_y = train_set["class"]      

        # partially fit model 
        model.partial_fit(train_set_X.values, train_set_y.values)    

        # get test data for the current timestep + 1 
        test_set = data[data["ts"] == ts + 1]
        test_set_X = test_set.iloc[:,:-1].values
        test_set_y = test_set["class"].values
    
        # predict test data for the current timestep + 1
        y_pred = model.predict(test_set_X)
        evaluation_f1 = f1_score(test_set_y, y_pred, average='binary', labels=np.unique(y_pred))
        evaluation_recall = recall_score(test_set_y, y_pred, average='binary', labels=np.unique(y_pred))
        evaluation_precision = precision_score(test_set_y, y_pred, average='binary', labels=np.unique(y_pred))
        evaluation_accuracy = accuracy_score(test_set_y, y_pred, normalize=True)
        
        # take note of predictions after timestep 34 (evaluation set)
        if ts+1 >= t_eval:
            true_test.append(test_set_y)
            predictions_test.append(y_pred)
            label_count = test_set["class"].value_counts()
            results_time.append({"timestep": ts + 1, 
                                 "score":evaluation_f1, 
                                 "total_pos_label": label_count.tolist()[1]}) 

            
    test_results = {}
    f1_score_test = f1_score(np.concatenate(true_test, axis=0), 
                        np.concatenate(predictions_test, axis=0), 
                        average='binary')
    recall_score_test = recall_score(np.concatenate(true_test, axis=0),   
                                np.concatenate(predictions_test, axis=0), 
                                average='binary')
    precision_score_test = precision_score(np.concatenate(true_test, axis=0),   
                                      np.concatenate(predictions_test, axis=0), 
                                      average='binary')
    accuracy_score_test = accuracy_score(np.concatenate(true_test, axis=0),   
                                    np.concatenate(predictions_test, axis=0), 
                                    normalize=True)
    confusion_matrix_test = confusion_matrix(np.concatenate(true_test, axis=0), 
                                             np.concatenate(predictions_test, axis=0))
    
    test_results["f1"] = round(f1_score_test, 3)   
    test_results["recall"] = round(recall_score_test, 3)   
    test_results["precision"] = round(precision_score_test, 3)   
    test_results["accuracy"] = round(accuracy_score_test, 3)   
    test_results["confusion_matrix"] = confusion_matrix_test  
    
    results["test_results"] = test_results
    results["time_metrics"] = pd.DataFrame(results_time)   

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

    print("Evaluating AXGBp")
    AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      max_window_size=max_window_size,
                                      min_window_size=min_window_size,
                                      detect_drift=detect_drift)
    experiment_3_results["AXGBp"][f_set] = evaluate_batch_incremental(AXGBp, data_eval, n_eval)

    # 4. Proposed Method
    print("Evaluating ASXGB")
    ASXGB = AdaptiveStackedBoostClassifier()
    experiment_3_results["ASXGB"][f_set] = evaluate_batch_incremental(ASXGB, data_eval, n_eval)
    
    elliptic_time_indexed_results(experiment_3_results)
    print(experiment_3_results)

# evaluate("AF", 35)

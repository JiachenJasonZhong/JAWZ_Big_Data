
# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import get_dimensions
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import time
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

###### AdaptiveXGBoost ###################################################

def ncl_smote(X, y, minority_class, majority_class, k_neighbors=5, smote_ratio=0.5):
    # print("Starting NCL-SMOTE sampling")
    # Create a binary mask for the minority class
    minority_mask = (y == minority_class)

    # Separate minority and majority class samples
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]
    X_majority = X[~minority_mask]
    y_majority = y[~minority_mask]

    # Find the k nearest neighbors for each minority class sample
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X_minority)

    # Count the number of majority class neighbors for each minority class sample
    majority_count = np.sum(y[indices] == majority_class, axis=1)

    # Calculate the selection probability for each minority class sample
    selection_prob = majority_count / k_neighbors

    # Create a dataset with minority and majority class samples
    X_resampled = np.concatenate((X_minority, X_majority))
    y_resampled = np.concatenate((y_minority, y_majority))

    # Apply SMOTE to the resampled dataset
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    # Select minority class samples based on the selection probability
    num_selected = int(X_resampled[y_resampled == minority_class].shape[0] * smote_ratio)
    selected_indices = np.random.choice(range(X_minority.shape[0]), size=num_selected, p=selection_prob/np.sum(selection_prob))
    X_selected = X_minority[selected_indices]
    y_selected = y_minority[selected_indices]

    # Combine the selected minority class samples with the majority class samples
    X_final = np.concatenate((X_selected, X_majority))
    y_final = np.concatenate((y_selected, y_majority))

    # Ensure class labels are consistent with the original labels
    y_final = np.where(y_final == minority_class, 1, 0)

    # print(f"Original dataset shape: {X.shape}")
    # print(f"Original minority class count: {np.sum(y == minority_class)}")
    # print(f"Original majority class count: {np.sum(y == majority_class)}")
    # print(f"Minority class samples shape: {X_minority.shape}")
    # print(f"Majority class samples shape: {X_majority.shape}")
    # print(f"Distances shape: {distances.shape}")
    # print(f"Indices shape: {indices.shape}")
    # print(f"Majority count shape: {majority_count.shape}")
    # print(f"Majority count: {majority_count}")
    # print(f"Selection probability shape: {selection_prob.shape}")
    # print(f"Selection probability: {selection_prob}")
    # print(f"Resampled dataset shape before SMOTE: {X_resampled.shape}")
    # print(f"Resampled labels shape before SMOTE: {y_resampled.shape}")
    # print(f"Unique labels in resampled dataset before SMOTE: {np.unique(y_resampled)}")
    # print(f"Resampled dataset shape after SMOTE: {X_resampled.shape}")
    # print(f"Resampled labels shape after SMOTE: {y_resampled.shape}")
    # print(f"Unique labels in resampled dataset after SMOTE: {np.unique(y_resampled)}")
    # print(f"Selected minority class samples: {X_selected.shape}")
    # print(f"Final dataset shape: {X_final.shape}")
    # print(f"Final minority class count: {np.sum(y_final == minority_class)}")
    # print(f"Final majority class count: {np.sum(y_final == majority_class)}")

    return X_final, y_final

# https://github.com/jacobmontiel/AdaptiveXGBoostClassifier
class AdaptiveXGBoostClassifier(BaseSKMObject, ClassifierMixin):
    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self,
                 n_estimators=30,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 detect_drift=False,
                 update_strategy='replace'):
        """
        Adaptive XGBoost classifier.

        Parameters
        ----------
        n_estimators: int (default=5)
            The number of estimators in the ensemble.

        learning_rate:
            Learning rate, a.k.a eta.

        max_depth: int (default = 6)
            Max tree depth.

        max_window_size: int (default=1000)
            Max window size.

        min_window_size: int (default=None)
            Min window size. If this parameters is not set, then a fixed size
            window of size ``max_window_size`` will be used.

        detect_drift: bool (default=False)
            If set will use a drift detector (ADWIN).

        update_strategy: str (default='replace')
            | The update strategy to use:
            | 'push' - the ensemble resembles a queue
            | 'replace' - oldest ensemble members are replaced by newer ones

        Notes
        -----
        The Adaptive XGBoost [1]_ (AXGB) classifier is an adaptation of the
        XGBoost algorithm for evolving data streams. AXGB creates new members
        of the ensemble from mini-batches of data as new data becomes
        available.  The maximum ensemble  size is fixed, but learning does not
        stop once this size is reached, the ensemble is updated on new data to
        ensure consistency with the current data distribution.

        References
        ----------
        .. [1] Montiel, Jacob, Mitchell, Rory, Frank, Eibe, Pfahringer,
           Bernhard, Abdessalem, Talel, and Bifet, Albert. “AdaptiveXGBoost for
           Evolving Data Streams”. In:IJCNN’20. International Joint Conference
           on Neural Networks. 2020. Forthcoming.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._ensemble = None
        self.detect_drift = detect_drift
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0
        if update_strategy not in self._UPDATE_STRATEGIES:
            raise AttributeError("Invalid update_strategy: {}\n"
                                 "Valid options: {}".format(update_strategy,
                                                            self._UPDATE_STRATEGIES))
        self.update_strategy = update_strategy
        self._configure()

    def _configure(self):
        if self.update_strategy == self._PUSH_STRATEGY:
            self._ensemble = []
        elif self.update_strategy == self._REPLACE_STRATEGY:
            self._ensemble = [None] * self.n_estimators
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {"silent": True,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = ADWIN()

    def reset(self):
        """
        Reset the estimator.
        """
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the data upon which
            the algorithm will create its model.

        y: Array-like
            An array of shape (, n_samples) containing the classification
            targets for all samples in X. Only binary data is supported.

        classes: Not used.

        sample_weight: Not used.

        Returns
        -------
        AdaptiveXGBoostClassifier
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self.window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self.window_size, :],
                                      y=self._y_buffer[0:self.window_size])
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

        # Support for concept drift
        if self.detect_drift:
            correctly_classifies = self.predict(X) == y
            # Check for warning
            self._drift_detector.add_element(int(not correctly_classifies))
            # Check if there was a change
            if self._drift_detector.detected_change():
                # Reset window size
                self._reset_window_size()
                if self.update_strategy == self._REPLACE_STRATEGY:
                    self._model_idx = 0

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        if self.update_strategy == self._REPLACE_STRATEGY:
            booster = self._train_booster(X, y, self._model_idx)
            # Update ensemble
            self._ensemble[self._model_idx] = booster
            self._samples_seen += X.shape[0]
            self._update_model_idx()
        else:   # self.update_strategy == self._PUSH_STRATEGY
            booster = self._train_booster(X, y, len(self._ensemble))
            # Update ensemble
            if len(self._ensemble) == self.n_estimators:
                self._ensemble.pop(0)
            self._ensemble.append(booster)
            self._samples_seen += X.shape[0]

    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx: int):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))
        # Get margins from trees in the ensemble
        margins = np.asarray([self._init_margin] * d_mini_batch_train.num_row())
        # Add logging to check if any model is None
        for j in range(last_model_idx):
            if self._ensemble[j] is None:
                print(f"Model at index {j} is None")  # You can replace print with logging.error if you use logging
            else:
                margins = np.add(margins,
                                self._ensemble[j].predict(d_mini_batch_train, output_margin=True))
    
        d_mini_batch_train.set_base_margin(margin=margins)
        booster = xgb.train(params=self._boosting_params,
                            dtrain=d_mini_batch_train,
                            num_boost_round=1,
                            verbose_eval=False)
        return booster

    def _update_model_idx(self):
        self._model_idx += 1
        if self._model_idx == self.n_estimators:
            self._model_idx = 0

    def predict(self, X):
        """
        Predict the class label for sample X

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the class label for.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.

        """
        if self._ensemble:
            if self.update_strategy == self._REPLACE_STRATEGY:
                trees_in_ensemble = sum(i is not None for i in self._ensemble)
            else:   # self.update_strategy == self._PUSH_STRATEGY
                trees_in_ensemble = len(self._ensemble)
            if trees_in_ensemble > 0:
                d_test = xgb.DMatrix(X)
                for i in range(trees_in_ensemble - 1):
                    margins = self._ensemble[i].predict(d_test, output_margin=True)
                    d_test.set_base_margin(margin=margins)
                predicted = self._ensemble[trees_in_ensemble - 1].predict(d_test)
                return np.array(predicted > 0.5).astype(int)
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        if self._ensemble:
            d_test = xgb.DMatrix(X)
            margins = np.zeros(X.shape[0])
            for booster in self._ensemble:
                if booster is not None:
                    margins += booster.predict(d_test, output_margin=True)
            # Normalize margins by the number of models to get the average score
            if self.update_strategy == self._PUSH_STRATEGY:
                margins /= len(self._ensemble)
            probabilities = 1 / (1 + np.exp(-margins))
            return np.vstack([1 - probabilities, probabilities]).T
        # Return no models are available
        return print('no models are available')

class AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4):
        
        self._first_run = True
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        
        # validate 'n_base_models' 
        if n_base_models <= 1:
            raise ValueError("'n_base_models' must be > 1")
        self._n_base_models = n_base_models
        # validate 'n_rounds_eval_base_model' 
        if n_rounds_eval_base_model > n_base_models or n_rounds_eval_base_model <= 0:
            raise ValueError("'n_rounds_eval_base_model' must be > 0 and <= to 'n_base_models'")
        self._n_rounds_eval_base_model = n_rounds_eval_base_model
        self._meta_learner = xgb.XGBClassifier(n_jobs=-1)
        self.meta_learner_train_ratio = meta_learner_train_ratio
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        # 3*N matrix 
        # 1st row - base-level model
        # 2nd row - evaluation rounds 
        self._base_models = [[None for x in range(n_base_models)] for y in range(3)]
        
        self._reset_window_size()
        
    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self._window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self._window_size = self._dynamic_window_size

        
    def partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, X.shape[1])
            self._y_buffer = np.array([])
            self._first_run = False
                           
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self._window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self._window_size, :],
                                      y=self._y_buffer[0:self._window_size])
            delete_idx = [i for i in range(self._window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)
    
    def _train_new_base_model(self, X_base, y_base, X_meta, y_meta):
        
        # new base-level model  
        new_base_model = xgb.XGBClassifier(n_jobs=-1)
        # first train the base model on the base-level training set 
        new_base_model.fit(X_base, y_base)
        # then extract the predicted probabilities to be added as meta-level features
        y_predicted = new_base_model.predict_proba(X_meta)   
        # once the meta-features for this specific base-model are extracted,
        # we incrementally fit this base-model to the rest of the data,
        # this is done so this base-model is trained on a full batch 
        new_base_model.fit(X_meta, y_meta, xgb_model=new_base_model.get_booster())
        return new_base_model, y_predicted
    
    def _construct_meta_features(self, meta_features):
        
        # get size of of meta-features
        meta_features_shape = meta_features.shape[1]  
        # get expected number of features,
        # binary probabilities from the total number of base-level models
        meta_features_expected = self._n_base_models * 2
        
        # since the base-level models list is not full, 
        # we need to fill the features until the list is full, 
        # so we set the remaining expected meta-features as 0
        if meta_features_shape < meta_features_expected:
            diff = meta_features_expected - meta_features_shape
            empty_features = np.zeros((meta_features.shape[0], diff))
            meta_features = np.hstack((meta_features, empty_features)) 
        return meta_features 
        
    def _get_weakest_base_learner(self):
        
        # loop rounds
        worst_model_idx = None 
        worst_performance = 1
        for idx in range(len(self._base_models[0])):
            current_round = self._base_models[1][idx]
            if current_round < self._n_rounds_eval_base_model:
                continue 
            
            current_performance = self._base_models[2][idx].sum()
            if current_performance < worst_performance:
                worst_performance = current_performance 
                worst_model_idx = idx

        return worst_model_idx
    
    def _train_on_mini_batch(self, X, y):
        
        # ----------------------------------------------------------------------------
        # STEP 1: split mini batch to base-level and meta-level training set
        # ----------------------------------------------------------------------------
        base_idx = int(self._window_size * (1.0 - self.meta_learner_train_ratio))
        X_base = X[0: base_idx, :]
        y_base = y[0: base_idx] 

        # this part will be used to train the meta-level model,
        # and to continue training the base-level models on the rest of this batch
        X_meta = X[base_idx:self._window_size, :]  
        y_meta = y[base_idx:self._window_size]
        
        # ----------------------------------------------------------------------------
        # STEP 2: train previous base-models 
        # ----------------------------------------------------------------------------
        meta_features = []
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len > 0: # check if we have any base-level models         
            base_model_performances = self._meta_learner.feature_importances_
            for b_idx in range(base_models_len): # loop and train and extract meta-level features 
                    
                # continuation of training (incremental) on base-level model,
                # using the base-level training set 
                base_model = self._base_models[0][b_idx]
                base_model.fit(X_base, y_base, xgb_model=base_model.get_booster())
                y_predicted = base_model.predict_proba(X_meta) # extract meta-level features 
                                
                # extract meta-features 
                meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
                
                # once the meta-features for this specific base-model are extracted,
                # we incrementally fit this base-model to the rest of the data,
                # this is done so this base-model is trained on a full batch 
                base_model.fit(X_meta, y_meta, xgb_model=base_model.get_booster())
                                
                # update base-level model list 
                self._base_models[0][b_idx] = base_model
                current_round = self._base_models[1][b_idx]
                last_performance = base_model_performances[b_idx * 2] + base_model_performances[(b_idx*2)+1] 
                self._base_models[2][b_idx][current_round%self._n_rounds_eval_base_model] = last_performance
                self._base_models[1][b_idx] = current_round + 1
                
        # ----------------------------------------------------------------------------
        # STEP 3: with each new batch, we create/train a new base model 
        # ----------------------------------------------------------------------------
        new_base_model, new_base_model_meta_features = self._train_new_base_model(X_base, y_base, X_meta, y_meta)

        insert_idx = base_models_len
        if base_models_len == 0:
            meta_features = new_base_model_meta_features
        elif base_models_len > 0 and base_models_len < self._n_base_models: 
            meta_features = np.hstack((meta_features, new_base_model_meta_features))     
        else: 
            insert_idx = self._get_weakest_base_learner()           
            meta_features[:, insert_idx * 2] = new_base_model_meta_features[:,0]
            meta_features[:, (insert_idx * 2) + 1] = new_base_model_meta_features[:,1]
            
        self._base_models[0][insert_idx] = new_base_model 
        self._base_models[1][insert_idx] = 0 
        self._base_models[2][insert_idx] = np.zeros(self._n_rounds_eval_base_model) 

        # STEP 4: train the meta-level model 
        meta_features = self._construct_meta_features(meta_features)
        
        if base_models_len == 0:
            self._meta_learner.fit(meta_features, y_meta)
        else:
            self._meta_learner.fit(meta_features, y_meta, xgb_model=self._meta_learner.get_booster())

    def predict(self, X):
      
        # only one model in ensemble use its predictions 
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len < self._n_base_models:
            predictions = []
            for i in range(base_models_len):
                tmp_predictions = self._base_models[0][i].predict(X)
                predictions.append(tmp_predictions)
            output = [int(Counter(col).most_common(1)[0][0]) for col in zip(*predictions)] 
            return output
        
        # predict via meta learner 
        meta_features = []           
        for b_idx in range(base_models_len):
            y_predicted = self._base_models[0][b_idx].predict_proba(X) 
            meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
        meta_features = self._construct_meta_features(meta_features)
        return self._meta_learner.predict(meta_features)
    
    def eval_proba(self, X):
        
        # only one model in ensemble use its predictions 
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len == 0:
            raise Exception("No base models have been trained.")

        meta_features = []
        for b_idx in range(base_models_len):
            y_predicted = self._base_models[0][b_idx].predict_proba(X)
            meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))
        
        meta_features = self._construct_meta_features(meta_features)
        return self._meta_learner.predict_proba(meta_features)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, lstm_dropout, learning_rate, weight_decay, target_value, scale_factor):

        # Initialize key variables 
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = hidden_size
        self.lstm_dropout = lstm_dropout 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_value = target_value
        self.scale_factor = scale_factor

        # Initialize weights and biases
        # print(f'hidden_size type = {hidden_size}')
        # print(f'input_size type = {input_size}')
        initialization_value = 0.1
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * initialization_value 
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * initialization_value 
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * initialization_value 
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * initialization_value
        self.Wy = np.random.randn(output_size, hidden_size) * initialization_value

        # Xavier initialization
        # self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + 2 * hidden_size))
        # self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + 2 * hidden_size))
        # self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + 2 * hidden_size))
        # self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + 2 * hidden_size))
        # self.Wy = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))

        # He initialization
        # self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        # self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        # self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        # self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        # self.Wy = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)

        # Bias initilization
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Initialize gradient matrices
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dWy = np.zeros_like(self.Wy)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dby = np.zeros_like(self.by)

    def reset_gradients(self):
        self.dWf.fill(0)
        self.dWi.fill(0)
        self.dWc.fill(0)
        self.dWo.fill(0)
        self.dWy.fill(0)
        self.dbf.fill(0)
        self.dbi.fill(0)
        self.dbc.fill(0)
        self.dbo.fill(0)
        self.dby.fill(0)

    def forward(self, x, h_prev, c_prev):
        num_samples = x.shape[0]
        
        y_preds = np.zeros((num_samples, self.output_size))
        i_list = np.zeros((num_samples, self.hidden_size, self.num_layers))
        c_bar_list = np.zeros((num_samples, self.hidden_size, self.num_layers))
        f_list = np.zeros((num_samples, self.hidden_size, self.num_layers))
        o_list = np.zeros((num_samples, self.hidden_size, self.num_layers))
        
        dropout_masks = np.zeros((num_samples, self.hidden_size, self.num_layers))
        
        for t in range(num_samples):
            xt = x[t].reshape(self.input_size, 1)  # Shape (input_size, 1)
            h_next_t = np.zeros((self.hidden_size, self.num_layers))
            c_next_t = np.zeros((self.hidden_size, self.num_layers))

            # Apply dropout
            h_next_t, mask = dropout(h_next_t, self.lstm_dropout)
            dropout_masks[t] = mask
            
            # Reshape and repeat operations
            h_prev_reshaped = h_prev.reshape(self.hidden_size, self.num_layers)
            c_prev_reshaped = c_prev.reshape(self.hidden_size, self.num_layers)
            xt_repeated = np.repeat(xt, self.num_layers, axis=1)
            concat = np.vstack((h_prev_reshaped, xt_repeated))
            
            # Gate activations
            f_t = sigmoid(np.dot(self.Wf, concat) + self.bf)
            i_t = sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_bar_t = np.tanh(np.dot(self.Wc, concat) + self.bc)
            o_t = sigmoid(np.dot(self.Wo, concat) + self.bo)
            
            # Cell state and hidden state
            c_next_t = f_t * c_prev_reshaped + i_t * c_bar_t
            h_next_t = o_t * np.tanh(c_next_t)
            
            # Output prediction
            yt = sigmoid(np.dot(self.Wy, h_next_t[:, -1].reshape(self.hidden_size, 1)) + self.by)
            y_preds[t] = yt.squeeze()  # Squeeze to remove extra dimensions if any

            # Store gate activations
            i_list[t] = i_t
            c_bar_list[t] = c_bar_t
            f_list[t] = f_t
            o_list[t] = o_t
            
            # Update previous states
            h_prev = h_next_t
            c_prev = c_next_t
        
        y_preds = y_preds.reshape(-1, self.output_size)

        # print(f'Shape of x: {x.shape}, expected: (num_samples, {self.input_size})')
        # print(f'Shape of h_prev: {h_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_prev: {c_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')

        # print(f'  Shape of h_prev_l: {h_prev_l.shape}, expected: ({self.hidden_size}, 1)')
        # print(f'  Shape of c_prev_l: {c_prev_l.shape}, expected: ({self.hidden_size}, 1)')
        # print(f'  Shape of concat: {concat.shape}, expected: ({self.hidden_size + self.input_size}, 1)')
        # print(f'  Shape of f_t[:, {l}]: {f_t[:, l].shape}, expected: ({self.hidden_size},)')
        # print(f'  Shape of i_t[:, {l}]: {i_t[:, l].shape}, expected: ({self.hidden_size},)')
        # print(f'  Shape of c_bar_t[:, {l}]: {c_bar_t[:, l].shape}, expected: ({self.hidden_size},)')
        # print(f'  Shape of c_next_t[:, {l}]: {c_next_t[:, l].shape}, expected: ({self.hidden_size},)')
        # print(f'  Shape of o_t[:, {l}]: {o_t[:, l].shape}, expected: ({self.hidden_size},)')
        # print(f'  Shape of h_next_t[:, {l}]: {h_next_t[:, l].shape}, expected: ({self.hidden_size},)')

        # print(f'Shape of h_next: {h_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_next: {c_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of y_preds: {y_preds.shape}, expected: (num_samples, {self.output_size})')

        return y_preds, h_next_t, c_next_t, i_list, c_bar_list, f_list, o_list

    def backward(self, x, y, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list):
        num_samples = x.shape[0]
        
        dh_next_t = np.zeros((self.hidden_size, self.num_layers))
        dc_next_t = np.zeros((self.hidden_size, self.num_layers))

        for t in reversed(range(num_samples)):  # Iterate over each sample in reverse order
            dE_dy = (y_preds[t] - y[t]) + self.scale_factor * 2 * (y_preds[t] - self.target_value)
            self.dWy += np.dot(dE_dy.reshape(self.output_size, 1), h_prev[:, -1].reshape(1, self.hidden_size))
            self.dby += dE_dy.reshape(self.output_size, 1)

            # Compute gradients for all layers simultaneously
            dh = np.dot(self.Wy.T, dE_dy.reshape(self.output_size, 1)) + dh_next_t
            dc = dc_next_t + dh * o_list[t] * (1 - np.square(np.tanh(c_prev)))
            do = dh * np.tanh(c_prev)
            dc_bar = dh * i_list[t]
            di = dh * c_bar_list[t]
            df = dh * c_prev

            xt = x[t].reshape(self.input_size, 1)
            concat = np.vstack((h_prev, np.repeat(xt, self.num_layers, axis=1)))

            sigmoid_der_f = sigmoid_derivative(f_list[t])
            sigmoid_der_i = sigmoid_derivative(i_list[t])
            sigmoid_der_o = sigmoid_derivative(o_list[t])
            tanh_der_c_bar = 1 - np.square(c_bar_list[t])

            self.dWf += np.dot((df * sigmoid_der_f), concat.T)
            self.dWi += np.dot((di * sigmoid_der_i), concat.T)
            self.dWc += np.dot((dc_bar * tanh_der_c_bar), concat.T)
            self.dWo += np.dot((do * sigmoid_der_o), concat.T)

            self.dbf += np.sum(df * sigmoid_der_f, axis=1, keepdims=True)
            self.dbi += np.sum(di * sigmoid_der_i, axis=1, keepdims=True)
            self.dbc += np.sum(dc_bar * tanh_der_c_bar, axis=1, keepdims=True)
            self.dbo += np.sum(do * sigmoid_der_o, axis=1, keepdims=True)

            dh_next_t = dh
            dc_next_t = dc

        return self.dWf, self.dWi, self.dWc, self.dWo, self.dWy, self.dbf, self.dbi, self.dbc, self.dbo, self.dby

    def update_weights(self, learning_rate, weight_decay):
        # Updates weights using gradients with L2 regularization.
        self.Wf -= learning_rate * (self.dWf + weight_decay * self.Wf)
        self.Wi -= learning_rate * (self.dWi + weight_decay * self.Wi)
        self.Wc -= learning_rate * (self.dWc + weight_decay * self.Wc)
        self.Wo -= learning_rate * (self.dWo + weight_decay * self.Wo)
        self.Wy -= learning_rate * (self.dWy + weight_decay * self.Wy)

    def save_weights(self, file_path):
        # Save the LSTM model's weights to a file
        weights = {
            'Wf': self.Wf,
            'Wi': self.Wi,
            'Wc': self.Wc,
            'Wo': self.Wo,
            'Wy': self.Wy,
            'bf': self.bf,
            'bi': self.bi,
            'bc': self.bc,
            'bo': self.bo,
            'by': self.by
        }
        np.savez(file_path, **weights)
        # print("Weights saved successfully.")
        # print("Saved weights:")
        # for key, value in weights.items():
        #     print(f"{key}: {value}")
    
    def load_weights(self, file_path):
        # Load the LSTM model's weights from a file
        with np.load(file_path) as data:
            self.Wf = data['Wf']
            self.Wi = data['Wi']
            self.Wc = data['Wc']
            self.Wo = data['Wo']
            self.Wy = data['Wy']
            self.bf = data['bf']
            self.bi = data['bi']
            self.bc = data['bc']
            self.bo = data['bo']
            self.by = data['by']

# Necessary variables for LSTM
batch_size = 100000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def dropout(x, dropout_rate):

    # Applies dropout by randomly setting a fraction of x to zero.
    if dropout_rate > 0:
        retain_prob = 1 - dropout_rate
        mask = np.random.binomial(1, retain_prob, size=x.shape)
        return x * mask, mask
    return x, np.ones_like(x)

def binary_cross_entropy(y_true, y_pred):
    # Avoid division by zero
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def modified_mse_loss(y_true, y_pred):
    mse_loss = np.mean((y_true - y_pred) ** 2)
    deviation_penalty = np.mean((y_pred - 0.5) ** 2)
    return mse_loss + deviation_penalty

def modified_binary_cross_entropy(y_true, y_pred, target_value, scale_factor):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    penalty_term = scale_factor * np.mean((target_value - y_pred) ** 2)
    return bce_loss + penalty_term

class LSTM_Base():
    def __init__(self,
                 lstm_units = None,
                 lstm_epochs = None, 
                 lstm_dropout = None, 
                 lstm_grad_clip_threshold = None,
                 learning_rate = None,
                 weight_decay = None,
                 output_size = 1,
                 target_value = None,
                 scale_factor = None,
                 threshold = None):
        
        self.hidden_dim = lstm_units
        self.lstm_epochs = lstm_epochs
        self.lstm_dropout = lstm_dropout 
        self.lstm_grad_clip_threshold = lstm_grad_clip_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._lstm_model = None
        self.target_value = target_value
        self.scale_factor = scale_factor
        self.threshold = threshold

        # print("\n" + "="*50)
        # print(f"SELF.LSTM_UNITS = {self.hidden_dim}")
        # print(f"SELF.LSTM_EPOCHS = {self.lstm_epochs}")
        # print(f"SELF.LSTM_DROPOUT = {self.lstm_dropout}")
        # print(f"SELF.LSTM_GRADCLIP = {self.lstm_grad_clip_threshold}")
        # print(f"SELF.LSTM_LEARNING = {self.learning_rate}")
        # print(f"SELF.LSTM_WEIGHTDECAY = {self.weight_decay}")
        # print(f"SELF.TARGET_VALUE = {self.target_value}")
        # print(f"SELF.SCALE_FACTOR = {self.scale_factor}")
        # print(f"SELF.THRESHOLD = {self.threshold}")
        # print("="*50 + "\n")

    def train_lstm(self, x, y):
        # Get the original class labels
        minority_class = 1  # Assuming minority class label is 1
        majority_class = 0  # Assuming majority class label is 0
        
        try:
            # Perform NCL-SMOTE oversampling on the training data
            X_train_resampled, y_train_resampled = ncl_smote(x, y, minority_class, majority_class)
            
            # Update the training data with the resampled data
            x = X_train_resampled
            y = y_train_resampled

        except ValueError as e:
            print(f"Skipping NCL-SMOTE due to error: {str(e)}")
            # Continue without applying NCL-SMOTE
            pass
                   
        # Train the LSTM model
        if self._lstm_model is None:
            self._lstm_model = LSTM(input_size=x.shape[1], hidden_size=self.hidden_dim, output_size=1, 
                                    lstm_dropout=self.lstm_dropout, learning_rate=self.learning_rate, 
                                    weight_decay=self.weight_decay, target_value=self.target_value, scale_factor=self.scale_factor)

        # Define num_samples based on X
        num_samples = x.shape[0]

        # Initialize a list to store predictions from the final epoch
        final_epoch_preds = []
        accuracy_over_epochs = []
        losses = []  # List to store loss values

        # Count the total number of 1s and 0s in the actual labels
        total_ones_actual = np.sum(y)
        total_zeros_actual = len(y) - total_ones_actual

        # print(f"Total number of actual 1s in y: {total_ones_actual}")
        # print(f"Total number of actual 0s in y: {total_zeros_actual}")

        y_preds_avg_over_epochs = []
        f1_scores = []

        # Iterate over mini-batches
        num_batches = (num_samples + batch_size - 1) // batch_size
    
        # Perform training loop
        for epoch in range(self.lstm_epochs):
            # print(f"Epoch {epoch+1}/{self.lstm_epochs} initiated.")

            # Temporary list for the current epoch predictions
            current_epoch_preds = []
            total_loss = 0
            total_correct = 0
            total_samples = 0
            total_ones = 0
            total_zeros = 0

            # Iterate over mini-batches
            for batch_index, batch_start in enumerate(range(0, num_samples, batch_size)):

                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                batch_size_actual = y_batch.shape[0]  

                # Initialize hidden state and cell state
                h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
                c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

                # Forward pass
                start_time_forward = time.time()
                y_preds, h_next, c_next, i_list, c_bar_list, f_list, o_list = self._lstm_model.forward(x_batch, h_prev, c_prev)
                end_time_forward = time.time()
                forward_time = end_time_forward - start_time_forward
                # print(f"Forward pass time: {forward_time:.2f} seconds")

                # Print the statistics of y_preds
                # print(f"y_preds average: {np.mean(y_preds)}")
                # print(f"y_preds minimum: {np.min(y_preds)}")
                # print(f"y_preds maximum: {np.max(y_preds)}")
                y_preds_avg_over_epochs.append(np.mean(y_preds))

                # print(y_preds)

                # Store predictions and loss from the current batch
                current_epoch_preds.append(y_preds)
                loss = modified_binary_cross_entropy(y_batch, y_preds.flatten(), target_value=self.target_value, scale_factor=1.0)
                total_loss += loss * batch_size_actual  # Weighting the loss by the batch size

                # Calculate and accumulate accuracy
                batch_predictions = (y_preds.flatten() > self.threshold).astype(int)
                total_correct += np.sum(batch_predictions == y_batch)
                total_samples += batch_size_actual

                # Calculate F1 score for the current batch
                batch_f1_score = f1_score(y_batch, batch_predictions)
                f1_scores.append(batch_f1_score)

                # Count the occurrences of 1s and 0s in the current batch
                total_ones += np.sum(batch_predictions)
                total_zeros += batch_size_actual - np.sum(batch_predictions)

                # print(f"Total number of 1s in y_preds: {total_ones}")
                # print(f"Total number of 0s in y_preds: {total_zeros}")

                start_time_backward = time.time()
                # Backward pass
                dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby = self._lstm_model.backward(x_batch, y_batch, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list)
                end_time_backward = time.time()
                backward_time = end_time_backward - start_time_backward
                # print(f"Backward pass time: {backward_time:.2f} seconds")

                # Clipping gradients
                grad_norm = np.sqrt(sum(np.sum(grad**2) for grad in [dWf, dWi, dWc, dWo, dWy]))
                if grad_norm > self.lstm_grad_clip_threshold:
                    clip_coef = self.lstm_grad_clip_threshold / (grad_norm + 1e-6)  # Avoid division by zero
                    dWf, dWi, dWc, dWo, dWy = [clip_coef * grad for grad in [dWf, dWi, dWc, dWo, dWy]]

                # Update weights and biases
                self._lstm_model.update_weights(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
                self._lstm_model.bf -= self.learning_rate * dbf
                self._lstm_model.bi -= self.learning_rate * dbi
                self._lstm_model.bc -= self.learning_rate * dbc
                self._lstm_model.bo -= self.learning_rate * dbo
                self._lstm_model.by -= self.learning_rate * dby

                # Reset gradients for the next batch
                self._lstm_model.reset_gradients()

                # After processing all batches in the current epoch
                if epoch == self.lstm_epochs - 1:  # Check if it's the final epoch
                    final_epoch_preds = current_epoch_preds  # Only store the final epoch's predictions
            
            # Compute average loss and accuracy for the epoch
            average_epoch_loss = total_loss / total_samples
            epoch_accuracy = total_correct / total_samples
            losses.append(average_epoch_loss)
            accuracy_over_epochs.append(epoch_accuracy)

            # print(f"Epoch {epoch+1}/{self.lstm_epochs} completed.")

            # Estimate the remaining time
            total_estimated_time = (forward_time + backward_time) * (self.lstm_epochs - (epoch+1))

            # print("\n" + "*"*50)
            # print(f"*{' ':48}*")
            # print(f"*  Estimated remaining time: {total_estimated_time:.2f} seconds  *")
            # print(f"*{' ':48}*")
            # print("*" * 50 + "\n")

        # print(f'y_preds_avg_over_epochs = {y_preds_avg_over_epochs}')
        
        # Generate epoch labels dynamically
        epochs = [f'Epoch {i+1}' for i in range(self.lstm_epochs)]

        # Plotting
        # plt.figure(figsize=(10, 6))
        # plt.plot(y_preds_avg_over_epochs, marker='o', linestyle='-', color='b', label='Average Prediction')
        # plt.plot(f1_scores, marker='o', linestyle='-', color='r', label='F1 Score')
        # plt.title('Average Predictions and F1 Scores Over Epochs')
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.xticks(range(self.lstm_epochs), epochs)
        # plt.legend()
        # plt.show()

        # Save model weights
        self._lstm_model.save_weights('lstm_weights.npz')

    def predict(self, X):
        self._lstm_model.load_weights('lstm_weights.npz')
        # print("Weights loaded for prediction.")
        
        # Print the loaded weights
        # print("Loaded weights:")
        # print(f"Wf: {self._lstm_model.Wf}")
        # print(f"Wi: {self._lstm_model.Wi}")
        # print(f"Wc: {self._lstm_model.Wc}")
        # print(f"Wo: {self._lstm_model.Wo}")
        # print(f"Wy: {self._lstm_model.Wy}")
        # print(f"bf: {self._lstm_model.bf}")
        # print(f"bi: {self._lstm_model.bi}")
        # print(f"bc: {self._lstm_model.bc}")
        # print(f"bo: {self._lstm_model.bo}")
        # print(f"by: {self._lstm_model.by}")
        
        # Initialize hidden state and cell state for LSTM prediction
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        
        # Process the input sequence X
        num_samples = X.shape[0]
        y_preds = np.zeros((num_samples, self._lstm_model.output_size))
        
        for t in range(num_samples):
            xt = X[t].reshape(self._lstm_model.input_size, 1)
            
            # Reshape and repeat operations
            h_prev_reshaped = h_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            c_prev_reshaped = c_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            xt_repeated = np.repeat(xt, self._lstm_model.num_layers, axis=1)
            concat = np.vstack((h_prev_reshaped, xt_repeated))
            
            # Gate activations using loaded weights
            f_t = sigmoid(np.dot(self._lstm_model.Wf, concat) + self._lstm_model.bf)
            i_t = sigmoid(np.dot(self._lstm_model.Wi, concat) + self._lstm_model.bi)
            c_bar_t = np.tanh(np.dot(self._lstm_model.Wc, concat) + self._lstm_model.bc)
            o_t = sigmoid(np.dot(self._lstm_model.Wo, concat) + self._lstm_model.bo)
            
            # Cell state and hidden state using loaded weights
            c_next_t = f_t * c_prev_reshaped + i_t * c_bar_t
            h_next_t = o_t * np.tanh(c_next_t)
            
            # Output prediction using loaded weights
            yt = sigmoid(np.dot(self._lstm_model.Wy, h_next_t[:, -1].reshape(self._lstm_model.hidden_size, 1)) + self._lstm_model.by)
            y_preds[t] = yt.squeeze()
            
            # Update previous states
            h_prev = h_next_t
            c_prev = c_next_t
        
        # Generate final predictions
        # Convert LSTM outputs to binary predictions (0 or 1)
        final_predictions = (y_preds.flatten() > self.threshold).astype(int)
        
        # print("Prediction completed.") 
        return final_predictions

    def eval_proba(self, X):
        self._lstm_model.load_weights('lstm_weights.npz')
        # print("Weights loaded for probability evaluation.")
        
        # Print the loaded weights
        # print("Loaded weights:")
        # print(f"Wf: {self._lstm_model.Wf}")
        # print(f"Wi: {self._lstm_model.Wi}")
        # print(f"Wc: {self._lstm_model.Wc}")
        # print(f"Wo: {self._lstm_model.Wo}")
        # print(f"Wy: {self._lstm_model.Wy}")
        # print(f"bf: {self._lstm_model.bf}")
        # print(f"bi: {self._lstm_model.bi}")
        # print(f"bc: {self._lstm_model.bc}")
        # print(f"bo: {self._lstm_model.bo}")
        # print(f"by: {self._lstm_model.by}")
        
        # Initialize hidden state and cell state for LSTM
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        
        # Process the input sequence X
        num_samples = X.shape[0]
        y_preds = np.zeros((num_samples, self._lstm_model.output_size))
        
        for t in range(num_samples):
            xt = X[t].reshape(self._lstm_model.input_size, 1)
            
            # Reshape and repeat operations
            h_prev_reshaped = h_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            c_prev_reshaped = c_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            xt_repeated = np.repeat(xt, self._lstm_model.num_layers, axis=1)
            concat = np.vstack((h_prev_reshaped, xt_repeated))
            
            # Gate activations using loaded weights
            f_t = sigmoid(np.dot(self._lstm_model.Wf, concat) + self._lstm_model.bf)
            i_t = sigmoid(np.dot(self._lstm_model.Wi, concat) + self._lstm_model.bi)
            c_bar_t = np.tanh(np.dot(self._lstm_model.Wc, concat) + self._lstm_model.bc)
            o_t = sigmoid(np.dot(self._lstm_model.Wo, concat) + self._lstm_model.bo)
            
            # Cell state and hidden state using loaded weights
            c_next_t = f_t * c_prev_reshaped + i_t * c_bar_t
            h_next_t = o_t * np.tanh(c_next_t)
            
            # Output prediction using loaded weights
            yt = sigmoid(np.dot(self._lstm_model.Wy, h_next_t[:, -1].reshape(self._lstm_model.hidden_size, 1)) + self._lstm_model.by)
            y_preds[t] = yt.squeeze()
            
            # Update previous states
            h_prev = h_next_t
            c_prev = c_next_t
        
        # Return the sigmoid outputs as probabilities
        probabilities = sigmoid(y_preds.flatten())
        
        # print("Probability evaluation completed.")
        # print("Probabilities:")
        # print(probabilities)
        
        return probabilities

class LSTM_AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4,
                 lstm_units=None,
                 lstm_epochs=None, 
                 lstm_dropout=None, 
                 lstm_grad_clip_threshold=None,
                 learning_rate=None,
                 weight_decay=None,
                 output_size = 1,
                 target_value = None,
                 scale_factor = None,
                 threshold = None):

        self.lstm_epochs = lstm_epochs
        self.lstm_dropout = lstm_dropout 
        self.lstm_grad_clip_threshold = lstm_grad_clip_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dim = lstm_units
        self._lstm_model = None
        self._first_run = True
        self._first_run = True
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.target_value = target_value
        self.scale_factor = scale_factor
        self.threshold = threshold

        self.lstm = LSTM(input_size=n_base_models * 2, hidden_size=self.hidden_dim, output_size=1, 
                            lstm_dropout=self.lstm_dropout, learning_rate=self.learning_rate, 
                            weight_decay=self.weight_decay, target_value=self.target_value, scale_factor=self.scale_factor)
        
        # validate 'n_base_models' 
        if n_base_models <= 1:
            raise ValueError("'n_base_models' must be > 1")
        self._n_base_models = n_base_models
        # validate 'n_rounds_eval_base_model' 
        if n_rounds_eval_base_model > n_base_models or n_rounds_eval_base_model <= 0:
            raise ValueError("'n_rounds_eval_base_model' must be > 0 and <= to 'n_base_models'")
        self._n_rounds_eval_base_model = n_rounds_eval_base_model
        self._meta_learner = xgb.XGBClassifier(n_jobs=-1)
        self.meta_learner_train_ratio = meta_learner_train_ratio
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        # 3*N matrix 
        # 1st row - base-level model
        # 2nd row - evaluation rounds 
        self._base_models = [[None for x in range(n_base_models)] for y in range(3)]
        
        self._reset_window_size()

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self._window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self._window_size = self._dynamic_window_size
     
    def partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, X.shape[1])
            self._y_buffer = np.array([])
            self._first_run = False
                        
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        
        total_mini_batches = self._X_buffer.shape[0] // self._window_size
        # print(f"Total mini-batches: {total_mini_batches}")
        
        train_count = 0
        while self._X_buffer.shape[0] >= self._window_size:
            # print(f"Training on mini-batch {train_count + 1}/{total_mini_batches}")
            self._train_on_mini_batch(X=self._X_buffer[0:self._window_size, :],
                                    y=self._y_buffer[0:self._window_size])
            delete_idx = [i for i in range(self._window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)
            train_count += 1
        
        # print(f"Total mini-batches trained: {train_count}")
    
    def _train_new_base_model(self, X_base, y_base, X_meta, y_meta):
        # new base-level model  
        new_base_model = xgb.XGBClassifier(n_jobs=-1)
        # first train the base model on the base-level training set 
        new_base_model.fit(X_base, y_base)
        # then extract the predicted probabilities to be added as meta-level features
        y_predicted = new_base_model.predict_proba(X_meta)   
        # once the meta-features for this specific base-model are extracted,
        # we incrementally fit this base-model to the rest of the data,
        # this is done so this base-model is trained on a full batch 
        new_base_model.fit(X_meta, y_meta, xgb_model=new_base_model.get_booster())
        return new_base_model, y_predicted
    
    def _construct_meta_features(self, meta_features):
        
        # get size of of meta-features
        meta_features_shape = meta_features.shape[1]  
        # get expected number of features,
        # binary probabilities from the total number of base-level models
        meta_features_expected = self._n_base_models * 2
        
        # since the base-level models list is not full, 
        # we need to fill the features until the list is full, 
        # so we set the remaining expected meta-features as 0
        if meta_features_shape < meta_features_expected:
            diff = meta_features_expected - meta_features_shape
            empty_features = np.zeros((meta_features.shape[0], diff))
            meta_features = np.hstack((meta_features, empty_features)) 
        return meta_features 
        
    def _get_weakest_base_learner(self):
        
        # loop rounds
        worst_model_idx = None 
        worst_performance = 1
        for idx in range(len(self._base_models[0])):
            current_round = self._base_models[1][idx]
            if current_round < self._n_rounds_eval_base_model:
                continue 
            
            current_performance = self._base_models[2][idx].sum()
            if current_performance < worst_performance:
                worst_performance = current_performance 
                worst_model_idx = idx

        return worst_model_idx
    
    def _train_on_mini_batch(self, X, y):
        
        # ----------------------------------------------------------------------------
        # STEP 1: split mini batch to base-level and meta-level training set
        # ----------------------------------------------------------------------------
        base_idx = int(self._window_size * (1.0 - self.meta_learner_train_ratio))
        X_base = X[0: base_idx, :]
        y_base = y[0: base_idx] 

        # this part will be used to train the meta-level model,
        # and to continue training the base-level models on the rest of this batch
        X_meta = X[base_idx:self._window_size, :]  
        y_meta = y[base_idx:self._window_size]
        
        # ----------------------------------------------------------------------------
        # STEP 2: train previous base-models 
        # ----------------------------------------------------------------------------
        meta_features = []
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len > 0: # check if we have any base-level models         
            base_model_performances = self._meta_learner.feature_importances_
            for b_idx in range(base_models_len): # loop and train and extract meta-level features 
                    
                # continuation of training (incremental) on base-level model,
                # using the base-level training set 
                base_model = self._base_models[0][b_idx]
                base_model.fit(X_base, y_base, xgb_model=base_model.get_booster())
                y_predicted = base_model.predict_proba(X_meta) # extract meta-level features 
                                
                # extract meta-features 
                meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
                
                # once the meta-features for this specific base-model are extracted,
                # we incrementally fit this base-model to the rest of the data,
                # this is done so this base-model is trained on a full batch 
                base_model.fit(X_meta, y_meta, xgb_model=base_model.get_booster())
                                
                # update base-level model list 
                self._base_models[0][b_idx] = base_model
                current_round = self._base_models[1][b_idx]
                last_performance = base_model_performances[b_idx * 2] + base_model_performances[(b_idx*2)+1] 
                self._base_models[2][b_idx][current_round%self._n_rounds_eval_base_model] = last_performance
                self._base_models[1][b_idx] = current_round + 1
                
        # ----------------------------------------------------------------------------
        # STEP 3: with each new batch, we create/train a new base model 
        # ----------------------------------------------------------------------------
        new_base_model, new_base_model_meta_features = self._train_new_base_model(X_base, y_base, X_meta, y_meta)

        insert_idx = base_models_len
        if base_models_len == 0:
            meta_features = new_base_model_meta_features
        elif base_models_len > 0 and base_models_len < self._n_base_models: 
            meta_features = np.hstack((meta_features, new_base_model_meta_features))     
        else: 
            insert_idx = self._get_weakest_base_learner()           
            meta_features[:, insert_idx * 2] = new_base_model_meta_features[:,0]
            meta_features[:, (insert_idx * 2) + 1] = new_base_model_meta_features[:,1]
            
        self._base_models[0][insert_idx] = new_base_model 
        self._base_models[1][insert_idx] = 0 
        self._base_models[2][insert_idx] = np.zeros(self._n_rounds_eval_base_model) 

        # STEP 4: train the meta-level model 
        meta_features = self._construct_meta_features(meta_features)
        
        if base_models_len == 0:
            self._meta_learner.fit(meta_features, y_meta)
        else:
            self._meta_learner.fit(meta_features, y_meta, xgb_model=self._meta_learner.get_booster())

        # STEP 5: train the LSTM model
        if self._lstm_model is None:
            self._lstm_model = self.lstm

        meta_features_reshaped = meta_features.reshape((-1, meta_features.shape[1]))
        # print(f'meta_features_reshaped shape (seq_length, input_dim): {meta_features_reshaped.shape}')

        # Define num_samples based on the number of rows in meta_features_reshaped
        num_samples = meta_features_reshaped.shape[0]

        # Initialize a list to store predictions from the final epoch
        final_epoch_preds = []
        accuracy_over_epochs = []
        losses = []  # List to store loss values

        # Count the total number of 1s and 0s in the actual labels
        total_ones_actual = np.sum(y)
        total_zeros_actual = len(y) - total_ones_actual

        # print(f"Total number of actual 1s in y: {total_ones_actual}")
        # print(f"Total number of actual 0s in y: {total_zeros_actual}")

        # Perform training loop
        for epoch in range(self.lstm_epochs):
            # Temporary list for the current epoch predictions
            current_epoch_preds = []
            total_loss = 0
            total_correct = 0
            total_samples = 0
            total_ones = 0
            total_zeros = 0

            # Iterate over mini-batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = meta_features_reshaped[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                batch_size_actual = y_batch.shape[0]  

                # Initialize hidden state and cell state
                h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
                c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

                # Forward pass
                y_preds, h_next, c_next, i_list, c_bar_list, f_list, o_list = self._lstm_model.forward(x_batch, h_prev, c_prev)

                # Print the statistics of y_preds
                # print(f"y_preds average: {np.mean(y_preds)}")
                # print(f"y_preds minimum: {np.min(y_preds)}")
                # print(f"y_preds maximum: {np.max(y_preds)}")

                # Store predictions and loss from the current batch
                current_epoch_preds.append(y_preds)
                loss = modified_binary_cross_entropy(y_batch, y_preds.flatten(), target_value=0.45, scale_factor=1.0)
                total_loss += loss * batch_size_actual  # Weighting the loss by the batch size

                # Calculate and accumulate accuracy
                batch_predictions = (y_preds.flatten() > 0.5).astype(int)
                total_correct += np.sum(batch_predictions == y_batch)
                total_samples += batch_size_actual

                # Count the occurrences of 1s and 0s in the current batch
                total_ones += np.sum(batch_predictions)
                total_zeros += batch_size_actual - np.sum(batch_predictions)

                # print(f"Total number of 1s in y_preds: {total_ones}")
                # print(f"Total number of 0s in y_preds: {total_zeros}")

                # Backward pass
                dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby = self._lstm_model.backward(x_batch, y_batch, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list)

                # Clipping gradients
                grad_norm = np.sqrt(sum(np.sum(grad**2) for grad in [dWf, dWi, dWc, dWo, dWy]))
                if grad_norm > self.lstm_grad_clip_threshold:
                    clip_coef = self.lstm_grad_clip_threshold / (grad_norm + 1e-6)  # Avoid division by zero
                    dWf, dWi, dWc, dWo, dWy = [clip_coef * grad for grad in [dWf, dWi, dWc, dWo, dWy]]

                # Update weights and biases
                self._lstm_model.update_weights(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
                self._lstm_model.bf -= self.learning_rate * dbf
                self._lstm_model.bi -= self.learning_rate * dbi
                self._lstm_model.bc -= self.learning_rate * dbc
                self._lstm_model.bo -= self.learning_rate * dbo
                self._lstm_model.by -= self.learning_rate * dby

                # Reset gradients for the next batch
                self._lstm_model.reset_gradients()

                # After processing all batches in the current epoch
                if epoch == self.lstm_epochs - 1:  # Check if it's the final epoch
                    final_epoch_preds = current_epoch_preds  # Only store the final epoch's predictions
            
            # Compute average loss and accuracy for the epoch
            average_epoch_loss = total_loss / total_samples
            epoch_accuracy = total_correct / total_samples
            losses.append(average_epoch_loss)
            accuracy_over_epochs.append(epoch_accuracy)

            # print(f"Epoch {epoch+1}/{self.lstm_epochs} completed.")

        # Save model weights
        self._lstm_model.save_weights('lstm_ASGXB_weights.npz')

        # # Concatenate all predictions from the final epoch
        # final_y_preds = np.concatenate([pred for pred in final_epoch_preds], axis=0)
        # print(f"Final predictions shape: {final_y_preds.shape}")
        # print(f'accuracy_over_epochs: {accuracy_over_epochs}')
        # print(f'losses: {losses}')

    def predict(self, X):
        self._lstm_model.load_weights('lstm_ASGXB_weights.npz')

        # only one model in ensemble use its predictions 
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len < self._n_base_models:
            predictions = []
            for i in range(base_models_len):
                tmp_predictions = self._base_models[0][i].predict(X)
                predictions.append(tmp_predictions)
            output = np.array([int(Counter(col).most_common(1)[0][0]) for col in zip(*predictions)])
            return output
        
        # predict via meta learner 
        meta_features = []           
        for b_idx in range(base_models_len):
            y_predicted = self._base_models[0][b_idx].predict_proba(X) 
            meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
        meta_features = self._construct_meta_features(meta_features)

        # Reconstruct meta features to match the expected input size for LSTM
        meta_features = self._construct_meta_features(meta_features)

        # Initialize hidden state and cell state for LSTM prediction
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        
        # Process the input sequence X
        num_samples = meta_features.shape[0]
        y_preds = np.zeros((num_samples, self._lstm_model.output_size))
        
        for t in range(num_samples):
            xt = meta_features[t].reshape(self._lstm_model.input_size, 1)
            
            # Reshape and repeat operations
            h_prev_reshaped = h_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            c_prev_reshaped = c_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            xt_repeated = np.repeat(xt, self._lstm_model.num_layers, axis=1)
            concat = np.vstack((h_prev_reshaped, xt_repeated))
            
            # Gate activations using loaded weights
            f_t = sigmoid(np.dot(self._lstm_model.Wf, concat) + self._lstm_model.bf)
            i_t = sigmoid(np.dot(self._lstm_model.Wi, concat) + self._lstm_model.bi)
            c_bar_t = np.tanh(np.dot(self._lstm_model.Wc, concat) + self._lstm_model.bc)
            o_t = sigmoid(np.dot(self._lstm_model.Wo, concat) + self._lstm_model.bo)
            
            # Cell state and hidden state using loaded weights
            c_next_t = f_t * c_prev_reshaped + i_t * c_bar_t
            h_next_t = o_t * np.tanh(c_next_t)
            
            # Output prediction using loaded weights
            yt = sigmoid(np.dot(self._lstm_model.Wy, h_next_t[:, -1].reshape(self._lstm_model.hidden_size, 1)) + self._lstm_model.by)
            y_preds[t] = yt.squeeze()
            
            # Update previous states
            h_prev = h_next_t
            c_prev = c_next_t
        
        # Generate final predictions
        # Convert LSTM outputs to binary predictions (0 or 1)
        final_predictions = (y_preds.flatten() > self.threshold).astype(int)
        return final_predictions

    def eval_proba(self, X):
        self._lstm_model.load_weights('lstm_ASGXB_weights.npz')

        # only one model in ensemble use its predictions 
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len == 0:
            raise Exception("No base models have been trained.")

        meta_features = []
        for b_idx in range(base_models_len):
            y_predicted = self._base_models[0][b_idx].predict_proba(X)
            meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))
        
        meta_features = self._construct_meta_features(meta_features)

        # Reconstruct meta features to match the expected input size for LSTM
        meta_features = self._construct_meta_features(meta_features)

        # Initialize hidden state and cell state for LSTM
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        
        # Process the input sequence X
        num_samples = meta_features.shape[0]
        y_preds = np.zeros((num_samples, self._lstm_model.output_size))
        
        for t in range(num_samples):
            xt = meta_features[t].reshape(self._lstm_model.input_size, 1)
            
            # Reshape and repeat operations
            h_prev_reshaped = h_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            c_prev_reshaped = c_prev.reshape(self._lstm_model.hidden_size, self._lstm_model.num_layers)
            xt_repeated = np.repeat(xt, self._lstm_model.num_layers, axis=1)
            concat = np.vstack((h_prev_reshaped, xt_repeated))
            
            # Gate activations using loaded weights
            f_t = sigmoid(np.dot(self._lstm_model.Wf, concat) + self._lstm_model.bf)
            i_t = sigmoid(np.dot(self._lstm_model.Wi, concat) + self._lstm_model.bi)
            c_bar_t = np.tanh(np.dot(self._lstm_model.Wc, concat) + self._lstm_model.bc)
            o_t = sigmoid(np.dot(self._lstm_model.Wo, concat) + self._lstm_model.bo)
            
            # Cell state and hidden state using loaded weights
            c_next_t = f_t * c_prev_reshaped + i_t * c_bar_t
            h_next_t = o_t * np.tanh(c_next_t)
            
            # Output prediction using loaded weights
            yt = sigmoid(np.dot(self._lstm_model.Wy, h_next_t[:, -1].reshape(self._lstm_model.hidden_size, 1)) + self._lstm_model.by)
            y_preds[t] = yt.squeeze()
            
            # Update previous states
            h_prev = h_next_t
            c_prev = c_next_t
        
        # Return the sigmoid outputs as probabilities
        probabilities = sigmoid(y_preds.flatten())
        return probabilities

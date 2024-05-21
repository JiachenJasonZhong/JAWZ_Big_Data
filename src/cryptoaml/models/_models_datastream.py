
# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import xgboost as xgb
from collections import Counter
from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import get_dimensions
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score


###### AdaptiveXGBoost ###################################################

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
    def __init__(self, input_size, hidden_size, output_size, lstm_dropout, learning_rate, weight_decay):

        # Initialize key variables 
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = hidden_size
        self.lstm_dropout = lstm_dropout 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize weights and biases
        # print(f'hidden_size type = {hidden_size}')
        # print(f'input_size type = {input_size}')
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
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

        # Initialize hidden state and cell state matrices
        self.h_next = np.zeros((hidden_size, self.num_layers))
        self.c_next = np.zeros((hidden_size, self.num_layers))

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
        # print(f'Shape of x: {x.shape}, expected: ({num_samples}, {self.input_size})')
        # print(f'Shape of h_prev: {h_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_prev: {c_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')

        dropout_masks = []
        y_preds = []
        i_list = []
        c_bar_list = []
        f_list = []
        o_list = []

        for t in range(x.shape[0]):  # Iterate over each sample
            xt = x[t].reshape(self.input_size, 1)  # Shape (input_size, 1)
            h_next_t = np.zeros((self.hidden_size, self.num_layers))
            c_next_t = np.zeros((self.hidden_size, self.num_layers))
            h_next_t, mask = dropout(h_next_t, self.lstm_dropout)
            dropout_masks.append(mask)
            i_t = np.zeros((self.hidden_size, self.num_layers))
            c_bar_t = np.zeros((self.hidden_size, self.num_layers))
            f_t = np.zeros((self.hidden_size, self.num_layers))
            o_t = np.zeros((self.hidden_size, self.num_layers))

            for l in range(self.num_layers):
                # print(f'Layer {l+1}:')
                h_prev_l = h_prev[:, l].reshape(self.hidden_size, 1)
                c_prev_l = c_prev[:, l].reshape(self.hidden_size, 1)
                # print(f'  Shape of h_prev_l: {h_prev_l.shape}, expected: ({self.hidden_size}, 1)')
                # print(f'  Shape of c_prev_l: {c_prev_l.shape}, expected: ({self.hidden_size}, 1)')

                concat = np.vstack((h_prev_l, xt))  # Shape (hidden_size + input_size, 1)
                # print(f'  Shape of concat: {concat.shape}, expected: ({self.hidden_size + self.input_size}, 1)')

                f_t[:, l] = sigmoid(np.dot(self.Wf, concat) + self.bf)[:, 0]
                # print(f'  Shape of f_t[:, {l}]: {f_t[:, l].shape}, expected: ({self.hidden_size},)')

                i_t[:, l] = sigmoid(np.dot(self.Wi, concat) + self.bi)[:, 0]
                # print(f'  Shape of i_t[:, {l}]: {i_t[:, l].shape}, expected: ({self.hidden_size},)')

                c_bar_t[:, l] = np.tanh(np.dot(self.Wc, concat) + self.bc)[:, 0]
                # print(f'  Shape of c_bar_t[:, {l}]: {c_bar_t[:, l].shape}, expected: ({self.hidden_size},)')

                c_next_t[:, l] = f_t[:, l] * c_prev_l[:, 0] + i_t[:, l] * c_bar_t[:, l]
                # print(f'  Shape of c_next_t[:, {l}]: {c_next_t[:, l].shape}, expected: ({self.hidden_size},)')

                o_t[:, l] = sigmoid(np.dot(self.Wo, concat) + self.bo)[:, 0]
                # print(f'  Shape of o_t[:, {l}]: {o_t[:, l].shape}, expected: ({self.hidden_size},)')

                h_next_t[:, l] = o_t[:, l] * np.tanh(c_next_t[:, l])
                # print(f'  Shape of h_next_t[:, {l}]: {h_next_t[:, l].shape}, expected: ({self.hidden_size},)')

            yt = np.dot(self.Wy, h_next_t[:, -1].reshape(self.hidden_size, 1)) + self.by
            y_preds.append(yt)
            i_list.append(i_t)
            c_bar_list.append(c_bar_t)
            f_list.append(f_t)
            o_list.append(o_t)

            h_prev = h_next_t
            c_prev = c_next_t

        y_preds = np.array(y_preds).reshape(-1, self.output_size)

        # print(f'Shape of h_next: {h_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_next: {c_next_t.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of y_preds: {y_preds.shape}, expected: ({num_samples}, {self.output_size})')

        return y_preds, h_next_t, c_next_t, i_list, c_bar_list, f_list, o_list

    def backward(self, x, y, y_preds, h_prev, c_prev, i_list, c_bar_list, f_list, o_list):
        # print(f'Shape of x: {x.shape}, expected: ({num_samples}, {self.input_size})')
        # print(f'Shape of y: {y.shape}, expected: ({num_samples},)')
        # print(f'Shape of y_preds: {y_preds.shape}, expected: ({num_samples}, {self.output_size})')
        # print(f'Shape of h_prev: {h_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of c_prev: {c_prev.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of dh_next: {dh_next.shape}, expected: ({self.hidden_size}, {self.num_layers})')
        # print(f'Shape of dc_next: {dc_next.shape}, expected: ({self.hidden_size}, {self.num_layers})')

        for t in reversed(range(x.shape[0])):  # Iterate over each sample in reverse order
            dE_dy = y_preds[t] - y[t] # the gradient of the binary cross-entropy loss 
            self.dWy += np.dot(dE_dy.reshape(self.output_size, 1), h_prev[:, -1].reshape(1, self.hidden_size))
            self.dby += dE_dy.reshape(self.output_size, 1)

            dh_next_t = np.zeros((self.hidden_size, self.num_layers))
            dc_next_t = np.zeros((self.hidden_size, self.num_layers))

            for l in reversed(range(self.num_layers)):
                dh = np.dot(self.Wy.T, dE_dy.reshape(self.output_size, 1)) + dh_next_t[:, l].reshape(self.hidden_size, 1)
                dc = dc_next_t[:, l].reshape(self.hidden_size, 1) + dh * o_list[t][:, l].reshape(self.hidden_size, 1) * (1 - np.square(np.tanh(c_prev[:, l].reshape(self.hidden_size, 1))))

                do = dh * np.tanh(c_prev[:, l].reshape(self.hidden_size, 1))
                dc_bar = dh * i_list[t][:, l].reshape(self.hidden_size, 1)
                di = dh * c_bar_list[t][:, l].reshape(self.hidden_size, 1)
                df = dh * c_prev[:, l].reshape(self.hidden_size, 1)

                xt = x[t].reshape(self.input_size, 1)
                concat = np.vstack((h_prev[:, l].reshape(self.hidden_size, 1), xt))

                self.dWf += np.dot(df * sigmoid_derivative(f_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)
                self.dWi += np.dot(di * sigmoid_derivative(i_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)
                self.dWc += np.dot(dc_bar * (1 - np.square(c_bar_list[t][:, l].reshape(self.hidden_size, 1))), concat.T)
                self.dWo += np.dot(do * sigmoid_derivative(o_list[t][:, l].reshape(self.hidden_size, 1)), concat.T)

                self.dbf += df * sigmoid_derivative(f_list[t][:, l].reshape(self.hidden_size, 1))
                self.dbi += di * sigmoid_derivative(i_list[t][:, l].reshape(self.hidden_size, 1))
                self.dbc += dc_bar * (1 - np.square(c_bar_list[t][:, l].reshape(self.hidden_size, 1)))
                self.dbo += do * sigmoid_derivative(o_list[t][:, l].reshape(self.hidden_size, 1))

                dh_next_t[:, l] = dh[:, 0]
                dc_next_t[:, l] = dc[:, 0]

        return self.dWf, self.dWi, self.dWc, self.dWo, self.dWy, self.dbf, self.dbi, self.dbc, self.dbo, self.dby

    def update_weights(self, learning_rate, weight_decay):
        # Updates weights using gradients with L2 regularization.
        self.Wf -= learning_rate * (self.dWf + weight_decay * self.Wf)
        self.Wi -= learning_rate * (self.dWi + weight_decay * self.Wi)
        self.Wc -= learning_rate * (self.dWc + weight_decay * self.Wc)
        self.Wo -= learning_rate * (self.dWo + weight_decay * self.Wo)
        self.Wy -= learning_rate * (self.dWy + weight_decay * self.Wy)

# Necessary functions for LSTM
lstm_units = 64
lstm_epochs = 5 
lstm_dropout = 0.2 
lstm_grad_clip_threshold = 1.0
learning_rate = 0.0001
weight_decay = 0.001

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

class LSTM_Base():
    def __init__(self,
                 lstm_units=lstm_units,
                 lstm_epochs=lstm_epochs, 
                 lstm_dropout=lstm_dropout, 
                 lstm_grad_clip_threshold=lstm_grad_clip_threshold,
                 learning_rate = learning_rate,
                 weight_decay = weight_decay,
                 output_size = 1):
        
        self.lstm_epochs = lstm_epochs
        self.lstm_dropout = lstm_dropout 
        self.lstm_grad_clip_threshold = lstm_grad_clip_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dim = lstm_units
        self._lstm_model = None

    def train_lstm(self, x, y):
                   
        # Train the LSTM model
        if self._lstm_model is None:
            self._lstm_model = LSTM(input_size=x.shape[1], hidden_size=self.hidden_dim, output_size=1, 
                        lstm_dropout=self.lstm_dropout, learning_rate=self.learning_rate, weight_decay=self.weight_decay)

        # Define num_samples based on X
        num_samples = x.shape[0]

        # Define the number of epochs and batch size
        batch_size = 100

        # Initialize a list to store predictions from the final epoch
        final_epoch_preds = []
        accuracy_over_epochs = []
        losses = []  # List to store loss values

        # Perform training loop
        for epoch in range(self.lstm_epochs):
            print(f"Epoch {epoch+1}/{self.lstm_epochs} initiated.")

            # Temporary list for the current epoch predictions
            current_epoch_preds = []
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            # Iterate over mini-batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                batch_size_actual = y_batch.shape[0]  

                # Initialize hidden state and cell state
                h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
                c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

                # Forward pass
                y_preds, h_next, c_next, i_list, c_bar_list, f_list, o_list = self._lstm_model.forward(x_batch, h_prev, c_prev)

                # Store predictions and loss from the current batch
                current_epoch_preds.append(y_preds)
                loss = binary_cross_entropy(y_batch, y_preds.flatten())
                total_loss += loss * batch_size_actual  # Weighting the loss by the batch size

                # Calculate and accumulate accuracy
                batch_predictions = (y_preds.flatten() > 0.5).astype(int)
                total_correct += np.sum(batch_predictions == y_batch)
                total_samples += batch_size_actual

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

            print(f"Epoch {epoch+1}/{self.lstm_epochs} completed.")

    def predict(self, X):

        # Initialize hidden state and cell state for LSTM prediction
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

        # Forward pass through LSTM model
        # Assume that we are processing the entire dataset X as one batch
        y_preds, _, _, _, _, _, _ = self._lstm_model.forward(X, h_prev, c_prev)
        
        # Generate final predictions
        # Convert LSTM outputs to binary predictions (0 or 1)
        final_predictions = (y_preds.flatten() > 0.5).astype(int)
        return final_predictions

    
    def eval_proba(self, X):

        # Initialize hidden state and cell state for LSTM
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

        # Forward pass through LSTM model assuming processing the entire dataset X as one batch
        y_preds, _, _, _, _, _, _ = self._lstm_model.forward(X, h_prev, c_prev)
        
        # Instead of converting to binary predictions, return the sigmoid outputs
        # as the probabilities. Adjust depending on your LSTM's output layer configuration.
        probabilities = sigmoid(y_preds.flatten())
        return probabilities

class LSTM_AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4,
                 lstm_units=lstm_units,
                 lstm_epochs=lstm_epochs, 
                 lstm_dropout=lstm_dropout, 
                 lstm_grad_clip_threshold=lstm_grad_clip_threshold,
                 learning_rate=learning_rate,
                 weight_decay=weight_decay,
                 output_size = 1):
        
        self.lstm = LSTM(input_size = n_base_models * 2, hidden_size = lstm_units, output_size = output_size, 
                         lstm_dropout = lstm_dropout, learning_rate = learning_rate, weight_decay = weight_decay)
        
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

        # STEP 5: train the LSTM model
        if self._lstm_model is None:
            self._lstm_model = self.lstm

        meta_features_reshaped = meta_features.reshape((-1, meta_features.shape[1]))
        # print(f'meta_features_reshaped shape (seq_length, input_dim): {meta_features_reshaped.shape}')

        # Define num_samples based on the number of rows in meta_features_reshaped
        num_samples = meta_features_reshaped.shape[0]

        # Define the number of epochs and batch size
        batch_size = 100

        # Initialize a list to store predictions from the final epoch
        final_epoch_preds = []
        accuracy_over_epochs = []
        losses = []  # List to store loss values

        # Perform training loop
        for epoch in range(self.lstm_epochs):
            # Temporary list for the current epoch predictions
            current_epoch_preds = []
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            x_shuffled = meta_features_reshaped[indices]
            y_shuffled = y[indices]

            # Iterate over mini-batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                batch_size_actual = y_batch.shape[0]  

                # Initialize hidden state and cell state
                h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
                c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

                # Forward pass
                y_preds, h_next, c_next, i_list, c_bar_list, f_list, o_list = self._lstm_model.forward(x_batch, h_prev, c_prev)

                # Store predictions and loss from the current batch
                current_epoch_preds.append(y_preds)
                loss = binary_cross_entropy(y_batch, y_preds.flatten())
                total_loss += loss * batch_size_actual  # Weighting the loss by the batch size

                # Calculate and accumulate accuracy
                batch_predictions = (y_preds.flatten() > 0.5).astype(int)
                total_correct += np.sum(batch_predictions == y_batch)
                total_samples += batch_size_actual

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

            print(f"Epoch {epoch+1}/{self.lstm_epochs} completed.")

        # # Concatenate all predictions from the final epoch
        # final_y_preds = np.concatenate([pred for pred in final_epoch_preds], axis=0)
        # print(f"Final predictions shape: {final_y_preds.shape}")
        # print(f'accuracy_over_epochs: {accuracy_over_epochs}')
        # print(f'losses: {losses}')

    def predict(self, X):

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

        # Forward pass through LSTM model
        # Assume that we are processing the entire dataset X as one batch
        y_preds, _, _, _, _, _, _ = self._lstm_model.forward(meta_features, h_prev, c_prev)
        
        # Generate final predictions
        # Convert LSTM outputs to binary predictions (0 or 1)
        final_predictions = (y_preds.flatten() > 0.5).astype(int)
        return final_predictions

    
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

        # Reconstruct meta features to match the expected input size for LSTM
        meta_features = self._construct_meta_features(meta_features)

        # Initialize hidden state and cell state for LSTM
        h_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))
        c_prev = np.zeros((self.hidden_dim, self._lstm_model.num_layers))

        # Forward pass through LSTM model assuming processing the entire dataset X as one batch
        y_preds, _, _, _, _, _, _ = self._lstm_model.forward(meta_features, h_prev, c_prev)
        
        # Instead of converting to binary predictions, return the sigmoid outputs
        # as the probabilities. Adjust depending on your LSTM's output layer configuration.
        probabilities = sigmoid(y_preds.flatten())
        return probabilities

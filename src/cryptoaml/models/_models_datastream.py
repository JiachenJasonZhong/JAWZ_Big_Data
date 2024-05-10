
# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import xgboost as xgb
from collections import Counter
from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import get_dimensions


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

class SimpleLSTM:
    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the SimpleLSTM class.

        Args:
            input_dim (int): The dimensionality of the input vectors.
            hidden_dim (int): The dimensionality of the hidden state vectors.
        """
        self.input_dim = input_dim
        print(f"input_dim has value: {self.input_dim} and theoretically should be an integer representing the dimensionality of the input vectors")

        self.hidden_dim = hidden_dim
        print(f"hidden_dim has value: {self.hidden_dim} and theoretically should be an integer representing the dimensionality of the hidden state vectors")

        # Initialize the input-to-hidden weights matrix with random values scaled by 0.1
        # The matrix has shape (4 * hidden_dim, input_dim + hidden_dim) to accommodate all gate weights
        self.weights_ih = np.random.rand(4 * hidden_dim, input_dim + hidden_dim) * 0.1
        print(f"weights_ih has shape: {self.weights_ih.shape} and theoretically should be ({4*hidden_dim, input_dim+hidden_dim})")

        # Initialize the hidden-to-output weights matrix with random values scaled by 0.1
        self.weights_ho = np.random.rand(hidden_dim) * 0.1
        print(f"weights_ho has shape: {self.weights_ho.shape} and theoretically should be ({hidden_dim})")

        # Initialize the bias vector with zeros
        self.bias = np.zeros(4 * hidden_dim)
        print(f"bias has shape: {self.bias.shape} and theoretically should be ({4*hidden_dim})")

    def step(self, x, h, c):
        """
        Performs a single step of the LSTM computation.

        Args:
            x (np.ndarray): The input vector at the current time step.
            h (np.ndarray): The hidden state vector from the previous time step.
            c (np.ndarray): The cell state vector from the previous time step.

        Returns:
            tuple: A tuple containing the updated hidden state vector and cell state vector.
        """
        # print(f"x has shape: {x.shape} and theoretically should be (input_dim,)")
        # print(f"h has shape: {h.shape} and theoretically should be (hidden_dim,)")
        # print(f"c has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Concatenate the hidden state and input vectors
        combined = np.hstack((h, x))
        # print(f"combined has shape: {combined.shape} and theoretically should be (hidden_dim+input_dim,)")

        # Compute the gate values by multiplying the weights with the combined vector and adding the bias
        gates = np.dot(self.weights_ih, combined) + self.bias
        # print(f"gates has shape: {gates.shape} and theoretically should be (4*hidden_dim,)")

        # Split the gate values into four parts: input gate, forget gate, output gate, and candidate cell state
        i, f, o, g = np.split(gates, 4)
        # print(f"i has shape: {i.shape} and theoretically should be (hidden_dim,)")
        # print(f"f has shape: {f.shape} and theoretically should be (hidden_dim,)")
        # print(f"o has shape: {o.shape} and theoretically should be (hidden_dim,)")
        # print(f"g has shape: {g.shape} and theoretically should be (hidden_dim,)")

        # Apply the sigmoid activation function to the input gate, forget gate, and output gate
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)

        # Apply the tanh activation function to the candidate cell state
        g = np.tanh(g)

        # Update the cell state by element-wise multiplying the forget gate with the previous cell state
        # and adding the element-wise multiplication of the input gate and candidate cell state
        c = f * c + i * g
        # print(f"c has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Update the hidden state by element-wise multiplying the output gate with the tanh of the updated cell state
        h = o * np.tanh(c)
        # print(f"h has shape: {h.shape} and theoretically should be (hidden_dim,)")

        return h, c

    def sigmoid(self, x):
        """
        Computes the sigmoid activation function.

        Args:
            x (np.ndarray): The input values.

        Returns:
            np.ndarray: The output values after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        Performs the forward pass of the LSTM on a sequence of inputs.

        Args:
            inputs (np.ndarray): The input sequence with shape (seq_length, input_dim) or (input_dim,).

        Returns:
            tuple: A tuple containing the output sequence, final hidden state, and final cell state.
        """
        print(f"inputs has shape: {inputs.shape} and theoretically should be (seq_length, input_dim) or (input_dim,)")

        # If the input is a 1D array, reshape it to a 2D array with a single sequence
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            print(f"inputs has been reshaped to: {inputs.shape}")

        # Initialize the hidden state and cell state vectors with zeros
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        # print(f"h has shape: {h.shape} and theoretically should be (hidden_dim,)")
        # print(f"c has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Initialize an empty list to store the output sequence
        outputs = []

        # Iterate over each input vector in the sequence
        for x in inputs:
            # Perform a single step of the LSTM computation
            h, c = self.step(x, h, c)
            # Append the updated hidden state to the output sequence
            outputs.append(h)

        outputs = np.array(outputs)
        print(f"outputs has shape: {outputs.shape} and theoretically should be (seq_length, hidden_dim)")

        # Convert the output sequence to a numpy array and return it along with the final hidden and cell states
        return outputs, h, c
    
class Simple_LSTM_AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4,
                 lstm_units=64,
                 lstm_dropout=0.2,
                 lstm_epochs=10):
        
        self.lstm = SimpleLSTM(input_dim=n_base_models * 2, hidden_dim=lstm_units)
        self.lstm_units = lstm_units

        print(f'self.lstm.input_dim: {self.lstm.input_dim}, which should be n_base_models * 2 = {n_base_models * 2}')
        print(f'self.lstm.hidden_dim: {self.lstm.hidden_dim}, which should be lstm_units = {lstm_units}')

        self.lstm_dropout = lstm_dropout
        self.lstm_epochs = lstm_epochs
        self._lstm_model = None
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
        print(f'meta_features_reshaped shape (seq_length, input_dim): {meta_features_reshaped.shape}')
        self._lstm_model.forward(meta_features_reshaped)

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

        # predict using the LSTM model
        lstm_outputs, _, _ = self._lstm_model.forward(meta_features.reshape((-1, meta_features.shape[1])))
        lstm_predictions = lstm_outputs[:, -1] 

        if np.isscalar(lstm_predictions):
            print('np value is a scalar value!')
            print(lstm_predictions)
            return np.full(X.shape[0], (lstm_predictions > 0.5).astype(int))
        else:
            return (lstm_predictions > 0.5).astype(int)
    
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
        # predict probabilities using the LSTM model
        lstm_outputs, _, _ = self._lstm_model.forward(meta_features.reshape((-1, meta_features.shape[1])))
        lstm_predictions = np.dot(lstm_outputs, self._lstm_model.weights_ho)
        probabilities = 1 / (1 + np.exp(-lstm_predictions))
        return probabilities

class LSTM:
    def __init__(self, input_dim, hidden_dim, dropout_prob, grad_clip_threshold):
        """
        Initializes the LSTM class.

        Args:
            input_dim (int): The dimensionality of the input vectors.
            hidden_dim (int): The dimensionality of the hidden state vectors.
            dropout_prob (float): The probability of dropping out units during training. Default is 0.2.
            grad_clip_threshold (float): The threshold for gradient clipping. Default is 1.0.
        """
        self.input_dim = input_dim
        print(f"input_dim has value: {self.input_dim} and theoretically should be an integer representing the dimensionality of the input vectors")

        self.hidden_dim = hidden_dim
        print(f"hidden_dim has value: {self.hidden_dim} and theoretically should be an integer representing the dimensionality of the hidden state vectors")

        # Initialize the input-to-hidden weights matrix with random values scaled by 0.1
        # The matrix has shape (4 * hidden_dim, input_dim + hidden_dim) to accommodate all gate weights
        self.weights_ih = np.random.rand(4 * hidden_dim, input_dim + hidden_dim) * 0.1
        # print(f"weights_ih has shape: {self.weights_ih.shape} and theoretically should be ({4*hidden_dim, input_dim+hidden_dim})")

        # Initialize the hidden-to-output weights matrix with random values scaled by 0.1
        self.weights_ho = np.random.rand(hidden_dim) * 0.1
        # print(f"weights_ho has shape: {self.weights_ho.shape} and theoretically should be ({hidden_dim})")

        # Initialize the bias vector with zeros
        self.bias = np.zeros(4 * hidden_dim)
        # print(f"bias has shape: {self.bias.shape} and theoretically should be ({4*hidden_dim})")

        # Store the dropout probability and gradient clipping threshold
        self.dropout_prob = dropout_prob
        self.grad_clip_threshold = grad_clip_threshold

        # Initialize the training flag to True
        self.training = True

    def step(self, x, h, c):
        """
        Performs a single step of the LSTM computation.

        Args:
            x (np.ndarray): The input vector at the current time step.
            h (np.ndarray): The hidden state vector from the previous time step.
            c (np.ndarray): The cell state vector from the previous time step.

        Returns:
            tuple: A tuple containing the updated hidden state vector and cell state vector.
        """
        # print(f"x has shape: {x.shape} and theoretically should be (input_dim,)")
        # print(f"h has shape: {h.shape} and theoretically should be (hidden_dim,)")
        # print(f"c has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Concatenate the hidden state and input vectors
        combined = np.hstack((h, x))
        # print(f"combined has shape: {combined.shape} and theoretically should be (hidden_dim+input_dim,)")

        # Compute the gate values by multiplying the weights with the combined vector and adding the bias
        gates = np.dot(self.weights_ih, combined) + self.bias
        # print(f"gates has shape: {gates.shape} and theoretically should be (4*hidden_dim,)")

        # Split the gate values into four parts: input gate, forget gate, output gate, and candidate cell state
        i, f, o, g = np.split(gates, 4)
        # print(f"i has shape: {i.shape} and theoretically should be (hidden_dim,)")
        # print(f"f has shape: {f.shape} and theoretically should be (hidden_dim,)")
        # print(f"o has shape: {o.shape} and theoretically should be (hidden_dim,)")
        # print(f"g has shape: {g.shape} and theoretically should be (hidden_dim,)")

        # Apply the sigmoid activation function to the input gate, forget gate, and output gate
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)

        # Apply the tanh activation function to the candidate cell state
        g = np.tanh(g)

        # Update the cell state by element-wise multiplying the forget gate with the previous cell state
        # and adding the element-wise multiplication of the input gate and candidate cell state
        c = f * c + i * g
        # print(f"c has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Update the hidden state by element-wise multiplying the output gate with the tanh of the updated cell state
        h = o * np.tanh(c)
        # print(f"h has shape: {h.shape} and theoretically should be (hidden_dim,)")

        # Apply dropout to the hidden state during training
        if self.training:
            # Create a dropout mask with the same shape as the hidden state
            mask = np.random.rand(*h.shape) < (1 - self.dropout_prob)
            # print(f"dropout mask has shape: {mask.shape} and theoretically should be the same as h")

            # Apply the dropout mask to the hidden state and scale by the inverse of the keep probability
            h = h * mask / (1 - self.dropout_prob)
            # print(f"hidden state after dropout has shape: {h.shape} and theoretically should be the same as before")

        return h, c

    def sigmoid(self, x):
        """
        Computes the sigmoid activation function.

        Args:
            x (np.ndarray): The input values.

        Returns:
            np.ndarray: The output values after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        """
        Performs the forward pass of the LSTM on a sequence of inputs.

        Args:
            inputs (np.ndarray): The input sequence with shape (seq_length, input_dim) or (input_dim,).

        Returns:
            tuple: A tuple containing the output sequence, final hidden state, and final cell state.
        """
        print(f"inputs has shape: {inputs.shape} and theoretically should be (seq_length, input_dim) or (input_dim,)")

        # If the input is a 1D array, reshape it to a 2D array with a single sequence
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            print(f"inputs has been reshaped to: {inputs.shape}")

        # Initialize the hidden state and cell state vectors with zeros
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        # print(f"initial hidden state has shape: {h.shape} and theoretically should be (hidden_dim,)")
        # print(f"initial cell state has shape: {c.shape} and theoretically should be (hidden_dim,)")

        # Initialize an empty list to store the output sequence
        outputs = []

        # Iterate over each input vector in the sequence
        for x in inputs:
            # Perform a single step of the LSTM computation
            h, c = self.step(x, h, c)
            # print(f"updated hidden state has shape: {h.shape} and theoretically should be (hidden_dim,)")
            # print(f"updated cell state has shape: {c.shape} and theoretically should be (hidden_dim,)")

            # Append the updated hidden state to the output sequence
            outputs.append(h)

        outputs = np.array(outputs)
        # print(f"outputs has shape: {outputs.shape} and theoretically should be (seq_length, hidden_dim)")

        # Convert the output sequence to a numpy array and return it along with the final hidden and cell states
        return outputs, h, c
    
    def train(self):
        """
        Sets the model to training mode.
        """
        self.training = True

    def eval(self):
        """
        Sets the model to evaluation mode.
        """
        self.training = False

    def clip_gradients(self):
        """
        Clips the gradients of the model parameters to the specified threshold.
        """
        self.weights_ih = np.clip(self.weights_ih, -self.grad_clip_threshold, self.grad_clip_threshold)
        # print(f"clipped weights_ih has shape: {self.weights_ih.shape} and theoretically should be (4*hidden_dim, input_dim+hidden_dim)")

        self.weights_ho = np.clip(self.weights_ho, -self.grad_clip_threshold, self.grad_clip_threshold)
        # print(f"clipped weights_ho has shape: {self.weights_ho.shape} and theoretically should be (4*hidden_dim)")

        self.bias = np.clip(self.bias, -self.grad_clip_threshold, self.grad_clip_threshold)
        # print(f"clipped bias has shape: {self.bias.shape} and theoretically should be (4*hidden_dim)")  

class MultiLayerLSTM:
    def __init__(self, input_dim, hidden_dims, dropout_prob, grad_clip_threshold):
        """
        Initializes the MultiLayerLSTM class.

        Args:
            input_dim (int): The dimensionality of the input vectors.
            hidden_dims (list): A list of integers representing the hidden dimensionalities of each LSTM layer.
            dropout_prob (float): The probability of dropping out units during training. Default is 0.2.
            grad_clip_threshold (float): The threshold for gradient clipping. Default is 1.0.
        """
        self.num_layers = len(hidden_dims)
        self.lstm_layers = []

        # Create multiple LSTM layers
        for i in range(self.num_layers):
            if i == 0:
                # First LSTM layer takes input_dim as input dimensionality
                lstm_layer = LSTM(input_dim, hidden_dims[i], dropout_prob, grad_clip_threshold)
            else:
                # Subsequent LSTM layers take the hidden dimensionality of the previous layer as input dimensionality
                lstm_layer = LSTM(hidden_dims[i-1], hidden_dims[i], dropout_prob, grad_clip_threshold)
            self.lstm_layers.append(lstm_layer)

    def forward(self, inputs):
        """
        Performs the forward pass of the multi-layer LSTM network.

        Args:
            inputs (np.ndarray): The input sequence with shape (seq_length, input_dim) or (input_dim,).

        Returns:
            tuple: A tuple containing the output sequence, final hidden states, and final cell states of all layers.
        """
        hidden_states = []
        cell_states = []

        # Iterate over LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                # First LSTM layer takes the original inputs
                outputs, h, c = lstm_layer.forward(inputs)
            else:
                # Subsequent LSTM layers take the outputs from the previous layer as inputs
                outputs, h, c = lstm_layer.forward(outputs)
            hidden_states.append(h)
            cell_states.append(c)

        return outputs, hidden_states, cell_states

    def train(self):
        """
        Sets all LSTM layers to training mode.
        """
        for lstm_layer in self.lstm_layers:
            lstm_layer.train()

    def eval(self):
        """
        Sets all LSTM layers to evaluation mode.
        """
        for lstm_layer in self.lstm_layers:
            lstm_layer.eval()

    def clip_gradients(self):
        """
        Clips the gradients of all LSTM layers.
        """
        for lstm_layer in self.lstm_layers:
            lstm_layer.clip_gradients()

class LSTM_AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4,
                 lstm_units=[64],
                 lstm_epochs=10, 
                 lstm_dropout=0.2, 
                 lstm_grad_clip_threshold=1.0):
        
        self.lstm = MultiLayerLSTM(input_dim=n_base_models * 2, hidden_dims=lstm_units,
                                   dropout_prob=lstm_dropout, grad_clip_threshold=lstm_grad_clip_threshold)        
        self.lstm_units = lstm_units

        # print(f'self.lstm.lstm_layers[0].input_dim: {self.lstm.lstm_layers[0].input_dim}, which should be n_base_models * 2 = {n_base_models * 2}')
        # print(f'self.lstm.lstm_layers[-1].hidden_dim: {self.lstm.lstm_layers[-1].hidden_dim}, which should be lstm_units[-1] = {lstm_units[-1]}')

        self.lstm_epochs = lstm_epochs
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
        self.lstm.train()
        self._lstm_model.forward(meta_features_reshaped)
        self.lstm.clip_gradients()

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

        # predict using the LSTM model
        self.lstm.eval()
        lstm_outputs, _, _ = self._lstm_model.forward(meta_features.reshape((-1, meta_features.shape[1])))
        lstm_predictions = lstm_outputs[:, -1] 

        if np.isscalar(lstm_predictions):
            print('np value is a scalar value!')
            print(lstm_predictions)
            return np.full(X.shape[0], (lstm_predictions > 0.5).astype(int))
        else:
            return (lstm_predictions > 0.5).astype(int)
    
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
        # predict probabilities using the LSTM model
        lstm_outputs, _, _ = self._lstm_model.forward(meta_features.reshape((-1, meta_features.shape[1])))
        lstm_predictions = np.dot(lstm_outputs, self._lstm_model.lstm_layers[-1].weights_ho)
        probabilities = 1 / (1 + np.exp(-lstm_predictions))
        return probabilities

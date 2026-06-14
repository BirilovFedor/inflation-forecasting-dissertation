import numpy as np
from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from itertools import combinations
from sklearn.preprocessing import StandardScaler




class WindowData:

    def __init__(self, dataset, val_size, test_size, lag_size, window_step, forecast_horizon,
                 target_name, train_size=None, difference_cols=None) -> None:
        self.dataset          = dataset.copy()
        self.original_dataset = dataset.copy()         
        self.val_size         = val_size
        self.test_size        = test_size
        self.train_size       = (len(self.dataset) - self.val_size - self.test_size
                                 if train_size is None else train_size)
        self.lag_size         = lag_size
        self.window_step      = window_step
        self.forecast_horizon = forecast_horizon
        self.target_name      = target_name
        self.difference_cols  = difference_cols if difference_cols is not None else []

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()


        self.last_level: dict = {}

    def apply_differencing(self):
        
        if not self.difference_cols:
            return

        # Save the last level value *before* differencing – used for inversion.
        # Index of the last training+val observation in the original (undifferenced) dataset.
        last_obs_idx = self.train_size + self.val_size - 1
        for col in self.difference_cols:
            self.last_level[col] = self.original_dataset[col].iloc[last_obs_idx]

        # Apply first difference
        for col in self.difference_cols:
            self.dataset[col] = self.dataset[col].diff()

        # Drop the first row (NaN after diff) and reset index
        self.dataset = self.dataset.iloc[1:].reset_index(drop=True)

        # Shrink sizes to reflect the lost row
        if self.train_size > 0:
            self.train_size -= 1

    def inverse_difference_forecast(self, y_pred_scaled, window_data=None):
        
        wd = window_data if window_data is not None else self

        # 1. Undo scaling  →  differenced values
        y_diff = y_pred_scaled * wd.scaler_y.scale_[0] + wd.scaler_y.mean_[0]

        # 2. Undo differencing via cumulative sum starting from last known level
        last_val = wd.last_level[wd.target_name]
        y_level  = np.cumsum(y_diff) + last_val

        return y_level

    def inverse_difference_actuals(self, y_scaled, window_data=None):

        wd = window_data if window_data is not None else self

        y_diff  = y_scaled * wd.scaler_y.scale_[0] + wd.scaler_y.mean_[0]
        last_val = wd.last_level[wd.target_name]
        y_level  = np.cumsum(y_diff) + last_val

        return y_level

    def scale_data(self):
        train_end = self.train_size
        val_end   = self.train_size + self.val_size

        X_cols = list(self.dataset.columns)
        y_col  = self.target_name

        X_train = self.dataset.iloc[:train_end][X_cols].values
        X_val   = self.dataset.iloc[train_end:val_end][X_cols].values if self.val_size  > 0 else None
        X_test  = self.dataset.iloc[val_end:][X_cols].values          if self.test_size > 0 else None

        y_train = self.dataset.iloc[:train_end][y_col].values.reshape(-1, 1)
        y_val   = self.dataset.iloc[train_end:val_end][y_col].values.reshape(-1, 1) if self.val_size  > 0 else None
        y_test  = self.dataset.iloc[val_end:][y_col].values.reshape(-1, 1)          if self.test_size > 0 else None

        self.scaler_X.fit(X_train)
        X_train_scaled = self.scaler_X.transform(X_train)
        X_val_scaled   = self.scaler_X.transform(X_val)  if self.val_size  > 0 else None
        X_test_scaled  = self.scaler_X.transform(X_test) if self.test_size > 0 else None

        self.scaler_y.fit(y_train)
        y_train_scaled = self.scaler_y.transform(y_train)
        y_val_scaled   = self.scaler_y.transform(y_val)  if self.val_size  > 0 else None
        y_test_scaled  = self.scaler_y.transform(y_test) if self.test_size > 0 else None

        total = self.train_size + self.test_size + self.val_size
        self.dataset_scaled  = self.dataset[:total].copy()
        self.response_scaled = self.dataset[:total].copy()

        if self.val_size > 0 and self.test_size > 0:
            self.dataset_scaled[X_cols]  = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
            self.response_scaled[y_col]  = np.vstack([y_train_scaled, y_val_scaled, y_test_scaled])
        elif self.val_size > 0:
            self.dataset_scaled[X_cols]  = np.vstack([X_train_scaled, X_val_scaled])
            self.response_scaled[y_col]  = np.vstack([y_train_scaled, y_val_scaled])
        elif self.test_size > 0:
            self.dataset_scaled[X_cols]  = np.vstack([X_train_scaled, X_test_scaled])
            self.response_scaled[y_col]  = np.vstack([y_train_scaled, y_test_scaled])
        else:
            self.dataset_scaled[X_cols]  = np.vstack([X_train_scaled])
            self.response_scaled[y_col]  = np.vstack([y_train_scaled])

    def create_rolling_windows(self):

        self.apply_differencing()
        self.scale_data()

        self.window_X_train = []
        self.window_y_train = []

        for iter in range(0, self.train_size - self.forecast_horizon - self.lag_size, self.window_step):
            self.window_X_train.append(self.dataset_scaled.iloc[iter: iter + self.lag_size, :].values)
            self.window_y_train.append(
                self.response_scaled[self.target_name].iloc[
                    iter + self.lag_size: iter + self.lag_size + self.forecast_horizon
                ]
            )

        self.window_X_val = []
        self.window_y_val = []
        for iter in range(
            self.train_size - self.lag_size + 1,
            self.train_size + self.val_size - self.lag_size - self.forecast_horizon + 1,
            self.window_step
        ):
            self.window_X_val.append(self.dataset_scaled.iloc[iter: iter + self.lag_size, :].values)
            self.window_y_val.append(
                self.response_scaled[self.target_name].iloc[
                    iter + self.lag_size: iter + self.lag_size + self.forecast_horizon
                ]
            )

        self.window_X_test = []
        self.window_y_test = []
        iter = self.train_size + self.val_size - self.lag_size
        self.window_X_test.append(self.dataset_scaled.iloc[iter: iter + self.lag_size, :].values)
        self.window_y_test.append(
            self.response_scaled[self.target_name].iloc[
                iter + self.lag_size: iter + self.lag_size + self.forecast_horizon
            ]
        )

        self.window_X_train = np.array(self.window_X_train)
        self.window_y_train = np.array(self.window_y_train)
        self.window_X_val   = np.array(self.window_X_val)
        self.window_y_val   = np.array(self.window_y_val)
        self.window_X_test  = np.array(self.window_X_test)
        self.window_y_test  = np.array(self.window_y_test)


class LSTM:

    def __init__(self, batch_size, number_of_features, hidden_size, lenght_of_output_window):

        '''
        Docstring for __init__

        :param c: long-term memory
        :param h: short-term memory

        :param wf: weights used in making a decision what percent of long-term memory to keep
        :param bf: bias used in making a decision what percent of long-term memory to keep
        :param wi: weights used in making a decision what percent of potential long-term memory to keep
        :param bi: bias used in making a decision what percent of potential long-term memory to keep
        :param wc: weights used in creating potential long-term memory
        :param bc: bias used in creating potential long-term memory
        :param wo: weights used in making a decision what percent of long-term memory to use in current short-term memory
        :param bo: bias used in making a decision what percent of long-term memory to use in current short-term memory
        '''
        


        np.random.seed(42) 
        self.wf = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bf = np.zeros((1, hidden_size))
        self.wi = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bi = np.zeros((1, hidden_size))
        self.wc = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bc = np.zeros((1, hidden_size))
        self.wo = np.random.randn(hidden_size + number_of_features, hidden_size) * 0.1
        self.bo = np.zeros((1, hidden_size))
        self.wy = np.random.randn(hidden_size, lenght_of_output_window) * 0.1
        self.by = np.zeros((1, lenght_of_output_window))
        self.memory = []
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_of_features = number_of_features

    def sigmoid_function(self, x):

        '''
        Docstring for sigmoid_function

        function takes x-axis value and returns corresponding y-axis value on the sigmoid function
        the returned value y in range [0, 1], can be treated as probability

        :param x: x-axis values
        '''

        y = 1 / (1 + np.exp(-x))
        return y
    
    def tahn_function(self, x):

        '''
        Docstring for tahn_function

        function takes x-axis value and returns corresponding y-axis value on the tahn function
        the returned value y in range [-1, 1]

        :param x: x-axis values
        '''

        y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return y

    def preactivation_function(self, x, w, b):

        '''
        Docstring for preactivation_function

        function that calculates the preactivation, linear transformation of x and w

        :param x: data + previous hidden layer(h or short-term memory)
        :param w: weights
        :param b: bias

        '''

        return np.dot(x, w) + b

    def memory_cleaner(self):
        '''
        Docstring for memory_cleaner

        function is used to clean the long-term and short term memories for new data
        '''
        self.memory = []

    def forward_sequence(self, x_sequence):
        '''
        Docstring for forward_sequence

        :param x_sequence: matrix of a size batch_size x number_of_input_years x number_of_features

        function runs forward pass for every year to update the long-term and short term memories
        '''

        self.memory_cleaner()

        self.c = np.zeros((x_sequence.shape[0], self.hidden_size))
        self.h = np.zeros((x_sequence.shape[0], self.hidden_size))

        for t in range (x_sequence.shape[1]):
            self.forward_pass(x_sequence[:, t, :])
        
    def forward_pass(self, x):

        ''' 
        Docstring for forward_pass

        :param x: parameter x is size of batch x number of features

        this function is taking 1 year of data and updating the long and short term memory of LSTM memory
        by using 3 steps: 
            1) decide what percent of long-term memory to keep
            2) decide how much to add to long-term memory
            3) decide how much of long-rem memory to reveal to short-term memory
        '''
        # save previous c values
        prev_c = self.c.copy()
        prev_h = self.h.copy()

        ################################## step 1: forget gate #########################################

        # linear combination of data and short-term memory
        f = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wf, self.bf)
        # use sigmoid to get percent of long-term memory that is kept from last iteration
        active_f = self.sigmoid_function(f)

        ################################## step 2: decide what amount of potential long-term meory to add ################

        # linear combination of data and short-term memory
        i = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wi, self.bi)
        # use sigmoid to get percent of potential long-term memory to add
        active_i = self.sigmoid_function(i)

        ################################## step 3: update gate #########################################

        # linear combination of data and short-term memory
        Ct = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wc, self.bc)
        # use tahn to get potential long-term memory to add
        active_ct = self.tahn_function(Ct)
        # keep part of old long-term memory and add new long-term memory
        self.c = self.c * active_f + active_ct * active_i

        ################################## step 4: output gate #########################################

        # linear combination of data and short-term memory
        o = self.preactivation_function(np.concatenate((self.h, x), axis=1), self.wo, self.bo)
        # use sigmoid to get percent of potential long-term memory to reveal to short-term memory in a current state
        active_o = self.sigmoid_function(o)

        self.h = active_o*self.tahn_function(self.c)

        self.memory.append({
            "o": active_o,
            "i": active_i,
            "ct": active_ct,
            "f": active_f,
            "prev_c": prev_c,
            "c": self.c.copy(),
            "prev_h": prev_h,
            "combined": np.concatenate((prev_h, x), axis=1)
        })

        return self.h
    
    def output_layer(self):

        y_pred = self.preactivation_function(self.h, self.wy, self.by)
        return y_pred

    def backpropagation(self, y_pred, y_true, learning_rate):

        '''
        Docstring for backpropagation()

        :param y_true: this parameter should store actual values of response variable

        function is a hand realization of backpropagation for LSTM model
        '''
        # initialize matrixes to store derivatives for weights and biases
        dL_dwf = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbf = np.zeros((1, self.hidden_size))
        dL_dwi = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbi = np.zeros((1, self.hidden_size))
        dL_dwc = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbc = np.zeros((1, self.hidden_size))
        dL_dwo = np.zeros((self.hidden_size + self.number_of_features, self.hidden_size))
        dL_dbo = np.zeros((1, self.hidden_size))
        
        # derivative of Loss with respect to output y: dL/dy = (y_pred - y_true)/batch_size
        # size is batch_size x lenght_output
        dL_dy = (y_pred - y_true)/self.batch_size

        # derivative of Loss with respect to weight wy: dL/dw = h * dL_dy
        # size is hidden_size x lenght_output
        dL_dwy = self.h.T @ dL_dy


        # derivative of Loss with respect to bias by: dL/db = sum(dL_dy) - summation over columns
        # size is 1 x length output
        dL_dby = np.sum(dL_dy, axis=0, keepdims=True)

        # derivative of Loss with respect to last short-term memory = dL_dh = wy * dL_dy
        # size batch_size x hidden_size
        dL_dh = dL_dy @ self.wy.T

        # initialize derivative of long-term memory coming from next step as zeroes
        dL_dc_next =  np.zeros((self.batch_size, self.hidden_size))
        # initialize derivative of short-term memory coming from next step
        dL_dh_next =  dL_dh


        for t in range(len(self.memory) - 1, -1, -1):

            # apply tahn function element_wise to long-term memory
            tahn_c_t = self.tahn_function(self.memory[t]["c"])

            # derivative of Loss function with respect to output gate: dL_do = tahn(c) * dh
            # size batch_size x hidden_size
            dL_do = dL_dh_next * tahn_c_t

            # derivative of Loss function with respect to long-term memory after update at time t
            # size batch_size x hidden_size
            dL_dc_t = dL_dh_next * self.memory[t]["o"] * (1 - (tahn_c_t ** 2))  + dL_dc_next

            # derivative of Loss function with respect to preactivated value of o
            dL_do_preact = (dL_do * self.memory[t]["o"] * (1 - self.memory[t]["o"]))

            # derivative of Loss function with respect to weight of output gate 
            # size is hidden+features x hidden
            dL_dwo += self.memory[t]["combined"].T @ dL_do_preact
            
            # derivative of Loss function with respect to bias of output gate 
            # size is 1 x hidden
            dL_dbo += np.sum(dL_do_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to active f
            # size batch_size x hidden_size
            dL_df = dL_dc_t * self.memory[t]["prev_c"]
            
            # derivative of Loss function with respect to active i
            # size batch_size x hidden_size
            dL_dct = dL_dc_t * self.memory[t]["i"]

            # derivative of Loss function with respect to active ct
            # size batch_size x hidden_size
            dL_di = dL_dc_t * self.memory[t]["ct"]

            # derivative of Loss function with respect to preactivated value of ct
            dL_dct_preact = dL_dct * (1 - (self.memory[t]["ct"] ** 2))
            # derivative of Loss function with respect to weight of ct gate 
            # size is hidden+features x hidden
            dL_dwc += self.memory[t]["combined"].T @ dL_dct_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbc += np.sum(dL_dct_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to preactivated value of i
            dL_di_preact = dL_di * self.memory[t]["i"] * (1 - self.memory[t]["i"])
            # derivative of Loss function with respect to weight of i gate 
            # size is hidden+features x hidden
            dL_dwi += self.memory[t]["combined"].T @ dL_di_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbi += np.sum(dL_di_preact, axis=0, keepdims=True)

            # derivative of Loss function with respect to preactivated value of i
            dL_df_preact = dL_df * self.memory[t]["f"] * (1 - self.memory[t]["f"])
            # derivative of Loss function with respect to weight of i gate 
            # size is hidden+features x hidden
            dL_dwf += self.memory[t]["combined"].T @ dL_df_preact
            # derivative of Loss function with respect to bias of ct gate 
            # size is 1 x hidden
            dL_dbf += np.sum(dL_df_preact, axis=0, keepdims=True)

            # derivative of the h_t-1 that is passed to the previous iteration
            dL_dh_next = dL_df_preact @ self.wf.T[:, :self.hidden_size] + \
                            dL_di_preact @ self.wi.T[:, :self.hidden_size] + \
                                dL_dct_preact @ self.wc.T[:, :self.hidden_size] + \
                                    dL_do_preact @ self.wo.T[:, :self.hidden_size]
            '''dL_dcombined = (dL_df_preact  @ self.wf.T +
                            dL_di_preact  @ self.wi.T +
                            dL_dct_preact @ self.wc.T +
                            dL_do_preact  @ self.wo.T)

            dL_dh_next = dL_dcombined[:, :self.hidden_size]'''
            # derivative of the c_t-1 that is passed to the previous iteration
            dL_dc_next = dL_dc_t * self.memory[t]["f"]


            ###################################### update weights and biases ###################################

        self.wc -= learning_rate * dL_dwc
        self.bc -= learning_rate * dL_dbc
        self.wf -= learning_rate * dL_dwf
        self.bf -= learning_rate * dL_dbf
        self.wi -= learning_rate * dL_dwi
        self.bi -= learning_rate * dL_dbi
        self.wo -= learning_rate * dL_dwo
        self.bo -= learning_rate * dL_dbo
        self.wy -= learning_rate * dL_dwy
        self.by -= learning_rate * dL_dby

    def training(self, X, y,  X_val, y_val, learning_rate, window_data,  num_epoches = 100):

        for epoch in range(num_epoches):

            # shaffle the data for each epoch to improve training
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            indices = np.random.permutation(X_val.shape[0])
            X_val_shuffled = X_val[indices]
            y_val_shuffled = y_val[indices]

            train_losses = []        
            # iterate over the whole dataset with a step of a batch size
            for start in range(0, (X.shape[0]//self.batch_size) * self.batch_size, self.batch_size):
                    
                batch_x = X_shuffled[start:start+self.batch_size]
                batch_y = y_shuffled[start:start+self.batch_size]

                self.forward_sequence(batch_x)
                y_pred = self.output_layer()
                self.backpropagation(y_pred, batch_y, learning_rate)

                train_losses.append(np.mean((y_pred - batch_y) ** 2))

            val_losses = []
            for start in range(0, (X_val.shape[0]//self.batch_size) * self.batch_size, self.batch_size):
                batch_x_val = X_val_shuffled[start:start + self.batch_size]
                batch_y_val = y_val_shuffled[start:start + self.batch_size]

                self.forward_sequence(batch_x_val)        # forward only — no backprop
                y_val_pred = self.output_layer()
                val_losses.append(np.mean((y_val_pred - batch_y_val) ** 2))

            val_rmse = np.sqrt(np.mean(val_losses))
            train_rmse  = np.sqrt(np.mean(train_losses))

            #

            print(f"Epoch {epoch + 1}/{num_epoches} — Train RMSE: {train_rmse*window_data.scaler_y.scale_[0]:.4f} Validation RMSE: {val_rmse*window_data.scaler_y.scale_[0]:.4f}")

        return train_rmse*window_data.scaler_y.scale_[0], val_rmse*window_data.scaler_y.scale_[0]

    def get_testing_results(self, X, y, window_data):

        self.forward_sequence(X)
        y_pred_scaled = self.output_layer()  
        y_true_scaled = y                   

        target_is_differenced = window_data.target_name in window_data.difference_cols

        if target_is_differenced:
            # undo scaling then undo differencing for both pred and actual
            original_y_pred = window_data.inverse_difference_forecast(
                y_pred_scaled.flatten(), window_data
            )
            original_y = window_data.inverse_difference_actuals(
                y_true_scaled.flatten(), window_data
            )
        else:
            original_y_pred = (y_pred_scaled * window_data.scaler_y.scale_[0]
                               + window_data.scaler_y.mean_[0])
            original_y      = (y_true_scaled * window_data.scaler_y.scale_[0]
                               + window_data.scaler_y.mean_[0])
            original_y_pred = original_y_pred.flatten()
            original_y      = original_y.flatten()

        #plt.plot(original_y_pred, color="red",  label="Predicted")
        #plt.plot(original_y,      color="blue", label="Actual")
        #plt.legend()
        #plt.title(f"LSTM Test Results – {window_data.target_name}")
        #lt.show()

        test_rmse = np.sqrt(np.mean((original_y_pred - original_y) ** 2))
        print(f"Test RMSE: {test_rmse:.4f}")

        return original_y_pred.flatten(), original_y.flatten()


def cross_validation_lags(dataset, lag_sizes, target_name, features, batch_size, hidden_size,
                           forecast_horizon=60, difference_cols=None):
    lag_values         = []
    rmses_values_train = []
    rmses_values_val   = []

    for lag in lag_sizes:
        split = 0
        cv_train, cv_val = [], []

        while forecast_horizon + 120 + 20 + split + forecast_horizon + 20 < 480:
            window = WindowData(dataset[features], train_size=forecast_horizon + 120 + 20 + split,
                                val_size=forecast_horizon + 20, test_size=0,
                                lag_size=lag, window_step=1, forecast_horizon=forecast_horizon,
                                target_name=target_name, difference_cols=difference_cols)
            window.create_rolling_windows()

            lstm_model = LSTM(batch_size, len(features), hidden_size, forecast_horizon)
            tr, va = lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                                          X_val=window.window_X_val, y_val=window.window_y_val,
                                          learning_rate=0.1, window_data=window, num_epoches=20)
            cv_train.append(tr)
            cv_val.append(va)
            split += 40

        lag_values.append(lag)
        rmses_values_train.append(np.mean(cv_train))
        rmses_values_val.append(np.mean(cv_val))

    plt.plot(lag_values, rmses_values_train, color='red',  linewidth=1, linestyle='-')
    plt.plot(lag_values, rmses_values_val,   color='blue', linewidth=2, linestyle='--')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--')]
    plt.legend(custom_lines, ['Train RMSE', 'Validation RMSE'])
    plt.xlabel("Lag Value", fontsize=12)
    plt.ylabel("RMSE",      fontsize=12)
    plt.xticks(lag_values,  fontsize=10, rotation=45)
    plt.title("LSTM Cross-Validation: Lag Value", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def cross_validation_hidden(dataset, lag_size, target_name, features, batch_size, hidden_sizes,
                             forecast_horizon=60, difference_cols=None):
    hidden_values      = []
    rmses_values_train = []
    rmses_values_val   = []

    for hidden_size in hidden_sizes:
        split = 0
        cv_train, cv_val = [], []

        while forecast_horizon + 120 + 20 + split + forecast_horizon + 20 < 480:
            window = WindowData(dataset[features], train_size=forecast_horizon + 120 + 20 + split,
                                val_size=forecast_horizon + 20, test_size=0,
                                lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon,
                                target_name=target_name, difference_cols=difference_cols)
            window.create_rolling_windows()

            lstm_model = LSTM(batch_size, len(features), hidden_size, forecast_horizon)
            tr, va = lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                                          X_val=window.window_X_val, y_val=window.window_y_val,
                                          learning_rate=0.1, window_data=window, num_epoches=20)
            cv_train.append(tr)
            cv_val.append(va)
            split += 40

        hidden_values.append(hidden_size)
        rmses_values_train.append(np.mean(cv_train))
        rmses_values_val.append(np.mean(cv_val))

    plt.plot(hidden_values, rmses_values_train, color='red',  linewidth=1, linestyle='-')
    plt.plot(hidden_values, rmses_values_val,   color='blue', linewidth=2, linestyle='--')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--')]
    plt.legend(custom_lines, ['Train RMSE', 'Validation RMSE'])
    plt.xlabel("Hidden Size", fontsize=12)
    plt.ylabel("RMSE",        fontsize=12)
    plt.xticks(hidden_values, fontsize=10, rotation=45)
    plt.title("LSTM Cross-Validation: Hidden Size", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def cross_validation_features(dataset, lag_size, target_name, features, batch_size, hidden_size,
                               forecast_horizon=60, difference_cols=None):
    all_cols   = list(features)
    other_cols = [c for c in all_cols if c != target_name]
    all_results = {}
    best_val_score = float('inf')
    best_features  = None

    for subset_size in range(1, len(all_cols)):
        for subset in combinations(other_cols, subset_size):
            features_set = [target_name] + list(subset)
            split = 0
            cv_train, cv_val = [], []

            while forecast_horizon + 120 + 20 + split + forecast_horizon + 20 < 480:
                window = WindowData(dataset[features_set],
                                    train_size=forecast_horizon + 120 + 20 + split,
                                    val_size=forecast_horizon + 20, test_size=0,
                                    lag_size=lag_size, window_step=1,
                                    forecast_horizon=forecast_horizon,
                                    target_name=target_name, difference_cols=difference_cols)
                window.create_rolling_windows()

                lstm_model = LSTM(batch_size, len(features_set), hidden_size, forecast_horizon)
                tr, va = lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                                              X_val=window.window_X_val, y_val=window.window_y_val,
                                              learning_rate=0.1, window_data=window, num_epoches=100)
                cv_train.append(tr)
                cv_val.append(va)
                split += 10

            subset_key = tuple(features_set)
            all_results[subset_key] = {
                'train': cv_train, 'val': cv_val,
                'mean_val': np.mean(cv_val), 'std_val': np.std(cv_val),
                'mean_train': np.mean(cv_train)
            }
            if all_results[subset_key]['mean_val'] < best_val_score:
                best_val_score = all_results[subset_key]['mean_val']
                best_features  = subset_key

    labels    = [str(list(k)) for k in all_results]
    mean_vals = [v['mean_val'] for v in all_results.values()]
    std_vals  = [v['std_val']  for v in all_results.values()]
    x = range(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 0.8), 5))
    plt.bar(x, mean_vals, yerr=std_vals, color='steelblue', alpha=0.7, capsize=4)
    plt.xticks(x, labels, rotation=90, fontsize=8)
    plt.ylabel('Mean Validation Score')
    plt.title('Cross-Validation: Mean Validation Score per Feature Subset')
    plt.axvline(x=list(all_results.keys()).index(best_features), color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    print(f"\nBest feature set : {list(best_features)}")
    print(f"Mean val score   : {best_val_score:.4f}")
    return all_results, best_features


def cross_validation_learning_rate(dataset, lag_size, target_name, features, batch_size,
                                    hidden_size, alphas, forecast_horizon=60, num_epoches=20,
                                    difference_cols=None):
    alpha_values       = []
    rmses_values_train = []
    rmses_values_val   = []

    for alpha in alphas:
        split = 0
        cv_train, cv_val = [], []

        while forecast_horizon + 120 + 20 + split + forecast_horizon + 20 < 480:
            window = WindowData(dataset[features],
                                train_size=forecast_horizon + 120 + 20 + split,
                                val_size=forecast_horizon + 20, test_size=0,
                                lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon,
                                target_name=target_name, difference_cols=difference_cols)
            window.create_rolling_windows()

            lstm_model = LSTM(batch_size, len(features), hidden_size, forecast_horizon)
            tr, va = lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                                          X_val=window.window_X_val, y_val=window.window_y_val,
                                          learning_rate=alpha, window_data=window, num_epoches=num_epoches)
            cv_train.append(tr)
            cv_val.append(va)
            split += 40

        alpha_values.append(alpha)
        rmses_values_train.append(np.mean(cv_train))
        rmses_values_val.append(np.mean(cv_val))

    plt.figure(figsize=(12, 7), dpi=600)
    plt.plot(alpha_values, rmses_values_train, color='red',  linewidth=1, linestyle='-')
    plt.plot(alpha_values, rmses_values_val,   color='blue', linewidth=2, linestyle='--')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--')]
    plt.legend(custom_lines, ['Train RMSE', 'Validation RMSE'])
    plt.xlabel("Learning Rate (Alpha)", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.xscale('log')
    plt.title("LSTM Cross-Validation: Learning Rate", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    best_alpha = alpha_values[np.argmin(rmses_values_val)]
    print(f"\nBest alpha    : {best_alpha}")
    print(f"Best val RMSE : {min(rmses_values_val):.4f}")
    return alpha_values, rmses_values_train, rmses_values_val, best_alpha


def cross_validation_epochs(dataset, lag_size, target_name, features, batch_size, hidden_size,
                             epoch_values, forecast_horizon=60, learning_rate=0.1,
                             difference_cols=None):
    epochs_list        = []
    rmses_values_train = []
    rmses_values_val   = []

    for num_epoches in epoch_values:
        split = 0
        cv_train, cv_val = [], []

        while forecast_horizon + 120 + 20 + split + forecast_horizon + 20 < 480:
            window = WindowData(dataset[features],
                                train_size=forecast_horizon + 120 + 20 + split,
                                val_size=forecast_horizon + 20, test_size=0,
                                lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon,
                                target_name=target_name, difference_cols=difference_cols)
            window.create_rolling_windows()

            lstm_model = LSTM(batch_size, len(features), hidden_size, forecast_horizon)
            tr, va = lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                                          X_val=window.window_X_val, y_val=window.window_y_val,
                                          learning_rate=learning_rate, window_data=window,
                                          num_epoches=num_epoches)
            cv_train.append(tr)
            cv_val.append(va)
            split += 40

        epochs_list.append(num_epoches)
        rmses_values_train.append(np.mean(cv_train))
        rmses_values_val.append(np.mean(cv_val))

    plt.figure(figsize=(12, 7), dpi=600)
    plt.plot(epochs_list, rmses_values_train, color='red',  linewidth=1, linestyle='-')
    plt.plot(epochs_list, rmses_values_val,   color='blue', linewidth=2, linestyle='--')
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--')]
    plt.legend(custom_lines, ['Train RMSE', 'Validation RMSE'])
    plt.xlabel("Number of Epochs", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("LSTM Cross-Validation: Number of Epochs", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    best_epochs = epochs_list[np.argmin(rmses_values_val)]
    print(f"\nBest epochs   : {best_epochs}")
    print(f"Best val RMSE : {min(rmses_values_val):.4f}")
    return epochs_list, rmses_values_train, rmses_values_val, best_epochs


def plot_LSTM_results(dataset, number_of_steps, features, file_name, target_name, lag_size,
                      hidden_size, forecast_horizon, num_epoches, difference_cols=None):
    
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    Data = Dataset(filepath, sep=';')
    plt.style.use('default')
    #plt.figure(figsize=(12,7), dpi=600)
    result = []
    plt.plot(dataset['Date'], dataset[target_name].values,
             label="Actual Data", color='red', linewidth=1)

    for step in range(1, number_of_steps):
        errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

        window = WindowData(Data.dataset[features], val_size=0, test_size=60 * step,
                            lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon,
                            target_name=target_name, difference_cols=difference_cols)
        window.create_rolling_windows()

        lstm_model = LSTM(10, len(features), hidden_size, forecast_horizon)
        lstm_model.training(X=window.window_X_train, y=window.window_y_train,
                             X_val=window.window_X_val, y_val=window.window_y_val,
                             learning_rate=0.1, window_data=window, num_epoches=num_epoches)

        y_pred, y_actual = lstm_model.get_testing_results(
            X=window.window_X_test, y=window.window_y_test, window_data=window
        )

        for ha in errors.keys():
            errors[ha] = np.sqrt(np.mean((y_pred[ha - 1] - y_actual[ha - 1]) ** 2))

        result.append(errors)
        time_slice = dataset['Date'][-60 * step: -60 * (step - 1) if step > 1 else None]
        plt.plot(time_slice, y_pred, label="Forecasted values",
                 color='blue', linestyle='--', linewidth=2)
        plt.axvline(x=dataset['Date'].values[-60 * step], color='black', linestyle='--', alpha=0.7)

    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--'),
                    Line2D([0], [0], color='black', lw=1, linestyle='--', alpha=0.7)]
    plt.legend(custom_lines, ['Actual Data', 'Forecasted values', 'Step boundaries'])
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(target_name, fontsize=12)
    plt.xticks(rotation=45)
    plt.title("LSTM Model", fontsize=14, fontweight='bold')
    #plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    print(pd.DataFrame(result))


def LSTM_energy():
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    Data = Dataset(filepath, sep=';')

    cross_validation_lags(dataset=Data.dataset, lag_sizes = [6, 12, 24, 36, 60, 120], target_name="CPI(Energy)", 
                 features = ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 
                 batch_size = 10, hidden_size = 24, forecast_horizon=60)

    cross_validation_hidden(dataset=Data.dataset, lag_size = 120, target_name="CPI(Energy)", 
                    features = ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_sizes = [10, 15, 24, 36, 60, 120], forecast_horizon=60)

    cross_validation_features  (dataset=Data.dataset, lag_size=120, target_name="CPI(Energy)", 
                    features = ['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 120, forecast_horizon=60)

    cross_validation_learning_rate(dataset=Data.dataset, lag_size=120, target_name="CPI(Energy)", 
                    features = ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 120, alphas = [0.001, 0.01, 0.1, 0.2], forecast_horizon=60, num_epoches=20)
    cross_validation_epochs(dataset=Data.dataset, lag_size=120, target_name="CPI(Energy)", 
                    features = ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 120, epoch_values = [1, 5, 10, 20, 40], forecast_horizon=60, learning_rate=0.1)
    
    plot_LSTM_results(dataset=Data.dataset, number_of_steps=8, features=['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYLSTMRESULTSSOLO.png", 
                  target_name="CPI(Energy)", lag_size=120, hidden_size=120, forecast_horizon=60, num_epoches=40)


def LSTM_food():
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    Data = Dataset(filepath, sep=';')

    cross_validation_lags(dataset=Data.dataset, lag_sizes = [6, 12, 24, 36, 60, 120], target_name="CPI(Food)", 
                 features = ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 
                 batch_size = 10, hidden_size = 24, forecast_horizon=60)

    cross_validation_hidden(dataset=Data.dataset, lag_size = 60, target_name="CPI(Food)", 
                    features = ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_sizes = [10, 15, 24, 36, 60, 120], forecast_horizon=60)

    cross_validation_features  (dataset=Data.dataset, lag_size=60, target_name="CPI(Food)", 
                    features = ['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 48, forecast_horizon=60)

    cross_validation_learning_rate(dataset=Data.dataset, lag_size=60, target_name="CPI(Food)", 
                    features = ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 48, alphas = [0.001, 0.01, 0.1, 0.2], forecast_horizon=60, num_epoches=20)
    cross_validation_epochs(dataset=Data.dataset, lag_size=60, target_name="CPI(Food)", 
                    features = ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 
                    batch_size = 10, hidden_size = 48, epoch_values = [1, 5, 10, 20, 40], forecast_horizon=60, learning_rate=0.1)
    
    plot_LSTM_results(dataset=Data.dataset, number_of_steps=8, features=['CPI(Food)', 'UNRATE', 'IP', 'GS5'], file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFoodLSTMRESULTSSOLO.png", 
                  target_name="CPI(Food)", lag_size=60, hidden_size=48, forecast_horizon=60, num_epoches=40)


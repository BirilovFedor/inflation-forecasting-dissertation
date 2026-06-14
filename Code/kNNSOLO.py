from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations
from sklearn.model_selection import TimeSeriesSplit  
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.ensemble import RandomForestRegressor

#lag = 2
# horizon = 4
# train = 10
# 0 1 2 
# 4 5  67 8 9 10 11
# validation = 5

from sklearn.neighbors import KNeighborsRegressor

class WindowData:

    def __init__(self, dataset, val_size, test_size, lag_size, window_step, forecast_horizon, target_name, diff_order) -> None:
        self.dataset = dataset
        self.original_dataset = dataset
        self.val_size = val_size
        self.test_size = test_size
        self.train_size = len(self.dataset) - self.val_size - self.test_size
        self.lag_size = lag_size
        self.window_step = window_step
        self.forecast_horizon = forecast_horizon
        self.target_name = target_name
        self.diff_order = diff_order

    def _apply_differencing(self, df, order):
        """Apply differencing of given order, storing initial values for inversion."""
        self.diff_initial_values = []  # stores the last value before each diff for inversion
        diffed = df.copy()
        for _ in range(order):
            self.diff_initial_values.append(diffed.iloc[0].copy())
            diffed = diffed.diff().dropna()
        return diffed

    def create_rolling_windows(self):
        
        if self.diff_order > 0:
            self.dataset = self._apply_differencing(self.dataset, self.diff_order)
            # Recalculate train_size after rows lost to differencing
            lost_rows = len(self.original_dataset) - len(self.dataset)
            self.train_size = self.train_size - lost_rows


        self.window_X_train = []
        self.window_y_train = []

        for iter in range(0, self.train_size - self.forecast_horizon-self.lag_size, self.window_step):
            self.window_X_train.append(self.dataset.iloc[iter : iter + self.lag_size, :].values.flatten())
            self.window_y_train.append(self.dataset[self.target_name].iloc[iter+self.lag_size+self.forecast_horizon - 1])

        self.window_X_val = []
        self.window_y_val = []
        self.preceding_y_val = []
        for iter in range(self.train_size-self.lag_size+1, self.train_size + self.val_size - self.lag_size - self.forecast_horizon+1, self.window_step):
            self.window_X_val.append(self.dataset.iloc[iter : iter + self.lag_size, :].values.flatten())
            self.window_y_val.append(self.dataset[self.target_name].iloc[iter+self.lag_size+self.forecast_horizon - 1])
            if self.diff_order > 0:
                target_idx_in_original = iter + self.lag_size + self.forecast_horizon - 1 + lost_rows
                self.preceding_y_val.append(self.original_dataset[self.target_name].iloc[target_idx_in_original - 1])


        self.window_X_test = []
        self.window_y_test = []
        iter = self.train_size + self.val_size - self.lag_size
        self.window_X_test.append(self.dataset.iloc[iter : iter + self.lag_size, :].values.flatten())
        self.window_y_test.append(self.dataset[self.target_name].iloc[iter+self.lag_size+self.forecast_horizon - 1])

        if self.diff_order > 0:
            test_target_idx_in_original = iter + self.lag_size + self.forecast_horizon - 1 + lost_rows
            self.last_original_value_before_test = self.original_dataset[self.target_name].iloc[test_target_idx_in_original - 1]

        self.window_X_train = np.array(self.window_X_train)
        self.window_y_train = np.array(self.window_y_train)
        self.window_X_val = np.array(self.window_X_val)
        self.window_y_val = np.array(self.window_y_val)
        self.window_X_test = np.array(self.window_X_test)
        self.window_y_test = np.array(self.window_y_test)

class kNN_Model:

    def __init__(self, dataset, test_size, val_size, target_name) -> None:
        self.dataset = dataset
        self.test_size = test_size
        self.val_size = val_size
        self.target_name = target_name

    def fit_kNN(self, n_neighbors, X_train, y_train, X_test, y_test, window_data, previous_prediction=None, return_predictions=False, test = True, cross_validation = False):

        scaler = StandardScaler()


        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, )
        knn_regressor.fit(scaled_X_train, y_train)

        y_pred = knn_regressor.predict(scaled_X_test)

        if test and window_data.diff_order > 0:
            last_val = window_data.last_original_value_before_test
            y_pred   = y_pred   + previous_prediction
            y_test   = y_test   + last_val

        if cross_validation:
            y_pred = y_pred
            y_test = y_test

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        if return_predictions:
            return rmse, y_pred, y_test
        return rmse


    
    def get_windowed_data(self, features, val_size, test_size, lag_size, forecast_horizon, diff_order,  window_step = 1):
        window_data = WindowData(self.dataset[features], val_size=val_size, test_size=test_size, lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon, target_name=self.target_name, diff_order=diff_order)
        window_data.create_rolling_windows()
        return window_data


    def cross_validation(self, Data, k_folds, lag_sizes, n_neighbors, metrics, weights,
                     forecast_horizon, diff):

        all_cols   = list(Data.columns)
        other_cols = [c for c in all_cols if c != self.target_name]
        max_vars   = len(all_cols)

        time_series_split = TimeSeriesSplit(n_splits=k_folds)

        results_dict = {
            "Variables": [], "Lag_size": [], "n_neighbors": [],
            "metric": [], "weights": [],
            "RMSE_mean": [], "RMSE_std": [], "validation_results": [], "score": []
        }

        for subset_size in range(1, max_vars):
            for subset in combinations(other_cols, subset_size):

                features = [self.target_name] + list(subset)

                for lag_size in lag_sizes:
                    window_data = self.get_windowed_data(
                        features, val_size=80, test_size=60,
                        lag_size=lag_size, window_step=1,
                        forecast_horizon=forecast_horizon, diff_order=diff
                    )

                    for n_neighbor in n_neighbors:
                        for metric in metrics:
                            for weight in weights:

                                fold_rmses = []

                                for train_idx, val_idx in time_series_split.split(window_data.window_X_train):
                                    X_train = window_data.window_X_train[train_idx]
                                    y_train = window_data.window_y_train[train_idx]
                                    X_val   = window_data.window_X_train[val_idx]
                                    y_val   = window_data.window_y_train[val_idx]

                                    fold_rmses.append(self.fit_kNN(
                                        n_neighbors=n_neighbor,
                                        metric=metric,
                                        weights=weight,
                                        X_train=X_train, y_train=y_train,
                                        X_test=X_val,   y_test=y_val,
                                        window_data=window_data,
                                        return_predictions=False, test=False, cross_validation=True
                                    ))

                                validation_results = self.fit_kNN(
                                    n_neighbors=n_neighbor,
                                    metric=metric,
                                    weights=weight,
                                    X_train=window_data.window_X_train,
                                    y_train=window_data.window_y_train,
                                    X_test=window_data.window_X_val,
                                    y_test=window_data.window_y_val,
                                    window_data=window_data,
                                    return_predictions=False, test=False, cross_validation=True
                                )

                                results_dict["Variables"].append(features)
                                results_dict["Lag_size"].append(lag_size)
                                results_dict["n_neighbors"].append(n_neighbor)
                                results_dict["metric"].append(metric)
                                results_dict["weights"].append(weight)
                                results_dict["RMSE_mean"].append(np.mean(fold_rmses))
                                results_dict["RMSE_std"].append(np.std(fold_rmses))
                                results_dict["validation_results"].append(validation_results)
                                results_dict["score"].append(
                                    validation_results + np.mean(fold_rmses) + np.std(fold_rmses)
                                )

        results_df = pd.DataFrame(results_dict)
        best_row = results_df.sort_values(
        by=["score"], ascending=True
        ).iloc[0]

        cols = ["Lag_size", "n_neighbors", "metric", "weights",
                "Variables", "RMSE_mean", "RMSE_std", "validation_results", "score"]

        print(f"################## MODEL FOR FORECAST HORIZON = {forecast_horizon} #####################")
        print("=" * 65)
        print(f"Best by val RMSE -> Lag: {int(best_row['Lag_size'])}, "
            f"Neighbors: {int(best_row['n_neighbors'])}, "
            f"Metric: {best_row['metric']}, "
            f"Weights: {best_row['weights']}, "
            f"Variables: {list(best_row['Variables'])}, "
            f"RMSE: {best_row['RMSE_mean']:.4f} ± {best_row['RMSE_std']:.4f}, "
            f"Val RMSE: {best_row['validation_results']:.4f}, "
            f"Score: {best_row['score']:.4f}")
        print("=" * 65)

        print("\nTop 20 by Score:")
        print(results_df.sort_values("score").head(20)[cols].to_string(index=False))

        best_config = (
        list(best_row['Variables']),
        int(best_row['Lag_size']),
        int(best_row['n_neighbors']),
        best_row['metric'],
        best_row['weights'],
        forecast_horizon
        )

        print("\nModel config entry:")
        print(f"    ({list(best_row['Variables'])}, "
            f"{int(best_row['Lag_size'])}, "
            f"{int(best_row['n_neighbors'])}, "
            f"'{best_row['metric']}', "
            f"'{best_row['weights']}', "
            f"{forecast_horizon}, ")

        return best_config
   

    def plot_kNN_results(self, number_of_steps, model_configs, diff_order, file_name):

        plt.style.use('default')  
        plt.figure(figsize=(12,7), dpi=600)

        # plot the results
        result = []
        plt.plot(self.dataset['Date'], self.dataset[self.target_name].values, label="Actual Data", color='red', linewidth=1)        

        for step in range(1, number_of_steps):
            
            used_dataset = self.dataset.iloc[:-(60*(step-1))] if step >1 else self.dataset
            previous_prediction = used_dataset[self.target_name].iloc[-61]
            all_forecasts = {} 
            forecast_results = []
            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
            for features, lag_size, n_neighbors, base_horizon, horizon_range in model_configs:
                
                for h in horizon_range:
                    rmse, y_pred = run_model_solo(
                        dataset=used_dataset,
                        features=features,
                        val_size=0,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=diff_order,
                        previous_prediction=previous_prediction,
                        n_neighbors=n_neighbors,
                        target_name=self.target_name
                    )
                    previous_prediction = y_pred
                    for ha in errors.keys():
                        if ha==h:
                            errors[h] = rmse
                    
                    all_forecasts[h] = (rmse, y_pred)
                    forecast_results.append(y_pred)
                    #print(f"Horizon {h:>3}m | model@{base_horizon}m | RMSE: {rmse:.4f}")
            result.append(errors)
            time_slice = self.dataset['Date'][-60*step : -60*(step-1) if step > 1 else None]
            plt.plot(
                time_slice, 
                forecast_results, 
                label="Forecasted values", 
                color='blue', 
                linestyle='--', 
                linewidth=2
            )
            plt.axvline(x= self.dataset['Date'].values[-60*step], color='black', linestyle='--', alpha=0.7)
                
        # Grid
        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='gray'
        )

        # Add a custom legend
        custom_lines = [
            Line2D([0], [0], color='red', lw=1),
            Line2D([0], [0], color='blue', lw=2, linestyle='--'),
            Line2D([0], [0], color='black', lw=1, linestyle='--', alpha=0.7)
        ]
        plt.legend(custom_lines, ['Actual Data', 'Forecasted values', 'Step boundaries'])
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(self.target_name, fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=10)
        plt.title("kNN Model", fontsize=14, fontweight='bold')

        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        print(pd.DataFrame(result))

def run_model_solo( dataset, features, val_size, test_size, lag_size, forecast_horizon, diff_order, n_neighbors, previous_prediction, target_name):
        model = kNN_Model(dataset, test_size, val_size, target_name)
        window_data = model.get_windowed_data(
            features, val_size=val_size, test_size=test_size,
            lag_size=lag_size, window_step=1,
            forecast_horizon=forecast_horizon, diff_order=diff_order
        )
        rmse, y_pred, y_test = model.fit_kNN(
            n_neighbors=n_neighbors,
            X_train=window_data.window_X_train,
            y_train=window_data.window_y_train,
            X_test=window_data.window_X_test,
            y_test=window_data.window_y_test,
            window_data=window_data,
            previous_prediction = previous_prediction,
            return_predictions=True
        )
        if forecast_horizon == 60:
            print(y_pred)
        return rmse, y_pred


def cross_validation_rf(dataset, target_name, features, lag_size, forecast_horizon,
                         diff_order, n_estimators, forecast_horizon_range,
                         n_splits=5, test_size=60):
    """
    Walk-forward cross-validation for Random Forest model.
    
    Parameters:
        dataset              : full dataset
        target_name          : target column name
        features             : list of feature columns
        lag_size             : lag window size
        forecast_horizon     : base forecast horizon
        diff_order           : differencing order
        n_estimators         : number of trees
        max_depth            : max depth of trees
        forecast_horizon_range: list of horizons e.g. [3,6,12,24,36,48,60]
        n_splits             : number of CV folds
        test_size            : size of each validation window
    """

    # ── Storage ────────────────────────────────────────────
    fold_results   = []     # one dict per fold
    rmse_per_horizon = {h: [] for h in forecast_horizon_range}

    # ── Walk-forward splits ────────────────────────────────
    # Each fold expands the training window by test_size
    total_size  = len(dataset)
    min_train   = lag_size + forecast_horizon + diff_order + 10

    # Build fold boundaries
    folds = []
    for fold in range(n_splits):
        # Expanding window — each fold adds test_size more training data
        fold_test_end   = total_size - (n_splits - fold - 1) * test_size
        fold_test_start = fold_test_end - test_size

        if fold_test_start < min_train:
            print(f"Fold {fold+1} skipped — not enough training data")
            continue

        folds.append((fold_test_start, fold_test_end))

    print(f"Running {len(folds)} folds...")
    print(f"{'─'*55}")

    for fold_idx, (test_start, test_end) in enumerate(folds):

        fold_dataset       = dataset.iloc[:test_end].copy()
        fold_errors        = {h: None for h in forecast_horizon_range}
        previous_prediction = fold_dataset[target_name].iloc[-(test_size + 1)]

        for h in forecast_horizon_range:
            try:
                model = kNN_Model(
                    dataset     = fold_dataset,
                    test_size   = test_size,
                    val_size=0,
                    target_name = target_name
                )
                window_data = model.get_windowed_data(
                    features         = features,
                    test_size        = test_size,
                    val_size=0,
                    lag_size         = lag_size,
                    forecast_horizon = h,
                    diff_order       = diff_order
                )
        


                rmse, y_pred = model.fit_kNN(
                    n_neighbors       = n_estimators,
                    X_train            = window_data.window_X_train,
                    y_train            = window_data.window_y_train,
                    X_test             = window_data.window_X_test,
                    y_test             = window_data.window_y_test,
                    window_data        = window_data,
                    previous_prediction = previous_prediction,
                    return_predictions = True,
                    test               = True,
                    cross_validation   = False
                )

                previous_prediction  = y_pred
                fold_errors[h]       = rmse
                rmse_per_horizon[h].append(rmse)

            except Exception as e:
                print(f"  Fold {fold_idx+1}, horizon={h} failed: {e}")
                continue

        fold_results.append({
            'fold'       : fold_idx + 1,
            'test_start' : test_start,
            'test_end'   : test_end,
            **{f'RMSE_h{h}': fold_errors[h] for h in forecast_horizon_range}
        })

        # Per-fold print
        rmse_str = '  '.join([f"h{h}={fold_errors[h]:.3f}" 
                               if fold_errors[h] is not None else f"h{h}=N/A"
                               for h in forecast_horizon_range])
        print(f"Fold {fold_idx+1}/{len(folds)} | {rmse_str}")

    print(f"{'─'*55}")

    # ── Summary ────────────────────────────────────────────
    summary = {
        h: {
            'mean_rmse' : np.mean(rmse_per_horizon[h]) if rmse_per_horizon[h] else None,
            'std_rmse'  : np.std(rmse_per_horizon[h])  if rmse_per_horizon[h] else None,
            'n_folds'   : len(rmse_per_horizon[h])
        }
        for h in forecast_horizon_range
    }

    print("\nSummary across folds:")
    print(f"{'Horizon':<10} {'Mean RMSE':<14} {'Std RMSE':<12} {'Folds'}")
    print(f"{'─'*45}")
    for h, stats in summary.items():
        if stats['mean_rmse'] is not None:
            print(f"h={h:<8} {stats['mean_rmse']:<14.4f} "
                  f"{stats['std_rmse']:<12.4f} {stats['n_folds']}")

    '''# ── Plot ───────────────────────────────────────────────
    horizons   = [h for h in forecast_horizon_range if summary[h]['mean_rmse'] is not None]
    mean_rmses = [summary[h]['mean_rmse'] for h in horizons]
    std_rmses  = [summary[h]['std_rmse']  for h in horizons]

    #plt.figure(figsize=(12, 7), dpi=600)
    plt.plot(
        horizons, mean_rmses,
        color='blue', linewidth=2, linestyle='--', marker='o'
    )
    plt.fill_between(
        horizons,
        np.array(mean_rmses) - np.array(std_rmses),
        np.array(mean_rmses) + np.array(std_rmses),
        color='blue', alpha=0.15, label='±1 Std'
    )
    plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, linestyle='--'),
        Line2D([0], [0], color='blue', lw=6, alpha=0.15),
    ]
    plt.legend(custom_lines, ['Mean RMSE', '± 1 Std'])
    plt.xlabel("Forecast Horizon", fontsize=12)
    plt.ylabel("RMSE",             fontsize=12)
    plt.xticks(horizons,           fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.title("Random Forest Cross-Validation: RMSE by Horizon",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()'''

    results_df = pd.DataFrame(fold_results)
    print("\nFull results table:")
    print(results_df.to_string(index=False))

    return results_df, summary



Date_columns = ['Date']
Response_columns = ['CPI(Food)', 'CPI(Energy)', 'CPI(Core)']
Explanatory_columns = ['PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5']
#0 path to the file with original dataset
filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
# create the instance that will be used to store data and process it 
Data = Dataset(filepath, sep=';')
print(Data.dataset['CPI(Energy)'].iloc[-1])
'''model_configs_energy_nodiff = [
            (['CPI(Energy)', 'GS5'],      1, 2,  3,  range(1,  5)),
            (['CPI(Energy)', 'GS5', 'IP'],           1, 9,  6,  range(5,  9)),
            (['CPI(Energy)', 'M2' , 'PPI', 'IP'],              1, 20,  12, range(9,  19)),
            (['CPI(Energy)', 'M2' , 'UNRATE', 'IP', 'GS5'],                    60,  9,  24, range(19, 31)),
            (['CPI(Energy)', 'PPI' , 'M2' , 'UNRATE', 'IP', 'GS5'],        60,  13,  36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'],                 48,  13,  48, range(43, 55)),
            (['CPI(Energy)', 'M2'],                 60,  2, 60, range(55, 61)),
        ]
model_configs_energy_diff = [
            (['CPI(Energy)', 'GS5' , 'PPI', 'IP'],      24, 2,  3,  range(1,  5)),
            (['CPI(Energy)', 'UNRATE', 'IP'],           12, 4,  6,  range(5,  9)),
            (['CPI(Energy)', 'PPI', 'IP'],              12, 7,  12, range(9,  19)),
            (['CPI(Energy)', 'GS5'],                    1,  5,  24, range(19, 31)),
            (['CPI(Energy)', 'M2', 'IP', 'GS5'],        1,  4,  36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE'],                 1,  7,  48, range(43, 55)),
            (['CPI(Energy)', 'GS5' , 'PPI', 'IP', 'UNRATE'],   6,  13, 60, range(55, 61)),
]'''





'''model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=3, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=6, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=12, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=24, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=36, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=48, diff = 1)

model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=60, diff = 1)

model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNN.png")
'''

'''

model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=3, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=6, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=12, diff =0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=24, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=36, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=48, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=60, diff = 0)


model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=3, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=6, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=12, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=24, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=36, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=48, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=60, diff = 1)

model_configs_food_diff = [
            (['CPI(Food)', 'UNRATE', 'IP'],      12, 20,  3,  range(1,  5)),
            (['CPI(Food)', 'UNRATE', 'PPI'],           6, 5,  6,  range(5,  9)),
            (['CPI(Food)', 'UNRATE'],              1, 13,  12, range(9,  19)),
            (['CPI(Food)', 'UNRATE' , 'GS5'],                    1,  13,  24, range(19, 31)),
            (['CPI(Food)', 'UNRATE'],        1,  20,  36, range(31, 43)),
            (['CPI(Food)', 'GS5' , 'PPI', 'IP', 'M2', 'UNRATE'],                 3,  13,  48, range(43, 55)),
            (['CPI(Food)', 'GS5'],   36,  5, 60, range(55, 61)),
]

model = kNN_Model(Data.dataset, 60, 0, "CPI(Food)")
model.plot_kNN_results(8, model_configs=model_configs_food_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODkNN.png")
'''

model_configs_food_nodiff = [
            (['CPI(Food)', 'PPI'],      60, 2,  3,  range(1,  5)),
            (['CPI(Food)', 'M2', 'GS5'],           60, 2,  6,  range(5,  9)),
            (['CPI(Food)', 'PPI', 'M2', 'GS5'],              60, 2,  12, range(9,  19)),
            (['CPI(Food)', 'M2' , 'IP'],                    48,  2,  24, range(19, 31)),
            (['CPI(Food)', 'PPI'],        36,  2,  36, range(31, 43)),
            (['CPI(Food)', 'GS5' , 'PPI', 'IP', 'M2', 'UNRATE'],                 24,  2,  48, range(43, 55)),
            (['CPI(Food)', 'PPI', 'IP', 'M2'],   12,  2, 60, range(55, 61)),
]


'''

model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=3, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=6, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=12, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=24, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=36, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=48, diff = 0)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=60, diff = 0)


model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=3, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=6, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=12, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=24, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=36, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=48, diff = 1)
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], k_folds = 5, lag_sizes = [1, 3, 6, 12, 20,  24, 36, 48, 60] , n_neighbors=[2, 4, 5, 7, 9, 13, 20] , forecast_horizon=60, diff = 1)



model_configs_core_diff = [
            (['CPI(Core)', 'M2', 'GS5'],      20, 20,  3,  range(1,  5)),
            (['CPI(Core)', 'M2', 'IP'],           3, 20,  6,  range(5,  9)),
            (['CPI(Core)', 'UNRATE'],              3, 20,  12, range(9,  19)),
            (['CPI(Core)', 'UNRATE' , 'IP' , 'GS5'],                    3,  9,  24, range(19, 31)),
            (['CPI(Core)', 'IP'],        3,  9,  36, range(31, 43)),
            (['CPI(Core)', 'GS5', 'UNRATE'],                 6,  13,  48, range(43, 55)),
            (['CPI(Core)', 'M2', 'PPI'],   1,  4, 60, range(55, 61)),
]

model = kNN_Model(Data.dataset, 60, 0, "CPI(Core)")
model.plot_kNN_results(8, model_configs=model_configs_core_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPICOREkNN.png")
'''


model_configs_core_nodiff = [
            (['CPI(Energy)', 'PPI', 'GS5'], 1, 4, 'manhattan', 'uniform', 3,  range(1,  5)),
            (['CPI(Energy)', 'IP', 'GS5'], 12, 6, 'manhattan', 'uniform', 6,  range(5,  9)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 6, 9, 'manhattan', 'uniform',12,  range(9,  19)),
            (['CPI(Energy)', 'PPI', 'GS5'], 12, 4, 'manhattan', 'distance',24,  range(19, 31)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 1, 4, 'manhattan', 'distance', 36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'GS5'], 12, 9, 'chebyshev', 'uniform', 48, range(43, 55)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 3, 6, 'manhattan', 'uniform', 60, range(55, 61)),
]

'''model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_nodiff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFF.png")
model = kNN_Model(Data.dataset, 60, 0, "CPI(Food)")
model.plot_kNN_results(8, model_configs=model_configs_food_nodiff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODkNNNODIFF.png")
model = kNN_Model(Data.dataset, 60, 0, "CPI(Core)")
model.plot_kNN_results(8, model_configs=model_configs_core_nodiff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPICOREkNNNODIFF.png")'''

model_configs_energy_nodiff = [
            (['CPI(Energy)', 'IP'], 1, 9, 'manhattan', 'uniform', 3,  range(1,  5)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'manhattan', 'uniform', 6,  range(5,  9)),
            (['CPI(Energy)', 'UNRATE', 'GS5'], 1, 20, 'manhattan', 'distance', 12, range(9,  19)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'IP', 'GS5'], 3, 2, 'euclidean', 'uniform', 24, range(19, 31)),
            (['CPI(Energy)', 'IP'], 1, 20, 'manhattan', 'uniform', 36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'euclidean', 'uniform', 48, range(43, 55)),
            (['CPI(Energy)', 'M2'], 60, 4, 'manhattan', 'distance', 60, range(55, 61)),
        ]

model_configs_energy_diff = [
            (['CPI(Energy)', 'PPI', 'GS5'], 1, 4, 'manhattan', 'uniform', 3,  range(1,  5)),
            (['CPI(Energy)', 'IP', 'GS5'], 12, 6, 'manhattan', 'uniform', 6,  range(5,  9)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 6, 9, 'manhattan', 'uniform',12,  range(9,  19)),
            (['CPI(Energy)', 'PPI', 'GS5'], 12, 4, 'manhattan', 'distance',24,  range(19, 31)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 1, 4, 'manhattan', 'distance', 36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'GS5'], 12, 9, 'chebyshev', 'uniform', 48, range(43, 55)),
            (['CPI(Energy)', 'UNRATE', 'M2', 'IP'], 3, 6, 'manhattan', 'uniform', 60, range(55, 61)),
]



'''model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")

model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFF.png")
model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_nodiff, diff_order=0, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYMSkNNNODIFF.png")
model.plot_kNN_results(8, model_configs=model_configs_energy_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYMSkNNDIFF.png")
'''



'''model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")
print(model.cross_validation(Data=Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Energy)")'''

'''model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 0))


model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Food)")
print(model.cross_validation(Data=Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 1))
'''



model_configs_food_nodiff = [
            (['CPI(Food)', 'M2'], 60, 2, 'manhattan', 'distance', 3,  range(1,  5)),
            (['CPI(Food)', 'M2'], 60, 2, 'manhattan', 'distance', 6,  range(5,  9)),
            (['CPI(Food)', 'M2'], 60, 2, 'manhattan', 'distance', 12, range(9,  19)),
            (['CPI(Food)', 'IP'], 48, 2, 'manhattan', 'distance', 24, range(19, 31)),
            (['CPI(Food)', 'PPI'], 36, 2, 'euclidean', 'distance', 36, range(31, 43)),
            (['CPI(Food)', 'PPI', 'M2', 'IP', 'GS5'], 24, 2, 'euclidean', 'distance', 48, range(43, 55)),
            (['CPI(Food)', 'PPI', 'IP'], 12, 2, 'manhattan', 'distance', 60, range(55, 61)),
        ]

model_configs_food_diff = [
            (['CPI(Food)', 'UNRATE', 'IP'], 12, 20, 'euclidean', 'uniform', 3,  range(1,  5)),
            (['CPI(Food)', 'UNRATE', 'IP'], 6, 20, 'manhattan', 'distance', 6,  range(5,  9)),
            (['CPI(Food)', 'UNRATE'], 1, 12, 'euclidean', 'uniform', 12,  range(9,  19)),
            (['CPI(Food)', 'UNRATE', 'M2'], 60, 4, 'chebyshev', 'distance', 24,  range(19, 31)),
            (['CPI(Food)', 'M2', 'GS5'], 48, 4, 'chebyshev', 'distance', 36, range(31, 43)),
            (['CPI(Food)', 'M2', 'GS5'], 36, 4, 'chebyshev', 'distance', 48, range(43, 55)),
            (['CPI(Food)', 'GS5'], 36, 9, 'euclidean', 'distance', 60, range(55, 61)),
]
#model = kNN_Model(Data.dataset, 60, 0, "CPI(Food)")
#model.plot_kNN_results(8, model_configs=model_configs_food_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODkNNDIFF.png")

#model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
#model.plot_kNN_results(8, model_configs=model_configs_energy_diff, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFF.png")
model_configs_energy_nodiff_solo = [
            (['CPI(Energy)'], 120, 5, 3,  range(1,  5)),
            (['CPI(Energy)'], 120, 5, 6,  range(5,  9)),
            (['CPI(Energy)'], 60, 5, 12, range(9,  19)),
            (['CPI(Energy)'], 48, 5, 24, range(19, 31)),
            (['CPI(Energy)'], 36, 5,  36, range(31, 43)),
            (['CPI(Energy)'], 48, 5,  48, range(43, 55)),
            (['CPI(Energy)'], 24, 5,  60, range(55, 61)),
        ]

model_configs_energy_nodiff_solo2 = [
            (['CPI(Energy)'], 1, 9,  3,  range(1,  5)),
            (['CPI(Energy)'], 1, 20,  6,  range(5,  9)),
            (['CPI(Energy)'], 1, 20,  12, range(9,  19)),
            (['CPI(Energy)'], 3, 2, 24, range(19, 31)),
            (['CPI(Energy)'], 1, 20,  36, range(31, 43)),
            (['CPI(Energy)'], 1, 20,  48, range(43, 55)),
            (['CPI(Energy)'], 60, 4,  60, range(55, 61)),
        ]



model_configs_food_diff_solo = [
            (['CPI(Food)'], 12, 20, 3,  range(1,  5)),
            (['CPI(Food)'], 6, 20, 6,  range(5,  9)),
            (['CPI(Food)'], 1, 12, 12,  range(9,  19)),
            (['CPI(Food)'], 60, 4,  24,  range(19, 31)),
            (['CPI(Food)'], 48, 4,  36, range(31, 43)),
            (['CPI(Food)'], 36, 4,  48, range(43, 55)),
            (['CPI(Food)'], 36, 9,  60, range(55, 61)),
]

model = kNN_Model(Data.dataset, 60, 0, "CPI(Food)")
model.plot_kNN_results(8, model_configs=model_configs_food_diff_solo, diff_order=1, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODkNNDIFFSOLO.png")

model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_nodiff_solo2, diff_order=0, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFFSOLO.png")


model_configs_energy_nodiff = [
            (['CPI(Energy)', 'IP'], 1, 9, 'manhattan', 'uniform', 3,  range(1,  5)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'manhattan', 'uniform', 6,  range(5,  9)),
            (['CPI(Energy)', 'UNRATE', 'GS5'], 1, 20, 'manhattan', 'distance', 12, range(9,  19)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'IP', 'GS5'], 3, 2, 'euclidean', 'uniform', 24, range(19, 31)),
            (['CPI(Energy)', 'IP'], 1, 20, 'manhattan', 'uniform', 36, range(31, 43)),
            (['CPI(Energy)', 'UNRATE', 'IP'], 1, 20, 'euclidean', 'uniform', 48, range(43, 55)),
            (['CPI(Energy)', 'M2'], 60, 4, 'manhattan', 'distance', 60, range(55, 61)),
        ]


'''model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(8, model_configs=model_configs_energy_nodiff, diff_order=0, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFF.png")'''



'''model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")

print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 0))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 0))


model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=3, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=6, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=12, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=24, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=36, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=48, diff = 1))
model = kNN_Model(Data.dataset, 60, 80, "CPI(Core)")
print(model.cross_validation(Data=Data.dataset[['CPI(Core)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5']], 
                       k_folds = 5, lag_sizes = [1, 3, 6, 12, 24, 36, 48, 60], 
                       n_neighbors=[2, 4, 6, 9, 12, 20],metrics = ['euclidean', 'manhattan', 'chebyshev'], weights=['uniform', 'distance'], forecast_horizon=60, diff = 1))'''



model_configs_food_nodiff = [
            (['CPI(Core)', 'M2', 'IP'], 60, 2, 'chebyshev', 'distance', 3,  range(1,  5)),
            (['CPI(Core)', 'IP', 'GS5'], 60, 2, 'euclidean', 'distance', 6,  range(5,  9)),
            (['CPI(Core)', 'M2'], 60, 2, 'manhattan', 'distance', 12, range(9,  19)),
            (['CPI(Core)', 'M2'], 60, 2, 'manhattan', 'distance', 24, range(19, 31)),
            (['CPI(Core)', 'GS5'], 60, 2, 'euclidean', 'distance', 36, range(31, 43)),
            (['CPI(Core)', 'M2'], 60, 2, 'manhattan', 'distance', 48, range(43, 55)),
            (['CPI(Core)', 'PPI', 'IP'], 60, 2, 'euclidean', 'distance', 60, range(55, 61)),
        ]

model_configs_food_diff = [
            (['CPI(Core)', 'M2', 'GS5'], 12, 4, 'chebyshev', 'uniform', 3,  range(1,  5)),
            (['CPI(Core)', 'UNRATE', 'M2', 'GS5'], 12, 20, 'chebyshev', 'uniform', 6,  range(5,  9)),
            (['CPI(Core)', 'PPI', 'UNRATE', 'M2'], 6, 12, 'chebyshev', 'distance', 12,  range(9,  19)),
            (['CPI(Core)', 'IP'], 1, 6, 'euclidean', 'uniform', 24,  range(19, 31)),
            (['CPI(Core)', 'PPI', 'M2'], 1, 4, 'manhattan', 'uniform', 36, range(31, 43)),
            (['CPI(Core)', 'UNRATE', 'GS5'], 24, 6, 'euclidean', 'uniform', 48, range(43, 55)),
            (['CPI(Core)', 'PPI', 'UNRATE', 'IP'], 24, 4, 'chebyshev', 'uniform', 60, range(55, 61)),
]

'''
model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
window_data = model.get_windowed_data(
                        ['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'IP', 'GS5'], val_size=0, test_size=60,
                        lag_size=60, window_step=1,
                        forecast_horizon=60, diff_order=0
                    )
'''

'''model = kNN_Model(Data.dataset, 60, 0, "CPI(Energy)")
model.plot_kNN_results(7, model_configs=model_configs_energy_nodiff_solo, diff_order=0, file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYkNNNODIFFSOLO.png")'''

def cross_validate_lag_size(target):
    results = []
    for lag in [6, 12, 24, 36, 48, 60, 120]:
        horizons = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
        results_df, summary = cross_validation_rf(
        dataset               = Data.dataset,
        target_name           = target,
        features              = [target],
        lag_size              = lag,
        forecast_horizon      = 60,
        diff_order            = 1,
        n_estimators          = 5,
        forecast_horizon_range = [3, 6, 12, 24, 36, 48, 60],
        n_splits              = 5,
        test_size             = 60
        )
        for h in horizons.keys():
            horizons[h] = summary[h]["mean_rmse"]
        results.append(horizons)


    lags = [6, 12, 24, 36, 48, 60, 120]

    # Convert results list of dicts into DataFrame
    df_results = pd.DataFrame(results, index=lags)
    df_results.index.name = "Lag"

    # Plot RMSE vs forecast horizon for each lag
    #plt.figure(figsize=(12, 6), dpi=600)
    for horizon in df_results.columns:
        plt.plot(df_results.index, df_results[horizon], marker='o', label=f'Horizon {horizon}')

    plt.xlabel("Lag")
    plt.ylabel("RMSE")
    plt.title("RMSE for Different Forecast Horizons Across Lags")
    plt.legend()
    plt.grid(True)
    #plt.savefig("/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFLAGSELECTION.png", bbox_inches='tight')
    plt.show()

#cross_validate_lag_size('CPI(Food)')


'''
target     = 'CPI(Food)'
all_features = ['PPI', 'UNRATE', 'M2', 'IP', 'GS5', 'GDP']  # target removed

X = Data.dataset[all_features][:-60]
y = Data.dataset[target][:-60]

# ── Train RF ───────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

# ── 1. Bar chart: raw feature importance ───────────────────
importances = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False)


importances.plot(kind='bar', color='steelblue', edgecolor='black')

        
plt.title('Random Forest Feature Importances For CPI(Food)')
plt.ylabel('Importance Score')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


print("\nFeature Importances:")
print(importances.round(4))

# ── 2. SHAP — fix: flatten to 2D if needed ─────────────────
# Use raw dataset, not window_X_train (which is 3D)
X_2d = X.values   # shape: (n_samples, n_features) — correct for TreeExplainer

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_2d)   # shape: (n_samples, n_features)

# ── 3. SHAP summary plot (best overall view) ───────────────
# Shows importance + direction + distribution in one plot
shap.summary_plot(shap_values, X_2d, feature_names=all_features, show=False)

# Get the current figure and save it
plt.gcf().set_size_inches(12, 7)  # optional: adjust size

# ── 4. SHAP bar plot (mean absolute SHAP per feature) ──────
plt.figure()
shap.summary_plot(
    shap_values,
    X_2d,
    feature_names = all_features,
    plot_type     = 'bar',
    show          = True
)

# ── 5. SHAP dependence plots for top 3 features ────────────
# Shows how each feature value affects prediction
top3 = importances.head(3).index.tolist()

for feat in top3:
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(
        feat,
        shap_values,
        X_2d,
        feature_names = all_features,
        show          = True
    )
'''
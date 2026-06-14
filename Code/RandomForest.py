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
from sklearn.model_selection import TimeSeriesSplit  
from sklearn.ensemble import RandomForestRegressor

class WindowData:

    def __init__(self, dataset, test_size, lag_size, window_step, forecast_horizon, target_name, diff_order) -> None:
        self.dataset = dataset.copy()
        self.original_dataset = dataset.copy()
        self.test_size = test_size
        self.train_size = len(self.dataset)  - self.test_size
        self.lag_size = lag_size
        self.window_step = window_step
        self.forecast_horizon = forecast_horizon
        self.target_name = target_name
        self.diff_order = diff_order

    def _apply_differencing(self, df, order):
        self.diff_initial_values = []
        diffed = df.copy()
        for _ in range(order):
            self.diff_initial_values.append(diffed.iloc[0].copy())
            diffed = diffed.diff().dropna()
        return diffed

    def create_rolling_windows(self):
        
        if self.diff_order > 0:
            self.dataset = self._apply_differencing(self.dataset, self.diff_order)
            lost_rows = len(self.original_dataset) - len(self.dataset)
            self.train_size = self.train_size - lost_rows

        self.window_X_train = []
        self.window_y_train = []

        for iter in range(0, self.train_size - self.forecast_horizon - self.lag_size, self.window_step):
            self.window_X_train.append(self.dataset.iloc[iter : iter + self.lag_size, :].values.flatten())
            self.window_y_train.append(self.dataset[self.target_name].iloc[iter + self.lag_size + self.forecast_horizon - 1])


        self.window_X_test = []
        self.window_y_test = []
        iter = self.train_size - self.lag_size
        self.window_X_test.append(self.dataset.iloc[iter : iter + self.lag_size, :].values.flatten())
        self.window_y_test.append(self.dataset[self.target_name].iloc[iter + self.lag_size + self.forecast_horizon - 1])

        if self.diff_order > 0:
            test_target_idx_in_original = iter + self.lag_size + self.forecast_horizon - 1 + lost_rows
            self.last_original_value_before_test = self.original_dataset[self.target_name].iloc[test_target_idx_in_original - 1]

        self.window_X_train = np.array(self.window_X_train)
        self.window_y_train = np.array(self.window_y_train)
        self.window_X_test = np.array(self.window_X_test)
        self.window_y_test = np.array(self.window_y_test)


class RandomForest_Model:

    def __init__(self, dataset, test_size, target_name) -> None:
        self.dataset = dataset
        self.test_size = test_size
        self.target_name = target_name

    def fit_RandomForest(self, n_estimators, max_depth, X_train, y_train, X_test, y_test, 
                         window_data, previous_prediction=None, return_predictions=False,
                         test=True, cross_validation=False):
        

        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        rf_regressor.fit(X_train, y_train)

        y_pred = rf_regressor.predict(X_test)

        if test and window_data.diff_order > 0:
            last_val = window_data.last_original_value_before_test
            y_pred = y_pred + previous_prediction
            y_test = y_test + last_val

        if cross_validation:
            y_pred = y_pred
            y_test = y_test

        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        if return_predictions:
            return rmse, y_pred
        return rmse

    def get_windowed_data(self, features, test_size, lag_size, forecast_horizon, diff_order, window_step=1):
        window_data = WindowData(self.dataset[features], test_size=test_size,
                                 lag_size=lag_size, window_step=1, forecast_horizon=forecast_horizon,
                                 target_name=self.target_name, diff_order=diff_order)
        window_data.create_rolling_windows()
        return window_data

    def cross_validation(self, Data, k_folds, lag_sizes, n_estimators_list, max_depth_list,
                     min_samples_split_list, forecast_horizon, diff):

        time_series_split = TimeSeriesSplit(n_splits=k_folds)


        all_cols = list(Data.columns)
        features = [self.target_name] + [c for c in all_cols if c != self.target_name]

        results_dict = {
            "Lag_size": [], "n_estimators": [], "max_depth": [], "min_samples_split": [],
            "RMSE_mean": [], "RMSE_std": [], "validation_results": [], "score": []
        }

        for lag_size in lag_sizes:
            window_data = self.get_windowed_data(
                features, val_size=80, test_size=60,
                lag_size=lag_size, window_step=1,
                forecast_horizon=forecast_horizon, diff_order=diff
            )

            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    for min_samples_split in min_samples_split_list:

                        fold_rmses = []

                        for train_idx, val_idx in time_series_split.split(window_data.window_X_train):
                            X_train = window_data.window_X_train[train_idx]
                            y_train = window_data.window_y_train[train_idx]
                            X_val   = window_data.window_X_train[val_idx]
                            y_val   = window_data.window_y_train[val_idx]

                            fold_rmses.append(self.fit_RandomForest(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                X_train=X_train, y_train=y_train,
                                X_test=X_val,   y_test=y_val,
                                window_data=window_data,
                                return_predictions=False, test=False, cross_validation=True
                            ))

                        validation_results = self.fit_RandomForest(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            X_train=window_data.window_X_train,
                            y_train=window_data.window_y_train,
                            X_test=window_data.window_X_val,
                            y_test=window_data.window_y_val,
                            window_data=window_data,
                            return_predictions=False, test=False, cross_validation=True
                        )

                        results_dict["Lag_size"].append(lag_size)
                        results_dict["n_estimators"].append(n_estimators)
                        results_dict["max_depth"].append(max_depth)
                        results_dict["min_samples_split"].append(min_samples_split)
                        results_dict["RMSE_mean"].append(np.mean(fold_rmses))
                        results_dict["RMSE_std"].append(np.std(fold_rmses))
                        results_dict["validation_results"].append(validation_results)
                        results_dict["score"].append(
                            validation_results + np.mean(fold_rmses) + np.std(fold_rmses)
                        )

        results_df = pd.DataFrame(results_dict)
        best_row = results_df.sort_values(
            by=["validation_results", "RMSE_mean"], ascending=[True, True]
        ).iloc[0]

        print(f"################## MODEL FOR FORECAST HORIZON = {forecast_horizon} #####################")
        print("=" * 65)
        print(f"Best -> Lag: {int(best_row['Lag_size'])}, "
            f"n_estimators: {int(best_row['n_estimators'])}, "
            f"max_depth: {best_row['max_depth']}, "
            f"min_samples_split: {int(best_row['min_samples_split'])}, "
            f"RMSE: {best_row['RMSE_mean']:.4f} ± {best_row['RMSE_std']:.4f}, "
            f"Val RMSE: {best_row['validation_results']:.4f}, "
            f"Score: {best_row['score']:.4f}")
        print("=" * 65)

        cols = ["Lag_size", "n_estimators", "max_depth", "min_samples_split",
                "RMSE_mean", "RMSE_std", "validation_results", "score"]

        print("\nTop 20 by Score:")
        print(results_df.sort_values("score").head(20)[cols].to_string(index=False))

        print("\nTop 20 by CV RMSE:")
        print(results_df.sort_values(["RMSE_mean", "validation_results"]).head(20)[cols].to_string(index=False))

        return results_df
    
    def plot_RandomForest_results(self, number_of_steps, model_configs, diff_order, file_name):

        plt.style.use('default')
        plt.figure(figsize=(12, 7), dpi=600)
        result = []
        plt.plot(self.dataset['Date'], self.dataset[self.target_name].values,
                 label="Actual Data", color='red', linewidth=1)

        for step in range(1, number_of_steps):

            used_dataset = self.dataset.iloc[:-(60 * (step - 1))] if step > 1 else self.dataset
            previous_prediction = used_dataset[self.target_name].iloc[-61]
            all_forecasts = {}
            forecast_results = []
            errors = {3: [], 6: [], 12: [], 24: [], 36: [], 48: [], 60: []}

            for features, lag_size, n_estimators, max_depth, base_horizon, horizon_range in model_configs:

                for h in horizon_range:
                    rmse, y_pred = run_model_RF(
                        dataset=used_dataset,
                        features=features,
                        test_size=60,
                        lag_size=lag_size,
                        forecast_horizon=h,
                        diff_order=diff_order,
                        previous_prediction=previous_prediction,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        target_name=self.target_name
                    )
                    previous_prediction = y_pred
                    for ha in errors.keys():
                        if ha == h:
                            errors[h] = rmse

                    all_forecasts[h] = (rmse, y_pred)
                    forecast_results.append(y_pred)

            result.append(errors)
            time_slice = self.dataset['Date'][-60 * step: -60 * (step - 1) if step > 1 else None]
            plt.plot(time_slice, forecast_results, label="Forecasted values",
                     color='blue', linestyle='--', linewidth=2)
            plt.axvline(x=self.dataset['Date'].values[-60 * step], color='black', linestyle='--', alpha=0.7)

        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

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
        plt.title("Random Forest Model", fontsize=14, fontweight='bold')

        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        print(pd.DataFrame(result))


def run_model_RF(dataset, features, test_size, lag_size, forecast_horizon,
              diff_order, n_estimators, max_depth, previous_prediction, target_name):
    model = RandomForest_Model(dataset, test_size, target_name)
    window_data = model.get_windowed_data(
        features,  test_size=test_size,
        lag_size=lag_size, window_step=1,
        forecast_horizon=forecast_horizon, diff_order=diff_order
    )
    rmse, y_pred = model.fit_RandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        X_train=window_data.window_X_train,
        y_train=window_data.window_y_train,
        X_test=window_data.window_X_test,
        y_test=window_data.window_y_test,
        window_data=window_data,
        previous_prediction=previous_prediction,
        return_predictions=True
    )
    return rmse, y_pred


def cross_validation_rf(dataset, target_name, features, lag_size, forecast_horizon,
                         diff_order, n_estimators, max_depth, forecast_horizon_range,
                         n_splits=5, test_size=60):


    fold_results   = []  
    rmse_per_horizon = {h: [] for h in forecast_horizon_range}


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
                model = RandomForest_Model(
                    dataset     = fold_dataset,
                    test_size   = test_size,
                    target_name = target_name
                )
                window_data = model.get_windowed_data(
                    features         = features,
                    test_size        = test_size,
                    lag_size         = lag_size,
                    forecast_horizon = h,
                    diff_order       = diff_order
                )

                rmse, y_pred = model.fit_RandomForest(
                    n_estimators       = n_estimators,
                    max_depth          = max_depth,
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


        rmse_str = '  '.join([f"h{h}={fold_errors[h]:.3f}" 
                               if fold_errors[h] is not None else f"h{h}=N/A"
                               for h in forecast_horizon_range])
        print(f"Fold {fold_idx+1}/{len(folds)} | {rmse_str}")

    print(f"{'─'*55}")


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


def cross_validation_rf_predifined_parameters(dataset, target_name, features, lag_size, forecast_horizon,
                         diff_order, n_estimators, max_depth, forecast_horizon_range, predefined_parameters,
                         n_splits=5, test_size=60):


    fold_results   = []
    rmse_per_horizon = {h: [] for h in forecast_horizon_range}


    total_size  = len(dataset)
    min_train   = lag_size + forecast_horizon + diff_order + 10


    folds = []
    for fold in range(n_splits):
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

        for h, lag_size in predefined_parameters:
            try:
                model = RandomForest_Model(
                    dataset     = fold_dataset,
                    test_size   = test_size,
                    target_name = target_name
                )
                window_data = model.get_windowed_data(
                    features         = features,
                    test_size        = test_size,
                    lag_size         = lag_size,
                    forecast_horizon = h,
                    diff_order       = diff_order
                )

                rmse, y_pred = model.fit_RandomForest(
                    n_estimators       = n_estimators,
                    max_depth          = max_depth,
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


        rmse_str = '  '.join([f"h{h}={fold_errors[h]:.3f}" 
                               if fold_errors[h] is not None else f"h{h}=N/A"
                               for h in forecast_horizon_range])
        print(f"Fold {fold_idx+1}/{len(folds)} | {rmse_str}")

    print(f"{'─'*55}")


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


def cross_validate_lag_size(target):

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

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
        n_estimators          = 40,
        max_depth             = None,
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


def cross_validate_diff(target):

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    results = []
    for diff in [0, 1]:
        horizons = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
        results_df, summary = cross_validation_rf(
        dataset               = Data.dataset,
        target_name           = target,
        features              = [target, 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],
        lag_size              = 120,
        forecast_horizon      = 60,
        diff_order            = diff,
        n_estimators          = 40,
        max_depth             = None,
        forecast_horizon_range = [3, 6, 12, 24, 36, 48, 60],
        n_splits              = 5,
        test_size             = 60
        )
        for h in horizons.keys():
            horizons[h] = summary[h]["mean_rmse"]
        results.append(horizons)


    diff = [0, 1]

    # Convert results list of dicts into DataFrame
    df_results = pd.DataFrame(results, index=diff)
    df_results.index.name = "Diff"

    # Plot RMSE vs forecast horizon for each lag
    #plt.figure(figsize=(12, 6), dpi=600)
    for horizon in df_results.columns:
        plt.plot(df_results.index, df_results[horizon], marker='o', label=f'Horizon {horizon}')

    plt.xlabel("Diff or non Diff")
    plt.ylabel("RMSE")
    plt.title("RMSE for Diff non Diff Data Across Lags")
    plt.legend()
    plt.grid(True)
    #plt.savefig("/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFDIFFSELECTION.png", bbox_inches='tight')
    plt.show()


def cross_validate_max_depth(target, predefined_parameters):

    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    results = []
    for max_depth in [1, 5, 10, 40]:
        horizons = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
        results_df, summary = cross_validation_rf_predifined_parameters(
        dataset               = Data.dataset,
        target_name           = target,
        features              = [target, 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],
        lag_size              = 120,
        forecast_horizon      = 60,
        diff_order            = 0,
        n_estimators          = 40,
        max_depth             = max_depth,
        forecast_horizon_range = [3, 6, 12, 24, 36, 48, 60],
        predefined_parameters = [(3, 6),
                              (6, 6),
                              (12, 12),
                              (24, 6),
                              (36, 48),
                              (48, 48),
                              (60, 48)],
        n_splits              = 5,
        test_size             = 60
        )
        for h in horizons.keys():
            horizons[h] = summary[h]["mean_rmse"]
        results.append(horizons)


    max_depth = [1, 5, 10, 40]

    # Convert results list of dicts into DataFrame
    df_results = pd.DataFrame(results, index=max_depth)
    df_results.index.name = "Diff"

    # Plot RMSE vs forecast horizon for each lag
    #plt.figure(figsize=(12, 6), dpi=600)
    for horizon in df_results.columns:
        plt.plot(df_results.index, df_results[horizon], marker='o', label=f'Horizon {horizon}')

    plt.xlabel("Max Depth")
    plt.ylabel("RMSE")
    plt.title("RMSE for different max depth")
    plt.legend()
    plt.grid(True)
    #plt.savefig("/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFMAXDEPTHSELECTION.png", bbox_inches='tight')
    plt.show()


def cross_validate_n_estimators(target, predefined_parameters):
    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    results = []
    for estimator_num in [1, 5, 10, 40, 100]:
        horizons = {3:[], 6:[], 12:[], 24:[], 36:[], 48:[], 60:[]}
        results_df, summary = cross_validation_rf_predifined_parameters(
        dataset               = Data.dataset,
        target_name           = target,
        features              = [target, 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],
        lag_size              = 120,
        forecast_horizon      = 60,
        diff_order            = 0,
        n_estimators          = estimator_num,
        max_depth             = 40,
        forecast_horizon_range = [3, 6, 12, 24, 36, 48, 60],
        predefined_parameters = [(3, 120),
                              (6, 120),
                              (12, 120),
                              (24, 120),
                              (36, 24),
                              (48, 6),
                              (60, 60)],
        n_splits              = 5,
        test_size             = 60
        )
        for h in horizons.keys():
            horizons[h] = summary[h]["mean_rmse"]
        results.append(horizons)


    estimator_num = [1, 5, 10, 40, 100]

    # Convert results list of dicts into DataFrame
    df_results = pd.DataFrame(results, index=estimator_num)
    df_results.index.name = "Diff"

    # Plot RMSE vs forecast horizon for each lag
    #plt.figure(figsize=(12, 6), dpi=600)
    for horizon in df_results.columns:
        plt.plot(df_results.index, df_results[horizon], marker='o', label=f'Horizon {horizon}')

    plt.xlabel("Number of Estimators")
    plt.ylabel("RMSE")
    plt.title("RMSE for different number of estimators")
    plt.legend()
    plt.grid(True)
    #plt.savefig("/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFNUMESTIMATORSSELECTION.png", bbox_inches='tight')
    plt.show()


def RF_energy():
    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')


    # MULTIVARIATE VERSION

    cross_validate_lag_size("CPI(Energy)")
    cross_validate_diff("CPI(Energy)")
    cross_validate_max_depth("CPI(Energy)", [(3, 120),
                              (6, 120),
                              (12, 120),
                              (24, 120),
                              (36, 24),
                              (48, 6),
                              (60, 60)])
    cross_validate_n_estimators("CPI(Energy)",[(3, 120),
                              (6, 120),
                              (12, 120),
                              (24, 120),
                              (36, 24),
                              (48, 6),
                              (60, 60)])
    
    model_configs_energy_mul = [
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],      120, 20, 10 ,  3,  range(1,  5)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           120, 20 ,10 , 6,  range(5,  9)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],              120, 20, 10,  12, range(9,  19)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],                    120,  20,10 ,  24, range(19, 31)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],        24,  20,10 ,  36, range(31, 43)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],                 6,  20,  10 ,48, range(43, 55)),
            (['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],                 60,  20, 10 ,60, range(55, 61)),
        ]
    

    rf = RandomForest_Model(Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5', 'Date']], 60, 'CPI(Energy)')
    rf.plot_RandomForest_results(7, model_configs_energy_mul, 0, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFRESULTS100.png")

    # Univariate VERSION

    cross_validate_lag_size("CPI(Energy)")
    cross_validate_diff("CPI(Energy)")
    cross_validate_max_depth("CPI(Energy)", [(3, 3),
                              (6, 6),
                              (12, 60),
                              (24, 60),
                              (36, 60),
                              (48, 48),
                              (60, 24)])
    cross_validate_n_estimators("CPI(Energy)",[(3, 3),
                              (6, 6),
                              (12, 60),
                              (24, 60),
                              (36, 60),
                              (48, 48),
                              (60, 24)])
    
    model_configs_energy_uni = [
            (['CPI(Energy)'],           3, 20, 20 ,  3,  range(1,  5)),
            (['CPI(Energy)'],           6, 20 ,20 , 6,  range(5,  9)),
            (['CPI(Energy)'],           60, 20, 20,  12, range(9,  19)),
            (['CPI(Energy)'],           60,  20,20 ,  24, range(19, 31)),
            (['CPI(Energy)'],           60,  20,20 ,  36, range(31, 43)),
            (['CPI(Energy)'],           48,  20,  20 ,48, range(43, 55)),
            (['CPI(Energy)'],           24,  20, 20 ,60, range(55, 61)),
        ]
    

    rf = RandomForest_Model(Data.dataset[['CPI(Energy)', 'Date']], 60, 'CPI(Energy)')
    rf.plot_RandomForest_results(7, model_configs_energy_uni, 0, "/home/fedor/Downloads/ImagesOverleaf/CPIENERGYRFRESULTSSOLO.png")


def RF_food():
    #0 path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')


    # MULTIVARIATE VERSION

    cross_validate_lag_size("CPI(Food)")
    cross_validate_diff("CPI(Food)")
    cross_validate_max_depth("CPI(Food)", [(3, 6),
                              (6, 6),
                              (12, 12),
                              (24, 6),
                              (36, 3),
                              (48, 3),
                              (60, 3)])
    cross_validate_n_estimators("CPI(Food)",[(3, 6),
                              (6, 6),
                              (12, 12),
                              (24, 6),
                              (36, 3),
                              (48, 3),
                              (60, 3)])
    
    model_configs_food_mul = [
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40, 20 ,  3,  range(1,  5)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6, 40 ,20 , 6,  range(5,  9)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           12, 40, 20,  12, range(9,  19)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           6,  40,20 ,  24, range(19, 31)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,20 ,  36, range(31, 43)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40,  20 ,48, range(43, 55)),
            (['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5'],           3,  40, 20 ,60, range(55, 61)),
        ]
    

    rf = RandomForest_Model(Data.dataset[['CPI(Food)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5', 'Date']], 60, 'CPI(Food)')
    rf.plot_RandomForest_results(7, model_configs_food_mul, 0, "/home/fedor/Downloads/ImagesOverleaf/CPIFOODRFRESULTS100.png")

    # Univariate VERSION

    cross_validate_lag_size("CPI(Food)")
    cross_validate_diff("CPI(Food)")
    cross_validate_max_depth("CPI(Food)", [(3, 24),
                              (6, 24),
                              (12, 12),
                              (24, 24),
                              (36, 3),
                              (48, 3),
                              (60, 3)])
    cross_validate_n_estimators("CPI(Food)",[(3, 24),
                              (6, 24),
                              (12, 12),
                              (24, 24),
                              (36, 3),
                              (48, 3),
                              (60, 3)])
    
    model_configs_food_solo = [
            (['CPI(Food)'],           24, 40, 20 ,  3,  range(1,  5)),
            (['CPI(Food)'],           24, 40 ,20 , 6,  range(5,  9)),
            (['CPI(Food)'],           12, 40, 20,  12, range(9,  19)),
            (['CPI(Food)'],           24,  40, 20 ,  24, range(19, 31)),
            (['CPI(Food)'],           3,  40, 20 ,  36, range(31, 43)),
            (['CPI(Food)'],           3,  40,  20 ,48, range(43, 55)),
            (['CPI(Food)'],           3,  40, 20 ,60, range(55, 61)),
        ]
    

    rf = RandomForest_Model(Data.dataset[['CPI(Food)', 'Date']], 60, 'CPI(Food)')
    rf.plot_RandomForest_results(7, model_configs_food_solo, 0, "/home/fedor/Downloads/ImagesOverleaf/CPIFoodRFRESULTS100SOLO.png")
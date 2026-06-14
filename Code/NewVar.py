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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class VAR_Data:

    def __init__(self, dataset, date) -> None:
        self.dataset = dataset
        self.diff_dataset = dataset.copy()
        self.diff_order = 0
        self.date = date

    def check_stability(self, order, testing_size, features):
        training_data = self.diff_dataset[:-testing_size][features]
        model = VAR(training_data)
        results = model.fit(order)
        return results.is_stable()
    
    def residual_correlation(self, order, testing_size, features, nlags):
        training_data = self.diff_dataset[:-testing_size][features]
        model = VAR(training_data)
        results = model.fit(order)

        return results.test_whiteness(nlags=nlags)
    
    def residual_normality(self, order, testing_size, features):
        training_data = self.diff_dataset[:-testing_size][features]
        model = VAR(training_data)
        results = model.fit(order)

        return results.test_normality()


    def check_stationarity(self):

        for col in self.dataset.columns:
            self.plot_test_stationarity(self.dataset[col], col)
             
    def plot_test_stationarity(self, data, col_name):
           
            plt.style.use('default')  
            
            data_diff = data.diff().dropna()

            
            print(adfuller(data))
            print(adfuller(data_diff))

                        
            plt.plot(data, label="Data", color='red', linewidth=1)
            plt.plot(data_diff, label="Differenced Data", color='blue', linewidth=1)
            
            # Grid
            plt.grid(
                True,
                linestyle='--',
                linewidth=0.7,
                alpha=0.6,
                color='gray'
            )

            plt.legend()
            plt.xlabel("Time", fontsize=12)
            plt.ylabel(f"{col_name}", fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(fontsize=10)
            plt.title(f"{col_name} over time", fontsize=14, fontweight='bold')
            plt.show()
    
    def differentiate(self, diff_order):
        '''
        Apply differencing for the dataset
        '''

        self.diff_order = diff_order
        for iter in range(0, diff_order):
            self.diff_dataset = self.diff_dataset.diff().dropna()
        

    def fit_VAR(self, order, forecast_steps, features, testing_size, target_column, is_actuals = False):
        
        training_data = self.diff_dataset[:-testing_size][features]
        model = VAR(training_data)
        results = model.fit(order)

        lagged_values = training_data.values[-order:]

        diff_forecast = results.forecast(y=lagged_values, steps=forecast_steps)

        target_idx = list(training_data.columns).index(target_column)
        target_diff_forecast = diff_forecast[:, target_idx]

        forecast_values = target_diff_forecast.copy()

        for d in range(self.diff_order, 0, -1):

            temp = self.dataset[target_column].copy()

            for i in range(d - 1):
                temp = temp.diff().dropna()

            last_known = temp.iloc[-(testing_size + 1)]
            forecast_values = np.concatenate([[last_known], forecast_values]).cumsum()[1:]

        subset_date = self.date.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.date.iloc[-forecast_steps:]

        forecast_series = pd.Series(forecast_values, index=subset_date, name=f"{target_column}_forecast")

        subset_data = self.dataset.iloc[-testing_size : -testing_size + forecast_steps] if testing_size > forecast_steps else self.dataset.iloc[-forecast_steps:]

        actuals = subset_data[target_column]

        mae  = np.mean(np.abs(forecast_series.values - actuals.values))
        rmse = np.sqrt(np.mean((forecast_series.values - actuals.values) ** 2))
        mape = np.mean(np.abs((forecast_series.values - actuals.values) / actuals.values)) * 100
        
        #print(f"\nForecast Accuracy ({target_column}):")
        #print(f"order = {order}, forecast steps = {forecast_steps}, features = {features}")
        #print(f"  MAE  : {mae:.4f}")
        #print(f"  RMSE : {rmse:.4f}")
        #print(f"  MAPE : {mape:.4f}%")

        if is_actuals:
            return forecast_series, actuals

        return forecast_series


    def VAR_horizons_training(self, order, max_testing_size, features, target_variable):
        
        result = []
        for test_size in range(max_testing_size, 59, -1 ):
            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}

            forecast_results, actuals = self.fit_VAR(order, 60, features, test_size, target_variable, True)
            for h in errors.keys():
                errors[h] = np.sqrt(np.mean((forecast_results.values[:h] - actuals.values[:h]) ** 2))
            result.append(errors)
        return pd.DataFrame(result)

    def plot_VECM_results(self, number_of_steps, order, features, target_variable, coint_rank=None):
        """
        Fit VECM and plot rolling 60-month forecasts.

        Parameters:
            number_of_steps : number of 60-month rolling windows to plot
            order           : lag order (in differences, same as VAR order - 1)
            features        : list of columns to include, target first
            target_variable : column to plot and evaluate
            coint_rank      : cointegration rank. If None, estimated automatically.
        """
        plt.style.use('default')
        result = []
        plt.plot(self.date, self.dataset[target_variable].values, 
                label="Actual Data", color='red', linewidth=1)

        for step in range(1, number_of_steps):

            testing_size  = 60 * step
            forecast_steps = 60

            # VECM uses levels, not differenced data
            training_data = self.dataset.iloc[:-testing_size][features]

            # Auto-select cointegration rank if not provided
            if coint_rank is None:
                rank_test  = select_coint_rank(training_data, det_order=0, k_ar_diff=order)
                rank       = rank_test.rank
                print(f"Step {step}: estimated cointegration rank = {rank}")
            else:
                rank = coint_rank

            if rank == 0:
                print(f"Step {step}: rank=0, no cointegration found — skipping window")
                continue

            try:
                model   = VECM(training_data, k_ar_diff=order, coint_rank=rank, deterministic="co")
                results = model.fit()

                # Forecast is in levels — no undifferencing needed
                diff_forecast = results.predict(steps=forecast_steps)

                target_idx     = features.index(target_variable)
                forecast_values = diff_forecast[:, target_idx]

                # Date and actuals slices
                if -testing_size + forecast_steps != 0:
                    time_slice = self.date.iloc[-testing_size : -testing_size + forecast_steps]
                    actuals    = self.dataset[target_variable].iloc[-testing_size : -testing_size + forecast_steps]
                else:
                    time_slice = self.date.iloc[-testing_size :]
                    actuals    = self.dataset[target_variable].iloc[-testing_size :]

                

                forecast_series = pd.Series(forecast_values, index=time_slice.values, 
                                            name=f"{target_variable}_forecast")

                # Compute errors at each horizon
                errors = {}
                for h in [3, 6, 12, 24, 36, 48, 60]:
                    errors[h] = np.sqrt(np.mean(
                        (forecast_series.values[:h] - actuals.values[:h]) ** 2
                    ))
                result.append(errors)

                print(f"Step {step} Forecast:\n{forecast_series}")
                print(f"Step {step} Actuals:\n{actuals.values}")

                plt.plot(
                    time_slice,
                    forecast_series,
                    label="Forecasted values",
                    color='blue',
                    linestyle='--',
                    linewidth=2
                )
                plt.axvline(x=self.date.values[-testing_size], color='black', 
                        linestyle='--', alpha=0.7)

            except Exception as e:
                print(f"Step {step} failed: {e}")
                continue

        # Grid and formatting
        plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

        custom_lines = [
            Line2D([0], [0], color='red', lw=1),
            Line2D([0], [0], color='blue', lw=2, linestyle='--'),
            Line2D([0], [0], color='black', lw=1, linestyle='--', alpha=0.7)
        ]
        plt.legend(custom_lines, ['Actual Data', 'Forecasted values', 'Step boundaries'])
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(target_variable, fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=10)
        plt.title(f"VECM(rank={coint_rank}, k_ar_diff={order})", fontsize=14, fontweight='bold')
        plt.show()

        print(pd.DataFrame(result))

    def select_order(self, target_variable, max_order=10, testing_size=60, max_vars=None, max_diff=2):
        """
        Search over all subsets of variables, lag orders, and differencing orders
        to find the best VAR specification by AIC and BIC.

        Parameters:
            target_variable : column that must appear in every subset
            max_order       : maximum lag order to test
            testing_size    : number of observations held out
            max_vars        : cap subset size. None = all subsets.
            max_diff        : maximum differencing order to test (default 2)
        """
        original_diff_order   = self.diff_order
        original_diff_dataset = self.diff_dataset.copy()

        all_cols  = list(self.dataset.columns)
        other_cols = [c for c in all_cols if c != target_variable]

        if max_vars is None:
            max_vars = len(all_cols)

        max_others = max_vars - 1
        results_dict = {"Diff_Order": [], "Variables": [], "Order": [], "AIC": [], "BIC": []}

        for diff_order in range(1, max_diff + 1):

            # Re-apply differencing from scratch each time
            self.diff_dataset = self.dataset.copy()
            self.diff_order   = diff_order
            for _ in range(diff_order):
                self.diff_dataset = self.diff_dataset.diff().dropna()

            training_data = self.diff_dataset.copy()

            for subset_size in range(1, max_others + 1):
                for subset in combinations(other_cols, subset_size):

                    features    = [target_variable] + list(subset)
                    subset_data = training_data[features]

                    for order in range(1, max_order + 1):
                        try:
                            model  = VAR(subset_data)
                            result = model.fit(order)
                            results_dict["Diff_Order"].append(diff_order)
                            results_dict["Variables"].append(tuple(features))
                            results_dict["Order"].append(order)
                            results_dict["AIC"].append(result.aic)
                            results_dict["BIC"].append(result.bic)
                        except Exception as e:
                            print(f"Failed for diff={diff_order}, {features}, order={order}: {e}")
                            continue

        results_df = pd.DataFrame(results_dict)
        #print(results_df[results_df["Variables"].apply(lambda x: x == ('CPI(Food)', 'M2', 'IP', 'GS5', 'PPI', 'GDP'))])

        # Best by AIC and BIC
        best_aic_row = results_df.loc[results_df["AIC"].idxmin()]
        best_bic_row = results_df.loc[results_df["BIC"].idxmin()]

        print("=" * 65)
        print(f"Best by AIC -> Diff: {int(best_aic_row['Diff_Order'])}, Variables: {list(best_aic_row['Variables'])}, Order: {int(best_aic_row['Order'])}, AIC: {best_aic_row['AIC']:.4f}")
        print(f"Best by BIC -> Diff: {int(best_bic_row['Diff_Order'])}, Variables: {list(best_bic_row['Variables'])}, Order: {int(best_bic_row['Order'])}, BIC: {best_bic_row['BIC']:.4f}")
        print("=" * 65)

        print("\nTop 10 by AIC:")
        print(results_df.nsmallest(10, "AIC")[["Diff_Order", "Variables", "Order", "AIC", "BIC"]].to_string(index=False))
        print("\nTop 10 by BIC:")
        print(results_df.nsmallest(10, "BIC")[["Diff_Order", "Variables", "Order", "AIC", "BIC"]].to_string(index=False))

        # Plot: one row per diff order, columns = AIC / BIC
        fig, axes = plt.subplots(max_diff, 2, figsize=(14, 5 * max_diff), squeeze=False)
        results_df["n_vars"] = results_df["Variables"].apply(len)

        for d in range(1, max_diff + 1):
            diff_df = results_df[results_df["Diff_Order"] == d]
            for n in sorted(diff_df["n_vars"].unique()):
                subset_df = diff_df[diff_df["n_vars"] == n]
                grouped   = subset_df.groupby("Order")[["AIC", "BIC"]].min()
                axes[d-1][0].plot(grouped.index, grouped["AIC"], marker='o', label=f"{n} vars")
                axes[d-1][1].plot(grouped.index, grouped["BIC"], marker='s', label=f"{n} vars")

            for ax, metric in zip(axes[d-1], ["AIC", "BIC"]):
                ax.set_xlabel("Lag Order", fontsize=12)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_title(f"Diff={d} — Best {metric} per Order by Subset Size", fontsize=13, fontweight='bold')
                ax.legend(title="Subset size")
                ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

        plt.tight_layout()
        plt.show()

        # Restore original state
        self.diff_order   = original_diff_order
        self.diff_dataset = original_diff_dataset

        return best_aic_row, best_bic_row, results_df

    def select_order_rmse(self, target_variable, max_order=10, testing_size=60, max_vars=None, max_diff=1, n_windows=8):
        """
        Search over all subsets of variables, lag orders, and differencing orders
        to find the best VAR specification by out-of-sample RMSE.
        Uses rolling windows of 60-month forecasts to get stable RMSE estimates.

        Parameters:
            target_variable : column that must appear in every subset
            max_order       : maximum lag order to test
            testing_size    : fixed forecast horizon (60 months)
            max_vars        : cap subset size. None = all subsets.
            max_diff        : maximum differencing order to test
            n_windows       : number of rolling windows to average RMSE over
        """
        original_diff_order   = self.diff_order
        original_diff_dataset = self.diff_dataset.copy()

        all_cols   = list(self.dataset.columns)
        other_cols = [c for c in all_cols if c != target_variable]

        if max_vars is None:
            max_vars = len(all_cols)

        max_others = max_vars - 1
        results_dict = {"Diff_Order": [], "Variables": [], "Order": [], "RMSE_mean": [], "RMSE_std": []}

        for diff_order in range(1, max_diff + 1):

            # Re-apply differencing from scratch each time
            self.diff_dataset = self.dataset.copy()
            self.diff_order   = diff_order
            for _ in range(diff_order):
                self.diff_dataset = self.diff_dataset.diff().dropna()

            for subset_size in range(1, max_others + 1):
                for subset in combinations(other_cols, subset_size):

                    features = [target_variable] + list(subset)

                    for order in range(1, max_order + 1):

                        window_rmses = []

                        for w in range(2, n_windows + 1):
                            window_test_size = testing_size * w

                            # Make sure we have enough training data
                            min_train = order + diff_order + 10
                            if len(self.diff_dataset) - window_test_size < min_train:
                                continue

                            try:
                                forecast, actuals = self.fit_VAR(
                                    order          = order,
                                    forecast_steps = testing_size,
                                    features       = features,
                                    testing_size   = window_test_size,
                                    target_column  = target_variable,
                                    is_actuals     = True
                                )
                                rmse = np.sqrt(np.mean((forecast.values - actuals.values) ** 2))
                                window_rmses.append(rmse)

                            except Exception as e:
                                print(f"Failed for diff={diff_order}, {features}, order={order}, window={w}: {e}")
                                continue

                        if len(window_rmses) == 0:
                            continue

                        results_dict["Diff_Order"].append(diff_order)
                        results_dict["Variables"].append(tuple(features))
                        results_dict["Order"].append(order)
                        results_dict["RMSE_mean"].append(np.mean(window_rmses))
                        results_dict["RMSE_std"].append(np.std(window_rmses))

        results_df = pd.DataFrame(results_dict)

        # Restore original state before any further operations
        self.diff_order   = original_diff_order
        self.diff_dataset = original_diff_dataset

        if results_df.empty:
            raise RuntimeError("No models were successfully evaluated.")

        best_rmse_row = results_df.loc[results_df["RMSE_mean"].idxmin()]

        print("=" * 65)
        print(f"Best by RMSE -> Diff: {int(best_rmse_row['Diff_Order'])}, "
            f"Variables: {list(best_rmse_row['Variables'])}, "
            f"Order: {int(best_rmse_row['Order'])}, "
            f"RMSE: {best_rmse_row['RMSE_mean']:.4f} ± {best_rmse_row['RMSE_std']:.4f}")
        print("=" * 65)

        print("\nTop 10 by RMSE:")
        print(results_df.nsmallest(10, "RMSE_mean")[
            ["Diff_Order", "Variables", "Order", "RMSE_mean", "RMSE_std"]
        ].to_string(index=False))

        # Plot: one row per diff order
        fig, axes = plt.subplots(max_diff, 1, figsize=(14, 5 * max_diff), squeeze=False)
        results_df["n_vars"] = results_df["Variables"].apply(len)

        for d in range(1, max_diff + 1):
            diff_df = results_df[results_df["Diff_Order"] == d]
            for n in sorted(diff_df["n_vars"].unique()):
                subset_df = diff_df[diff_df["n_vars"] == n]
                grouped   = subset_df.groupby("Order")["RMSE_mean"].min()
                axes[d-1][0].plot(grouped.index, grouped.values, marker='o', label=f"{n} vars")

            axes[d-1][0].set_xlabel("Lag Order", fontsize=12)
            axes[d-1][0].set_ylabel("RMSE (mean over windows)", fontsize=12)
            axes[d-1][0].set_title(f"Diff={d} — Best RMSE per Order by Subset Size", fontsize=13, fontweight='bold')
            axes[d-1][0].legend(title="Subset size")
            axes[d-1][0].grid(True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

        plt.tight_layout()
        plt.show()

        return best_rmse_row, results_df


# path to the file with original dataset
filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
# create the instance that will be used to store data and process it 
Data = Dataset(filepath, sep=';')


# create class instance
cpifood = VAR_Data(Data.dataset.drop(["CPI(Core)", "CPI(Energy)", "Date"], axis=1),Data.dataset["Date"])


# differentiate data
cpifood.differentiate(1)

rank_test = select_coint_rank(
    cpifood.dataset[['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'PPI', 'M2']].iloc[:-60],
    det_order=0, 
    k_ar_diff=7
)
print(rank_test.summary())

# Then plot
cpifood.plot_VECM_results(
    number_of_steps = 9,
    order           = 7,      # k_ar_diff = VAR order - 1
    features        = ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'PPI', 'M2'],
    target_variable = "CPI(Food)",
    coint_rank      = rank_test.rank
)


''' CPI(Food)



#select best models using AIC BIC
best_aic_row, best_bic_row, results_df = cpifood.select_order(target_variable="CPI(Food)")
#select best model using RMSE without data leakage
best_row, results_df = cpifood.select_order_rmse(target_variable="CPI(Food)")

# check stability for selected models
print(cpifood.check_stability(2, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
print(cpifood.check_stability(3, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
print(cpifood.check_stability(5, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))

# check normality and serial correlation for each variable
print(cpifood.residual_correlation(2, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 3))
print(cpifood.residual_normality(2, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))

print(cpifood.residual_correlation(3, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 4))
print(cpifood.residual_normality(3, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))

print(cpifood.residual_correlation(5, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 6))
print(cpifood.residual_normality(5, 60,['CPI(Food)', 'UNRATE', 'IP', 'GS5']))

# see the performarce of each model in a rolling window of 6 months predictions
print(cpifood.VAR_horizons_training(2, 80, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], "CPI(Food)"))
print(cpifood.VAR_horizons_training(3, 80, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], "CPI(Food)"))
print(cpifood.VAR_horizons_training(5, 80, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], "CPI(Food)"))

'''



''' CPI(Core)

# create class instance
cpicore = VAR_Data(Data.dataset.drop(["CPI(Energy)", "CPI(Food)", "Date"], axis=1),Data.dataset["Date"])

# differentiate data
cpicore.differentiate(1)

#select best models using AIC BIC
best_aic_row, best_bic_row, results_df = cpicore.select_order(target_variable="CPI(Core)")

# check stability for selected models
print(cpicore.check_stability(2, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))
print(cpicore.check_stability(3, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))
print(cpicore.check_stability(8, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))

# check normality and serial correlation for each variable
print(cpicore.residual_correlation(2, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5'], 3))
print(cpicore.residual_normality(2, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))

print(cpicore.residual_correlation(3, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5'], 4))
print(cpicore.residual_normality(3, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))

print(cpicore.residual_correlation(8, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5'], 9))
print(cpicore.residual_normality(8, 60,['CPI(Core)', 'UNRATE', 'IP', 'GS5']))

# see the performarce of each model in a rolling window of 6 months predictions
print(cpicore.VAR_horizons_training(2, 80, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)"))
print(cpicore.VAR_horizons_training(3, 80, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)"))
print(cpicore.VAR_horizons_training(8, 80, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)"))

# plot forecasting results for each of the models
cpicore.plot_VAR_results(9, 2, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)")
cpicore.plot_VAR_results(9, 3, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)")
cpicore.plot_VAR_results(9, 8, ['CPI(Core)', 'UNRATE', 'IP', 'GS5'], "CPI(Core)")
'''

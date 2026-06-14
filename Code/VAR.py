from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from statsmodels.tsa.api import VAR
from itertools import combinations
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class VAR_Data:
    
    def __init__(self, dataset, date) -> None:
        self.dataset = dataset
        self.diff_dataset = dataset.copy()
        self.diff_order = 0
        self.date = date

    def plot_residual_diagnostics(self, order, testing_size, features, target_column, file_name=''):

        training_data = self.diff_dataset[:-testing_size][features]
        model = VAR(training_data)
        results = model.fit(order)
        residuals = results.resid[target_column]

        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
        fig.suptitle(
            f"Residual Diagnostics — VAR({order})  |  Target: {target_column}",
            fontsize=14, fontweight='bold', y=1.01
        )

        grid_kw = dict(linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

        # Residuals over time
        ax = axes[0, 0]
        train_dates = self.date.iloc[:len(training_data)]
        resid_dates = train_dates.iloc[order:]          # VAR loses first `order` rows
        ax.plot(resid_dates, residuals.values, color='steelblue', linewidth=0.9)
        ax.axhline(0, color='red', linewidth=1, linestyle='--')
        ax.set_title("Residuals over Time", fontweight='bold')
        ax.set_xlabel("Time")
        ax.set_ylabel("Residual")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, **grid_kw)

        # Histogram
        ax = axes[0, 1]
        ax.hist(residuals, bins=30, density=True,
                color='steelblue', edgecolor='white', alpha=0.7, label='Histogram')
        kde_x = np.linspace(residuals.min(), residuals.max(), 200)
        kde   = stats.gaussian_kde(residuals)
        ax.plot(kde_x, kde(kde_x), color='black', linewidth=1.5, label='KDE')
        mu, sigma = residuals.mean(), residuals.std()
        ax.plot(kde_x, stats.norm.pdf(kde_x, mu, sigma),
                color='red', linewidth=1.5, linestyle='--', label='Normal fit')
        ax.set_title("Histogram + KDE", fontweight='bold')
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, **grid_kw)

        # Q-Q plot
        ax = axes[1, 0]
        (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, color='steelblue', s=12, alpha=0.7, label='Sample quantiles')
        ref_line = np.array([min(osm), max(osm)])
        ax.plot(ref_line, slope * ref_line + intercept,
                color='red', linewidth=1.5, linestyle='--', label='Normal reference')
        ax.set_title("Q-Q Plot", fontweight='bold')
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=9)
        ax.grid(True, **grid_kw)

        # ACF of residuals
        ax = axes[1, 1]
        plot_acf(residuals, lags=30, ax=ax, color='steelblue',
                title='', zero=False)
        ax.set_title("ACF of Residuals", fontweight='bold')
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, **grid_kw)

        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, bbox_inches='tight')

        plt.show()

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
        
    def fit_VAR(self, order, forecast_steps, features, testing_size, target_column, is_actuals = False, is_stable_check = False):
        
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
            if is_stable_check:
                return forecast_series, actuals, results.is_stable()
            else:
                return forecast_series, actuals

        
        if is_stable_check:
            return forecast_series, results.is_stable()
        else:
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

    def plot_VAR_results(self, number_of_steps, order, features, target_variable, file_name=''):


        plt.style.use('default')  
        plt.figure(figsize=(12,7), dpi=600)
        # plot the results
        result = []
        plt.plot(self.date, self.dataset[target_variable].values, label="Actual Data", color='red', linewidth=1)

        for step in range(1, number_of_steps):

            errors = {3:[],6:[],12:[],24:[],36:[],48:[],60:[]}
            forecast_results, actuals = self.fit_VAR(order, 60, features, 60*step, target_variable, True)
            for h in errors.keys():
                errors[h] = np.sqrt(np.mean((forecast_results.values[h-1] - actuals.values[h-1]) ** 2))
            result.append(errors)

            time_slice = self.date[-60*step : -60*(step-1) if step > 1 else None]
            plt.plot(
                time_slice, 
                forecast_results, 
                label="Forecasted values", 
                color='blue', 
                linestyle='--', 
                linewidth=2
            )
            plt.axvline(x= self.date.values[-60*step], color='black', linestyle='--', alpha=0.7)
                
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
        plt.ylabel(target_variable, fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=10)
        plt.title("VAR()", fontsize=14, fontweight='bold')

        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
        print(pd.DataFrame(result))

    def select_order(self, target_variable, max_order=20, testing_size=60, max_vars=None, max_diff=2, file_name=""):

        original_diff_order   = self.diff_order
        original_diff_dataset = self.diff_dataset.copy()

        all_cols  = list(self.dataset.columns)
        other_cols = [c for c in all_cols if c != target_variable]

        if max_vars is None:
            max_vars = len(all_cols)

        max_others = max_vars - 1
        results_dict = {"Diff_Order": [], "Variables": [], "Order": [], "AIC": [], "BIC": []}

        for diff_order in range(1, max_diff + 1):

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
                            if result.is_stable():
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

        self.diff_order   = original_diff_order
        self.diff_dataset = original_diff_dataset

        return best_aic_row, best_bic_row, results_df

    def select_order_rmse(self, target_variable, max_order=20, testing_size=60, max_vars=None, max_diff=1, n_windows=8, file_name = ''):
        


        original_diff_order   = self.diff_order
        original_diff_dataset = self.diff_dataset.copy()

        all_cols   = list(self.dataset.columns)
        other_cols = [c for c in all_cols if c != target_variable]

        if max_vars is None:
            max_vars = len(all_cols)

        max_others = max_vars - 1
        results_dict = {"Diff_Order": [], "Variables": [], "Order": [], "RMSE_mean": [], "RMSE_std": []}

        for diff_order in range(1, max_diff + 1):


            self.diff_dataset = self.dataset.copy()
            self.diff_order   = diff_order
            for _ in range(diff_order):
                self.diff_dataset = self.diff_dataset.diff().dropna()

            for subset_size in range(1, max_others + 1):
                for subset in combinations(other_cols, subset_size):

                    features = [target_variable] + list(subset)

                    for order in range(1, max_order + 1):
                        stable_count = 0
                        total_count = 0
                        window_rmses = []

                        for w in range(2, n_windows + 1):
                            window_test_size = testing_size * w

                            # Make sure we have enough training data
                            min_train = order + diff_order + 10
                            if len(self.diff_dataset) - window_test_size < min_train:
                                continue

                            try:
                                forecast, actuals, is_stable = self.fit_VAR(
                                    order          = order,
                                    forecast_steps = 60,
                                    features       = features,
                                    testing_size   = window_test_size,
                                    target_column  = target_variable,
                                    is_actuals     = True,
                                    is_stable_check = True
                                )


                                stable_count += 1 if is_stable else 0   # count stable windows
                                total_count  += 1


                                if is_stable:
                                    rmse = np.sqrt(np.mean((forecast.values - actuals.values) ** 2))
                                    window_rmses.append(rmse) 

                            except Exception as e:
                                print(f"Failed for diff={diff_order}, {features}, order={order}, window={w}: {e}")
                                continue
                        
                        stability_ratio = stable_count / total_count if total_count > 0 else 0

                        if len(window_rmses) == 0 or stability_ratio < 0.8:
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


def VAR_energy():

    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    # create class instance
    cpienergy = VAR_Data(Data.dataset.drop(["CPI(Core)", "CPI(Food)", "Date"], axis=1),Data.dataset["Date"])

    # differentiate data
    cpienergy.differentiate(1) 

    #select best models using AIC BIC
    best_aic_row, best_bic_row, results_df = cpienergy.select_order(target_variable="CPI(Energy)", file_name = '/home/fedor/Downloads/ImagesOverleaf/CPIENERGYVARAICBIC.png')
    #select best model using RMSE
    best_row, results_df = cpienergy.select_order_rmse(target_variable="CPI(Energy)", file_name = '/home/fedor/Downloads/ImagesOverleaf/CPIENERGYVARRMSE.png')

    # make residual correlation, normality and stability of a model
    print(cpienergy.residual_correlation(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 30))
    print(cpienergy.residual_correlation(10, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 30))
    print(cpienergy.residual_correlation(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5', 'GDP'], 30))
    print(cpienergy.residual_correlation(4, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], 30))

    print(cpienergy.residual_normality(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))
    print(cpienergy.residual_normality(10, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))
    print(cpienergy.residual_normality(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5', 'GDP']))
    print(cpienergy.residual_normality(4, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))

    print(cpienergy.check_stability(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))
    print(cpienergy.check_stability(10, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))
    print(cpienergy.check_stability(9, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5', 'GDP']))
    print(cpienergy.check_stability(4, 60, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5']))


    #  plot resutls ofr a final model
    cpienergy.plot_VAR_results(9, 9, ['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], "CPI(Energy)", file_name="/home/fedor/Downloads/ImagesOverleaf/CPIENERGYVARRESULTS.png")
    #plot residual diagnosis for a final model
    cpienergy.plot_residual_diagnostics(order=9, testing_size=60, features=['CPI(Energy)', 'UNRATE', 'IP', 'GS5'], target_column='CPI(Energy)', file_name='/home/fedor/Downloads/ImagesOverleaf/CPIENERGYVARRESIDUALDIAGNOSIS.png')


def VAR_food():


    # path to the file with original dataset
    filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
    # create the instance that will be used to store data and process it 
    Data = Dataset(filepath, sep=';')

    # create class instance
    cpienergy = VAR_Data(Data.dataset.drop(["CPI(Core)", "CPI(Food)", "Date"], axis=1),Data.dataset["Date"])
    cpifood = VAR_Data(Data.dataset.drop(["CPI(Core)", "CPI(Energy)", "Date"], axis=1),Data.dataset["Date"])

    # differentiate data
    cpifood.differentiate(1)


    #select best models using AIC BIC
    best_aic_row, best_bic_row, results_df = cpifood.select_order(target_variable="CPI(Food)", file_name = '/home/fedor/Downloads/ImagesOverleaf/CPIFOODVARAICBIC.png')
    #select best model using RMSE
    best_row, results_df = cpifood.select_order_rmse(target_variable="CPI(Food)", file_name = '/home/fedor/Downloads/ImagesOverleaf/CPIFOODVARRMSE.png')

    # check residual correlation, normality and stabilit of a model
    print(cpifood.residual_correlation(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 30))
    print(cpifood.residual_correlation(10, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 30))
    print(cpifood.residual_correlation(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'GDP'], 30))
    print(cpifood.residual_correlation(4, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], 30))

    print(cpifood.residual_normality(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.residual_normality(10, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.residual_normality(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'GDP']))
    print(cpifood.residual_normality(4, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))

    print(cpifood.check_stability(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.check_stability(10, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.check_stability(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'GDP']))
    print(cpifood.check_stability(4, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.check_stability(7, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.check_stability(8, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5']))
    print(cpifood.check_stability(3, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'GDP']))
    print(cpifood.check_stability(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'PPI', 'M2']))
    print(cpifood.check_stability(9, 60, ['CPI(Food)', 'UNRATE', 'IP', 'GS5', 'PPI']))



    # plot results for a final model
    cpifood.plot_VAR_results(9, 7, ['CPI(Food)', 'UNRATE', 'IP', 'GS5'], "CPI(Food)", file_name="/home/fedor/Downloads/ImagesOverleaf/CPIFOODVARRESULTS.png")


